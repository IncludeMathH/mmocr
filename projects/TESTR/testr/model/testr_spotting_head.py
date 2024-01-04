# Copyright (c) OpenMMLab. All rights reserved.
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob, constant_init
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.builder import HEADS
import numpy as np
from mmcv.ops.nms import batched_nms

from .matcher import build_matcher
from .losses import SetCriterion
from .deformable_transformer import DeformableTransformer
from .pos_encoding import PositionalEncoding1D, PositionalEncoding2D
from .misc import inverse_sigmoid_offset, sigmoid_offset, box_xyxy_to_cxcywh


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@HEADS.register_module()
class TESTRSpottingHead(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout,
                 num_feature_levels,
                 dec_n_points, enc_n_points, num_proposals, pos_embed_scale, num_ctrl_points, num_classes, max_text_len,
                 voc_size, use_polygon, aux_loss, loss_cfg, test_score_threshold, train_cfg=None, test_cfg=None,
                 nms=None):
        super(TESTRSpottingHead, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.num_feature_levels = num_feature_levels
        self.dec_n_points = dec_n_points
        self.enc_n_points = enc_n_points
        self.num_proposals = num_proposals
        self.pos_embed_scale = pos_embed_scale
        self.num_ctrl_points = num_ctrl_points
        self.num_classes = num_classes
        self.max_text_len = max_text_len
        self.voc_size = voc_size
        self.sigmoid_offset = not use_polygon
        self.use_polygon = use_polygon
        self.activation = "relu"
        self.return_intermediate_dec = True
        self.aux_loss = aux_loss
        self.loss_cfg = loss_cfg
        self.test_score_threshold = test_score_threshold
        self.nms = nms

        self.box_matcher, self.point_matcher = build_matcher(loss_cfg)
        enc_losses = ['labels', 'boxes']
        dec_losses = ['labels', 'ctrl_points', 'texts']
        self.criterion = SetCriterion(num_classes, num_decoder_layers, self.box_matcher, self.point_matcher, enc_losses,
                                      dec_losses, num_ctrl_points, loss_cfg)
        N_steps = self.d_model // 2
        self.positional_encoding = PositionalEncoding2D(N_steps, normalize=True)

        self.text_pos_embed = PositionalEncoding1D(self.d_model, normalize=True, scale=self.pos_embed_scale)
        self.transformer = DeformableTransformer(
            d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers, dim_feedforward=self.dim_feedforward,
            dropout=self.dropout, activation=self.activation, return_intermediate_dec=self.return_intermediate_dec,
            num_feature_levels=self.num_feature_levels, dec_n_points=self.dec_n_points,
            enc_n_points=self.enc_n_points, num_proposals=self.num_proposals,
        )
        self.ctrl_point_class = nn.Linear(self.d_model, self.num_classes)
        self.ctrl_point_coord = MLP(self.d_model, self.d_model, 2, 3)
        self.bbox_coord = MLP(self.d_model, self.d_model, 4, 3)
        self.bbox_class = nn.Linear(self.d_model, self.num_classes)
        self.text_class = nn.Linear(self.d_model, self.voc_size + 1)

        # shared prior between instances (objects)
        self.ctrl_point_embed = nn.Embedding(self.num_ctrl_points, self.d_model)
        self.text_embed = nn.Embedding(self.max_text_len, self.d_model)

        if self.num_feature_levels > 1:
            strides = [8, 16, 32]
            num_channels = [512, 1024, 2048]
            num_backbone_outs = len(strides)
            input_proj_list = []
            for i in range(num_backbone_outs):
                in_channels = num_channels[i]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.d_model, kernel_size=1),
                    nn.GroupNorm(32, self.d_model),
                ))
            for _ in range(self.num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.d_model,
                              kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, self.d_model),
                ))
                in_channels = self.d_model
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            strides = [32]
            num_channels = [2048]
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(
                        num_channels[0], self.d_model, kernel_size=1),
                    nn.GroupNorm(32, self.d_model),
                )])

        self._init_weights()

        num_pred = self.num_decoder_layers
        self.ctrl_point_class = nn.ModuleList(
            [self.ctrl_point_class for _ in range(num_pred)])
        self.ctrl_point_coord = nn.ModuleList(
            [self.ctrl_point_coord for _ in range(num_pred)])

    def _init_weights(self):
        prior_prob = 0.01
        bias_value = -np.log((1 - prior_prob) / prior_prob)
        self.ctrl_point_class.bias.data = torch.ones(self.num_classes) * bias_value
        self.bbox_class.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.ctrl_point_coord.layers[-1].weight.data, 0)
        nn.init.constant_(self.ctrl_point_coord.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        nn.init.constant_(self.bbox_coord.layers[-1].bias.data[2:], 0.0)
        self.transformer.bbox_class_embed = self.bbox_class
        self.transformer.bbox_embed = self.bbox_coord

    def forward(self, mlvl_feats, img_metas):
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        mlvl_srcs = []
        for l, feat in enumerate(mlvl_feats):
            mlvl_srcs.append(self.input_proj[l](feat))
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_srcs[-1], mlvl_masks[-1]).to(feat.dtype))

        if self.num_feature_levels > len(mlvl_srcs):
            _len_srcs = len(mlvl_srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](mlvl_feats[-1])
                else:
                    src = self.input_proj[l](mlvl_srcs[-1])
                mask = F.interpolate(
                    img_masks[None], size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.positional_encoding(src, mask).to(src.dtype)
                mlvl_srcs.append(src)
                mlvl_masks.append(mask)
                mlvl_positional_encodings.append(pos_l)

        # n_points, embed_dim --> n_objects, n_points, embed_dim
        ctrl_point_embed = self.ctrl_point_embed.weight[None, ...].repeat(self.num_proposals, 1, 1)
        text_pos_embed = self.text_pos_embed(self.text_embed.weight)[None, ...].repeat(self.num_proposals, 1, 1)
        text_embed = self.text_embed.weight[None, ...].repeat(self.num_proposals, 1, 1)

        hs, hs_text, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(
            mlvl_srcs, mlvl_masks, mlvl_positional_encodings, ctrl_point_embed, text_embed, text_pos_embed,
            text_mask=None)

        outputs_classes = []
        outputs_coords = []
        outputs_texts = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid_offset(reference, offset=self.sigmoid_offset)
            outputs_class = self.ctrl_point_class[lvl](hs[lvl])
            tmp = self.ctrl_point_coord[lvl](hs[lvl])
            if reference.shape[-1] == 2:
                tmp += reference[:, :, None, :]
            else:
                assert reference.shape[-1] == 4
                tmp += reference[:, :, None, :2]
            outputs_texts.append(self.text_class(hs_text[lvl]))
            outputs_coord = sigmoid_offset(tmp, offset=self.sigmoid_offset)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_text = torch.stack(outputs_texts)

        out = {'pred_logits': outputs_class[-1],
               'pred_ctrl_points': outputs_coord[-1],
               'pred_texts': outputs_text[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord, outputs_text)

        enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
        out['enc_outputs'] = {
            'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out

    def forward_train(self, x, img_metas, gt_bboxes, gt_labels, gt_texts, gt_polygons):
        output = self(x, img_metas)
        loss = self.loss(output, gt_bboxes, gt_labels, gt_texts, gt_polygons, img_metas)
        return loss

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_text):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_ctrl_points': b, 'pred_texts': c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_text[:-1])]

    def prepare_targets(self, gt_bboxes_list, gt_labels_list, gt_text_list, gt_ctrl_points_list, img_metas):
        targets = []
        for i in range(len(gt_bboxes_list)):
            h, w, _ = img_metas[i]['img_shape']
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
            gt_boxes = gt_bboxes_list[i] / image_size_xyxy.type_as(gt_bboxes_list[i])
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            raw_ctrl_points = gt_ctrl_points_list[i]
            gt_ctrl_points = raw_ctrl_points.reshape(-1, self.num_ctrl_points, 2) / torch.as_tensor([w, h],
                                                                                                    dtype=torch.float)[
                                                                                    None, None, :].type_as(
                gt_bboxes_list[i])
            targets.append({'labels': gt_labels_list[i], 'boxes': gt_boxes, 'ctrl_points': gt_ctrl_points,
                            'texts': gt_text_list[i]})
        return targets

    def loss(self, output, gt_bboxes_list, gt_labels_list, gt_text_list, gt_ctrl_points_list, img_metas):

        targets = self.prepare_targets(gt_bboxes_list, gt_labels_list, gt_text_list, gt_ctrl_points_list, img_metas)
        loss_dict = self.criterion(output, targets)
        weight_dict = self.criterion.weight_dict
        for k in loss_dict.keys():
            if k in weight_dict:
                loss_dict[k] *= weight_dict[k]
        return loss_dict

    def simple_test(self, feats, img_metas, rescale=False):
        """Test det bboxes without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """
        # forward of this head requires img_metas
        outs = self.forward(feats, img_metas)
        results_list = self.get_bboxes(outs, img_metas, rescale=rescale)
        return results_list

    def get_bboxes(self, output, img_metas, rescale):
        ctrl_point_cls = output["pred_logits"]
        ctrl_point_coord = output["pred_ctrl_points"]
        text_pred = output["pred_texts"]

        results = []

        text_pred = torch.softmax(text_pred, dim=-1)
        # print(text_pred.shape)
        prob = ctrl_point_cls.mean(-2).sigmoid()
        scores, labels = prob.max(-1)

        for scores_per_image, labels_per_image, ctrl_point_per_image, text_per_image, img_meta in zip(
                scores, labels, ctrl_point_coord, text_pred, img_metas
        ):
            selector = scores_per_image >= self.test_score_threshold
            scores_per_image = scores_per_image[selector]
            labels_per_image = labels_per_image[selector]
            ctrl_point_per_image = ctrl_point_per_image[selector]
            text_per_image = text_per_image[selector]
            image_size = img_meta['img_shape']
            # print(image_size, img_meta['scale_factor'])
            # scale_factor = img_meta['scale_factor']
            result = {}
            # result['scores'] = scores_per_image
            if scores_per_image.shape[0] == 0:
                result['labels'] = labels_per_image
                result['bboxes'] = ctrl_point_per_image.new_zeros((0, 5))
                result['polygons'] = ctrl_point_per_image.new_zeros((0, self.num_ctrl_points, 2))
                result['texts'] = text_per_image.new_zeros((0, self.max_text_len, 2))
            else:
                result['labels'] = labels_per_image
                text_scores, _ = text_per_image.max(-1)
                ctrl_point_per_image[..., 0] *= image_size[1]  # / img_meta['scale_factor'][1]
                ctrl_point_per_image[..., 1] *= image_size[0]  # / img_meta['scale_factor'][0]
                # ctrl_point_per_image /= img_meta['scale_factor']
                x1y1, _ = ctrl_point_per_image.min(1)
                x2y2, _ = ctrl_point_per_image.max(1)
                result['bboxes'] = torch.cat([x1y1, x2y2, scores_per_image.unsqueeze(-1)], dim=-1)
                if self.use_polygon:
                    result['polygons'] = ctrl_point_per_image.flatten(1)
                else:
                    result['beziers'] = ctrl_point_per_image.flatten(1)
                _, topi = text_per_image.topk(1)
                # print(topi.shape, text_scores.shape)
                result['texts'] = torch.cat([topi.float(), text_scores.unsqueeze(-1)], dim=-1)

                if self.nms is not None:
                    _, keep = batched_nms(result['bboxes'][:, :4], result['bboxes'][:, -1], result['labels'], self.nms)
                    for k, v in result.items():
                        result[k] = v[keep]
                # print(result['texts'].shape)
            results.append(result)
        return results
