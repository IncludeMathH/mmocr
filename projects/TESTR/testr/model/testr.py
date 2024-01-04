# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import SingleStageDetector
import numpy as np


def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)
        labels (torch.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]


@DETECTORS.register_module()
class TESTR(SingleStageDetector):
    def __init__(self, backbone, bbox_head, neck=None, train_cfg=None, test_cfg=None, pretrained=None, init_cfg=None):
        super(TESTR, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained, init_cfg)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_texts=None,
                      gt_polygons=None,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_texts, gt_polygons)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        out_list = []
        for results in results_list:
            new_results = {}
            labels = results['labels']
            if labels.shape[0] == 0:
                new_results['bboxes'] = [np.zeros((0, 5), dtype=np.float32) for i in range(self.bbox_head.num_classes)]
                # new_results['scores'] = [np.zeros((0,), dtype=np.float32) for i in range(self.bbox_head.num_classes)]
                # new_results['recog_scores'] = [np.zeros((0, self.bbox_head.max_text_len), dtype=np.float32) for i in range(self.bbox_head.num_classes)]
                new_results['polygons'] = [np.zeros((0, 16, 2), dtype=np.float32) for i in
                                           range(self.bbox_head.num_classes)]
                new_results['texts'] = [np.zeros((0, self.bbox_head.max_text_len, 2), dtype=np.float32) for i in
                                        range(self.bbox_head.num_classes)]
            else:
                labels = labels.detach().cpu().numpy()
                for k, v in results.items():
                    if k == 'labels':
                        continue
                    v = v.detach().cpu().numpy()
                    new_results[k] = [v[labels == i, ...] for i in range(self.bbox_head.num_classes)]
            out_list.append(new_results)
        return out_list
