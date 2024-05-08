# Modified from original TextSnake
from typing import Dict, Optional, Sequence
from mmocr.structures import TextDetDataSample

import torch
import torch.nn as nn

from mmocr.registry import MODELS
from .single_stage_text_detector import SingleStageTextDetector


@MODELS.register_module()
class MVTextSnake(SingleStageTextDetector):
    """The class for implementing TextSnake text detector: TextSnake: A
    Flexible Representation for Detecting Text of Arbitrary Shapes.

    [https://arxiv.org/abs/1807.01544]
    """
    def __init__(self,
                 backbone: Dict,
                 det_head: Dict,
                 neck: Optional[Dict] = None,
                 data_preprocessor: Optional[Dict] = None,
                 init_cfg: Optional[Dict] = None,
                 process_multi_view: Optional[Dict] = None):
        super().__init__(backbone, det_head, neck, data_preprocessor, init_cfg)
        if process_multi_view is not None:
            # TODO: 仔细思考这个参数里面应该包含哪些关键字
            self.process_multi_view = True
            input_channel, output_channel = process_multi_view['input_channel'], process_multi_view['output_channel']
            self.fusion = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channel, output_channel, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
            )
        else:
            self.process_multi_view = False
            
    def extract_feat(self, inputs: torch.Tensor, predict: bool = False) -> torch.Tensor:
        """Extract features.

        Args:
            inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            Tensor or tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        if not predict and self.process_multi_view is not None:
            inputs, refimg_1, refimg_2 = torch.split(inputs, inputs.shape[1] // 3, dim=1)

        inputs = self.backbone(inputs)
        if self.with_neck:
            inputs = self.neck(inputs)

        if not predict and self.process_multi_view is not None:
            refimg_1 = self.backbone(refimg_1)
            if self.with_neck:
                refimg_1 = self.neck(refimg_1)

            refimg_2 = self.backbone(refimg_2)
            if self.with_neck:
                refimg_2 = self.neck(refimg_2)

            # ============feature fusion================
            cross_corr1 = torch.sum(inputs * refimg_1, dim=1, keepdim=True)
            cross_corr2 = torch.sum(inputs * refimg_2, dim=1, keepdim=True)
            mv_inputs = torch.cat([inputs, refimg_1, refimg_2, cross_corr1, cross_corr2], dim=1)   # 32 + 32 + 32 + 1 + 1
            inputs = self.fusion(mv_inputs) + inputs

        return inputs

    def predict(self, inputs: torch.Tensor,
                data_samples: Sequence[TextDetDataSample]
                ) -> Sequence[TextDetDataSample]:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (torch.Tensor): Images of shape (N, C, H, W).
            data_samples (list[TextDetDataSample]): A list of N
                datasamples, containing meta information and gold annotations
                for each of the images.

        Returns:
            list[TextDetDataSample]: A list of N datasamples of prediction
            results.  Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - polygons (list[np.ndarray]): The length is num_instances.
                    Each element represents the polygon of the
                    instance, in (xn, yn) order.
        """
        assert inputs.shape[1] == 3, "inputs has wrong shape"
        x = self.extract_feat(inputs, predict=True)
        return self.det_head.predict(x, data_samples)
