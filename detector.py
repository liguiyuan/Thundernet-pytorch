from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.jit.annotations import Tuple, List, Dict, Optional

from thundernet.snet import SNet49
from thundernet.module import CEM, SAM, RCNNSubNetHead, ThunderNetPredictor

from src.roi_layers.ps_roi_align import PSRoIAlign
from src.bbox_tools import generate_anchors
from src.rpn import AnchorGenerator
from src.rpn import RegionProposalNetwork
from src.rpn import RPNHead
from src.roi_layers.poolers import MultiScaleRoIAlign
from src.roi_heads import RoIHeads
from src.transform import GeneralizedRCNNTransform

from collections import OrderedDict
import warnings


class DetectNet(nn.Module):
    def __init__(self, backbone, num_classes=None,
        # RPN parameters
        rpn_anchor_generator=None, rpn_head=None,
        rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=100,
        rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,

        rpn_mns_thresh=0.7,
        rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,

        # Box parameters
        box_ps_roi_align=None, box_head=None, box_predictor=None,
        box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
        box_fg_iou_thresh=0.5,box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512, box_positive_fraction=0.25,
        bbox_reg_weights=None):
        super(DetectNet, self).__init__()
        
        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_ps_roi_align, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        out_channels = backbone.out_channels    # 245
        
        self.backbone = backbone

        self.cem = CEM()     # CEM module
        self.sam = SAM()     # SAM module

        # rpn
        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
        
        if rpn_head is None:
            rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        self.rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_mns_thresh)


        # ps roi align
        if box_ps_roi_align is None:
            box_ps_roi_align = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2)

        # R-CNN subnet
        if box_head is None:
            resolution = box_ps_roi_align.output_size[0]    # size: (7, 7)
            representation_size = 1024
            box_out_channels = 5
            box_head = RCNNSubNetHead(
                box_out_channels * resolution ** 2,         # 5 * 7 * 7
                representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = ThunderNetPredictor(representation_size, num_classes)

        self.roi_heads = RoIHeads(
            box_ps_roi_align, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

        self.transform = GeneralizedRCNNTransform()


    def forward(self, images, targets2=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]])
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """

        targets = []
        t2 = {}
        for t in targets2:
            t2["boxes"] = t[:, 0:4]
            t2["labels"] = t[:, 4]
            targets.append(t2)

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))   # (h, w)


        # backbone
        _, c4_feature, c5_feature = self.backbone(images)
        images, targets = self.transform(images, targets)

        # cem
        cem_feature = self.cem(c4_feature, c5_feature)
        cem_feature_output = cem_feature

        if isinstance(cem_feature, torch.Tensor):
            cem_feature = OrderedDict([('0', cem_feature)])
        
        # rpn
        proposals, proposal_losses, rpn_output = self.rpn(images, cem_feature, targets)

        # sam
        sam_feature = self.sam(rpn_output, cem_feature_output)
        
        if isinstance(sam_feature, torch.Tensor):
            sam_feature = OrderedDict([('0', sam_feature)])

        detections, detector_losses = self.roi_heads(sam_feature, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        return detector_losses, proposal_losses

def ThunderNet():
    snet = SNet49()
    snet.out_channels = 245
    thundernet = DetectNet(snet, num_classes=80)

    return thundernet


#if __name__ == '__main__':
#    thundernet = ThunderNet()
#    print('thundernet: ', thundernet)
    
