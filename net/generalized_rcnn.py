# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

from collections import OrderedDict
import torch
from torch import nn
import warnings
from torch.jit.annotations import Tuple, List, Dict, Optional
from torch import Tensor


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.
    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, cem, sam, rpn, roi_heads, transform):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.cem = cem
        self.sam = sam
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
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
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)
        _, c4_feature, c5_feature = self.backbone(images.tensors)

        cem_feature = self.cem(c4_feature, c5_feature)
        print('cem_feature shape: ', cem_feature.shape)

        if isinstance(cem_feature, torch.Tensor):
            cem_feature = OrderedDict([('0', cem_feature)])
        proposals, proposal_losses, rpn_output = self.rpn(images, cem_feature, targets)
        sam_feature = self.sam(rpn_output, cem_feature)

        detections, detector_losses = self.roi_heads(sam_feature, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return (losses, detections)
        else:
            return self.eager_outputs(losses, detections)