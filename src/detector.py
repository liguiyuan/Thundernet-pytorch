from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from thundernet.snet import SNet49
from src.roi_layers.ps_roi_align import PSRoIAlign
from src.bbox_tools import generate_anchors

from src.rpn import AnchorGenerator
from src.rpn import RegionProposalNetwork
from src.rpn import RPNHead
from src.roi_layers.poolers import MultiScaleRoIAlign
from src.generalized_rcnn import GeneralizedRCNN

#from torchvision.models.detection.roi_heads import RoIHeads
from src.roi_heads import RoIHeads
#from torchvision.models.detection.transform import GeneralizedRCNNTransform
from src.transform import GeneralizedRCNNTransform

class CEM(nn.Module):
    def __init__(self):
        super(CEM, self).__init__()
        self.conv1 = nn.Conv2d(120, 245, kernel_size=1, stride=1, padding=0)

        self.conv2 = nn.Conv2d(512, 245, kernel_size=1, stride=1, padding=0)

        self.avg_pool = nn.AvgPool2d(10)
        self.conv3 = nn.Conv2d(512, 245, kernel_size=1, stride=1, padding=0)

    def forward(self, c4_feature, c5_feature):
        # c4
        c4 = c4_feature
        c4_lat = self.conv1(c4)             # output: [245, 20, 20]

        # c5
        c5 = c5_feature
        c5_lat = self.conv2(c5)             # output: [245, 10, 10]
        # upsample x2
        c5_lat = F.interpolate(input=c5_lat, size=[20, 20], mode="nearest") # output: [245, 20, 20]

        c_glb = self.avg_pool(c5)           # output: [512, 1, 1]
        c_glb_lat = self.conv3(c_glb)       # output: [245, 1, 1]

        out = c4_lat + c5_lat + c_glb_lat   # output: [245, 20, 20]
        return out

class SAM(nn.Module):
    def __init__(self):
        super(SAM, self).__init__()
        self.conv = nn.Conv2d(256, 245, 1, 1, 0, bias=False) # input channel = 245 ?
        self.bn = nn.BatchNorm2d(245)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, rpn_feature, cem_feature):
        cem = cem_feature      # feature map of CEM: [245, 20, 20]
        rpn = rpn_feature      # feature map of RPN: [256, 20, 20]

        sam = self.conv(rpn)
        sam = self.bn(sam)
        sam = self.sigmoid(sam)
        out = cem * sam     # output: [245, 20, 20]

        return out


class DetectNet(GeneralizedRCNN):
    def __init__(self, backbone, num_classes=None,
        # transform parameters
        min_size=800, max_size=1333,
        image_mean=None, image_std=None,
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

        # CEM module
        cem = CEM() 

        # SAM module
        sam = SAM()

        # rpn
        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
        
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_mns_thresh)


        # ps roi align
        if box_ps_roi_align is None:
            box_ps_roi_align = MultiScaleRoIAlign(          # ps roi align
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
            box_predictor = ThunderNetPredictor(
                representation_size,
                num_classes)

        roi_heads = RoIHeads(
            # Box
            box_ps_roi_align, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)


        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(DetectNet, self).__init__(backbone, cem, sam, rpn, roi_heads, transform)

class RCNNSubNetHead(nn.Module):
    """
    Standard heads for FPN-based models
    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(RCNNSubNetHead, self).__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)  # in_channles: 7*7*5=245  representation_size:1024

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))

        return x
  
class ThunderNetPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.
    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """
    def __init__(self, in_channels, num_classes):
        super(ThunderNetPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):       # x: [1024, 1, 1]
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas
        

def ThunderNet():
    snet = SNet49()
    snet.out_channels = 245
    thundernet = DetectNet(snet, num_classes=80)

    return thundernet


#if __name__ == '__main__':
#    thundernet = ThunderNet()
#    print('thundernet: ', thundernet)
    
