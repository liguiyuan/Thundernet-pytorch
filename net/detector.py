from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Snet49 import ShuffleNetV2
from roi_layers.ps_roi_align import PSRoIAlign
from bbox_tools import generate_anchors

from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.rpn import RegionProposalNetwork
from torchvision.models.detection.rpn import RPNHead

class CEM(nn.Module):
    def __init__(self):
        super(CEM, self).__init__()
        self.conv1 = nn.Conv2d(120, 245, kernel_size=1, stride=1, padding=0)

        self.conv2 = nn.Conv2d(512, 245, kernel_size=1, stride=1, padding=0)

        self.avg_pool = nn.AvgPool2d(10)
        self.conv3 = nn.Conv2d(512, 245, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        # c4
        c4 = inputs[0]
        c4_lat = self.conv1(c4)             # output: [245, 20, 20]

        # c5
        c5 = inputs[1]
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
        
    def forward(self, input):
        cem = input[0]      # feature map of CEM: [245, 20, 20]
        rpn = input[1]      # feature map of RPN: [256, 20, 20]

        sam = self.conv(rpn)
        sam = self.bn(sam)
        sam = self.sigmoid(sam)
        out = cem * sam     # output: [245, 20, 20]

        return out

class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()
        # RPN
        self.dw5x5 = nn.Conv2d(245, 245, kernel_size=5, stride=1, padding=2, groups=245, bias=False)
        self.bn0 = nn.BatchNorm2d(245)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(245, 256, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(256)
        #self.conv2 = nn.Conv2d(num_anchors, (1, 1))         # class
        #self.conv3 = nn.Conv2d(num_anchors * 4, (1, 1))     # region

        anchor_generator = generate_anchors()

    def forward(self, x):   # x: CEM output feature (20x20x245)
        # RPN
        x = self.dw5x5(x)   # output: [245, 20, 20]
        x = self.bn0(x)
        x = self.relu(x)
        x = self.conv1(x)   # output: [256, 20, 20]
        x = self.bn1(x)
        x = self.relu(x)
        return x

class RCNN_Subnet(nn.Module):
    def __init__(self, nb_classes):
        super(RCNN_Subnet, self).__init__()
        self.linear = nn.Linear(245, 1024)       # fc

        # classification
        self.linear_cls = nn.Linear(1024, nb_classes)
        self.softmax = nn.Softmax(dim=0) 

        # localization
        self.linear_reg = nn.Linear(1024, 4 * (nb_classes - 1))

    def forward(self, x):       # x: 7x7x5 
        x = torch.flatten(x)
        out = self.linear(x)                    # output: [1, 1024]

        # classification
        out_score = self.linear_cls(out)        # output: [nb_classes]
        out_class = self.softmax(out_score)

        # localization
        out_regressor = self.linear_reg(out)

        return [out_class, out_regressor]              
        

def detecter():
    img = torch.randn(1, 3, 320, 320)

    snet = ShuffleNetV2()
    snet_feature, c4_feature, c5_feature = snet(img)

    cem = CEM()
    cem_input = [c4_feature, c5_feature] # c4: [120, 20, 20]  c5: [512, 10, 10]
    cem_output = cem(cem_input)          # output: [245, 20, 20]     

    rpn = RPN()
    rpn_output = rpn(cem_output)            # output: [256, 20, 20]

    sam = SAM()
    sam_input = [cem_output, rpn_output]
    sam_output = sam(sam_input)             # output: [245, 20, 20]

    # PS ROI Align
    roi_regions = 7
    # (Tensor[K, 5] or List[Tensor[L, 4]]): the box coordinates in (x1, y1, x2, y2)
    ps_roi_align = PSRoIAlign(output_size=[roi_regions, roi_regions], spatial_scale=1.0, sampling_ratio=-1)
    #ps_roi_align_output = ps_roi_align(input=sam_output, rois=input_rois)

    
    feature_roi = torch.randn(1, 5, 7, 7)
    nb_classes = 80
    rcnn = RCNN_Subnet(nb_classes)
    #rcnn_output = rcnn(ps_roi_align_output)
    rcnn_output = rcnn(feature_roi)
    

class ThunderNet(nn.Module):
    def __init__(self, backbone,
        # RPN parameters
        rpn_anchor_generator=None, rpn_head=None,
        rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=100,
        rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,

        rpn_mns_thresh=0.7,
        rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,

        ):
        super(ThunderNet, self).__init__()

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        
        out_channels = backbone.out_channels

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

if __name__ == '__main__':
    #detecter()

    snet = ShuffleNetV2()
    snet.out_channels = 1024


    thundernet = ThunderNet(snet)
    
