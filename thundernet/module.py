import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CEM(nn.Module):
    def __init__(self):
        super(CEM, self).__init__()
        self.conv1 = nn.Conv2d(120, 245, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(512, 245, kernel_size=1, stride=1, padding=0)
        self.avg_pool = nn.AvgPool2d(10)
        self.conv3 = nn.Conv2d(512, 245, kernel_size=1, stride=1, padding=0)

    def forward(self, c4_feature, c5_feature):
        c4 = c4_feature
        c4_lat = self.conv1(c4)             # output: [245, 20, 20]

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