import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO
class CEM(nn.Module):
    def __init__(self, arg):
        super(CEM, self).__init__()
        self.conv4_lat = nn.Conv2d(245, 245, kernel_size=1, stride=1, padding=1)

        # self.conv5 = stage(4)
        self.conv5_lat = nn.Conv2d(245, 245, kernel_size=1, stride=1, padding=1)
        #self.conv5_upsample = 

        self.avg_pool = nn.AvgPool2d(10)
        self.conv_glb = nn.Conv2d(245, 245, kernel_size=1, stride=1, padding=1)
        # self.broadcast = 



        slef.c4_lat

    def forward(slef, input):
        c4 = x
        c4_lat = self.conv_lat(c4)

        c5_lat = self.conv()
        
        c_glb = self.avgpool()
        c_glb_lat = self.conv_glb(c_glb)

        out = c4_lat + c5_lat + cglb_lat

        return out



class SAM(nn.Module):
    def __init__(self):
        super(SAM, self).__init__()
        self.conv1 = nn.Conv2d(245, 245, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(245)
        
    def forward(self, input):
        cem = input[0]
        rpn = input[1]

        sam = slef.conv1(rpn)
        sam = self.bn(sam)
        sam = F.sigmoid(sam)
        out = cem * sam

        return out


