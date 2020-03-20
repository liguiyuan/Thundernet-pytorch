import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup // 2

        if self.benchmodel == 1:    # basic unit
            self.branch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 5, stride, padding=2, groups=oup_inc, bias=False),  # padding=2
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
        else:   # down sample (2x)
            self.branch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 5, stride, padding=2, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

            self.branch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 5, stride, padding=2, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if 1 == self.benchmodel:
            x1 = x[:, :(x.shape[1]//2), :, :]
            x2 = x[:, (x.shape[1]//2):, :, :]
            out = self._concat(x1, self.branch2(x2))
        elif 2 == self.benchmodel:
            out = self._concat(self.branch1(x), self.branch2(x))

        return channel_shuffle(out, 2)

class ShuffleNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224):
        super(ShuffleNetV2, self).__init__()
        assert input_size % 32 == 0

        self.stage_repeats = [4, 8, 4]
        
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        self.stage_out_channels = [-1, 24, 60, 120, 240, 512]

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(3, input_channel, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        self.features1 = []
        self.features2 = []
        self.features3 = []

        """
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]
            for i in range(numrepeat):
                if i == 0:
                    # (inp, oup, stride, benchmodel)
                    self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel

        """
        # stage2
        numrepeat = self.stage_repeats[0]
        output_channel = self.stage_out_channels[2]
        for i in range(numrepeat):
            if i == 0:
                # (inp, oup, stride, benchmodel)
                self.features1.append(InvertedResidual(input_channel, output_channel, 2, 2)) 
            else:
               self.features1.append(InvertedResidual(input_channel, output_channel, 1, 1)) 
            input_channel = output_channel

        # stage3
        numrepeat = self.stage_repeats[1]
        output_channel = self.stage_out_channels[3]
        for i in range(numrepeat):
            if i == 0:
                # (inp, oup, stride, benchmodel)
                self.features2.append(InvertedResidual(input_channel, output_channel, 2, 2)) 
            else:
               self.features2.append(InvertedResidual(input_channel, output_channel, 1, 1)) 
            input_channel = output_channel

        # stage4
        numrepeat = self.stage_repeats[2]
        output_channel = self.stage_out_channels[4]
        for i in range(numrepeat):
            if i == 0:
                # (inp, oup, stride, benchmodel)
                self.features3.append(InvertedResidual(input_channel, output_channel, 2, 2)) 
            else:
               self.features3.append(InvertedResidual(input_channel, output_channel, 1, 1)) 
            input_channel = output_channel
        

        # make it nn.Sequential
        #self.features = nn.Sequential(*self.features)
        self.features1 = nn.Sequential(*self.features1)
        self.features2 = nn.Sequential(*self.features2)
        self.features3 = nn.Sequential(*self.features3)

        # building last several layers
        self.conv5 = conv_1x1_bn(input_channel, self.stage_out_channels[-1])
        self.globalpool = nn.Sequential(nn.AvgPool2d(int(input_size/32)))

        # building classifier
        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class))

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        #x = self.features(x)    # stage2, stage3, stage4

        x = self.features1(x)       # stage2
        x = self.features2(x)    # stage3
        out_c4 = x
        x = self.features3(x)    # stage4

        x = self.conv5(x)
        out_c5 = x
        x = self.globalpool(x)
        x = x.view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x, out_c4, out_c5


class CEM(nn.Module):
    def __init__(self):
        super(CEM, self).__init__()
        self.conv4 = nn.Conv2d(120, 245, kernel_size=1, stride=1, padding=1)

        self.conv5 = nn.Conv2d(512, 245, kernel_size=1, stride=1, padding=1)
        #self.conv5_upsample = nn.Upsample((10, 10), (2, 2), 'bilinear')

        self.avg_pool = nn.AvgPool2d(10)
        self.conv_glb = nn.Conv2d(512, 245, kernel_size=1, stride=1, padding=1)
        # self.broadcast = 

    def forward(slef, inputs):
        c4 = inputs[0]  # stage3 output feature map
        c4_lat = self.conv4(c4)

        c5 = inputs[1]  # conv5 output feature map
        c5_lat = self.conv5(c5)
        c5_lat = F.interpolate(input=c5_lat, scale_factor=(2, 2), mode='bilinear')

        c_glb = self.avg_pool(c5)
        c_glb_lat = self.conv_glb(c_glb)

        out = c4_lat + c5_lat + c_glb_lat

        return out

class SAM(nn.Module):
    def __init__(self):
        super(SAM, self).__init__()
        self.conv1 = nn.Conv2d(245, 245, 1, 1, 0, bias=False) # input channel = 245 ?
        self.bn = nn.BatchNorm2d(245)
        
    def forward(self, input):
        cem = input[0]      # feature map of CEM
        rpn = input[1]      # feature map of RPN

        sam = slef.conv1(rpn)
        sam = self.bn(sam)
        sam = F.sigmoid(sam)
        out = cem * sam

        return out

class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()
        self.arg = arg

        self.dw5_5 = nn.Conv2d(in_channel, in_channel, kernel_size=5, stride=1, padding=2, groups=in_channel)
        self.conv1 = nn.Conv2d(in_channel, 256, 1, 1, 0)
        #self.bn1 = nn.BatchNorm2d(in_channel)



    def forward(self, x):
        x = self.dw5_5()
        out1 = self.conv1(x)

        
        

def Snet():
    snet, out1, out2 = ShuffleNetV2()

    cem_input = out1, out2
    f_cem = CEM(cem_input)

    return snet

        
if __name__ == '__main__':
    img = torch.randn(1, 3, 224, 224)

    snet = ShuffleNetV2()
    feature, out1, out2 = snet(img)
    #sprint(snet)
        
