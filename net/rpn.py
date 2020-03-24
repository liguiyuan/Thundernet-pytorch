import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.append("../")
from utils.bbox_tools import generage_anchor_base

class RPN(nn.Module):
    def __init__(self, anchor_scales=[8, 16, 32], ratios=[0.5, 1, 2], feat_stride=16):
        super(RPN, self).__init__()

        self.anchor_base = generage_anchor_base(anchor_scales=anchor_scales, ratios=ratios)


    def forward(self, x):   # x: CEM output feature (20x20x245)

        n, _, hh, ww = x.shape
        anchor = _enumerate_shifted_anchor(
            np.array(self.anchor_base), self.feat_stride, hh, ww)
        




