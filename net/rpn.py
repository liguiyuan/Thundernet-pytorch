import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from bbox_tools import generate_anchors

class RPN(nn.Module):
    def __init__(self, anchor_scales=[8, 16, 32], ratios=[0.5, 1, 2], feat_stride=16,
        proposal_creator_params=dict()):
        super(RPN, self).__init__()

        self.anchor_base = generate_anchors(ratios=ratios)
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)

    def forward(self, x):   # x: CEM output feature (20x20x245)
        pass
        #n, _, hh, ww = x.shape
        #anchor = _enumerate_shifted_anchor(
        #    np.array(self.anchor_base), self.feat_stride, hh, ww)
      

if __name__ == '__main__':
      rpn = RPN()





