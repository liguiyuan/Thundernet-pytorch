import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from bbox_tools import generate_anchors
import _utils as det_utils

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
      

class RegionProposalNetwork(nn.Module):
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
        'pre_nms_top_n': Dict[str, int],
        'post_nms_top_n': Dict[str, int],
    }

    def __init__(self, 
                 anchor_generator, 
                 head, 
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 pre_nms_top_n, pos_nms_top_n, nms_thresh):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # used during training
        self.box_similarity = box_ops.box_iou

        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=True,
        )

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction
        )
        # used during testing
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = 1e-3

    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']


        


if __name__ == '__main__':
      rpn = RegionProposalNetwork()






