from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn


class PSRoiAlignPooling(nn.Module):
    def __init__(self, pool_size, num_rois, alpha, **kwargs):
        super(PSRoiAlignPooling, self).__init__()
        self.pool_size = pool_size
        self.num_rois = num_rois    # number of rois
        self.alpha_channels = alpha

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]    # ?

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.alpha_channels

    def forward(self, x, mask=None):
        assert(len(x) == 2)
        total_bins = 1
        img = x[0]      # x[0] is image shape(rows, cols, channels)

        rois = x[1]     # x[1] is roi shape(num_rois, 4) with ordering (x, y, w, h)
 
        bin_crop_size = []
        for num_bins, crop_dim in zip((7, 7), (14, 14)):
            assert num_bins >= 1
            assert crop_dim % num_bins == 0
            total_bins *= num_bins
            bin_crop_size.append(crop_dim // num_bins)

         xmin, ymin, xmax, ymax = torch.unbind(rois[0], dim=1)
         spatial_bins_y =  spatial_bins_x = 7
         step_y = (ymax - ymin) / spatial_bins_y
         step_x = (xmax - xmin) / spatial_bins_x

         # gen bins
         position_sensitive_boxes = []
         for bin_x in range(self.pool_size):
            for bin_y in range(self.pool_size):
                box_coordinates = [
                    ymin + bin_y * step_y,
                    xmin + bin_x * step_x,
                    ymin + (bin_y + 1) * step_y,
                    xmin + (bin_x + 1) * step_x 
                ]
                position_sensitive_boxes.append(torch.stack(box_coordinates, dim=1))

        img_splits = torch.split(img, total_bins, dim=3)
        box_image_indices = np.zeros(self.num_rois)

        feature_crops = []
        for split, box in zip(img_splits, position_sensitive_boxes):
            crop = CropAndResizeFunction.apply(split, box, box_image_indices, bin_crop_size[0], bin_crop_size[1], 0)

            crop_1 = torch.max(crop, dim=1, keepdim=False, out=None)
            crop_2 = torch.max(crop, dim=2, keepdim=False, out=None)
            crop = torch.stack(crop_1, crop_2)
            crop = crop.unsqueeze(1)

            feature_crops.append(crop)

        final_output = torch.cat(feature_crops, dim=1)

        final_output = final_output.reshape(1, self.num_rois, self.pool_size, self.pool_size, self.alpha_channels)

        final_output = final_output.permute(0, 1, 2, 3, 4)

        return final_output



class RPN(nn.Module):
    def __init__(self, in_channels, num_anchors, nb_classes, in_channels2):
        super(RPN, self).__init__()
        #rpn part

        # SAM module
        #self.conv1 = Conv_1x1(in_channels=in_channels, out_channels=245, strides=1, groups=1)
        self.conv1 = nn.Conv2d(245, 245, 1, 1, 0, bias=False) # input channel = 245 ?
        self.bn1 = nn.BatchNorm2d(245)

        # RPN
        self.dw5x5 = nn.Conv2d(245, 245, kernel_size=5, stride=1, padding=1, groups=245)
        self.conv1_1 = nn.Conv2d(245, 256, kernel_size=1, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_anchors, (1, 1))         # class
        self.conv3 = nn.Conv2d(num_anchors * 4, (1, 1))     # region

        # classifier part
        self.dropout = nn.Dropout(0.5)
        self.batchnorm = nn.BatchNorm2d()
        self.linear = nn.Linear(1024)
        self.linear_cls = nn.Linear(nb_classes)
        self.softmax = nn.Softmax()
        self.linear_reg = nn.Linear(4 * (nb_classes - 1))

    def forward(self, x):   # x: CEM output feature (20x20x245)
        # RPN
        x = self.dw5x5(x)
        x = self.conv1_1(x)
        x_class = F.sigmoid(self.conv2(x))
        x_regr = self.conv3(x)

        return [x_class, x_regr]

    def classifier(self, input, input_rois, num_rois, nb_classes=3):
        # SAM module
        feature_cem = input[0]      # feature map of CEM: 20x20x5
        feature_rpn = input[1]      # feature map of RPN

        rpn = self.conv1(rpn)
        rpn = self.bn1(rpn)
        rpn = F.sigmoid(rpn)
        sam_output = feature_cem * rpn
        
        # PSRoI align
        pooling_regions = 7
        alpha = 5
        out_roi_pool = PSRoiAlignPooling(pooling_regions, num_rois, alpha)([sam_output, input_rois])

        # fc
        out = torch.flatten(out_roi_pool)   
        out = self.linear(out)                  # output 1-dim: 1024
        #out = self.dropout(out)                # dropout

        out_score = self.linear_cls(out)        # 
        out_class = self.softmax(out_score)     # classification

        out_regr = self.linear_reg(out)         # localization

        return [out_class, out_regr]

        

