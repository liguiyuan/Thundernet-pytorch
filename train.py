from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import torch

from torchvision import transforms
from torch.utils.data import DataLoader

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Simple training parameter for training a SNet.')

    parser.add_argument('--train_txt', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--test_txt', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--batch_size',  help='Batch size', type=int, default=128)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--start_epoch', help='start epoch', type=int, default=1)
    parser.add_argument('--gpus', help='Use CUDA on the listed devides', nargs='+', type=int, default=[])
    parser.add_argument('--seed', help='Random seed', type=int, default=1234)
    parser.add_argument('--input_size', help='Image size', type=int, default=128)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def main(args=None):
    
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_set = CocoDataset(txt_path=args.train_txt, transform=transform)
    test_set = CocoDataset(txt_path=args.test_txt, transform=transform)




if __name__ == '__main__':
    args = parse_args()
    main(args)