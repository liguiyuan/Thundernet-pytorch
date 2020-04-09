from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import argparse
import torch

from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim

from net.detector import ThunderNet
from load_data import CocoDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()
print('device available: {}'.format(device))

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Simple training parameter for training a SNet.')

    parser.add_argument('--data_path', type=str, default='data/COCO', help='the path folder of dataset')
    parser.add_argument('--batch_size',  help='Batch size', type=int, default=128)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=1000)
    parser.add_argument('--start_epoch', help='start epoch', type=int, default=1)
    parser.add_argument('--gpus', help='Use CUDA on the listed devides', nargs='+', type=int, default=[])
    parser.add_argument('--seed', help='Random seed', type=int, default=1234)
    parser.add_argument('--input_size', help='Image size', type=int, default=128)
    parser.add_argument('--saved_path', help='save path', type=str, default='./checkpoint')
    parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args



def main(args=None):
    
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_set = CocoDataset(root_dir=args.data_path, set_name='train2017', transform=transform)
    val_set = CocoDataset(root_dir=args.data_path, set_name='val2017', transform=transform)

    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False)

    model = ThunderNet()

    if not os.path.isdir(args.saved_path):
        os.makedirs(args.saved_path)

    if use_cuda:
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)

    # update lr
    milestones = [500, 800, 1200, 1500]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)

    for epoch in range(args.start_epoch, 2):
        train(train_loader, model, epoch, scheduler)

        test(test_loader, model)

        scheduler.step()


def train(train_loader, model, epoch, scheduler):
    model.train()
    epoch_loss = []

    for i, data in enumerate(train_loader):  
        
        cls_loss, reg_loss = mode(data['img'].cuda().float(), data['annot'].cuda())

        cls_loss = cls_loss.mean()
        reg_loss = reg_loss.mean()
        loss = cls_loss + reg_loss

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        epoch_loss.append(float(loss))
        total_loss = np.mean(epoch_loss)

        if (i+1)%50 == 0:
            learning_rate = scheduler.get_lr()[0]   # get learning rate
            print('classification loss: {:1.5f} | regression loss: {:1.5f} | total loss: {:1.5f} | lr: {}'.format(
            cls_loss, reg_loss, np.mean(loss), learning_rate))


def test(test_loader, model):
    model.eval()

    loss_regression_ls = []
    loss_classification_ls = []
    for i, data in enumerate(test_loader):
        with torch.no_grad():
            cls_loss, reg_loss = model(data['img'].cuda().float(), data['annot'].cuda())

            cls_loss = cls_loss.mean()
            reg_loss = reg_loss.mean()

            loss_classification_ls.append(float(cls_loss))
            loss_regression_ls.append(float(reg_loss))

    cls_loss = np.mean(loss_classification_ls)
    reg_loss = np.mean(loss_regression_ls)
    loss = cls_loss + reg_loss

    print('classification loss: {:1.5f} | regression loss: {:1.5f} | total loss: {:1.5f}'.format(
        cls_loss, reg_loss, np.mean(loss)))


if __name__ == '__main__':
    args = parse_args()
    main(args)