from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import argparse
import torch

import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from detector import ThunderNet
from load_data import CocoDataset, Resizer, Normalizer, Augmenter, collater
from tqdm.autonotebook import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()
print('device available: {}'.format(device))

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Simple training parameter for training a SNet.')

    parser.add_argument('--data_path', type=str, default='data/COCO', help='the path folder of dataset')
    parser.add_argument('--batch_size',  help='Batch size', type=int, default=32)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=1000)
    parser.add_argument('--start_epoch', help='start epoch', type=int, default=1)
    parser.add_argument('--gpus', help='Use CUDA on the listed devides', nargs='+', type=int, default=[])
    parser.add_argument('--seed', help='Random seed', type=int, default=1234)
    parser.add_argument('--saved_path', help='save path', type=str, default='./checkpoint')
    parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def main(args=None):
    transform_train = transforms.Compose([
        Normalizer(),
        Augmenter(),
        Resizer()
    ])

    transform_test = transforms.Compose([
        Normalizer(),
        Resizer()
    ])

    num_gpus = 1
    train_params = {
        "batch_size": 64,
        "shuffle": True,
        "drop_last": True,
        "collate_fn": collater,
        "num_workers": 8
    }

    test_params = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "drop_last": False,
        "collate_fn": collater,
        "num_workers": 8
    }

    train_set = CocoDataset(root_dir=args.data_path, set_name='train2017', transform=transform_train)
    val_set = CocoDataset(root_dir=args.data_path, set_name='val2017', transform=transform_test)

    train_loader = DataLoader(dataset=train_set, **train_params)
    test_loader = DataLoader(dataset=val_set, **test_params)

    num_iter = len(train_loader)

    model = ThunderNet()

    save_path = args.saved_path
    if not os.path.isdir(args.saved_path):
        os.makedirs(args.saved_path)

    if use_cuda:
        torch.cuda.set_device(args.gpus[0])
        torch.cuda.manual_seed(args.seed)
        model = model.cuda()
        #model = torch.nn.DataParallel(model)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)

    # update lr
    milestones = [500, 800, 1200, 1500]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)

    writer = SummaryWriter(log_dir='./checkpoint/summary')

    for epoch in range(args.start_epoch, 2):
        train_loss = train(train_loader, model, optimizer, args, num_iter, epoch, scheduler)
        #test(test_loader, model)

        writer.add_scalar('train loss', train_loss)
        scheduler.step()

        save_name = '{}/thundernet_{}.pth.tar'.format(save_path, epoch)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            }, filename=save_name)

    writer.export_scalars_to_json('./checkpoint/summary/' + 'pretrain' + 'all_scalars.json')
    writer.close()


def train(train_loader, model, optimizer, args, num_iter, epoch, scheduler):
    model.train()
    epoch_loss = []

    progress_bar = tqdm(train_loader)
    for i, data in enumerate(progress_bar):  
        
        input_data = data['img'].cuda().float()
        input_labels = data['annot'].cuda()

        loss_dict = model(input_data, input_labels)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss.append(float(losses))
        if (i+1)%50 == 0:
            learning_rate = scheduler.get_lr()[0]   # get learning rate
            print('Epoch: {}/{}. Iter: {}/{} loss: {:.5f}'.format(
                epoch, args.epochs, (i+1), num_iter, losses.item()))

    train_loss = np.mean(epoch_loss)
    return train_loss

def test(test_loader, model):
    model.eval()
    total_loss = []

    progress_bar = tqdm(test_loader)
    for i, data in enumerate(progress_bar):
        with torch.no_grad():
            input_data = data['img'].cuda().float()
            input_labels = data['annot'].cuda()

            loss_dict = model(input_data, input_labels)
            losses = sum(loss for loss in loss_dict.values())
            print('losses: ', losses)

            total_loss.append(losses.item())
            #cls_loss = cls_loss.mean()
            #reg_loss = reg_loss.mean()

    sum_loss = np.mean(total_loss)
    print('test total loss: {:1.5f}'.format(sum_loss))

def save_checkpoint(state, filename):
    print('save model: {}\n'.format(filename))
    torch.save(state, filename)

if __name__ == '__main__':
    args = parse_args()
    main(args)