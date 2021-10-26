"""

Script for training a regular UNET

"""
import os
import argparse
import time
import random
from pathlib import Path

from tqdm import tqdm
import visdom
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset

from models import unet
from dataset import coco_for_cascade, drive
from utility import VisdomLinePlotter

parser = argparse.ArgumentParser(description='Unet')
parser.add_argument('-output_path', type=str, default='../output/unet/', help='parent directery to the experiment that contains types directories')
parser.add_argument('-visdom_env_name', type=str, default=None, help='to specify explicitly environment name for visdom')
parser.add_argument('-lr', type=float, default=0.001)
parser.add_argument('-max_epoch', type=int, default=50)
parser.add_argument('-batch_size', type=int, default=1)
parser.add_argument('-use_visdom', type=bool, default=True)
parser.add_argument('-cuda', type=int, default=0)
parser.add_argument('-channels', type=int, default=1)
args = parser.parse_args()

output_path = f'{args.output_path}'
Path(output_path).mkdir(parents=True, exist_ok=True)

if args.visdom_env_name is not None:
    visdom_env_name = args.visdom_env_name
lr = args.lr
max_epoch = args.max_epoch
batch_size = args.batch_size
use_visdom = args.use_visdom
device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available else 'cpu')
print('Using ', device)

# Dataset, dataloader
params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0, 'drop_last': False}

dataset = drive(mode='train')
# Random split for train & valid
dtlen=15
train_dataset, valid_dataset = random_split(dataset, [dtlen, dataset.__len__()-dtlen])
test_dataset = drive(mode='test')

train_loader = DataLoader(train_dataset, **params)
valid_loader = DataLoader(valid_dataset, **params)

# Initialize model 
model = unet(n_channels=args.channels).to(device)
# model.load_state_dict(torch.load('saved_blur/blur_nonorm/116.pth'))

# Loss functions
maeloss = nn.L1Loss()

# Optimizer & scheduler
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=2)

# Logger
if use_visdom:
    visdom_plotter = VisdomLinePlotter(env_name=visdom_env_name, filepath=f'{output_path}log.visdom')

def showtestresults(model, test_dataset, winname, id):
    input, mask, gt = test_dataset.__getitem__(id)
    input = input.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    gt = dataset.__detransform__(gt).numpy()
    output = model(input)
    input = dataset.__detransform__(input.squeeze(0).detach().cpu()).numpy()
    output= dataset.__detransform__(output.squeeze(0).detach().cpu()).numpy()
    visdom_plotter.draw_3_images([input, gt, output], win_name=winname)

for epoch in range(max_epoch):
    print(f'Epoch: {epoch}')

    model.train()
    trainloss = 0.0
    # for img_orig, img_mask, img_gt in tqdm(train_loader, desc='train phase'):
    for img_orig, img_mask, img_gt in train_loader:
        img_orig = img_orig.to(device)
        img_mask = img_mask.to(device)
        img_gt = img_gt.to(device)

        output = model(img_orig)

        loss = maeloss(output, img_gt)
        
        trainloss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    validloss = 0.0
    for img_orig, img_mask, img_gt in tqdm(valid_loader, desc='validation phase'):
        img_orig = img_orig.to(device)
        img_mask = img_mask.to(device)
        img_gt = img_gt.to(device)

        output = model(img_orig)

        loss = maeloss(output, img_gt)

        validloss += loss.item()

    # print('Average training loss', trainloss/len(train_loader))
    # print('Average validation loss', validloss/len(valid_loader))

    if use_visdom:
        visdom_plotter.plot(legend_name='train', title_name='Train loss', x=epoch, y=trainloss/len(train_loader))
        visdom_plotter.plot(legend_name='valid', title_name='Train loss', x=epoch, y=validloss/len(valid_loader))

    if (epoch + 1 ) % 5 == 0:
        state_dict = model.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, f'{output_path}{epoch}.pth')

        if use_visdom:
            for idn, draw_id in enumerate(random.sample(range(valid_dataset.__len__()), 5)):
                showtestresults(model, valid_dataset, winname=f'input_gt_output_{idn}', id=draw_id)

    scheduler.step(validloss/len(valid_loader))