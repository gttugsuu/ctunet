"""

Script for training cascaded UNET

"""
import enum
import os
from pathlib import Path
import argparse
import time
import random
from numpy.core.numeric import ones_like
from torch._C import device

from tqdm import tqdm
import visdom
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset

from models import cascadable_unet
from dataset import tunable_data, drive
from utility import VisdomLinePlotter

parser = argparse.ArgumentParser(description='Cascaded Unet')
parser.add_argument('-num_unet', type=int, default=4, help='number of unets to cascade')
parser.add_argument('-output_path', type=str, default='../output/drive_cascaded_unet/', help='parent directery to the experiment that contains types directories')
parser.add_argument('-env', type=str, default=None, help='to specify explicitly environment name for visdom')
parser.add_argument('-lr', type=float, default=0.0001)
parser.add_argument('-start_epoch', type=int, default=0)
parser.add_argument('-max_epoch', type=int, default=50)
parser.add_argument('-batch_size', type=int, default=15)
# cuda
parser.add_argument('-cudas', nargs='+', default=['0','1','2','3'], help='cuda numbers to use')
parser.add_argument('-cuda', type=int, default=0)
# visdom decision
parser.add_argument('-visdom', dest='use_visdom', action='store_true')
parser.add_argument('-no_visdom', dest='use_visdom', action='store_false')
parser.set_defaults(use_visdom=True)
args = parser.parse_args()

# for arg in args.__dict__:
#     print(arg,':', args.__dict__[arg])

torch.manual_seed(30)

if args.env == None:
    env = args.lr
else:
    env = args.env
output_path = f'{args.output_path}{args.num_unet}_{env}/'
visdom_env_name = f'drive-cunet/{args.num_unet}_{env}'

os.system(f'mkdir {output_path}')
Path(output_path).mkdir(parents=True, exist_ok=True)

lr = args.lr
start_epoch = args.start_epoch
max_epoch = args.max_epoch
batch_size = args.batch_size
use_visdom = args.use_visdom
devices = [torch.device(f'cuda:{int(args.cudas[i])}') for i in range(args.num_unet)]
print('Using', devices)

params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 16, 'drop_last': False}

dataset = drive(mode='train', img_type='L')
# Random split for train & valid
dtlen=15
train_dataset, valid_dataset = random_split(dataset, [dtlen, dataset.__len__()-dtlen])
test_dataset  = drive(mode='test', img_type='L')

train_loader = DataLoader(train_dataset, **params)
valid_loader = DataLoader(valid_dataset, **params)
test_loader  = DataLoader(test_dataset, **params)

# Initialize models & load trained weights
models = []
for i in range(args.num_unet):
    model = cascadable_unet(i_channels=1, o_channels=1, use_sigmoid=False).to(devices[i])
    models.append(model)

# Loss functions
maeloss = nn.L1Loss()

# Optimizer & scheduler
parameters = []
for model in models:
    param = list(model.parameters())
    parameters += param
optimizer = torch.optim.Adam(parameters, lr = lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=10)

# Logger
if use_visdom:
    visdom_plotter = VisdomLinePlotter(env_name=visdom_env_name, filepath=f'{output_path}log.visdom')

def showtestresults(models, the_dataset, winname, id):
    img_orig, img_mask, img_gt = the_dataset.__getitem__(id)
    img_orig = img_orig.unsqueeze(0).to(devices[0])
    img_mask = img_mask.unsqueeze(0).to(devices[0])

    img_orig = img_orig * img_mask

    for m_n, model in enumerate(models):
        if m_n == 0:
            output, conv1, conv2, conv3 = model(img_orig)
        else:
            output = output.to(devices[m_n])
            conv1 = conv1.to(devices[m_n])
            conv2 = conv2.to(devices[m_n])
            conv3 = conv3.to(devices[m_n])
            output, conv1, conv2, conv3 = model(output, skipc1=conv1, skipc2=conv2, skipc3=conv3)
    
    input = dataset.__detransform__(img_orig.squeeze(0).detach().cpu(), orig=True).numpy()
    gt = dataset.__detransform__(img_gt, orig=False).numpy()
    output= dataset.__detransform__(output.squeeze(0).detach().cpu()).numpy()
    images = [input, gt, output]
    visdom_plotter.draw_3_images(images, win_name=winname)

def run_through(models, devices, loader, maeloss, optimizer):
    phase='training phase' if models[0].training else 'validating phase'
    loss_sum = 0.0
    for img_orig, img_mask, img_gt in loader:
        img_orig = img_orig.to(devices[0])
        img_mask = img_mask.to(devices[0])
        img_gt = img_gt.to(devices[-1])
        
        img_orig = img_orig * img_mask

        for m_n, model in enumerate(models):
            if m_n == 0:
                output, conv1, conv2, conv3 = model(img_orig)
            else:
                output = output.to(devices[m_n])
                conv1 = conv1.to(devices[m_n])
                conv2 = conv2.to(devices[m_n])
                conv3 = conv3.to(devices[m_n])
                output, conv1, conv2, conv3 = model(output, skipc1=conv1, skipc2=conv2, skipc3=conv3)
        
        loss = maeloss(output, img_gt)
        loss_sum += loss.item()

        if models[0].training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return loss_sum

for epoch in tqdm(range(start_epoch, max_epoch), desc='Epoch'):
    # print(f'Epoch: {epoch}')

    for model in models: model.train()
    trainloss = run_through(models, devices, train_loader, maeloss, optimizer)
    
    for model in models: model.eval()
    validloss = run_through(models, devices, train_loader, maeloss, optimizer)

    if use_visdom:
        visdom_plotter.plot(legend_name='train', title_name='Train loss', x=epoch, y=trainloss/len(train_loader))
        visdom_plotter.plot(legend_name='valid', title_name='Train loss', x=epoch, y=validloss/len(valid_loader))

    if (epoch + 1 ) % 100 == 0 or epoch == 0:
        for m_n, model in enumerate(models):
            state_dict = model.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict, f'{output_path}{epoch}_{m_n}.pth')

        if use_visdom:
            for idn, draw_id in enumerate(random.sample(range(valid_dataset.__len__()), 3)):
                showtestresults(models, valid_dataset, winname=f'valid_{idn}', id=draw_id)
            for idn, draw_id in enumerate(random.sample(range(test_dataset.__len__()), 2)):
                showtestresults(models, test_dataset, winname=f'test_{idn}', id=draw_id)

    scheduler.step(validloss/len(valid_loader))