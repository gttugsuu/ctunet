"""

Sciprt for training cascaded tunable UNETs

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

from models import cascadable_TUNET
from dataset import tunable_data, drive
from utility import VisdomLinePlotter

parser = argparse.ArgumentParser(description='Cascaded Unet')
parser.add_argument('-types', nargs='+', default=['cannyedge', 'binary', 'dilation', 'erosion'], help='{binary cannyedge dilation erosion hist_eq}')
parser.add_argument('-tuning_ps', nargs='+', default=[2,2,2,8])
parser.add_argument('-output_path', type=str, default='../output/drive_cascaded_ctunet/', help='parent directery to the experiment that contains types directories')
parser.add_argument('-env', type=str, default=None, help='to specify explicitly environment name for visdom')
parser.add_argument('-lr', type=float, default=0.0001)
parser.add_argument('-start_epoch', type=int, default=0)
parser.add_argument('-max_epoch', type=int, default=50)
parser.add_argument('-batch_size', type=int, default=15)
# cuda
parser.add_argument('-cudas', nargs='+', default=['0','1','2','3'], help='cuda numbers to use')
parser.add_argument('-cuda', type=int, default=0)
# visdom decision
parser.add_argument('-no_visdom', dest='use_visdom', action='store_false')
parser.set_defaults(use_visdom=True)
args = parser.parse_args()

# for arg in args.__dict__:
#     print(arg,':', args.__dict__[arg])

torch.manual_seed(30)

experiment_type = ''
for arg in args.types: experiment_type += arg+'_'
for arg in args.tuning_ps: experiment_type += f'{arg}_'
experiment_type = experiment_type[:-1]

if args.env == None:
    env = args.lr
else:
    env = args.env
output_path = f'{args.output_path}{experiment_type}_{env}/'
visdom_env_name = f'drive-ctunet/{experiment_type}_{env}'

Path(output_path).mkdir(parents=True, exist_ok=True)

ps = [int(pn) for pn in args.tuning_ps]

lr = args.lr
start_epoch = args.start_epoch
max_epoch = args.max_epoch
batch_size = args.batch_size
use_visdom = args.use_visdom
devices = [torch.device(f'cuda:{int(args.cudas[i])}') for i in range(len(args.types))]
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
for i, type in enumerate(args.types):
    model = cascadable_TUNET(n_channels=1,o_channels=1,use_sigmoid=False).to(devices[i])
    model.load_state_dict(torch.load(f'../ctunet_modules_gray/{type}.pth'))
    # model.load_state_dict(torch.load(f'{output_path}{start_epoch-1}_{i}.pth'))
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

def model_run(models, img_orig, ps):
    for m_n, model in enumerate(models):
            if m_n == 0:
                output, conv1, conv2, conv3 = model(img_orig, ps[0].to(devices[0]))
            else:
                output = output.to(devices[m_n])
                conv1 = conv1.to(devices[m_n])
                conv2 = conv2.to(devices[m_n])
                conv3 = conv3.to(devices[m_n])
                p = ps[m_n].to(devices[m_n])
                output, conv1, conv2, conv3 = model(output, p=p, skipc1=conv1, skipc2=conv2, skipc3=conv3)
    return output

def showtestresults(models, the_dataset, winname, id, pns, type):
    img_orig, img_mask, img_gt = the_dataset.__getitem__(id)
    img_orig = img_orig.unsqueeze(0).to(devices[0])
    img_mask = img_mask.unsqueeze(0).to(devices[0])
    ps = []
    for i, pn in enumerate(pns):
        ps.append(torch.tensor([pn]).unsqueeze(-1).float().to(devices[i]))

    img_orig = img_orig * img_mask

    output = model_run(models, img_orig, ps)
    
    input = dataset.__detransform__(img_orig.squeeze(0).detach().cpu(), orig=True).numpy()
    gt = dataset.__detransform__(img_gt, orig=False).numpy()
    output= dataset.__detransform__(output.squeeze(0).detach().cpu()).numpy()
    images = [input, gt, output]
    visdom_plotter.draw_3_images(images, win_name=winname, caption=f'{type}: ps={pns}')

def run_through(models, devices, loader, maeloss, optimizer, pns):
    phase='training phase' if models[0].training else 'validating phase'
    loss_sum = 0.0
    for img_orig, img_mask, img_gt in loader:
        img_orig = img_orig.to(devices[0])
        img_mask = img_mask.to(devices[0])
        img_gt = img_gt.to(devices[-1])
        ps = []
        for i, pn in enumerate(pns):
            ps.append(torch.tensor([pn]*batch_size).unsqueeze(-1).float().to(devices[i]))
        
        img_orig = img_orig * img_mask

        output = model_run(models, img_orig, ps)
        
        loss = maeloss(output, img_gt)
        loss_sum += loss.item()

        if models[0].training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return loss_sum

for epoch in tqdm(range(start_epoch, max_epoch), desc='Epoch'):

    for model in models: model.train()
    trainloss = run_through(models, devices, train_loader, maeloss, optimizer, ps)
    
    for model in models: model.eval()
    validloss = run_through(models, devices, train_loader, maeloss, optimizer, ps)

    if use_visdom:
        visdom_plotter.plot(legend_name='train', title_name='Train loss', x=epoch, y=trainloss/len(train_loader))
        visdom_plotter.plot(legend_name='valid', title_name='Train loss', x=epoch, y=validloss/len(valid_loader))

    if (epoch + 1 ) % 50 == 0 or epoch == 0:

        # show sample outputs in visdom
        if use_visdom:
            # for idn, sample_id in enumerate(random.sample(range(train_dataset.__len__()), 2)):
            for sample_id in range(2):
                showtestresults(models, train_dataset, winname=f'train_{sample_id}', id=sample_id, pns=ps, type='Train')
                showtestresults(models, valid_dataset, winname=f'valid_{sample_id}', id=sample_id, pns=ps, type='Valid')
                showtestresults(models, test_dataset, winname=f'test_{sample_id}', id=sample_id, pns=ps, type='Test')

        # save model
        if (epoch + 1) % 100 == 0:
            for m_n, model in enumerate(models):
                state_dict = model.state_dict()
                for key in state_dict.keys():
                    state_dict[key] = state_dict[key].to(torch.device('cpu'))
                torch.save(state_dict, f'{output_path}{epoch}_{m_n}.pth')

    scheduler.step(validloss/len(valid_loader))