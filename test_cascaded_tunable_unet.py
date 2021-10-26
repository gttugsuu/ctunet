"""

Script for training cascaded tunable UNET combinations

"""

import csv
import enum
import os
import argparse
import time
import random
from numpy.core.numeric import ones_like
from torch._C import device
from torch.functional import cartesian_prod
from torch.nn.modules import loss

from tqdm import tqdm
import visdom
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset

from torchvision.utils import save_image, make_grid
from torchvision.transforms.functional import to_pil_image

from models import cascadable_TUNET
from dataset import tunable_data, drive
from utility import VisdomLinePlotter, DiceLoss

parser = argparse.ArgumentParser(description='Test Cascaded Unet')
parser.add_argument('-types', nargs='+', default=['cannyedge', 'binary', 'dilation', 'erosion'], help='{binary cannyedge dilation erosion hist_eq}')
parser.add_argument('-output_path', type=str, default='../output/drive_cascaded_ctunet/', help='parent directery to the experiment that contains types directories')
parser.add_argument('-env', type=str, default=None, help='to specify explicitly environment name for visdom')
parser.add_argument('-lr', type=float, default=0.00001)
parser.add_argument('-max_epoch', type=int, default=50)
parser.add_argument('-batch_size', type=int, default=20)
# cuda
# parser.add_argument('-cudas', nargs='+', default=['0','1','2','3'], help='cuda numbers to use')
parser.add_argument('-cudas', nargs='+', default=['3','3','3','3'], help='cuda numbers to use')
parser.add_argument('-cuda', type=int, default=0)
# visdom decision
parser.add_argument('-visdom', dest='use_visdom', action='store_true')
parser.add_argument('-no_visdom', dest='use_visdom', action='store_false')
parser.set_defaults(use_visdom=True)
args = parser.parse_args()

torch.manual_seed(30)

experiment_type = ''
for arg in args.types: experiment_type += arg+'_'
experiment_type = experiment_type[:-1]

if args.env == None:
    env = args.lr
else:
    env = args.env
output_path = f'{args.output_path}{experiment_type}_{env}/'
visdom_env_name = f'test-drive-ctunet/{experiment_type}_{env}'

lr = args.lr
max_epoch = args.max_epoch
batch_size = args.batch_size
use_visdom = args.use_visdom
devices = [torch.device(f'cuda:{cuda}') for cuda in args.cudas]
print('Using', devices)

params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 16, 'drop_last': False}

test_dataset  = drive(mode='test', img_type='L')
train_dataset = drive(mode='train', img_type='L')
train_loader = DataLoader(train_dataset, **params)
test_loader  = DataLoader(test_dataset, **params)

# Initialize models & load trained weights
models = []
for i, type in enumerate(args.types):
    model = cascadable_TUNET(n_channels=1, o_channels=1).to(devices[i])
    model.load_state_dict(torch.load(f'../ctunet_modules_gray/{type}.pth'))
    # model.load_state_dict(torch.load(f'{output_path}{max_epoch-1}_{i}.pth'))
    models.append(model)

# Loss functions
maeloss = nn.L1Loss(reduction='mean')
l2_loss = nn.MSELoss(reduction='mean')

# Logger
if use_visdom:
    visdom_plotter = VisdomLinePlotter(env_name=visdom_env_name, filepath=f'../ctunet_modules_gray/log_test.visdom')

for model in models: model.eval()

# For models with 4 modules
# pnss = [[2,2,2,2],
#         [5,5,5,5],
#         [8,8,8,8]]
pnss = []
pps = [2, 5, 8]
pps = [1,2,3,4,5,6,7,8,9]
for p1 in pps:
    for p2 in pps:
        for p3 in pps:
            for p4 in pps:
                pnss.append([p1,p2,p3,p4])
# For models with only one module
# pnss = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]

min_loss = 9999999999
min_ps = []

def run_through(models, devices, loader, loss_fn, pns):
    ps = []
    for i, pn in enumerate(pns):
        ps.append(torch.tensor([pn]*batch_size).unsqueeze(-1).float().to(devices[i]))
    
    loss_sum = 0
    
    # for each batch
    for img_orig, img_mask, img_gt in loader:
        img_orig = img_orig.to(devices[0])
        img_mask = img_mask.to(devices[0])
        img_gt = img_gt.to(devices[-1])

        img_orig = img_orig*img_mask

        for m_n, model in enumerate(models):
            if m_n == 0:
                output, conv1, conv2, conv3 = model(img_orig, ps[0])
            else:
                output = output.to(devices[m_n])
                conv1 = conv1.to(devices[m_n])
                conv2 = conv2.to(devices[m_n])
                conv3 = conv3.to(devices[m_n])
                output, conv1, conv2, conv3 = model(output, p=ps[m_n], skipc1=conv1, skipc2=conv2, skipc3=conv3)
        loss_sum += loss_fn(output, img_gt)
    
    return loss_sum.item()

for pns in pnss:
    # print(pns)

    loss_sum = run_through(models, devices, train_loader, l2_loss, pns)
    # print(loss_sum)
    # make tensor inputs from scalar values
    # ps = []
    # for i, pn in enumerate(pns):
    #     ps.append(torch.tensor([pn]*batch_size).unsqueeze(-1).float().to(devices[i]))

    # loss_sum = 0
    # # for each batch
    # for img_orig, img_mask, img_gt in train_loader:
    #     img_orig = img_orig.to(devices[0])
    #     img_mask = img_mask.to(devices[0])
    #     img_gt = img_gt.to(devices[-1])

    #     for m_n, model in enumerate(models):
    #         if m_n == 0:
    #             output, conv1, conv2, conv3 = model(img_orig*img_mask, ps[0])
    #             img_orig = None
    #             img_mask = None
    #         else:
    #             output = output.to(devices[m_n])
    #             conv1 = conv1.to(devices[m_n])
    #             conv2 = conv2.to(devices[m_n])
    #             conv3 = conv3.to(devices[m_n])
    #             output, conv1, conv2, conv3 = model(output, p=ps[m_n], skipc1=conv1, skipc2=conv2, skipc3=conv3)
    #     loss_sum += l2_loss(output, img_gt)

    # loss = loss_sum/train_dataset.__len__()
    
    if loss_sum <= min_loss:
        min_loss = loss_sum
        min_ps = pns
        # print('Update: ', min_ps)
print(f'The best combination is: {min_ps}, loss: {min_loss:.3}')


row = args.types + min_ps + [min_loss]

with open('test_all.csv', 'a+', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(row)

