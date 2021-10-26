"""

Script for training a (cascadable) Tunable UNET module

"""
import enum
import os
from pathlib import Path
import argparse
import time
import random
from numpy.core.numeric import ones_like

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
from dataset import tunable_data
from utility import VisdomLinePlotter

parser = argparse.ArgumentParser(description='Cascaded Unet')
parser.add_argument('-type', type=str, default='binary', help='{binary cannyedge dilation erosion hist_eq}')
parser.add_argument('-output_path', type=str, default='../output/ctunet/', help='parent directery to the experiment that contains types directories')
parser.add_argument('-env', type=str, default=None, help='to specify explicitly environment name for visdom')
parser.add_argument('-lr', type=float, default=0.001)
parser.add_argument('-max_epoch', type=int, default=50)
parser.add_argument('-batch_size', type=int, default=10)
parser.add_argument('-cuda', type=int, default=0)
parser.add_argument('-tuning_ps', nargs='+', default=[2,5,8], help='tuning parameters')
parser.add_argument('-ds_length', type=float, default=1.0, help='percentile of the dataset to use')
# visdom decision
parser.add_argument('-visdom', dest='use_visdom', action='store_true')
parser.add_argument('-no_visdom', dest='use_visdom', action='store_false')
parser.set_defaults(use_visdom=True)
# sigmoid decision
parser.add_argument('-sigmoid', dest='use_sigmoid', action='store_true')
parser.add_argument('-no_sigmoid', dest='use_sigmoid', action='store_false')
parser.set_defaults(use_sigmoid=False)
# argument for bias in fully connected network
parser.add_argument('-bias', dest='fc_bias', action='store_true')
parser.add_argument('-no_bias', dest='fc_bias', action='store_false')
parser.set_defaults(fc_bias=False)
# args
args = parser.parse_args()

for hmm in args.__dict__:
    print(hmm,':', args.__dict__[hmm])

torch.manual_seed(30)

experiment_type = args.type
if args.env == None:
    env = args.lr
else:
    env = f'{args.env}{args.lr}'
if args.use_sigmoid:
    env = f'{env}_sigmoid'
else:
    env = f'{env}_nosigmoid'
output_path = f'{args.output_path}{experiment_type}_{env}/'
visdom_env_name = f'ctunet/{experiment_type}_{env}'

Path(output_path).mkdir(parents=True, exist_ok=True)

lr = args.lr
max_epoch = args.max_epoch
batch_size = args.batch_size
use_visdom = args.use_visdom
tuning_ps = args.tuning_ps
ds_length = args.ds_length
use_sigmoid = args.use_sigmoid
device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available else 'cpu')
print('Using ', device)

params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'drop_last': False}

train_dataset = tunable_data(mode='train', type=experiment_type, tuning_ps=tuning_ps, ds_length=ds_length, img_type='L')
valid_dataset = tunable_data(mode='valid', type=experiment_type, tuning_ps=tuning_ps, ds_length=ds_length, img_type='L')
test_dataset  = tunable_data(mode='test', type=experiment_type, tuning_ps=tuning_ps, ds_length=ds_length, img_type='L')
# Five random ids to visualize
one_test_length = int(test_dataset.__len__()/len(tuning_ps))
random_ids = random.sample(range(one_test_length), 5)

train_loader = DataLoader(train_dataset, **params)
valid_loader = DataLoader(valid_dataset, **params)
# test_loader  = DataLoader(test_dataset, **params)

# Initialize model
model = cascadable_TUNET(n_channels=1, o_channels=1, use_sigmoid=use_sigmoid).to(device)
# model.load_state_dict(torch.load('saved_blur/blur_nonorm/116.pth'))

# Loss functions
maeloss = nn.L1Loss()

# Optimizer & scheduler
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=5)

# Logger
if use_visdom:
    visdom_plotter = VisdomLinePlotter(env_name=visdom_env_name, filepath=f'{output_path}log.visdom')

def showtestresults_7(model, test_dataset, winname, id, tuning_ps, one_test_length):
    # get original images for testing
    img_gts = []
    for i, p in enumerate(tuning_ps):
        img_orig, img_gt, pgt = test_dataset.__getitem__(id+i*one_test_length)
        img_gts.append(img_gt)
        if pgt != p:
            print(f'id: {id}, one length: {one_test_length}, test len: {test_dataset.__len__()}')
            exit(f'Test dataset indexing error: pgt={pgt}, p={p}')
    # add batch dimension
    img_orig = img_orig.unsqueeze(0).to(device)
    # make tensors for each p
    p_tensors = [torch.tensor(p).unsqueeze(-1).unsqueeze(-1).float().to(device) for p in tuning_ps]
    # run images through the model for each p
    outputs = [model(img_orig, p_tensor)[0] for p_tensor in p_tensors]
    
    # denormalize & convert to image
    gt_images = [test_dataset.__detransform__(img_gt).numpy() for img_gt in img_gts]
    orig_image = test_dataset.__detransform__(img_orig.squeeze(0).detach().cpu()).numpy()
    output_images = [test_dataset.__detransform__(output.squeeze(0).detach().cpu()).numpy() for output in outputs]

    # draw with visdom
    images = [orig_image] + gt_images + [np.ones_like(orig_image)] + output_images
    visdom_plotter.draw_7_images(images, win_name=winname, caption = f'Input, GT, p={tuning_ps[0]}, p={tuning_ps[1]}, p={tuning_ps[2]}') 

def showtestresults(model, test_dataset, winname, id):
    img_orig, img_gt, p = test_dataset.__getitem__(id)
    
    img_orig = img_orig.unsqueeze(0).to(device)
    p_tensor = torch.tensor(p).unsqueeze(-1).unsqueeze(-1).float().to(device)
    
    img_gt = test_dataset.__detransform__(img_gt).numpy()

    output, _, _, _  = model(img_orig, p_tensor)

    img_orig = test_dataset.__detransform__(img_orig.squeeze(0).detach().cpu()).numpy()
    output= test_dataset.__detransform__(output.squeeze(0).detach().cpu()).numpy()

    visdom_plotter.draw_3_images([img_orig, img_gt, output], win_name=winname, caption = f'Input, GT, Output, p={p}')

for epoch in range(max_epoch):
    print(f'Epoch: {epoch}')

    model.train()
    trainloss = 0.0
    for img_orig, img_gt, p in tqdm(train_loader, desc='train phase'):
        img_orig = img_orig.to(device)
        img_gt = img_gt.to(device)
        p = p.unsqueeze(-1).float().to(device)

        output, _, _, _ = model(img_orig, p)

        loss = maeloss(output, img_gt)
        
        trainloss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    validloss = 0.0
    for img_orig, img_gt, p in tqdm(valid_loader, desc='validation phase'):
        img_orig = img_orig.to(device)
        img_gt = img_gt.to(device)
        p = p.unsqueeze(-1).float().to(device)

        output, _, _, _ = model(img_orig, p)
        loss = maeloss(output, img_gt)
        validloss += loss.item()

    if use_visdom:
        visdom_plotter.plot(legend_name='train', title_name='Train loss', x=epoch, y=trainloss/len(train_loader))
        visdom_plotter.plot(legend_name='valid', title_name='Train loss', x=epoch, y=validloss/len(valid_loader))

    

    if (epoch + 1) % 50 == 0:
        
        if (epoch + 1 ) % 50 == 0:
            state_dict = model.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict, f'{output_path}{epoch}.pth')

        # if use_visdom:
        #     one_test_length = int(test_dataset.__len__()/len(tuning_ps))
        #     random_id = random.sample(range(one_test_length), 1)[0]
        #     for i, p_int in enumerate(tuning_ps):
        #         showtestresults(model, test_dataset, winname=f'p={p_int}', id=random_id+one_test_length*i)

        if use_visdom:
            for vi, id in enumerate(random_ids):
                showtestresults_7(model, test_dataset, winname=f'{vi}', id=id, tuning_ps=tuning_ps, one_test_length=one_test_length)
        

    scheduler.step(validloss/len(valid_loader))