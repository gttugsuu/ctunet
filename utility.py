import os
import numpy as np
from visdom import Visdom
import torch
import torch.nn as nn

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main', filepath='mse_runs/log.log'):
        self.viz = Visdom(port=6006, log_to_filename=filepath)
        if os.path.exists(filepath):
            self.viz.replay_log(filepath)
        self.env = env_name
        self.plots = {}
        self.images = {}

    def plot(self, legend_name, title_name, x, y):
        if title_name not in self.plots:
            self.plots[title_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]),
                                                env=self.env, 
                                                win=title_name,
                                                opts=dict(
                                                    legend=[legend_name],
                                                    title=title_name,
                                                    xlabel='Epochs',
                                                    ylabel='Loss value'
                                                    ),
                                                )
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env,
                            win=title_name, name=legend_name,
                            update = 'append')

    def draw_image(self, image, win_name = 'An_image', caption = 'An image'):
        self.viz.image(image,
                    env = self.env,
                    win = win_name,
                    opts = dict(
                        caption = caption,
                        width = 500,
                        height = 500,
                        store_history = True
                        )
                    )

    def draw_3_images(self, images, win_name = '3_images', caption = 'Input, GT, Output'):
        self.viz.images(images,
                    env = self.env,
                    win = win_name,
                    nrow = 3,
                    opts = dict(
                        padding = 5,
                        caption = caption,
                        width = 800,
                        height = 400,
                        store_history = True,
                        )
                    )

    def draw_4_images(self, images, win_name = '4_images', caption = 'Input, GT, Output'):
        self.viz.images(images,
                    env = self.env,
                    win = win_name,
                    nrow = 4,
                    opts = dict(
                        padding = 5,
                        caption = caption,
                        width = 1100,
                        height = 400,
                        store_history = True,
                        )
                    )

    def draw_5_images(self, images, win_name = '5_images', caption = 'Input, GT, Output'):
        self.viz.images(images,
                    env = self.env,
                    win = win_name,
                    nrow = 5,
                    opts = dict(
                        padding = 5,
                        caption = caption,
                        width = 1400,
                        height = 400,
                        store_history = True,
                        )
                    )

    def draw_7_images(self, images, win_name = '7_images', caption = 'Input, GT, Outputs'):
        self.viz.images(images,
                    env = self.env,
                    win = win_name,
                    nrow=4,
                    opts = dict(
                        padding = 5,
                        caption = caption,
                        width = 1400,
                        height = 600,
                        store_history = True,
                        )
                    )

    def draw_images(self, images, nrow=2, win_name = 'Arbitrary_images', caption = 'Input, GT, Outputs'):
        self.viz.images(images,
                    env = self.env,
                    win = win_name,
                    nrow=nrow,
                    opts = dict(
                        padding = 5,
                        caption = caption,
                        width = 280*nrow,
                        height = 280*int(len(images)/nrow),
                        store_history = True,
                        )
                    )

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice