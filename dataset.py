"""

Script for defining datasets

"""
import os
from glob import glob
import random

from PIL import Image, ImageOps

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms.transforms import Lambda

coco_transforms = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor(),
    transforms.Normalize(   (0.4984, 0.5092, 0.5008),
                            (0.2316, 0.2321, 0.2332))
            ])

coco_reverse_transforms = transforms.Compose([
    transforms.Normalize(   (-0.4984/0.2316, -0.5092/0.2321, -0.5008/0.2332),
                            (1/0.2316, 1/0.2321, 1/0.2332)),
])

bn_transforms = transforms.Compose([
            transforms.Resize([256,256]),
            transforms.ToTensor()
            ])    
to_tensor_transforms = transforms.Compose([
            transforms.ToTensor()
            ])    

'''
Dataset for training tunable UNET
'''
class tunable_data(Dataset):
    '''
    type is the directory name:     binary, cannyedge, dilation, erosion, hist_eq,
    '''
    def __init__(self, root='/home/tugs/Tunable_unet/tunable_dataset/', mode='train', type='binary', tuning_ps=[2,5,8], ds_length=1.0, img_type='RGB'):

        self.root = root
        self.mode = mode
        self.type = type
        self.img_type = img_type
        self.length = ds_length
        if self.img_type == 'RGB':
            self.transform = coco_transforms
            self.reverse_transform = coco_reverse_transforms
        elif self.img_type == 'L':
            self.transform = transforms.Compose([
                transforms.Resize([256,256]),
                transforms.ToTensor(),
                transforms.Normalize(   0.4502,
                                        0.2225)
                ])

            self.reverse_transform = transforms.Compose([
                transforms.Normalize(   -0.4502/0.2225,
                                        1/0.2225),
                ])

        self.orig_paths = glob(f'{root}orig_{mode}/*')

        self.gt_images = []
        self.tuning_ps = []
        for p in tuning_ps:
            files = glob(f'{self.root}{self.type}/{self.mode}/{p}/*')
            files, _ = np.split(files, [int(len(files)*self.length)])
            files = files.tolist()
            
            self.gt_images += files
            self.tuning_ps += len(files)*[p]

    def __len__(self):
        return len(self.gt_images)

    def __getitem__(self, index):

        gt = self.transform(Image.open(self.gt_images[index]).convert(self.img_type))
        orig = self.transform(Image.open(f'{self.root}orig_{self.mode}/{os.path.basename(self.gt_images[index])}').convert(self.img_type))
        
        return orig, gt, self.tuning_ps[index]

    def __detransform__(self, tensor):
        
        if tensor.shape[0] > 3:
            tensor = tensor[0]

        tensor = self.reverse_transform(tensor)
        
        # method 1
        # for i in range(3):
        #     minFrom= tensor[i].min()
        #     maxFrom= tensor[i].max()
        #     tensor[i] = (tensor[i] - minFrom) / (maxFrom - minFrom)

        # method 2
        minFrom = tensor.min()
        maxFrom = tensor.max()
        tensor = (tensor - minFrom) / (maxFrom - minFrom)

        # method 3
        # tensor[tensor>1.0] = 1.0
        # tensor[tensor<0.0] = 0.0

        return tensor

'''
DRIVE dataset for fine-tuning cascaded tunable UNET
'''
class drive(Dataset):
    '''train or test mode'''
    def __init__(self, root='../DRIVE/', mode='train', img_type='RGB'):
        self.root = root
        self.mode = mode
        self.img_type = img_type
        if self.img_type == 'RGB':
            self.transform = coco_transforms
            self.reverse_transform = coco_reverse_transforms
        elif self.img_type == 'L':
            self.transform = transforms.Compose([
                transforms.Resize([256,256]),
                transforms.ToTensor(),
                transforms.Normalize(   0.4502,
                                        0.2225)
                ])
            self.reverse_transform = transforms.Compose([
                transforms.Normalize(   -0.4502/0.2225,
                                        1/0.2225),
                ])
        self.mask_transform = transforms.Compose([
            transforms.Resize([256,256]),
            transforms.ToTensor()
            ])   

        self.images = sorted(glob(f'{self.root}{self.mode}/images/*.jpg'))
        self.masks = sorted(glob(f'{self.root}{self.mode}/mask/*.jpg'))
        if self.mode == 'train':
            self.gts = sorted(glob(f'{self.root}{self.mode}/1st_manual/*.jpg'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        orig = self.transform(Image.open(f'{self.images[index]}').convert(self.img_type))
        mask = self.mask_transform(Image.open(f'{self.masks[index]}').convert('1'))
        if self.mode == 'train':
            gt = self.transform(Image.open(f'{self.gts[index]}').convert('1'))
            return orig, mask, gt
        else:
            return orig, mask, mask

    def __detransform__(self, tensor, orig=False):
        
        # if tensor.shape[0] > 3:
        #     tensor = tensor[0]

        tensor = self.reverse_transform(tensor)
        
        # method 1
        # for i in range(3):
        #     minFrom= tensor[i].min()
        #     maxFrom= tensor[i].max()
        #     tensor[i] = (tensor[i] - minFrom) / (maxFrom - minFrom)

        # method 2
        minFrom = tensor.min()
        maxFrom = tensor.max()
        tensor = (tensor - minFrom) / (maxFrom - minFrom)

        if orig == True:
            return 1 - tensor
        else:
            return tensor
        
class coco_for_cascade(Dataset):
    '''
    type is the directory name:     binary, cannyedge, dilation, erosion, hist_eq,
    '''
    def __init__(self, root='/home/gt/github/Datasets/COCO_2014/', mode='train', type='binary', return_path=False):

        self.root = root
        self.mode = mode
        self.type = type
        self.return_path = return_path
        self.transform = coco_transforms

        orig_paths = glob(f'{root}{mode}/*')
        self.image_names = [os.path.basename(path) for path in orig_paths]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):

        orig = 1.0 - self.transform(Image.open(f'{self.images[index]}').convert('RGB'))
        mask = 1.0 - self.transform(Image.open(f'{self.masks[index]}').convert('RGB'))
        if self.mode == 'train':
            gt = 1.0 - self.transform(Image.open(f'{self.gts[index]}').convert('RGB'))
            return orig, mask, gt
        else:
            return orig, mask, mask

    def __detransform__(self, tensor, orig=False):
        
        if tensor.shape[0] > 3:
            tensor = tensor[0]
            
        if orig: tensor = 1.0 - tensor

        tensor = self.reverse_transform(tensor)
        
        # method 1
        # for i in range(3):
        #     minFrom= tensor[i].min()
        #     maxFrom= tensor[i].max()
        #     tensor[i] = (tensor[i] - minFrom) / (maxFrom - minFrom)

        # method 2
        minFrom = tensor.min()
        maxFrom = tensor.max()
        tensor = (tensor - minFrom) / (maxFrom - minFrom)
        return tensor

class bncoco(Dataset):
    '''
    mode = {train, valid, test}
    '''
    def __init__(self, root="/home/gt/github/Datasets/for_tunet/coco_2017/", mode='train', bn_modes=[2,5,8], return_name=False, transforms=coco_transforms, bn_transforms=bn_transforms):

        self.transform = coco_transforms

        self.transforms_bn = bn_transforms

        self.mode = mode
        self.return_name = return_name
        self.datapath = f"{root}{mode}/"

        filenames = []
        for file in glob(f"{self.datapath}orig/*"):
            filenames.append(os.path.basename(file)[:-4])
        
        self.data = []
        for mode in bn_modes:
            for filename in filenames:
                self.data.append([filename, mode])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        img_orig = Image.open(f"{self.datapath}orig/{self.data[index][0]}.jpg").convert('RGB')
        img_orig = self.transform(img_orig)
        img_bn = Image.open(f"{self.datapath}{self.data[index][1]}/{self.data[index][0]}.jpg")
        img_bn = self.transforms_bn(img_bn)
        
        mode = self.data[index][1]
        filename = self.data[index][0]

        if self.return_name:
            return img_orig, img_bn, mode, filename
        else:
            return img_orig, img_bn, mode

    

# tr = transforms.Compose(transforms_)
# img = Image.open(glob('/home/gt/github/Datasets/for_tunet/coco_2017/train/orig/*')[0])
# img = tr(img)
# print(img.shape )

# '''
# Calculate mean and std of a training dataset
# '''
# class train_data(Dataset):
#     def __init__(self, root="/home/tugs/Tunable_unet/tunable_dataset/", mode='train', bn_modes=[2,5,8]):
#         self.transform = transforms.Compose([
#                             transforms.Resize([256,256]),
#                             transforms.ToTensor()
#                             ])    
#         self.filelist = glob(f"/home/tugs/Tunable_unet/tunable_dataset/orig_train/*.jpg")
#     def __len__(self):
#         return len(self.filelist)
#     def __getitem__(self, index):
#         img_orig = Image.open(self.filelist[index]).convert('L')
#         img_orig = self.transform(img_orig)
#         return img_orig
# dataset = train_data()
# from torch.utils.data import DataLoader
# loader = DataLoader(dataset,
#                          batch_size=10,
#                          num_workers=5,
#                          shuffle=False)
# mean = 0.
# std = 0.
# for images in loader:
#     batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
#     images = images.view(batch_samples, images.size(1), -1)
#     mean += images.mean(2).sum(0)
#     std += images.std(2).sum(0)
# mean /= len(loader.dataset)
# std /= len(loader.dataset)
# print(mean)
# print(std)