#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 22:30:39 2022

@author: hal
"""

import os

from matplotlib import pyplot as plt
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import KFold

from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader as DataLoader

import torchvision

import tiffile as tiff

import random

import albumentations as auglib
import albumentations.pytorch as pytaugs

GLOBAL_SEED = 42


def check_csv(csv_file):
    csv = pd.read_csv(csv_file)
    print(csv.info())
    print(csv.iloc[0, 7][-1])

class TissueDataset(Dataset):
    def __init__(self, train_csv, img_size, batch_size, use_numerics=False, kfolds=7,  debug=True):   
        self.tiss_df = pd.read_csv(train_csv)
        self.use_numerics = use_numerics

        self.num_folds = kfolds
        self.train_batch_size = batch_size
        self.test_batch_size = self.train_batch_size // 2
        self.debug = debug
        self.kfold = KFold(n_splits=self.num_folds, shuffle=True, random_state=GLOBAL_SEED)
        
        self.img_size = img_size
        self.img_dir = "./train_images/"
        self.img_size = img_size
        
        self.enc_orgs = self.encode_onehot('organ')
        self.enc_sex = self.encode_onehot('sex')
        self.no_transform = torchvision.transforms.Resize(size=self.img_size, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        
        
        self.transforms = auglib.Compose([auglib.Resize(self.img_size, self.img_size),
                                          auglib.Normalize(),
                                          auglib.RandomResizedCrop(height=self.img_size, width=self.img_size, p=0.4),
                                          auglib.Flip(p=0.3), auglib.RandomBrightnessContrast(p=0.2), pytaugs.transforms.ToTensorV2(transpose_mask=True)])
    
    def rle2mask(self, rle, shape=(3000, 3000)):
        s = rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape)
    
        
    
    def encode_onehot(self, categorical_col):
         ohe =  pd.get_dummies(self.tiss_df[categorical_col], prefix="id_")
         onehot = ohe.values
         return onehot
     

    def __len__(self):
        return self.tiss_df.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        
        img_id = self.tiss_df.iloc[idx, 0]
        img_rle = self.tiss_df.iloc[idx, 7]
        img_path = os.path.join(self.img_dir+str(img_id)+".tiff")
        img_name = str(img_id)
        
        image = tiff.imread(img_path)
        
        mask = self.rle2mask(img_rle, shape=image.shape[:2])
        
        mask = np.expand_dims(mask, axis=2)
        
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        
        image = torch.permute(image, dims=(2, 1, 0))
        mask = torch.permute(mask, dims=(2, 1, 0))
        
        image = self.no_transform(image)
        mask = self.no_transform(mask)
        
        image = image / 255.0
        
        #transformed = self.transforms(image=image, mask=mask)
        
        #image = transformed['image']
        #mask = transformed['mask']
        
        organ = self.enc_orgs[idx]
        age = self.tiss_df.iloc[idx, 8]
        sex = self.tiss_df.iloc[idx, 9]
        
        if self.debug ==True and self.use_numerics ==True:
            return (img_id, image, mask, organ, age, sex, img_name)
        elif self.debug == True and self.use_numerics ==False:
            return (img_id, image, mask, img_name)
        
        elif self.debug == False and self.use_numerics == True:
            return (image, mask, organ, age, sex, img_name)
        elif self.debug == False and self.use_numerics == False:
            return (image, mask, img_name)
    
    def show_example(self):
        assert self.debug == True, "Debug must be set to true to show example"
        rand_idx = random.randint(0, self.__len__())
        rand_example = self.__getitem__(rand_idx)
        
        rand_img = rand_example[1]
        rand_mask = rand_example[2]
        rand_img = torch.permute(rand_img, dims=(2, 1, 0))
        rand_mask = torch.permute(rand_mask, dims=(2, 1, 0))
        
        plt.figure()
        plt.imshow(rand_img)
        plt.axis("off")
        
        plt.imshow(rand_mask, cmap='coolwarm', alpha=0.5)
        plt.axis("off")
        
    def get_train_test_split(self):
        train_size = int(0.8* self.__len__())
        test_size = self.__len__() - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset=self, lengths=[train_size, test_size], generator=torch.Generator().manual_seed(GLOBAL_SEED))
        
        #print("Split Loaded")
        return train_dataset, test_dataset
    
    def get_train_kfold(self):
        train_dataset, _ = self.get_train_test_split()
        train_indices, test_indices = [], []
        
        for (train_idx, test_idx) in self.kfold.split(train_dataset):
            train_indices.append(train_idx)
            test_indices.append(test_idx)
        
        train_subsamplers = [torch.utils.data.SubsetRandomSampler(train_ix, generator=torch.Generator().manual_seed(GLOBAL_SEED)) for train_ix in train_indices]
        test_subsamplers = [torch.utils.data.SubsetRandomSampler(test_ix, generator=torch.Generator().manual_seed(GLOBAL_SEED)) for test_ix in test_indices]
        
        trainloaders = [DataLoader(dataset=train_dataset, 
                                   batch_size=self.train_batch_size, sampler=tr_sampler, generator=torch.Generator().manual_seed(GLOBAL_SEED), num_workers=6) for tr_sampler in train_subsamplers]
        
        testloaders = [DataLoader(dataset=train_dataset, 
                                  batch_size=self.test_batch_size, sampler=ts_sampler, generator=torch.Generator().manual_seed(GLOBAL_SEED), num_workers=6) for ts_sampler in test_subsamplers]
        
        loader_tups = list(zip(trainloaders, testloaders))
        
        return loader_tups
        
        
def load_data():
    tissue_data = TissueDataset(train_csv="./train.csv", img_size=224, use_numerics=True,)
    tissue_data.show_example()
    
if __name__ == "__main__":
    load_data()
    
    
    
