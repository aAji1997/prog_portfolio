#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 11:46:39 2022

@author: hal
"""
import numpy as np
import pandas as pd
import os

from glob import glob
import tifffile as tiff
import torch
from model_cast_torch import UDTransNet

from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader as DataLoader

import albumentations as auglib
import albumentations.pytorch as pytaugs
from training_torch import get_model_config

from cv2 import cv2

class TissueSubmitter(Dataset):
    def __init__(self, img_size=224, batch_size=16):
        self.img_size = img_size
        self.batch_size = batch_size
        
        self.test_df = pd.read_csv("./test.csv")
        self.test_img_dir = "./test_images"
        self.model_name = "UDTRANS"
        self.load_path = f"./torch_models/best_model-{self.model_name}.pth.tar"
        
        self.model_config = get_model_config()
        self.model = UDTransNet(self.model_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.cuda(device=self.device)
        
        self.load_checkpoint(self.load_path)
        self.mask_transforms = None
                                          
    def __len__(self):
        return self.test_df.shape[0]
    
    def mask2rle (self, img):
        '''
        img: numpy array, 1 - mask, 0 - background
        Returns run length as string formated
        '''
        bytes = np.where(img.flatten()==1)[0]
        runs = []
        prev = -2
        for b in bytes:
            if (b>prev+1): runs.extend((b+1, 0))
            runs[-1] += 1
            prev = b
        
        return ' '.join([str(i) for i in runs])
    
    def load_checkpoint(self, load_path):
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        this_fold = checkpoint['fold']
        
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
            
        img_id = self.test_df.iloc(idx, 0)
        img_path = os.path.join(self.img_dir+str(img_id)+".tiff")
        
        image = tiff.imread(img_path)
        mask_transforms = auglib.Compose([auglib.Resize(image.shape[0], image.shape[1], cv2.INTER_NEAREST_EXACT),
                                          pytaugs.transforms.ToTensorV2(transpose_mask=True)])
        self.mask_transforms = mask_transforms
        image_transforms = auglib.Compose([auglib.Resize(self.img_size, self.img_size, cv2.INTER_NEAREST_EXACT), auglib.Normalize(
        ), pytaugs.transforms.ToTensorV2(transpose_mask=True)])
        image = image_transforms(image)
        
        return img_id, image
    
    def get_pred_rles(self):
        self.model.eval()
        infer_loader = DataLoader(self, batch_size=16, num_workers=6)
        print("-------------------------Beginning Prediction-------------------------------")
        
        submission_raw = []
        for (i, batch) in enumerate(infer_loader):
            img_ids, images = batch[0], batch[1]
            img_ids, images = img_ids, images.cuda(device=self.device)
            
            pred_masks = self.model(images)
            pred_masks = pred_masks.cpu().detach().numpy()
            pred_rles = [self.mask2rle(pred_mask) for pred_mask in pred_masks]
            
            img_ids = img_ids.detach().numpy()
            
            for entry in (img_ids, pred_rles):
                submission_raw.append(entry)
            
            return submission_raw
            
        
    def get_subm_df(self):
        submission_raw = self.get_pred_rles()
        subm_frame = pd.DataFrame()
        subm_frame['id'] = submission_raw[0].tolist()
        subm_frame['rle'] = submission_raw[1].tolist()
        subm_frame.to_csv('submission.csv', index=False)
        print("\nSubmission Saved\n")
            
if __name__ == '__main__':
    submitter = TissueSubmitter()
    submitter.get_subm_df()
            
        
            
        
        
        
        
        
        
        
        
    