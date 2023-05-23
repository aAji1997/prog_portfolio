#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 17:27:27 2022

@author: hal
"""
import tensorflow as tf
import keras
from matplotlib import pyplot as plt

from model_cast import Augment, UDTransNet
import ml_collections


#Configuration
class model_config:
    def __init__(self):
        self.config = ml_collections.ConfigDict()
        self.config.transformer = ml_collections.ConfigDict()
        self.config.transformer.num_heads = 4
        self.config.transformer.num_layers = 4
        self.config.expand_ratio = 2
        
        
        self.config.transformer.embedding_channels = 32*self.config.transformer.num_heads
        self.config.KV_size = self.config.transformer.embedding_channels * 4
        self.config.KV_size_S = self.config.transformer.embedding_channels
        self.config.transformer.attention_dropout_rate = 0.1
        self.config.transformer.dropout_rate = 0.1
        self.config.patch_sizes = [16, 8, 4, 2]
        self.config.base_channel = 32
        self.config.decoder_channels = [32, 64, 128, 256, 512]
        
        self.config.image_size = 224
        self.config.num_classes = 3
        self.config.num_channels = 3
        
    def get_config(self):
        
        return self.config

class image_model(keras.Model):
    '''
    There are 3 classes determined by the model: (1) FTU mask pixel, (2) Pixel bordering the pet (3) Surrounding pixel 
    '''
    def __init__(self, seed, config, augment=True, augs=['flip'], inspect_augment=True, aug_prop=0.2):
        super.__init__()
        self.seed = seed
        self.augment = augment
        self.augs = augs
        self.inspect_augment = inspect_augment
        self.aug_prop = aug_prop
        self.augmentation = Augment(self.augs, self.seed) if self.augment else None
        self.segmenter = UDTransNet(config=config, n_channels=config.num_channels, n_classes=config.num_classes, img_size=config.img_size)
    
    def call(self, images, masks):
        '''
        Augmentation component: Augment (aug_prop*100)% of the images in a given batch
        '''
        num_to_aug = images.shape[0] //(1/self.aug_prop)
        aug_images, aug_masks = self.augmentation(images[:num_to_aug], masks[:num_to_aug]) if self.augment else images[:num_to_aug], masks[:num_to_aug]
        norm_images, norm_masks = images[num_to_aug:], masks[num_to_aug:]
        images, truths = tf.concat([aug_images, norm_images], axis=0), tf.concat([aug_masks, norm_masks], axis=0) #recombine and send to other components
        
        if self.augment and self.inspect_augment:
            #inspection for visualization and debugging
            images = tf.transpose(images, perm=(0, 3, 2, 1))
            masks = tf.transpose(masks, perm=(0, 3, 2, 1))
            
            plt.figure()
            plt.imshow(images[0].numpy())
            plt.imshow(masks[0].numpy())
            assert self.inspect_augment, "Inspect Image and Mask"
            
        predicts = self.segmenter(images)
        
        return predicts, truths
            
        
            
        
            
            
            
    
