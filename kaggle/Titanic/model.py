#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 01:14:11 2022

@author: hal
"""
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Sequential 
from tensorflow.keras.initializers import HeUniform as He

def titan_model(inp_shape, num_classes):
    model = Sequential()
    model.add(layers.InputLayer(input_shape=inp_shape))
    model.add(layers.Dense(32, activation=layers.LeakyReLU(), kernel_initializer=He()))
    model.add(layers.Dense(32, activation=layers.LeakyReLU(), kernel_initializer=He()))
    #model.add(layers.Dense(64, activation=layers.LeakyReLU(), kernel_initializer=He()))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

    
