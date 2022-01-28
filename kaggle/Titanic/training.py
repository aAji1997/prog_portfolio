#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 09:19:53 2022

@author: hal
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras as keras
import pandas as pd

from ETL import load_data
from sklearn.model_selection import KFold
from model import titan_model

from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam

def setup():
    
    #TTraining params
    batch_size = 64
    num_classes = 2
    epochs = 50
    optimizer = Adam()
    loss_function = categorical_crossentropy
    verbosity = 1
    num_folds = 7
    
    #Getting data and defining metrics
    (X_train, y_train), (X_test, y_test) = load_data()
    print("Training Shape: ", X_train.shape, y_train.shape)
    print("Testing Shape: ", X_test.shape, y_test.shape)
    
    #Merging datasets for k-fold cross-validation
    features = np.concatenate((X_train, X_test), axis=0)
    targets = np.concatenate((y_train, y_test), axis=0)
    
    inp_shape = features.shape[1:] #Features shape
    print(inp_shape)
    #print(np.argmax(y_train, axis=1))
    
    #Metrics
    acc_per_fold = []
    loss_per_fold = []
    
    #K-fold Cross-validator
    kfold = KFold(n_splits=num_folds, shuffle=True)
    
    fold_no = 1
    
    #Training and Evaluation
    for train, test in kfold.split(features, targets):
        #Get model and compile
        model = titan_model(inp_shape, num_classes)
        model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
        print("-----------------------------------------------")
        print(f'Training for fold {fold_no} ...')
        
        #Fit to model
        history = model.fit(x=features[train], y=targets[train], batch_size=batch_size, epochs=epochs, validation_data=(features[test], targets[test]),
                             workers=6, use_multiprocessing=True, verbose=verbosity)
        #Generate score metrics
        scores = model.evaluate(features[test], targets[test], verbose=0)
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        
        #Increase fold number
        fold_no = fold_no + 1
        
    #Get average scores
    print('------------------------------------------------------------------------')
    #plt.figure()
    print('Score per fold')
    
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
        #plt.plot(i, loss_per_fold[i])
        
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')
    
    
    fig1, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(loss_per_fold, color=color)
    ax1.set_xlabel('fold', color=color)
    ax1.set_ylabel('total loss', color=color)
    ax1.tick_params(axis='y', color=color)
    
    
    fig2, ax2 = plt.subplots()  
    color = 'tab:blue'
    ax2.set_ylabel('accuracy', color=color)  # we already handled the x-label with ax1
    ax2.plot(acc_per_fold, color=color)
    ax2.tick_params(axis='y', color=color)
    fig1.tight_layout()
    
    model.save_weights('ti_weights.h5')
    print("Model Saved")
    print(model.summary())
    

        
        
        
    
    
    

def main():
    setup()
    
main()