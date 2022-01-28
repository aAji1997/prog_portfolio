#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 14:05:00 2022

@author: hal
"""
from IPython.display import display
import pandas as pd
import numpy as np
import tensorflow as tf
import string
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing
from tensorflow.keras.utils import to_categorical

        
        
def load_data():
    #Initial loading
    ti_df_train = pd.read_csv('data/train.csv')
    #ti_df_train = ti_df_train.dropna()
    #ti_df_test = pd.read_csv("data/test.csv")
    #print(ti_df_train.dtypes)
    #print(ti_df_train['Survived'].median())
    
    allowed_types = [np.float64, np.float32,np.int32, np.int64]
    ti_non_num = ti_df_train.select_dtypes(exclude=allowed_types)
    
    #Cleaning and pruning
    ti_non_num = ti_non_num.fillna("NAN")
    
    ti_numerical = ti_df_train.select_dtypes(include=allowed_types)
    ti_numerical = ti_numerical.apply(lambda x: x.fillna(x.median()),axis=0)
    ti_numerical = ti_numerical.drop(['PassengerId'], axis=1)
    
    
    
    y = ti_numerical['Survived']
    y = y.values
    ti_numerical = ti_numerical.drop(['Survived'], axis=1)
    display(ti_numerical)
    ti_non_num = ti_non_num.drop(['Ticket','Cabin','Name'], axis=1)
    print(ti_numerical.columns)
    
    ti_non_num = ti_non_num.values
    ti_numerical = ti_numerical.values
    print(np.unique(ti_non_num))
    print(ti_numerical.shape)
    
    #Encoding
    ohe = OneHotEncoder(sparse=False)
    ti_non_num = ohe.fit_transform(ti_non_num)
    
    #print(ti_non_num.shape, ti_numerical.shape)
    
    ti_numerical = preprocessing.StandardScaler().fit(ti_numerical).transform(ti_numerical)
    
    #X = np.concatenate((ti_numerical, ti_non_num), axis=1)
    X_train_numerical, X_test_numerical, X_train_non_num, X_test_non_num, y_train, y_test = train_test_split(ti_numerical, ti_non_num,
                                                                                                             y, test_size=0.2, random_state=42)
    #print(X_train_non_num.shape, X_test_non_num.shape)
    
    X_train_numerical = preprocessing.StandardScaler().fit(X_train_numerical).transform(X_train_numerical)
    X_test_numerical = preprocessing.StandardScaler().fit(X_test_numerical).transform(X_test_numerical)
    
    X_train = np.concatenate((X_train_numerical, X_train_non_num), axis=1)
    X_test = np.concatenate((X_test_numerical, X_test_non_num), axis=1)
    y_train_enc = to_categorical(y_train, num_classes=2, dtype='int64')
    y_test_enc = to_categorical(y_test, num_classes=2, dtype='int64')
    print(X_test.shape, y_test_enc.shape)
    
    print("Target Correspondence:", y_train_enc[0], y_train[0])
    
    return (X_train, y_train_enc),(X_test, y_test_enc)


    