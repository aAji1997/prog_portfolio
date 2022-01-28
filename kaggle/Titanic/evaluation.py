#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 12:01:53 2022

@author: hal
"""

from IPython.display import display
import pandas as pd
import numpy as np
import tensorflow as tf
import string
from sklearn.preprocessing import OneHotEncoder
#from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing
from ETL import load_data
from model import titan_model

def load_test():
    #Initial loading
    ti_df_valid = pd.read_csv('data/test.csv')
    #ti_df_valid = ti_df_valid.dropna()
    #ti_df_test = pd.read_csv("data/test.csv")
    print(ti_df_valid.dtypes)
    
    allowed_types = [np.float64, np.float32,np.int32, np.int64]
    ti_non_num = ti_df_valid.select_dtypes(exclude=allowed_types)
    
    #Cleaning and pruning
    ti_non_num = ti_non_num.fillna("NAN")
    
    ti_numerical = ti_df_valid.select_dtypes(include=allowed_types)
    ti_numerical = ti_numerical.apply(lambda x: x.fillna(x.median()),axis=0)
    ti_numerical = ti_numerical.drop(['PassengerId'], axis=1)
    
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
    
    X = np.concatenate((ti_numerical, ti_non_num), axis=1)
    inp_shape = X.shape[1:]
    print("------------Train data --------------------------------")
    (x_train, y_train), (x_test, y_test) = load_data()
    print(x_train.shape, X.shape)
    
    model = titan_model(inp_shape, 2)
    model.load_weights('ti_weights.h5')
    
    raw_predictions = model.predict(X, batch_size=64)
    predictions = np.argmax(raw_predictions, axis=1)
    print(predictions)
    
    return ti_df_valid, predictions

    
def main():
    load_test()
    
main()
