#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 22:54:16 2023

@author: hal
"""
import pandas as pd
from etna.datasets import generate_periodic_df

from etna.datasets import TSDataset

from etna.models import CatBoostPerSegmentModel

from etna.transforms import LagTransform

import h5py
import pickle

def load_test():
    horizon = 7
    with open("data.pkl", "rb") as data_file:
        ts = pickle.load(data_file)
        
    with open("cat.pkl", "rb") as mod_file:
        model = pickle.load(mod_file)
        
    future = ts.make_future(horizon)
    print("Testing Dataset..\n")
    
    forecast = model.forecast(future)
    print("Loaded Model Forecast frame\n")
    print(forecast[:, :, "target"])

def gen_test():
    print("generating dataset...")
    
    classic_df = generate_periodic_df(
    
        periods=100,
    
        start_time="2020-01-01",
    
        n_segments=4,
    
        period=7,
    
        sigma=3
    
    )
    
    df = TSDataset.to_dataset(df=classic_df)
    
    ts = TSDataset(df, freq="D")
    
    horizon = 7
    
    transforms = [
    
        LagTransform(in_column="target", lags=[horizon, horizon+1, horizon+2])
    
    ]
    
    ts.fit_transform(transforms=transforms)
    
    future = ts.make_future(horizon)
    
    mod_params = {"logging_level": "Verbose",
                  "thread_count": 6,
                  "task_type": "GPU",
                  
                  }
    
    model = CatBoostPerSegmentModel()
    print("Fitting model.....")
    model.fit(ts=ts)
    CatBoostPerSegmentModel(iterations = None, depth = None, learning_rate = None,
    logging_level = 'Silent', l2_leaf_reg = None, thread_count = None, )
    
    forecast = model.forecast(future)

    print(forecast[:, :, "target"])
    
    with open("data.pkl", "wb") as data_file:
        pickle.dump(ts, data_file, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open("cat.pkl", "wb") as mod_file:
        pickle.dump(model, mod_file, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("Data and model pickled")
    

def check_template():
    
    

    
load_test()