#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 22:18:49 2022

@author: hal
"""
import pandas as pd
import numpy as np
from evaluation import load_test
from IPython.display import display

def submission():
    ti_df_valid, predictions = load_test()
    
    #Append predictions to dataframe
    ti_df_valid['Survived'] = predictions.tolist()
    #display(ti_df_valid)
    subm_df = ti_df_valid[['PassengerId', 'Survived']]
    subm_df = subm_df.drop([418])
    display(subm_df)
    subm_df.to_csv('AA_submission.csv', index=False)
def main():
    submission()
    
main()