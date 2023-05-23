import pandas as pd
import xgboost as xg
from tqdm import tqdm
import numpy as np



def smape(y_true, y_pred):
    smap = np.zeros(len(y_true))
    
    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)
    
    pos_ind = (y_true!=0)|(y_pred!=0)
    smap[pos_ind] = num[pos_ind] / dem[pos_ind]
    
    return 100 * np.mean(smap)

def vsmape(y_true, y_pred):
    smap = np.zeros(len(y_true))
    
    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)
    
    pos_ind = (y_true!=0)|(y_pred!=0)
    smap[pos_ind] = num[pos_ind] / dem[pos_ind]
    
    return 100 * smap

def build_features(raw, target='microbusiness_density', target_act='active_tmp', lags = 6):
    feats = []
    for lag in range(1, lags):
        raw[f'mbd_lag_{lag}'] = raw.groupby('cfips')[target].shift(lag)
        raw[f'act_lag_{lag}'] = raw.groupby('cfips')[target_act].diff(lag)
        feats.append(f'mbd_lag_{lag}')
        feats.append(f'act_lag_{lag}')
        
    lag = 1
    for window in [2, 4, 6]:
        raw[f'mbd_rollmea{window}_{lag}'] = raw.groupby('cfips')[f'mbd_lag_{lag}'].transform(lambda s: s.rolling(window, min_periods=1).sum())        
        #raw[f'mbd_rollmea{window}_{lag}'] = raw[f'mbd_lag_{lag}'] - raw[f'mbd_rollmea{window}_{lag}']
        feats.append(f'mbd_rollmea{window}_{lag}')
        
    return raw, feats

def load_data():
    #Acquisition And sorting
    census_df = pd.read_csv("./census_starter.csv")
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    train_df["first_day_of_month"] = pd.to_datetime(train_df["first_day_of_month"])
    test_df["first_day_of_month"] = pd.to_datetime(test_df["first_day_of_month"])
    #print(train_df["first_day_of_month"].head())

    train_df["istest"] = 0
    test_df["istest"] = 1
    raw = pd.concat((train_df, test_df)).sort_values(['cfips','row_id']).reset_index(drop=True)
    raw['first_day_of_month'] = pd.to_datetime(raw["first_day_of_month"])
    raw['county'] = raw.groupby('cfips')['county'].ffill()
    raw['state'] = raw.groupby('cfips')['state'].ffill()
    raw["year"] = raw["first_day_of_month"].dt.year
    raw["month"] = raw["first_day_of_month"].dt.month
    raw["dcount"] = raw.groupby(['cfips'])['row_id'].cumcount()
    raw['county_i'] = (raw['county'] + raw['state']).factorize()[0]
    raw['state_i'] = raw['state'].factorize()[0]
    #print("Last 10 Values of Raw Dataframe:\n")
    #print(raw.tail(10))

    #Anomaly Handling
    lag = 1
    raw[f'mbd_lag_{lag}'] = raw.groupby('cfips')['microbusiness_density'].shift(lag).bfill()
    raw['dif'] = (raw['microbusiness_density'] / raw[f'mbd_lag_{lag}']).fillna(1).clip(0, None) - 1
    raw.loc[(raw[f'mbd_lag_{lag}']==0), 'dif'] = 0
    raw.loc[(raw[f'microbusiness_density']>0) & (raw[f'mbd_lag_{lag}']==0), 'dif'] = 1
    raw['dif'] = raw['dif'].abs()
    
    outliers = []
    cnt = 0
    for o in tqdm(raw.cfips.unique()):
        indices = (raw['cfips']==o)
        tmp = raw.loc[indices].copy().reset_index(drop=True)
        var = tmp.microbusiness_density.values.copy()
        #vmax = np.max(var[:38]) - np.min(var[:38])
        
        for i in range(37, 2, -1):
            thr = 0.20*np.mean(var[:i])
            difa = abs(var[i]-var[i-1])
            if (difa>=thr):
                var[:i] *= (var[i]/var[i-1])
                outliers.append(o)
                cnt+=1
        var[0] = var[1]*0.99
        raw.loc[indices, 'microbusiness_density'] = var
        
    outliers = np.unique(outliers)
    #print(len(outliers), cnt)

    lag = 1
    raw[f'mbd_lag_{lag}'] = raw.groupby('cfips')['microbusiness_density'].shift(lag).bfill()
    raw['dif'] = (raw['microbusiness_density'] / raw[f'mbd_lag_{lag}']).fillna(1).clip(0, None) - 1
    raw.loc[(raw[f'mbd_lag_{lag}']==0), 'dif'] = 0
    raw.loc[(raw[f'microbusiness_density']>0) & (raw[f'mbd_lag_{lag}']==0), 'dif'] = 1
    raw['dif'] = raw['dif'].abs()
    #raw.groupby('dcount')['dif'].sum().plot()
    
    #transform target for SMAPE Evaluation
    raw['target'] = raw.groupby('cfips')['microbusiness_density'].shift(-1)
    raw['target'] = raw['target']/raw['microbusiness_density'] - 1
    
    raw.loc[raw['cfips']==28055, 'target'] = 0.0
    raw.loc[raw['cfips']==48269, 'target'] = 0.0

    raw['lastactive'] = raw.groupby('cfips')['active'].transform('last')

    dt = raw.loc[raw.dcount==28].groupby('cfips')['microbusiness_density'].agg('last')
    raw['lasttarget'] = raw['cfips'].map(dt)
    
    #raw['lastactive'].clip(0, 8000).hist(bins=30)
    
    #Build Features with respect to lag of target
    raw, feats = build_features(raw, 'target', 'active', lags = 4)
    features = ['state_i']
    features += feats

    return raw, features

if __name__ == "__main__":
    load_data()
