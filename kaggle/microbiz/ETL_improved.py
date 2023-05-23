import pandas as pd
import xgboost as xg
from tqdm import tqdm
import numpy as np
from etna.datasets import TSDataset
from etna.transforms import (DensityOutliersTransform, 
                             TimeSeriesImputerTransform, 
                             DifferencingTransform,
                             LagTransform,
                             StandardScalerTransform
                             )



#import warnings
#warnings.filterwarnings("ignore", category=RuntimeWarning) 

class MicroBusDataset:
    def __init__(self, train_csv, test_csv, census_csv):
        self.census_df = pd.read_csv(census_csv)
        self.train_df = pd.read_csv(train_csv)
        self.test_df = pd.read_csv(test_csv)
        self.raw_df = pd.DataFrame()
        self.train_ts = 39
        self.test_ts = 8
        
        self.num_states = None
        self.num_counties = None

    def common_load(self):
        self.train_df["first_day_of_month"] = pd.to_datetime(self.train_df["first_day_of_month"])
        self.test_df["first_day_of_month"] = pd.to_datetime(self.test_df["first_day_of_month"])

        self.train_df["istest"] = 0
        self.test_df["istest"] = 1
        raw = pd.concat((self.train_df, self.test_df)).sort_values(["cfips", "row_id"]).reset_index(drop=True)
        raw['first_day_of_month'] = pd.to_datetime(raw["first_day_of_month"])
        raw['county'] = raw.groupby('cfips')['county'].ffill()#.astype("category")
        raw['state'] = raw.groupby('cfips')['state'].ffill()#.astype("category")
        
        raw["year"] = raw["first_day_of_month"].dt.year.astype(str)#.astype("category")
        raw["month"] = raw["first_day_of_month"].dt.month.astype(str)#.astype("category")
        
        #print(raw.groupby(['cfips', 'row_id']).sum())
        raw["dcount"] = raw.groupby(['cfips'])['row_id'].cumcount()
        raw['county'] = (raw['county'] + raw['state']).factorize()[0]
        raw['state'] = raw['state'].factorize()[0]
        
        #raw["county"] = raw["county"].astype("category")
        #raw["state"] = raw["state"].astype("category")
        
        
        
        self.num_states = len(raw["state"].unique())
        self.num_counties = len(raw["county"].unique())
        
        #raw = raw.drop(["county_i", "state_i"], axis=1)
        #print(raw[raw.istest==0].isnull().sum())
        
        
        #print(raw["dcount"])
        #print(raw.info())
        self.raw_df = raw

    def check_empty(self):
        if self.raw_df.empty:
            self.common_load()
    
    def xgb_featurize(self, raw, target='microbusiness_density', target_act='active_tmp', lags = 6):
        """
        Build XGB Features with respect to lag of target
        """
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
        #print(raw["mbd_lag_2"].head())
        return raw, feats
    
    def xgb_prune(self, raw):
        """
        XGB Anomaly Handling
        """
        
        lag = 1
        raw[f'mbd_lag_{lag}'] = raw.groupby('cfips')['microbusiness_density'].shift(lag).bfill()
        raw['dif'] = (raw['microbusiness_density'] / raw[f'mbd_lag_{lag}']).fillna(1).clip(0, None) - 1
        raw.loc[(raw[f'mbd_lag_{lag}']==0), 'dif'] = 0
        raw.loc[(raw[f'microbusiness_density']>0) & (raw[f'mbd_lag_{lag}']==0), 'dif'] = 1
        raw['dif'] = raw['dif'].abs()
        #print(raw[raw.istest==0].isnull().sum())
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
        #print(raw[raw.istest==0].isnull().sum())
        #print(len(outliers), cnt)

        lag = 1
        raw[f'mbd_lag_{lag}'] = raw.groupby('cfips')['microbusiness_density'].shift(lag).bfill()
        raw['dif'] = (raw['microbusiness_density'] / raw[f'mbd_lag_{lag}']).fillna(1).clip(0, None) - 1
        raw.loc[(raw[f'mbd_lag_{lag}']==0), 'dif'] = 0
        raw.loc[(raw[f'microbusiness_density']>0) & (raw[f'mbd_lag_{lag}']==0), 'dif'] = 1
        raw['dif'] = raw['dif'].abs()
        #print(raw.groupby(['cfips', 'first_day_of_month']).sum().head(50))
        #print(raw[raw.istest==0].isnull().sum())
        return raw
    
    def xgb_transform(self, raw):
        """
        transform target for SMAPE Evaluation
        """
        raw['target'] = raw.groupby('cfips')['microbusiness_density'].shift(-1)
        raw['target'] = raw['target']/raw['microbusiness_density'] - 1
        
        raw.loc[raw['cfips']==28055, 'target'] = 0.0
        raw.loc[raw['cfips']==48269, 'target'] = 0.0

        raw['lastactive'] = raw.groupby('cfips')['active'].transform('last')

        dt = raw.loc[raw.dcount==28].groupby('cfips')['microbusiness_density'].agg('last')
        raw['lasttarget'] = raw['cfips'].map(dt)
        
        #targ_check = raw.groupby(['cfips', 'first_day_of_month']).sum()
        #targ_check = targ_check[targ_check.istest==0]
        #targ_check = raw[raw.istest==0].isnull().sum()
        #print(targ_check)
        
        return raw
    

    def load_xgb(self):
        self.check_empty()
        raw_xgb = self.raw_df.copy()
        raw_xgb = self.xgb_prune(raw_xgb)
        raw_xgb = self.xgb_transform(raw_xgb)
        raw_xgb, feats = self.xgb_featurize(raw_xgb, 'target', 'active', lags=4)
        features = ['state_i']
        features += feats
        
        return raw_xgb, features
    

    
    def etna_transforms(self, n_imputer_wins, num_lags):
        lag_cols = [f"lag_{lg}" for lg in range(1, num_lags)]
        outlier_remover = outlier_remover = DensityOutliersTransform(window_size=10, distance_coef=1.5, in_column="target")
        outlier_imputer = TimeSeriesImputerTransform(in_column="target", strategy="running_mean", window=n_imputer_wins)
        diff_transformer = DifferencingTransform(in_column="target", period=1, order=1, inplace=False, out_column="diff")
        lag_transformer = LagTransform(in_column="target", lags=range(1, num_lags), out_column="lag")
        
        

        
        transforms = [outlier_remover,
                      outlier_imputer,
                      diff_transformer,
                      lag_transformer,
                      
                      ]
        
        
        
        return transforms
    
    def prep_pred(self, pred):
        pred = pred.to_pandas(flatten=True)
        pred = pred.drop(columns=["active", "target"])
        pred = TSDataset.to_dataset(pred)
        return pred
    

    
    def convert_to_etna(self, raw, exog):
        raw = TSDataset.to_dataset(raw)
        exog = TSDataset.to_dataset(exog)
        raw = TSDataset(df=raw, freq="MS", df_exog=exog, known_future=["state", "county", "month", "year"])
        #assert True!=True, "Stooped"
        
        raw_train, raw_pred = raw.train_test_split(test_size=9)
        #raw_train, raw_val = raw_train.train_test_split(test_size=10)
        
        n_imputer_wins = 5
        n_lags = 5
        
        
        transforms =self.etna_transforms(n_imputer_wins, num_lags=n_lags)
        
        
        print(f"Imputing outliers with running {n_imputer_wins}-windowed mean")
        print("Differencing and adding lags")
        raw_train.fit_transform(transforms)
        #raw_train.df.fillna(0, inplace=True)
        
        raw_pred = self.prep_pred(raw_pred)
        
       

        
        raw_dict = {"train": raw_train,
                    "pred": raw_pred
                    }
        
        return raw_dict
    
  
    
    def load_etna(self):
        self.check_empty()
        raw_etna = self.raw_df.copy()
        raw_etna = raw_etna.drop(["dcount","row_id", "istest"], axis=1)
        raw_etna = raw_etna.rename(columns={"first_day_of_month": "timestamp", "cfips": "segment", 
                                  "microbusiness_density": "target"})
        raw_etna = raw_etna[["timestamp", "segment", "year", "month", "county", "state", "active", "target"]]
        
        exog_etna = raw_etna[["timestamp", "segment", "state", "county", "month", "year", "active"]]
        raw_etna = raw_etna.drop(["state", "county", "month", "year", "active"], axis=1)
        print(raw_etna.columns)
        #assert True!=True, "Stooped"
        raw_etna = self.convert_to_etna(raw_etna, exog=exog_etna)

        return raw_etna
    
    def get_num_states(self):
        assert self.num_states is not None, "Set up num_states"
        return self.num_states
    
    def get_num_counties(self):
        assert self.num_counties is not None, "Set up num_counties"
        return self.num_counties
        

if __name__ == "__main__":
    train_csv = "./train.csv"
    test_csv = "./test.csv"
    census_csv = "./census_starter.csv"
    
    microset = MicroBusDataset(train_csv, test_csv, census_csv)
    #raw_xgb, features = microset.load_xgb()
    etna_ds = microset.load_etna()
    print("etna Dataset Loaded")
    print(etna_ds["pred"].head())
    #print(etna_ds.tail(10))
    #print("-------Dataset Statistics--------\n")
    
    #print(raw_xgb.head())
    
        
