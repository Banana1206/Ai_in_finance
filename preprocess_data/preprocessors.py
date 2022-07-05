import fxcmpy
import pandas as pd
import numpy as np
import datetime as dt
import gym
import gym_anytrading
from stockstats import StockDataFrame as Sdf

# from stable_baselines3.ppo.policies import MlpPolicy
# from stable_baselines3 import A2C, PPO, DDPG
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.cmd_util import make_vec_env

# input token to crawl api
token = '5b6cbe5a6c2f112df27c190ee25b6ecf5ca253ad' 
name_file_dl = 'data_train'

instrument = 'USD/JPY' #
# 'EUR/USD',
#  'USD/JPY',
#  'GBP/USD',
#  'USD/CHF',
#  'EUR/CHF',
#  'AUD/USD',
#  'USD/CAD',
#  'NZD/USD',
#  'EUR/GBP',
#  'EUR/JPY',
#  'GBP/JPY',
#  'CHF/JPY',
#  'GBP/CHF',
#  'EUR/AUD',
#  'EUR/CAD',
#  'AUD/CAD',
#  'AUD/JPY',
#  'CAD/JPY',
#  'NZD/JPY'

# start = dt.datetime(2020, 6,1 )
# stop = dt.datetime(2021, 6, 16)

# def download_fx_data(start_date, end_date, ):
#     con =  fxcmpy.fxcmpy(access_token = token, log_level = 'error', log_file = None)
#     df = 


# TRAINING_DATA_FILE = "data/dow_30_2009_2020.csv"

# now = datetime.datetime.now()
# TRAINED_MODEL_DIR = f"trained_models/{now}"
# os.makedirs(TRAINED_MODEL_DIR)
# TURBULENCE_DATA = "data/dow30_turbulence_index.csv"

# TESTING_DATA_FILE = "test.csv"


def load_dataset(file_name: str):
    """
    load csv dataset from path
    :return: (df) pandas dataframe
    """
    #_data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
    _data = pd.read_csv(file_name)
    return _data

def add_technical_indicator(df):
    """
    calcualte technical indicators
    use stockstats package to add technical inidactors
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    stock = Sdf.retype(df.copy())

#     stock['close'] = stock['adjcp']
#     unique_ticker = stock.tic.unique()

    macd = pd.DataFrame()
    rsi = pd.DataFrame()
    cci = pd.DataFrame()
    dx = pd.DataFrame()

    #temp = stock[stock.tic == unique_ticker[0]]['macd']

    temp_macd = stock['macd']
    temp_macd = pd.DataFrame(temp_macd)
    macd = macd.append(temp_macd, ignore_index=True)
    ## rsi
    temp_rsi = stock['rsi_30']
    temp_rsi = pd.DataFrame(temp_rsi)
    rsi = rsi.append(temp_rsi, ignore_index=True)
    ## cci
    temp_cci = stock['cci_30']
    temp_cci = pd.DataFrame(temp_cci)
    cci = cci.append(temp_cci, ignore_index=True)
    ## adx
    temp_dx = stock['dx_30']
    temp_dx = pd.DataFrame(temp_dx)
    dx = dx.append(temp_dx, ignore_index=True)


    df['macd'] = macd
    df['rsi'] = rsi
    df['cci'] = cci
    df['adx'] = dx

    return df


def preprocess_data(file_name):
    """data preprocessing pipeline"""
    
    df = load_dataset(file_name)
    df_final=add_technical_indicator(df)
    # fill the missing values at the beginning
    df_final.fillna(method='bfill',inplace=True)
    return df_final


