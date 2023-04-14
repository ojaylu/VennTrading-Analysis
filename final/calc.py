import pandas as pd
import pandas_ta as ta
import numpy as np
from macd import get_macd
from obv import *
from implement_stoch_macd_strategy_optimization import *
from stochastic import *
from pandas import Series, datetime

def calc(data):

    #Stochastic Oscillator Calculation
    data['%k'], data['%d'] = get_stoch_osc(data['high'], data['low'], data['close'], 14, 3)
    data['%k'].fillna(value=data['%k'].mean(), inplace=True)
    data['%d'].fillna(value=data['%d'].mean(), inplace=True)

    data['macd'] = get_macd(data['close'], 26, 12, 9)[0]
    data['macd_signal'] = get_macd(data['close'], 26, 12, 9)[1]
    data['macd_hist'] = get_macd(data['close'], 26, 12, 9)[2]
    #data = data.dropna()

    length = 14
    data.ta.rsi(close='close', length=length, append=True, signal_indicators=True, xa=50, xb=55)

    df = data
    data["EMA10"] = ta.ema(data["close"], length=10)
    data["EMA30"] = ta.ema(data["close"], length=30)
    data['ATR'] = df.ta.atr(df['high'].values.any(), df['low'].values.any(), df['close'].values.any(), timeperiod=14)
    #data['ADX'] = df.ta.adx(df['high'].values.any(), df['low'].values.any(), df['close'].values.any(), timeperiod=14)
    data['EMA10'].fillna(value=data['EMA10'].mean(), inplace=True)
    data['EMA30'].fillna(value=data['EMA30'].mean(), inplace=True)

    
    data['ClgtEMA10'] = np.where(data['close'] > data['EMA10'], 1, -1)
    data['EMA10gtEMA30'] = np.where(data['EMA10'] > data['EMA30'], 1, -1)
    data['MACDSIGgtMACD'] = np.where(data['macd_signal'] > data['macd'], 1, -1)

    data['Return'] = data['close'].pct_change(1).shift(-1)
    data['Return'].fillna(value=data['Return'].mean(), inplace=True)
    #data['target_cls'] = np.where(data.Return > 0, 1, 0)
    #data['target_rgs'] = data['Return']

    #data = data.dropna()
    if len(data) > 14:
        data['RSI_14'].fillna(value=data['RSI_14'].mean(), inplace=True)
        data['ATR'].fillna(value=data['ATR'].mean(), inplace=True)
    else:
        data['RSI_14'] = 0
        data['ATR'].fillna(value=0, inplace=True)

    OBVcalculation(data)
    data = data.fillna(value=np.nan)
    return data