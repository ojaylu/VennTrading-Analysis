# https://www.datacamp.com/tutorial/lstm-python-stock-market?utm_source=google&utm_medium=paid_search&utm_campaignid=19589720821&utm_adgroupid=143216588577&utm_device=c&utm_keyword=&utm_matchtype=&utm_network=g&utm_adpostion=&utm_creative=645433043010&utm_targetid=aud-438999696879:dsa-1947282172981&utm_loc_interest_ms=&utm_loc_physical_ms=9069537&utm_content=dsa~page~community-tuto&utm_campaign=230119_1-sea~dsa~tutorials_2-b2c_3-row-p1_4-prc_5-na_6-na_7-le_8-pdsh-go_9-na_10-na_11-na-jrjan23&gclid=Cj0KCQiAt66eBhCnARIsAKf3ZNEL_yPdPrDI1yVl6Ko_R7iGuGHxiN_TFfUB0c_1uA_lBlA-pH9_mNIaApbnEALw_wcB
# https://www.analyticsvidhya.com/blog/2021/01/bear-run-or-bull-run-can-reinforcement-learning-help-in-automated-trading/
# https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
# https://medium.com/auquan/https-medium-com-auquan-machine-learning-techniques-trading-b7120cee4f05

import pandas as pd
import pandas_ta as ta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import requests
import matplotlib.pyplot as plt
from math import floor
from termcolor import colored as cl
from binance.client import Client

import datetime as dt
import json
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeRegressor

from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from itertools import product
import math
from numpy import nan

from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
#import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler

# from keras.models import Sequential
# from keras.layers import LSTM
# from keras.layers import Dropout
# from keras.layers import Dense

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)

url = 'https://api.binance.com/api/v3/klines'
symbol = 'ADAUSDT'
interval = '1d'
start = str(int(dt.datetime(2022,3,1).timestamp()*1000))
end = str(int(dt.datetime(2022,11,1).timestamp()*1000))
par = {'symbol': symbol, 'interval': interval, 'startTime': start, 'endTime': end}
data = pd.DataFrame(json.loads(requests.get(url, params= par).text))
#format columns name
data.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume','close_time', 'qav', 'num_trades','taker_base_vol', 'taker_quote_vol', 'ignore']
data.index = [dt.datetime.fromtimestamp(x/1000.0) for x in data.datetime]
data=data.astype(float)

df = data

def rsi_buy_sell(signal, col1, col2):
        signPriceBuy = []
        signPriceSell = []
        flag = -1
        #Loop through the length of the data set
        for i in range(0, len(signal)):
                #If RSI_14 > 50 Then Buy --> col1 => 'Lower boundary' and col2 => 'upper boundary'
                if (signal[col1][i] == 1 and signal[col2][i] == 0) and flag != 1:
                        signPriceBuy.append(signal['close'][i])
                        signPriceSell.append(np.nan)
                        flag = 1
                # If OBV < OBV_EMA Then Sell
                elif signal[col1][i] == 0 and signal[col2][i] == 1 and flag != 0:
                        signPriceSell.append(signal['close'][i])
                        signPriceBuy.append(np.nan)
                        flag = 0
                else:
                        signPriceSell.append(np.nan)
                        signPriceBuy.append(np.nan)
        newsignPriceBuy = [item for item in signPriceBuy if not(math.isnan(item)) == True]
        newsignPriceSell = [item for item in signPriceSell if not(math.isnan(item)) == True]
        surplus = sum(newsignPriceSell) - sum(newsignPriceBuy)
        return (signPriceBuy, signPriceSell)

def rsi_buy_sell_surplus(signal, col1, col2):
        signPriceBuy = []
        signPriceSell = []
        flag = -1
        #Loop through the length of the data set
        for i in range(0, len(signal)):
                #If RSI_14 > 50 Then Buy --> col1 => 'Lower boundary' and col2 => 'upper boundary'
                if (signal[col1][i] == 1 and signal[col2][i] == 0) and flag != 1:
                        signPriceBuy.append(signal['close'][i])
                        signPriceSell.append(np.nan)
                        flag = 1
                # If OBV < OBV_EMA Then Sell
                elif signal[col1][i] == 0 and signal[col2][i] == 1 and flag != 0:
                        signPriceSell.append(signal['close'][i])
                        signPriceBuy.append(np.nan)
                        flag = 0
                else:
                        signPriceSell.append(np.nan)
                        signPriceBuy.append(np.nan)
        newsignPriceBuy = [item for item in signPriceBuy if not(math.isnan(item)) == True]
        newsignPriceSell = [item for item in signPriceSell if not(math.isnan(item)) == True]
        surplus = sum(newsignPriceSell) - sum(newsignPriceBuy)
        return surplus

def get_macd(price, slow, fast, smooth):
        exp1 = price.ewm(span = fast, adjust = False).mean()
        exp2 = price.ewm(span = slow, adjust = False).mean()
        macd = pd.DataFrame(exp1 - exp2).rename(columns = {'close':'macd'})
        signal = pd.DataFrame(macd.ewm(span = smooth, adjust = False).mean()).rename(columns = {'macd':'signal'})
        hist = pd.DataFrame(macd['macd'] - signal['signal']).rename(columns = {0:'hist'})
        return macd, signal, hist

data['macd'] = get_macd(data['close'], 26, 12, 9)[0]
data['macd_signal'] = get_macd(data['close'], 26, 12, 9)[1]
data['macd_hist'] = get_macd(data['close'], 26, 12, 9)[2]
data = data.dropna()

length = 14
data.ta.rsi(close='close', length=length, append=True, signal_indicators=True, xa=50, xb=55)

data["EMA10"] = ta.ema(data["close"], length=10)
data["EMA30"] = ta.ema(data["close"], length=30)
data['ATR'] = df.ta.atr(df['high'].values.any(), df['low'].values.any(), df['close'].values.any(), timeperiod=14)
#data['ADX'] = df.ta.adx(df['high'].values.any(), df['low'].values.any(), df['close'].values.any(), timeperiod=14)

data['ClgtEMA10'] = np.where(data['close'] > data['EMA10'], 1, -1)
data['EMA10gtEMA30'] = np.where(data['EMA10'] > data['EMA30'], 1, -1)
data['MACDSIGgtMACD'] = np.where(data['macd_signal'] > data['macd'], 1, -1)

data['Return'] = data['close'].pct_change(1).shift(-1)
data['target_cls'] = np.where(data.Return > 0, 1, 0)
data['target_rgs'] = data['Return']

data = data.dropna()

predictors_list = ['ATR','RSI_14', 'ClgtEMA10', 'EMA10gtEMA30', 'MACDSIGgtMACD']
X = data[predictors_list]

y_cls = data.target_cls
y_rgs = data.target_rgs

y=y_cls
X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(X, y, test_size=0.3, random_state=432, stratify=y)

train_length = int(len(data)*0.70)
X_rgs_train = X[:train_length]
X_rgs_test = X[train_length:]
y_rgs_train = y_rgs[:train_length]
y_rgs_test = y_rgs[train_length:]

lstm_training_set = data.iloc[:, 1:2].values

sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(lstm_training_set)

X_train = []
y_train = []
for i in range(60, 200):
    X_train = np.append(X_train, training_set_scaled[i, 0])
    y_train = np.append(y_train, training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], 1))

# model = Sequential()
# model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[0], 1)))
# model.add(Dropout(0.2))
# model.add(LSTM(units=50,return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(units=50,return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(units=50))
# model.add(Dropout(0.2))
# model.add(Dense(units=1))
# model.compile(optimizer='adam',loss='mean_squared_error')
# model.fit(X_train,y_train,epochs=100,batch_size=32)

real_stock_price = data.iloc[:, 1:2].values

dataset_total = pd.concat((data['open'], data['open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(data) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 76):
    np.append(X_test, inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], 1))
# predicted_stock_price = model.predict(X_test)
# predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# print('the predicted price is ', predicted_stock_price)

