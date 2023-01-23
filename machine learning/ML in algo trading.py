import warnings
warnings.filterwarnings('ignore')

import os, sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import cm
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz, _tree
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error, make_scorer
import graphviz

import statsmodels.api as sm


import pandas_ta as ta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import requests
from math import floor
from termcolor import colored as cl
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)
from binance.client import Client
import datetime as dt
import json
from sklearn import tree
from sklearn.metrics import classification_report
from utils import MultipleTimeSeriesCV

sys.path.insert(1, os.path.join(sys.path[0], '..'))

sns.set_style('white')

results_path = Path('results', 'decision_trees')
if not results_path.exists():
    results_path.mkdir(parents=True)

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

lin_reg = LinearRegression()
lin_reg.fit(X_cls_train, y_cls_train)
print(lin_reg.intercept_, lin_reg.coef_)

log_reg_sm = sm.Logit(endog=y_cls_train, exog=sm.add_constant(X_cls_train))
log_result = log_reg_sm.fit()
print(log_result.summary())
log_reg_sk = LogisticRegression()
log_reg_sk.fit(X_cls_train, y_cls_train)
print(log_reg_sk.coef_)

y_cls_pred = log_reg_sk.predict(X_cls_test)
report = classification_report(y_cls_test, y_cls_pred)
print(report)
print(y_cls_pred)

n_splits = 10
train_period_length = 60
test_period_length = 6
lookahead = 1

cv = MultipleTimeSeriesCV(n_splits=n_splits,
                          train_period_length=train_period_length,
                          test_period_length=test_period_length,
                          lookahead=lookahead)