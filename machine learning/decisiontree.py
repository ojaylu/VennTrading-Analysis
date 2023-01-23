#reference: 
# https://www.projectpro.io/recipes/find-optimal-parameters-using-gridsearchcv
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# https://medium.com/auquan/https-medium-com-auquan-machine-learning-techniques-trading-b7120cee4f05
# https://www.pythonpool.com/python-remove-nan-from-list/
# https://www.geeksforgeeks.org/how-to-get-column-names-in-pandas-dataframe/
# https://www.tutorialkart.com/python/python-range/python-range-to-list/
# https://www.quantstart.com/static/ebooks/sat/sample.pdf
# https://medium.com/auquan/https-medium-com-auquan-machine-learning-techniques-trading-b7120cee4f05
# https://blog.quantinsti.com/decision-tree/
import pandas as pd
import pandas_ta as ta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import requests
import matplotlib.pyplot as plt
from math import floor
from termcolor import colored as cl
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)
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

clf = DecisionTreeClassifier(criterion='gini', splitter='best',max_depth=3,min_samples_split=2,min_samples_leaf=6,min_weight_fraction_leaf=0.0,
                                    max_features=None,random_state=42,max_leaf_nodes=None,min_impurity_decrease=0.0)
clf = clf.fit(X_cls_train, y_cls_train)

dot_data = tree.export_graphviz(clf, out_file=None,filled=True,feature_names=predictors_list)

y_cls_pred = clf.predict(X_cls_test)
report = classification_report(y_cls_test, y_cls_pred)

print(report)
print(y_cls_pred)
print(graphviz.Source(dot_data).view())

dtr = DecisionTreeRegressor(criterion='squared_error', max_depth=3, min_samples_leaf = 6)
dtr = dtr.fit(X_rgs_train, y_rgs_train)

dot_data2 = tree.export_graphviz(dtr,out_file = None ,filled=True,feature_names=predictors_list)
print(graphviz.Source(dot_data2).view())

parameters = {'max_depth': [3,4,5,6,7,8,9,10],
                'min_samples_split'    : [2],
                'min_samples_leaf' : [6,7,8,9,10],
                'min_weight_fraction_leaf'    : [0.0, 0.1, 0.2]}

grid_clf = GridSearchCV(estimator=clf, param_grid = parameters, cv = 5, n_jobs=-1)
grid_clf.fit(X_cls_train, y_cls_train)

print(" Results from Grid Search " )
print("\n The best estimator across ALL searched params:\n",grid_clf.best_estimator_)
print("\n The best score across ALL searched params:\n",grid_clf.best_score_)
print("\n The best parameters across ALL searched params:\n",grid_clf.best_params_)

#RSI optimization (calculation)


xa = list(range(31,50,5))
xb = list(range(51,70,5))
day = [7, 14, 31]
strat_params_list = list(product(xa, xb, day))
print(strat_params_list)
strat_params_dict_list = {'xa': [40, 50, 60],'xb': [45, 55, 65],'day' : [7, 14, 31]}
surplus = 0
spl = len(strat_params_list)
x = 0
strategyID = 0
for i, spl in enumerate(strat_params_list):
        print("Strategy %s out of %s..." % (i+1, spl))
        flag = data.ta.rsi(close='close', length=strat_params_list[i][2], append=True, signal_indicators=True, xa=strat_params_list[i][0], xb=strat_params_list[i][1])
        print(flag)
        if rsi_buy_sell_surplus(data, flag.columns[1], flag.columns[2]) > surplus:
                surplus = rsi_buy_sell_surplus(data, flag.columns[1], flag.columns[2])
                x = rsi_buy_sell(data, flag.columns[1], flag.columns[2])
                #print("This strategy is : ",buy_sell(data, flag.columns[1], flag.columns[2]))
                strategyID = i
                print("This strategy help you earn",rsi_buy_sell_surplus(data, flag.columns[1], flag.columns[2]))
        else:
                #print("This strategy is : ",buy_sell(data, flag.columns[1], flag.columns[2]))
                print("This strategy help you earn",rsi_buy_sell_surplus(data, flag.columns[1], flag.columns[2]))
                continue

print("the optimized strategy",strategyID,"you will use is ", x)
print("The money you will get using the optimized strategy",strategyID,"is", surplus)

#MACD optimization



