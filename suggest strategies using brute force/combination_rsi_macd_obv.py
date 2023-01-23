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

    #Stochastic Oscillator Calculation

def get_stoch_osc(high, low, close, k_lookback, d_lookback):
        lowest_low = low.rolling(k_lookback).min()
        highest_high = high.rolling(k_lookback).max()
        k_line = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        d_line = k_line.rolling(d_lookback).mean()
        return k_line, d_line

data['%k'], data['%d'] = get_stoch_osc(data['high'], data['low'], data['close'], 14, 3)
data.tail()

    # MACD CALCULATION

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
data.tail()

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

def implement_stoch_macd_strategy_optimization(prices, k, d, macd, macd_signal, lb, ub):    
        buy_price = []
        sell_price = []
        stoch_macd_signal = []
        signal = 0

        for i in range(len(prices)):
            if k[i] < lb and d[i] < lb and macd[i] < 0 and macd_signal[i] < 0:
                if signal != 1:
                    buy_price.append(prices[i])
                    sell_price.append(np.nan)
                    signal = 1
                    stoch_macd_signal.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    stoch_macd_signal.append(0)
                    
            elif k[i] > ub and d[i] > ub and macd[i] > 0 and macd_signal[i] > 0:
                if signal != -1 and signal != 0:
                    buy_price.append(np.nan)
                    sell_price.append(prices[i])
                    signal = -1
                    stoch_macd_signal.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    stoch_macd_signal.append(0)
            
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                stoch_macd_signal.append(0)
                
        return buy_price, sell_price, stoch_macd_signal

def combination_MACD_RSI_surplus(signal, col1, col2, prices, k, d, macd, macd_signal, MACDLB, MACDUB):
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

    buy_price, sell_price, stoch_macd_signal = implement_stoch_macd_strategy_optimization(prices, k, d, macd, macd_signal, MACDLB, MACDUB)
    newsignPriceBuy = [item for item in signPriceBuy if not(math.isnan(item)) == True]
    newsignPriceSell = [item for item in signPriceSell if not(math.isnan(item)) == True]
    surplus = sum(newsignPriceSell) - sum(newsignPriceBuy)
    return surplus


#Calculate the On Balance Volume (OBV)
def OBVcalculation(data):
    OBV = []
    OBV.append(0)

    #Loop through the data set (close price) from the second row (index 1) to the end of the data set
    for i in range(1, len(data.close)):
        if data.close[i] > data.close[i-1]:
            OBV.append(OBV[-1] + data.volume[i])
        elif data.close[i] < data.close[i-1]:
            OBV.append(OBV[-1] - data.volume[i])
        else:
            OBV.append(OBV[-1])

    #Store the OBV and OBV Expontential Moving Average (EMA) into new columns
    data['OBV'] = OBV
    data['OBV_EMA'] = data['OBV'].ewm(span=20).mean()

    return data

def OBV_buy_sell(signal, col1, col2):
    signPriceBuy = []
    signPriceSell = []
    flag = -1
    #Loop through the length of the data set
    for i in range(0, len(signal)):
        #If OBV > OBV_EMA Then Buy --> col1 => 'OBV' and col2 => 'OBV_EMA'
        if signal[col1][i] > signal[col2][i] and flag != 1:
            signPriceBuy.append(signal['close'][i])
            signPriceSell.append(np.nan)
            flag = 1
        # If OBV < OBV_EMA Then Sell
        elif signal[col1][i] < signal[col2][i] and flag != 0:
            signPriceSell.append(signal['close'][i])
            signPriceBuy.append(np.nan)
            flag = 0
        else:
            signPriceSell.append(np.nan)
            signPriceBuy.append(np.nan)
    
    return (signPriceBuy, signPriceSell)


RSILB = list(range(31,50,5))
RSIUB = list(range(51,70,5))
day = [14, 31]
MACDLB = list(range(45,50,1))
MACDUB = list(range(50,55,1))
strat_params_list = list(product(RSILB, RSIUB, day, MACDLB, MACDUB))
strat_params_dict_list = {'xa': [40, 50, 60],'xb': [45, 55, 65],'day' : [7, 14, 31]}
surplus = 0
highestsurplus = 0
spl = len(strat_params_list)
MACDstrategy = 0
RSIstrategy = 0
OBVstrategy = 0
strategyID = 0

for i, spl in enumerate(strat_params_list):
    print("Strategy %s out of %s..." % (i+1, spl))
    RSIflag = data.ta.rsi(close='close', length=strat_params_list[i][2], append=True, signal_indicators=True, xa=strat_params_list[i][0], xb=strat_params_list[i][1])
    macd_buy_price, macd_sell_price, stoch_macd_signal = implement_stoch_macd_strategy_optimization(data['close'], data['%k'], data['%d'], data['macd'], data['macd_signal'], strat_params_list[i][3], strat_params_list[i][4])
    rsi_buy_price, rsi_sell_price = rsi_buy_sell(data, RSIflag.columns[1], RSIflag.columns[2])
    obv_buy_price, obv_sell_price = OBV_buy_sell(OBVcalculation(data), 'OBV', 'OBV_EMA')
    for j in range(len(macd_buy_price)):
        if not(math.isnan(macd_buy_price[j])) == True and not(math.isnan(rsi_buy_price[j])) == True:
            macd_buy_price.remove(macd_buy_price[j])
        if not(math.isnan(macd_sell_price[j])) == True and not(math.isnan(rsi_sell_price[j])) == True:
            macd_sell_price.remove(macd_sell_price[j])
    newMACDPriceBuy = [item for item in macd_buy_price if not(math.isnan(item)) == True]
    newMACDPriceSell = [item for item in macd_sell_price if not(math.isnan(item)) == True]
    newRSIPriceBuy = [item for item in rsi_buy_price if not(math.isnan(item)) == True]
    newRSIPriceSell = [item for item in rsi_sell_price if not(math.isnan(item)) == True]
    newOBVPriceBuy = [item for item in obv_buy_price if not(math.isnan(item)) == True]
    newOBVPriceSell = [item for item in obv_sell_price if not(math.isnan(item)) == True]
    surplus = sum(newMACDPriceSell) + sum(newRSIPriceSell) + sum(newOBVPriceSell) - sum(newMACDPriceBuy) - sum(newRSIPriceBuy)- sum(newOBVPriceBuy)

    print("This strategy help you earn",surplus)

    if surplus > highestsurplus:
        highestsurplus = surplus
        strategyID = i
        MACDstrategy = macd_buy_price, macd_sell_price
        RSIstrategy = rsi_buy_price, rsi_sell_price
        OBVstrategy = obv_buy_price, obv_sell_price

print("the optimized MACD strategy is strategy",strategyID,"Below is the strategy trading details ", MACDstrategy)
print("the optimized RSI strategy is strategy",strategyID,"Below is the strategy trading details ", RSIstrategy)
print("the optimized OBV strategy is strategy",strategyID,"Below is the strategy trading details ", OBVstrategy)
print("The money you will get using the optimized strategy",strategyID,"is", highestsurplus)

