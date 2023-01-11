import pandas as pd
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

def MACD():

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

    print(data)

    # TRADING STRATEGY

    def implement_stoch_macd_strategy(prices, k, d, macd, macd_signal):    
        buy_price = []
        sell_price = []
        stoch_macd_signal = []
        signal = 0

        for i in range(len(prices)):
            if k[i] < 30 and d[i] < 30 and macd[i] < 0 and macd_signal[i] < 0:
                if signal != 1:
                    buy_price.append(prices[i])
                    sell_price.append(np.nan)
                    signal = 1
                    stoch_macd_signal.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    stoch_macd_signal.append(0)
                    
            elif k[i] > 70 and d[i] > 70 and macd[i] > 0 and macd_signal[i] > 0:
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
                
    buy_price, sell_price, stoch_macd_signal = implement_stoch_macd_strategy(data['close'], data['%k'], data['%d'], data['macd'], data['macd_signal'])



    #position change

    position = []
    for i in range(len(stoch_macd_signal)):
        if stoch_macd_signal[i] > 1:
            position.append(0)
        else:
            position.append(1)
            
    for i in range(len(data['close'])):
        if stoch_macd_signal[i] == 1:
            position[i] = 1
        elif stoch_macd_signal[i] == -1:
            position[i] = 0
        else:
            position[i] = position[i-1]
            
    close_price = data['close']
    k_line = data['%k']
    d_line = data['%d']
    macd_line = data['macd']
    signal_line = data['macd_signal']
    stoch_macd_signal = pd.DataFrame(stoch_macd_signal).rename(columns = {0:'stoch_macd_signal'}).set_index(data.index)
    position = pd.DataFrame(position).rename(columns = {0:'stoch_macd_position'}).set_index(data.index)

    frames = [close_price, k_line, d_line, macd_line, signal_line, stoch_macd_signal, position]
    strategy = pd.concat(frames, join = 'inner', axis = 1)

    #print(buy_price, sell_price, stoch_macd_signal)
    #print(strategy)
    #def indicator1(date,time_start,time_end,user_max, user_min):

    df = data

    #Visually show the stock price
    plt.figure(figsize=(12.2, 4.5))
    plt.plot(df['close'], label = 'close')
    plt.title('Close Price')
    plt.xlabel('Date', fontsize = 18)
    plt.ylabel('Price USD', fontsize = 18)
    plt.show()

    #Create and plot the graph
    plt.figure(figsize=(12.2, 4.5))
    plt.plot(df['%k'], label = 'K-line', color = 'purple')
    plt.plot(df['%d'], label = 'D-line', color = 'blue')
    plt.title('Stoch Ocillator')
    plt.xlabel('Date', fontsize = 18)
    plt.ylabel('Price USD', fontsize = 18)
    plt.show()

    plt.figure(figsize=(12.2, 4.5))
    plt.plot(df['macd'], label = 'MACD', color = 'orange')
    plt.plot(df['macd_signal'], label = 'MACD Signal', color = 'green')
    plt.ylim(-0.2, 0.2)
    plt.title('MACD')
    plt.xlabel('Date', fontsize = 18)
    plt.ylabel('Price USD', fontsize = 18)
    plt.show()

    df['Buy_Signal_Price'] =  buy_price
    df['Sell_Signal_Price'] = sell_price

    #Plot the buy and sell prices
    plt.figure(figsize=(12.2, 4.5))
    plt.plot(df['close'], label = 'Close', alpha = 0.35)
    plt.scatter(df.index, df['Buy_Signal_Price'], label = 'Buy Signal', marker = '^', alpha = 1, color = 'green')
    plt.scatter(df.index, df['Sell_Signal_Price'], label = 'Sell Signal', marker = 'v', alpha = 1, color = 'red')
    plt.title('Buy & Sell Signals')
    plt.xlabel('Date', fontsize = 18)
    plt.ylabel('Price USD', fontsize = 18)
    plt.legend(loc='upper left')
    plt.show()

    return {"MACD": df['macd'], "MACD signal": df['macd_signal'], 'K-line': df['%k'], 'D-line': df['%d'], 'Buy_Signal_Price': df['Buy_Signal_Price'],'Sell_Signal_Price': df['Sell_Signal_Price']}


print(MACD())