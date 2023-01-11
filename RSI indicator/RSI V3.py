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

def RSI():
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

    length = 14
    # Load the data
    #df = pd.read_csv('wilder-rsi-data.csv', header=0).set_index(['period'])
    # Calculate the RSI via pandas_ta
    #df.ta.rsi(close='close', length=length, append=True)
    # Calculate with signal_indicators
    #df.ta.rsi(close='close', length=length, append=True, signal_indicators=True)
    # define custom overbought/oversold thresholds
    df.ta.rsi(close='close', length=length, append=True, signal_indicators=True, xa=50, xb=55)

    print(df)

    #Store the OBV and OBV Expontential Moving Average (EMA) into new columns
    df['RSI_14_EMA'] = df['RSI_14'].ewm(span=20).mean()

    #Visually show the stock price
    plt.figure(figsize=(12.2, 4.5))
    plt.plot(df['close'], label = 'close')
    plt.title('Close Price')
    plt.xlabel('Date', fontsize = 18)
    plt.ylabel('Price USD', fontsize = 18)
    plt.show()

    #Create and plot the graph
    plt.figure(figsize=(12.2, 4.5))
    plt.plot(df['RSI_14'], label = 'RSI_14', color = 'orange')
    plt.plot(df['RSI_14_EMA'], label = 'RSI_14_EMA', color = 'purple')
    plt.title('RSI_14')
    plt.xlabel('Date', fontsize = 18)
    plt.ylabel('Price USD', fontsize = 18)
    plt.show()

    #Create a function to signal when to buy and sell the stock
    # If OBV > OBV_EMA Then Buy
    # If OBV < OBV_EMA Then Sell
    # Else Do Nothing
    def buy_sell(signal, col1, col2):
        signPriceBuy = []
        signPriceSell = []
        flag = -1
        #Loop through the length of the data set
        for i in range(0, len(signal)):
            #If RSI_14 > 50 Then Buy --> col1 => 'OBV' and col2 => 'OBV_EMA'
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
        
        return (signPriceBuy, signPriceSell)

    x = buy_sell(df, 'RSI_14_A_50', 'RSI_14_B_55')


    df['Buy_Signal_Price'] =  x[0]
    df['Sell_Signal_Price'] = x[1]
    #Show the data set
    print(df)
    
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

    return df['RSI_14'], df['Buy_Signal_Price'],df['Sell_Signal_Price']

print(RSI())