import csv
import pandas as pd
import pandas_ta as ta
from plotly.subplots import make_subplots
import plotly.graph_objects as go

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

def RSI(upperboundary, lowerboundary, df):

    # Load our sample CSV data as DataFrame
    df = pd.read_csv('./wilder-rsi-data.csv', header=0).set_index(['period'])
    #print(df.head())

    # Calculate Price Differences
    df['diff'] = df.diff(1)

    # Calculate Avg. Gains/Losses
    df['gain'] = df['diff'].clip(lower=0).round(2)
    df['loss'] = df['diff'].clip(upper=0).abs().round(2)

    window_length = 14

    # Get initial Averages
    df['avg_gain'] = df['gain'].rolling(window=window_length, min_periods=window_length).mean()[:window_length+1]
    df['avg_loss'] = df['loss'].rolling(window=window_length, min_periods=window_length).mean()[:window_length+1]

    # View first SMA value
    #print(df.iloc[window_length-1: window_length+2])

    # Get WMS averages
    # Average Gains
    for i, row in enumerate(df['avg_gain'].iloc[window_length+1:]):
        df['avg_gain'].iloc[i + window_length + 1] =\
            (df['avg_gain'].iloc[i + window_length] *
            (window_length - 1) +
            df['gain'].iloc[i + window_length + 1])\
            / window_length
    # Average Losses
    for i, row in enumerate(df['avg_loss'].iloc[window_length+1:]):
        df['avg_loss'].iloc[i + window_length + 1] =\
            (df['avg_loss'].iloc[i + window_length] *
            (window_length - 1) +
            df['loss'].iloc[i + window_length + 1])\
            / window_length
    # View initial results
    #print(df[window_length-1:window_length+5])

    # Calculate RS Values
    df['rs'] = df['avg_gain'] / df['avg_loss']

    # Calculate RSI
    df['rsi'] = 100 - (100 / (1.0 + df['rs']))

    #RSI with panda_ta

    length = 14

    # Load the data
    #df = pd.read_csv('wilder-rsi-data.csv', header=0).set_index(['period'])
    # Calculate the RSI via pandas_ta
    df.ta.rsi(close='close', length=length, append=True)
    # View the result
    print(df)

    # Calculate with signal_indicators
    #df.ta.rsi(close='price', length=14, append=True, signal_indicators=True)
    # define custom overbought/oversold thresholds
    df.ta.rsi(close='close', length=length, append=True, signal_indicators=True, xa=lowerboundary, xb=upperboundary)

    print(df.ta.rsi(close='close', length=length, append=True, signal_indicators=True, xa=lowerboundary, xb=upperboundary))

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

    data.ta.rsi(close='close', length=length, append=True, signal_indicators=True, xa=lowerboundary, xb=upperboundary)

    print(data)

    visualization(data)




def validation():
    #Validation list
    WILDER_RSI_VALUES = [
        74.36, 74.55, 65.75, 59.68, 61.98, 66.44, 65.75, 67.0, 71.43,
        70.5, 72.14, 67.95, 60.78, 55.56, 56.71, 49.49, 48.19, 52.38,
        50.0, 43.5, 45.36, 42.53, 44.14, 44.75
    ]
    #Load in the validation data + create NaN values preceding our first RSI calculations
    v = pd.DataFrame(pd.concat([pd.Series(["NaN"] * (window_length)), pd.Series(WILDER_RSI_VALUES)])).reset_index(level=0).drop(['index'], axis=1)
    v.index = list(range(1, len(v) + 1))  # reindex starting with 0
    #Calculate differences
    df['diff_rsi'] = ((df['rsi'] - v.values).abs())
    df['diff_pct'] = ((df['rsi'] - v.values) / v.values * 100).abs()
    #Round off for easy comparison
    df['diff_rsi'] = df['diff_rsi'].apply(lambda x: round(x, 2))
    df['diff_pct'] = df['diff_pct'].apply(lambda x: round(x, 2))




def visualization(data):
    # Create Figure
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_width=[0.25, 0.75])
    # Inspect Result
    print(fig)

    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        increasing_line_color='#ff9900',
        decreasing_line_color='black',
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['RSI_14'],
        line=dict(color='#ff9900', width=2),
        showlegend=False,
    ), row=2, col=1
    )

    # Add upper/lower bounds
    fig.update_yaxes(range=[-10, 110], row=2, col=1)
    fig.add_hline(y=0, col=1, row=2, line_color="#666", line_width=2)
    fig.add_hline(y=100, col=1, row=2, line_color="#666", line_width=2)
    # Add overbought/oversold
    fig.add_hline(y=30, col=1, row=2, line_color='#336699', line_width=2, line_dash='dash')
    fig.add_hline(y=70, col=1, row=2, line_color='#336699', line_width=2, line_dash='dash')
    # Customize font, colors, hide range slider
    layout = go.Layout(
        plot_bgcolor='#efefef',
        # Font Families
        font_family='Monospace',
        font_color='#000000',
        font_size=20,
        xaxis=dict(
            rangeslider=dict(
                visible=False
            )
        )
    )
    # update and display
    fig.update_layout(layout)
    fig.show()



RSI(40, 60, 80)