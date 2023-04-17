from calc import calc
import pandas as pd
from bfs import bfs
import matplotlib.pyplot as plt
from numpy import nan
import lime
from ema import *
from bruteforce import *

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from datacollection import *
from itertools import chain

import io
import base64

imageList = {}
def lstmingrid(df):

    pd.options.mode.chained_assignment = None
    tf.random.set_seed(0)

    y = df['close'].fillna(method='ffill')
    y = y.values.reshape(-1, 1)

    # scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(y)
    y = scaler.transform(y)

    # generate the input and output sequences
    n_lookback = 60  # length of input sequences (lookback period)
    n_forecast = 30  # length of output sequences (forecast period)

    X = []
    Y = []

    for i in range(n_lookback, len(y) - n_forecast + 1):
        X.append(y[i - n_lookback: i])
        Y.append(y[i: i + n_forecast])

    X = np.array(X)
    Y = np.array(Y)

    # fit the model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(n_forecast))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, Y, epochs=100, batch_size=32, verbose=0)

    # generate the forecasts
    X_ = y[- n_lookback:]  # last available input sequence
    X_ = X_.reshape(1, n_lookback, 1)

    Y_ = model.predict(X_).reshape(-1, 1)
    Y_ = scaler.inverse_transform(Y_)

    # organize the results in a data frame
    df_past = df[['close']].reset_index()
    df_past.rename(columns={'index': 'datetime', 'close': 'Actual'}, inplace=True)
    df_past['datetime'] = pd.to_datetime(df_past['datetime'])
    df_past['Forecast'] = np.nan
    df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]

    df_future = pd.DataFrame(columns=['datetime', 'Actual', 'Forecast'])
    df_future['datetime'] = pd.date_range(start=df_past['datetime'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
    df_future['Forecast'] = Y_.flatten()
    df_future['Actual'] = np.nan

    results = df_past.append(df_future).set_index('datetime')

    # plot the results
    results.plot(title='BTC closing price')

    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    predCloseLSTM  = base64.b64encode(buf.getvalue()).decode('utf-8')
    fig.clear()
    imageList['predCloseLSTM'] = predCloseLSTM

    future = list(chain.from_iterable(Y_))

    #print(results['Forecast'].to_markdown())

    return results, future

def lstmingrid_h(df):

    pd.options.mode.chained_assignment = None
    tf.random.set_seed(0)

    y = df['high'].fillna(method='ffill')
    y = y.values.reshape(-1, 1)

    # scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(y)
    y = scaler.transform(y)

    # generate the input and output sequences
    n_lookback = 60  # length of input sequences (lookback period)
    n_forecast = 30  # length of output sequences (forecast period)

    X = []
    Y = []

    for i in range(n_lookback, len(y) - n_forecast + 1):
        X.append(y[i - n_lookback: i])
        Y.append(y[i: i + n_forecast])

    X = np.array(X)
    Y = np.array(Y)

    # fit the model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(n_forecast))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, Y, epochs=100, batch_size=32, verbose=0)

    # generate the forecasts
    X_ = y[- n_lookback:]  # last available input sequence
    X_ = X_.reshape(1, n_lookback, 1)

    Y_ = model.predict(X_).reshape(-1, 1)
    Y_ = scaler.inverse_transform(Y_)

    # organize the results in a data frame
    df_past = df[['high']].reset_index()
    df_past.rename(columns={'index': 'datetime', 'high': 'Actual'}, inplace=True)
    df_past['datetime'] = pd.to_datetime(df_past['datetime'])
    df_past['Forecast'] = np.nan
    df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]

    df_future = pd.DataFrame(columns=['datetime', 'Actual', 'Forecast'])
    df_future['datetime'] = pd.date_range(start=df_past['datetime'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
    df_future['Forecast'] = Y_.flatten()
    df_future['Actual'] = np.nan

    results = df_past.append(df_future).set_index('datetime')

    # plot the results
    results.plot(title='BTC high price')

    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    predHighLSTM  = base64.b64encode(buf.getvalue()).decode('utf-8')
    fig.clear()
    imageList['predHighLSTM '] = predHighLSTM 

    future = list(chain.from_iterable(Y_))

    #print(results['Forecast'].to_markdown())

    return results, future

def lstmingrid_l(df):

    pd.options.mode.chained_assignment = None
    tf.random.set_seed(0)

    y = df['low'].fillna(method='ffill')
    y = y.values.reshape(-1, 1)

    # scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(y)
    y = scaler.transform(y)

    # generate the input and output sequences
    n_lookback = 60  # length of input sequences (lookback period)
    n_forecast = 30  # length of output sequences (forecast period)

    X = []
    Y = []

    for i in range(n_lookback, len(y) - n_forecast + 1):
        X.append(y[i - n_lookback: i])
        Y.append(y[i: i + n_forecast])

    X = np.array(X)
    Y = np.array(Y)

    # fit the model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(n_forecast))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, Y, epochs=100, batch_size=32, verbose=0)

    # generate the forecasts
    X_ = y[- n_lookback:]  # last available input sequence
    X_ = X_.reshape(1, n_lookback, 1)

    Y_ = model.predict(X_).reshape(-1, 1)
    Y_ = scaler.inverse_transform(Y_)

    # organize the results in a data frame
    df_past = df[['low']].reset_index()
    df_past.rename(columns={'index': 'datetime', 'low': 'Actual'}, inplace=True)
    df_past['datetime'] = pd.to_datetime(df_past['datetime'])
    df_past['Forecast'] = np.nan
    df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]

    df_future = pd.DataFrame(columns=['datetime', 'Actual', 'Forecast'])
    df_future['datetime'] = pd.date_range(start=df_past['datetime'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
    df_future['Forecast'] = Y_.flatten()
    df_future['Actual'] = np.nan

    results = df_past.append(df_future).set_index('datetime')

    # plot the results
    results.plot(title='BTC low price')

    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    predLowLSTM  = base64.b64encode(buf.getvalue()).decode('utf-8')
    fig.clear()
    imageList['predLowLSTM '] = predLowLSTM 

    future = list(chain.from_iterable(Y_))

    #print(results['Forecast'].to_markdown())

    return results, future

def lstmingrid_v(df):

    pd.options.mode.chained_assignment = None
    tf.random.set_seed(0)

    y = df['volume'].fillna(method='ffill')
    y = y.values.reshape(-1, 1)

    # scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(y)
    y = scaler.transform(y)

    # generate the input and output sequences
    n_lookback = 60  # length of input sequences (lookback period)
    n_forecast = 30  # length of output sequences (forecast period)

    X = []
    Y = []

    for i in range(n_lookback, len(y) - n_forecast + 1):
        X.append(y[i - n_lookback: i])
        Y.append(y[i: i + n_forecast])

    X = np.array(X)
    Y = np.array(Y)

    # fit the model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(n_forecast))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, Y, epochs=100, batch_size=32, verbose=0)

    # generate the forecasts
    X_ = y[- n_lookback:]  # last available input sequence
    X_ = X_.reshape(1, n_lookback, 1)

    Y_ = model.predict(X_).reshape(-1, 1)
    Y_ = scaler.inverse_transform(Y_)

    # organize the results in a data frame
    df_past = df[['volume']].reset_index()
    df_past.rename(columns={'index': 'datetime', 'volume': 'Actual'}, inplace=True)
    df_past['datetime'] = pd.to_datetime(df_past['datetime'])
    df_past['Forecast'] = np.nan
    df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]

    df_future = pd.DataFrame(columns=['datetime', 'Actual', 'Forecast'])
    df_future['datetime'] = pd.date_range(start=df_past['datetime'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
    df_future['Forecast'] = Y_.flatten()
    df_future['Actual'] = np.nan

    results = df_past.append(df_future).set_index('datetime')

    # plot the results
    results.plot(title='BTC volume')

    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    predVolLSTM  = base64.b64encode(buf.getvalue()).decode('utf-8')
    fig.clear()
    imageList['predVolLSTM '] = predVolLSTM 

    future = list(chain.from_iterable(Y_))

    #print(results['Forecast'].to_markdown())

    return results, future

def mylstm(data, clf, plotbfsgraph):

    result_c, x_future_c = lstmingrid(data)
    result_h, x_future_h = lstmingrid_h(data)
    result_l, x_future_l = lstmingrid_l(data)
    result_v, x_future_v = lstmingrid_v(data)
    x_future = pd.DataFrame()
    x_future['close'] = [tensor.item() for tensor in x_future_c]
    x_future['high'] = [tensor.item() for tensor in x_future_h]
    x_future['low'] = [tensor.item() for tensor in x_future_l]
    x_future['volume'] = [tensor.item() for tensor in x_future_v]
    # Get current axis
    plt.cla()
    ax = plt.gca()
    data.index = pd.to_datetime(data.index)
    x_future['datetime'] = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=len(x_future_c))
    x_future.plot(kind='line',y='close',color='yellow', ax=ax)
    x_future.plot(kind='line',y='high',color='black', ax=ax)
    x_future.plot(kind='line', y='low',color='orange', ax=ax)
    plt.title('the predicted price')

    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    combinedCLHLSTM = base64.b64encode(buf.getvalue()).decode('utf-8')
    fig.clear()
    imageList['combinedCLHLSTM '] = combinedCLHLSTM 

    plt.show()
    x_future = calc(x_future)
    print("The prediction of future price using LSTM is \n", x_future)


    if len(x_future_c) < 22:
        print("since the prediction period is not too long, BFS will be used to generate buy/sell signal")
        bfsfuture, bfsfinal, bfsfrontier = bfs('b', x_future_c, plotbfsgraph=plotbfsgraph)
        bfssignal = bfsfuture[1]

    else:
        print("Then it will perform brute force using the predicted price by LSTM: ")
        x_future_2 = x_future
        x_future_2, combineflag, macdlb, macdub, rsilb, rsiub, newsurplus = bruteforce(x_future_2, plotgraph=False,plotmacdgraph=False,
                                                                                       plotrsigraph=False,
                                                                                       plotobvgraph=False,
                                                                                       plotemagraph=False)
        print("now these list of buy/sell signal will be the standard signal.")
        print("Then we will use the ML classifier to generate buy/sell signal using the same data predicted by LSTM")
        x_future = x_future.loc[:, ['close', 'RSI_14', 'EMA10', 'EMA30', 'macd','OBV', 'ATR','ClgtEMA10', 'EMA10gtEMA30', 'MACDSIGgtMACD', '%k', '%d']]
        x_future = x_future.fillna(value=x_future['RSI_14'].mean())
        y_cls_pred = clf.predict(x_future)
        print('According to the ML classifier, the buy/sell signal corresponding to future price is ', y_cls_pred)

    sell = []
    buy = []
    if len(x_future_c) <= 22:
        for i in range(len(bfssignal)):
            if bfsfinal[i] == 1:
                buy.append(x_future['close'][i])
                sell.append(nan)
            elif bfsfinal[i] == -1:
                buy.append(nan)
                sell.append(x_future['close'][i])
            else:
                buy.append(nan)
                sell.append(nan)
        x_future['buy price signal'] = pd.DataFrame(buy)
        x_future['sell price signal'] = pd.DataFrame(sell)
    else:
        for i in range(len(y_cls_pred)):
            if y_cls_pred[i] == 1:
                buy.append(x_future['close'][i])
                sell.append(nan)
            elif y_cls_pred[i] == -1:
                buy.append(nan)
                sell.append(x_future['close'][i])
            else:
                buy.append(nan)
                sell.append(nan)
        x_future['buy price signal'] = pd.DataFrame(buy)
        x_future['sell price signal'] = pd.DataFrame(sell)

    plt.cla()
    data.index = pd.to_datetime(data.index)
    x_future['datetime'] = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=len(y_cls_pred))
    x_future = x_future.set_index('datetime')

    plt.figure(figsize=(15,8))
    plt.plot(x_future.index, x_future['close'])
    plt.scatter(x_future.index, x_future['buy price signal'], color = 'red')
    plt.scatter(x_future.index, x_future['sell price signal'], color = 'green')
    plt.legend()

    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    LSTMBSPos = base64.b64encode(buf.getvalue()).decode('utf-8')
    fig.clear()
    imageList['LSTMBSPos'] = LSTMBSPos
    plt.show()

    if len(x_future_c) <= 22:
        return bfssignal, imageList
    else:
        return y_cls_pred, imageList
