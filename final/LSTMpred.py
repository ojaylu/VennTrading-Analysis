from calc import calc
import pandas as pd
from bfs import bfs
import matplotlib.pyplot as plt
from numpy import nan
import lime
from ema import *
from bruteforce import *


def lstmpred(data):
    from matplotlib.pylab import rcParams
    from pandas import Series, datetime
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split, KFold, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_curve, auc
    from statsmodels.tsa.arima_model import ARIMA
    import random
    from xgboost import XGBClassifier
    import seaborn as sns
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import pandas_ta as ta
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    import pandas as pd
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from datetime import date
    #separate line

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import requests
    from math import floor
    from termcolor import colored as cl
    from binance.client import Client
    import datetime as dt
    import json
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import tree
    import graphviz
    from sklearn.tree import DecisionTreeRegressor
    from sklearn import datasets
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import GradientBoostingClassifier
    from itertools import product
    import math
    from datacollection import datacollection
    plt.style.use('fivethirtyeight')
    plt.rcParams['figure.figsize'] = (20,10)

    djia = data.to_csv('output.csv')
    #news_w_label = pd.read_csv("Datasets/Combined_News_DJIA.csv")
    djia = pd.read_csv('output.csv')
    #djia = pd.read_csv("./Datasets/upload_DJIA_table.csv")
    # djia_merged = news_w_label.merge(djia) 
    # djia_merged.fillna("", inplace=True) # Set the NaN values as empty string
    # djia_merged.isnull().sum()
    djia_merged = djia
    # Count of labels
    #sns.countplot(x='Label', data=djia_merged)
    #plt.show()

    # Histpgrams for existing columns
    cols = list(djia.columns[1:-2])
    cnt = 0
    fig, ax = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(13, 10))
    fig.subplots_adjust(hspace=0.3, wspace=0.1)
    for i in range(2):
        for j in range(2):
            ax[i, j].hist(djia_merged[cols[cnt]])
            ax[i, j].set_title(cols[cnt])
            cnt += 1
    #plt.show()
    # Histograms for differences
    fig, ax = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(13, 5))
    fig.subplots_adjust(hspace=0.3, wspace=0.1)
    ax[0].hist(djia_merged['close']-djia_merged['open'])
    ax[0].set_title('close - open')
    ax[1].hist(djia_merged['high']-djia_merged['low'])
    ax[1].set_title('high - low')
    #plt.show()
    # Information about difference of closing price with the previous day's closing price
    data = np.array(np.array(djia_merged['close'][1:]) - np.array(djia_merged['close'][:-1]))
    df = pd.DataFrame(data)
    interval_perc = 0.2
    #print(float(df.quantile(0.5-(interval_perc/2))), float(df.quantile(0.5+(interval_perc/2))))
    # Plots for difference of closing price with the previous day's closing price
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    ax[0].hist(data, bins = 25, edgecolor="red")
    ax[0].set_title('Histogram')
    ax[1].boxplot(data, notch=True, vert=False, showfliers=False)
    ax[1].set_title('Boxplot')
    #plt.show()
    # Basic time series
    plt.plot(djia_merged['datetime'], djia_merged['close'])
    datelist = [djia_merged['datetime'][i] for i in range(0, 200, 20)]
    plt.xticks(ticks=datelist, labels=datelist, rotation = 45)
    #plt.show()

    new_df = pd.DataFrame(djia_merged[['datetime', 'close']])
    # new_df['10_Day_MA'] = ta.ema(djia_merged["close"], length=10)
    # new_df['30_Day_MA'] = ta.ema(djia_merged["close"], length=30)
    new_df['EMA10'] = ta.ema(djia_merged["close"], length=10)
    new_df['EMA30'] = ta.ema(djia_merged["close"], length=30)


    new_df, signal, ema_buy_price, ema_sell_price = calc_signal(new_df, 10, 30)

    # Plotting MAs
    print('No. of Buy signals: ', sum(new_df['Buy']))
    print('No. of Sell signals: ', sum(new_df['Sell']))
    #show_graph(new_df, 10, 30)


    new_df['datetime'] = pd.to_datetime(new_df.datetime,format='%Y-%m-%d')
    fig = plt.figure(figsize=(10, 10))
    plt.plot(new_df['datetime'], new_df['close'], label='Close Price history')
    plt.legend()
    X=new_df[['EMA10','EMA30']]
    y=new_df['close']
    y=np.asarray(y)
    # Split the initial 80% of the data as training set and the remaining 20% data as the testing set
    X_train=X[:int(0.8*len(X))]
    X_train_date=new_df['datetime'][:int(0.8*len(X))]
    X_test=X[int(0.8*len(X)):]
    X_test_date=new_df['datetime'][int(0.8*len(X)):]
    y_train=y[:int(0.8*len(y))]
    y_test=y[int(0.8*len(y)):]
    minn=np.min(X_train)
    maxx=np.max(X_train)
    X_train=(X_train-np.min(X_train))/(np.max(X_train)-np.min(X_train))
    X_test=(X_test-minn)/(maxx-minn)
    fig = plt.figure(figsize=(15, 10))
    plt.plot(X_train_date,y_train, 'blue', label='Training Data')
    plt.plot(X_test_date,y_test, 'green', label='Testing Data')
    plt.legend()

    history = y_train

    #LSTM-----------------------------------------------------------------------------------------------------------------------------------------LSTM

    def split_data(data_raw, lookback):
        data = []
        
        # create all possible sequences of length seq_len
        for index in range(len(data_raw) - lookback): 
            data.append(data_raw[index: index + lookback])
        
        data = np.array(data)
        test_set_size = int(np.round(0.2*data.shape[0]))
        train_set_size = data.shape[0] - (test_set_size)
        
        x_train = data[:train_set_size,:-1,:]
        y_train = data[:train_set_size,-1,:]
        
        x_test = data[train_set_size:,:-1]
        y_test = data[train_set_size:,-1,:]
        
        return [x_train, y_train, x_test, y_test]


    close = (djia['close']).values.reshape(-1,1)
    open = (djia['open']).values.reshape(-1,1)
    high = (djia['high']).values.reshape(-1,1)
    low = (djia['low']).values.reshape(-1,1)
    volume = (djia['volume']).values.reshape(-1,1)
    sc = MinMaxScaler(feature_range = (0, 1))
    open = sc.fit_transform(open)
    close = sc.fit_transform(close)
    high = sc.fit_transform(high)
    low = sc.fit_transform(low)
    volume = sc.fit_transform(volume)
    lookback = 14 
    x_train_o, y_train_o, x_test_o, y_test_o = split_data(open, lookback)
    x_train_h, y_train_h, x_test_h, y_test_h = split_data(high, lookback)
    x_train_l, y_train_l, x_test_l, y_test_l = split_data(low, lookback)
    x_train_c, y_train_c, x_test_c, y_test_c = split_data(close, lookback)
    x_train_v, y_train_v, x_test_v, y_test_v = split_data(volume, lookback)
    plt.figure(figsize=(20, 10))


    print('training price is', x_train_c)
    plt.subplot(1, 2, 1)
    plt.plot(y_train_c, color="blue", label="Closing Price")
    plt.plot(y_train_o, color="red", linestyle='--', label="Open")
    plt.plot(y_train_h, color="green", linestyle='--', label="High")
    plt.plot(y_train_l, color="purple", linestyle='--', label="Low")
    plt.xlabel('time period')
    plt.ylabel('Scale price (open/close/high/low)')


    plt.subplot(1, 2, 2)
    plt.plot(y_test_c, color="blue", label="Closing Price")
    plt.plot(y_test_o, color="red", linestyle='--', label="Open")
    plt.plot(y_test_h, color="green", linestyle='--', label="High")
    plt.plot(y_test_l, color="purple", linestyle='--', label="Low")
    plt.xlabel('time period')
    plt.ylabel('Scaled price (open/close/high/low)')
    #plt.show()

    x_train_c = torch.from_numpy(x_train_c).type(torch.Tensor)
    x_test_c = torch.from_numpy(x_test_c).type(torch.Tensor)
    y_train_c = torch.from_numpy(y_train_c).type(torch.Tensor)
    y_test_c = torch.from_numpy(y_test_c).type(torch.Tensor)

    x_train_h = torch.from_numpy(x_train_h).type(torch.Tensor)
    x_test_h = torch.from_numpy(x_test_h).type(torch.Tensor)
    y_train_h = torch.from_numpy(y_train_h).type(torch.Tensor)
    y_test_h = torch.from_numpy(y_test_h).type(torch.Tensor)

    x_train_l = torch.from_numpy(x_train_l).type(torch.Tensor)
    x_test_l = torch.from_numpy(x_test_l).type(torch.Tensor)
    y_train_l = torch.from_numpy(y_train_l).type(torch.Tensor)
    y_test_l = torch.from_numpy(y_test_l).type(torch.Tensor)

    x_train_v = torch.from_numpy(x_train_v).type(torch.Tensor)
    x_test_v = torch.from_numpy(x_test_v).type(torch.Tensor)
    y_train_v = torch.from_numpy(y_train_v).type(torch.Tensor)
    y_test_v = torch.from_numpy(y_test_v).type(torch.Tensor)

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    input_dim = 1     # Number of features/columns used in training/testing. If we plann to use close price and open price for prediction, this would be changed to 2
    hidden_dim = 32   # Hidden dimension
    num_layers = 1    # Number for LSTM/GRU layer(s) used. We are using only 1. If we plan to use bi-directional RNN then this would be changed to 2
    output_dim = 1    # Dimension of the output we are trying to predict (either close price/ open price/ high / low)

    num_epochs = 500 # we train our LSTM/GRU models for 10000 epochs
    # we use Adam Optimizer and MSE loss fucntion for trainig the models

    class LSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
            super(LSTM, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)

            # we need to detach h0 and c0 here since we are doing back propagation through time (BPTT)
            out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
            out = self.fc(out[:, -1, :]) 
            return out

    def train(net, x_train, y_train, x_test, y_test, criterion, optimizer):
        print(net)
        net.to(device)
        start_time = time.time()

        hist = []
        
        for t in range(num_epochs):
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            
            y_train_pred = net(x_train)
            loss = criterion(y_train_pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            hist.append(loss.item())
            if t !=0 and t % 100 == 0 :
                print(' Epoch: {:.6f} \t Training loss: {:.6f} ' .format(t, loss.item()))
                test_loss = criterion(net(x_test.to(device)), y_test.to(device)).item()
                print(' Epoch: {:.6f} \t Test loss: {:.6f} ' .format(t, test_loss))
        
        training_time = time.time()-start_time
        print("Training time: {}".format(training_time))

        return np.array(hist)

    net_lstm_c = LSTM(input_dim, hidden_dim, num_layers, output_dim)
    net_lstm_h = LSTM(input_dim, hidden_dim, num_layers, output_dim)
    net_lstm_l = LSTM(input_dim, hidden_dim, num_layers, output_dim)
    net_lstm_v = LSTM(input_dim, hidden_dim, num_layers, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net_lstm_h.parameters(), lr=0.0005)
    hist_lstm_c = train(net_lstm_c, x_train_c, y_train_c, x_test_c, y_test_c, criterion, optimizer)
    hist_lstm_h = train(net_lstm_h, x_train_h, y_train_h, x_test_h, y_test_h, criterion, optimizer)
    hist_lstm_l = train(net_lstm_l, x_train_l, y_train_l, x_test_l, y_test_l, criterion, optimizer)
    hist_lstm_v = train(net_lstm_v, x_train_v, y_train_v, x_test_v, y_test_v, criterion, optimizer)


    # Plotting
    fig = plt.figure(figsize=(5, 5))
    plt.plot(hist_lstm_h, color="blue", label="Training MSE for lstm")
    #plt.plot(hist_gru, color="red", linestyle='--', label="Training MSE for gru")
    plt.xlabel('time period')
    plt.ylabel('MSE loss')
    #plt.show()

    #pred_gru_h = net_gru_h(x_test_h.to(device)).detach().cpu()
    pred_lstm_c = net_lstm_c(x_test_c.to(device)).detach().cpu()
    pred_lstm_h = net_lstm_h(x_test_h.to(device)).detach().cpu()
    pred_lstm_l = net_lstm_l(x_test_l.to(device)).detach().cpu()
    pred_lstm_v = net_lstm_v(x_test_v.to(device)).detach().cpu()
    # Plotting
    fig = plt.figure(figsize=(12, 12))
    plt.plot(y_test_h, color="blue", label="High Price")
    plt.plot(pred_lstm_h, color="red", linestyle='--', label="LSTM predicted High price")
    #plt.plot(pred_gru_h, color="yellow", linestyle='--', label="GRU predicted High price")

    plt.xlabel('time period')
    plt.ylabel('Scaled price (High)')

    plt.legend()
    plt.show()

    # train prediction
    #trainPred = net_gru_h(x_train_h.to(device)).detach().cpu()
    trainPredPlot = np.empty_like(high)
    trainPredPlot[:, :] = np.nan
    #trainPredPlot[lookback:len(trainPred)+lookback, :] = trainPred

    #test preediction
    testPredPlot = np.empty_like(high)
    testPredPlot[:] = np.nan
    #testPredPlot[len(trainPred)+lookback-1:len(high)-1, :] = pred_gru_h

    plt.figure(figsize=(15, 10))
    plt.plot(open, color="green", label = "Original training high price")
    plt.plot(trainPredPlot, color="blue", label="Predicted train")
    plt.plot(testPredPlot, color="red", label="Predicted test")
    plt.legend()

    plt.xlabel('time period')
    plt.ylabel('Scaled price (High)')

    plt.show()

    # features = ["close"]
    # lime_explainer = lime.lime_tabular.LimeTabularExplainer(x_train_c, class_names=features,verbose=True, mode='regression')
    # lime_explanations = lime_explainer.explain_instance(x_test_c[50], pred_lstm_c, num_features=4)


    return pred_lstm_c, pred_lstm_h, pred_lstm_l, pred_lstm_v



def mylstm(data, clf):

    x_future_c, x_future_h, x_future_l, x_future_v = lstmpred(data)
    # x_future = [0.4, 0.7, 0,9, 0.8, 0.3, 0.56, 0.7, 0.8, 0.23, 0.19, 0.78, 
    #             0.65, 0.87, 0.42, 0.49, 0.1, 0.3, 0.5, 0.7, 0.2, 0,4, 0.6, 0.8, 0.45, 0.63, 0.97, 0.46, 0.55, 0.73, 0.85, 0.09]
    if len(x_future_c) < 22:
        bfsfuture = bfs('b', x_future_c)
        bfssignal = bfsfuture[1]
    x_future = pd.DataFrame()
    x_future['close'] = [tensor.item() for tensor in x_future_c]
    x_future['high'] = [tensor.item() for tensor in x_future_h]
    x_future['low'] = [tensor.item() for tensor in x_future_l]
    x_future['volume'] = [tensor.item() for tensor in x_future_v]
    x_future = calc(x_future)
    print("here is the strategies suggested by AI")
    x_future_2 = x_future
    x_future_2, combineflag, macdlb, macdub, rsilb, rsiub, newsurplus = bruteforce(x_future_2)
    x_future = x_future.loc[:, ['close', 'RSI_14', 'EMA10', 'EMA30', 'macd','OBV', 'ATR','ClgtEMA10', 'EMA10gtEMA30', 'MACDSIGgtMACD']]
    x_future = x_future.fillna(value=x_future['RSI_14'].mean())
    y_cls_pred = clf.predict(x_future)
    print('The future price prediceted by LSTM is ', x_future_c)
    print('The buy/sell signal corresponding future price is ', y_cls_pred)

    sell = []
    buy = []
    if len(x_future_c) <= 22:
        for i in range(len(bfssignal)):
            if bfssignal[i] == 1:
                buy.append(x_future['close'][i])
                sell.append(nan)
            elif bfssignal[i] == -1:
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

    plt.figure(figsize=(15,8))
    plt.plot(x_future.index, x_future['close'])
    plt.scatter(x_future.index, x_future['buy price signal'], color = 'red')
    plt.scatter(x_future.index, x_future['sell price signal'], color = 'green')
    #plt.scatter(x_future.index[x_future['signal' == -1.0]], x_future['signal' == -1.0], color = 'blue', label = 'sell')
    plt.legend()
    plt.show()

    if len(x_future_c) <= 22:
        return bfssignal
    else:
        return y_cls_pred
