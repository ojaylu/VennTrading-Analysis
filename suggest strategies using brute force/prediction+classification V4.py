#-------------LSTM--------------------------------------------------------------------------------------------------------

def lstmpred():
    import pandas as pd
    import numpy as np
    import matplotlib
    from matplotlib import pyplot
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
    import matplotlib.pyplot as plt
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
    from sklearn.svm import SVC
    from sklearn.metrics import roc_auc_score
    from datetime import date
    #separate line

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import numpy as np
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
    from numpy import nan
    plt.style.use('fivethirtyeight')
    plt.rcParams['figure.figsize'] = (20,10)

    url = 'https://api.binance.com/api/v3/klines'
    symbol = 'ADAUSDT'
    interval = '1d'
    start = str(int(dt.datetime(2020,1,1).timestamp()*1000))
    end = str(int(dt.datetime(2023,1,1).timestamp()*1000))
    par = {'symbol': symbol, 'interval': interval, 'startTime': start, 'endTime': end}
    data = pd.DataFrame(json.loads(requests.get(url, params= par).text))
    #format columns name
    data.columns = ['datetime', 'Open', 'High', 'Low', 'Close', 'volume','close_time', 'qav', 'num_trades','taker_base_vol', 'taker_quote_vol', 'ignore']
    data.index = [dt.datetime.fromtimestamp(x/1000.0) for x in data.datetime]
    data=data.astype(float)

    djia = data.to_csv('output.csv')
    news_w_label = pd.read_csv("Datasets/Combined_News_DJIA.csv")
    #print(news_w_label.shape)
    #print(news_w_label.columns)
    djia = pd.read_csv('output.csv')
    #djia = pd.read_csv("./Datasets/upload_DJIA_table.csv")
    #print(djia.shape)
    #print(djia.columns)
    # djia_merged = news_w_label.merge(djia) 
    # djia_merged.head(20)
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
    ax[0].hist(djia_merged['Close']-djia_merged['Open'])
    ax[0].set_title('Close - Open')
    ax[1].hist(djia_merged['High']-djia_merged['Low'])
    ax[1].set_title('High - Low')
    #plt.show()
    # Information about difference of closing price with the previous day's closing price
    data = np.array(np.array(djia_merged['Close'][1:]) - np.array(djia_merged['Close'][:-1]))
    df = pd.DataFrame(data)
    interval_perc = 0.2
    #print(float(df.quantile(0.5-(interval_perc/2))), float(df.quantile(0.5+(interval_perc/2))))
    df.describe()
    # Plots for difference of closing price with the previous day's closing price
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    ax[0].hist(data, bins = 25, edgecolor="red")
    ax[0].set_title('Histogram')
    ax[1].boxplot(data, notch=True, vert=False, showfliers=False)
    ax[1].set_title('Boxplot')
    #plt.show()
    # Basic time series
    plt.plot(djia_merged['datetime'], djia_merged['Close'])
    datelist = [djia_merged['datetime'][i] for i in range(0, 200, 20)]
    plt.xticks(ticks=datelist, labels=datelist, rotation = 45)
    #plt.show()

    new_df = pd.DataFrame(djia_merged[['datetime', 'Close']])
    #print(new_df)
    new_df['10_Day_MA'] = ta.ema(djia_merged["Close"], length=10)
    new_df['30_Day_MA'] = ta.ema(djia_merged["Close"], length=30)
    #Function to calculate the signal (1: True, 0: False -> Both 0 indicate 'Hold')
    def calc_signal(df, short_days, long_days):
        short_term = str(short_days) + '_Day_MA'
        long_term = str(long_days) + '_Day_MA'
        df['Buy'] = 0
        df['Sell'] = 0
        for i in range(1, len(df['datetime'])):
            if (df[short_term][i] > df[long_term][i]) and (df[short_term][i-1] <= df[long_term][i-1]):
                new_df['Buy'][i] = 1 #Buy
            elif (df[short_term][i] < df[long_term][i]) and (df[short_term][i-1] >= df[long_term][i-1]):
                new_df['Sell'][i] = 1 #Sell
        return df
    def show_graph(df, short_days, long_days):
        short_term = str(short_days) + '_Day_MA'
        long_term = str(long_days) + '_Day_MA'
        fig = plt.figure(figsize=(15, 15))
        plt.plot(df['Close'][long_days:], color="black", label="Closing Price")
        plt.plot(df[short_term][long_days:], color="magenta", linestyle='--', label=str(short_days)+" Days Moving Average")
        plt.plot(df[long_term][long_days:], color="blue", linestyle='--', label=str(long_days)+" Days Moving Average")
        plt.plot(df[long_days:][df['Buy'] == 1][short_term], marker='o', color="green", linestyle='None', label="Golden Cross (Buy)")
        plt.plot(df[long_days:][df['Sell'] == 1][short_term], marker='o', color="red", linestyle='None', label="Death Cross (Sell)")
        plt.legend()
        datelist = [djia_merged['Date'][i] for i in range(long_days, 2000, 100)]
        plt.xticks(ticks=list(range(long_days, 2000, 100)), labels=datelist, rotation = 45)
        plt.show()

    new_df = calc_signal(new_df, 10, 30)
    #print(new_df)

    # Plotting MAs
    print('No. of Buy signals: ', sum(new_df['Buy']))
    print('No. of Sell signals: ', sum(new_df['Sell']))
    #show_graph(new_df, 10, 30)


    new_df['datetime'] = pd.to_datetime(new_df.datetime,format='%Y-%m-%d')
    fig = plt.figure(figsize=(10, 10))
    plt.plot(new_df['datetime'], new_df['Close'], label='Close Price history')
    plt.legend()
    X=new_df[['10_Day_MA','30_Day_MA']]
    y=new_df['Close']
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

    import statsmodels.api as sm

    predictions = np.array([])
    for t in range(len(y_test)):
        model = sm.tsa.arima.ARIMA(history, order=(5,1,0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions=np.append(predictions,yhat)
        obs = y_test[t]
        history=np.append(history,obs)
    error = np.sqrt(np.mean(np.square(predictions-y_test)))
    print('RMS error: ',error)
    fig = plt.figure(figsize=(15, 10))
    plt.plot(X_test_date,predictions, 'blue', label='Prediction')
    plt.plot(X_test_date,y_test, 'green', label='Testing Data')
    plt.legend()
    #plt.show()

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


    close = (djia['Close']).values.reshape(-1,1)
    open = (djia['Open']).values.reshape(-1,1)
    high = (djia['High']).values.reshape(-1,1)
    low = (djia['Low']).values.reshape(-1,1)
    sc = MinMaxScaler(feature_range = (0, 1))
    open = sc.fit_transform(open)
    close = sc.fit_transform(close)
    high = sc.fit_transform(high)
    low = sc.fit_transform(low)
    lookback = 14 
    x_train_o, y_train_o, x_test_o, y_test_o = split_data(open, lookback)
    x_train_h, y_train_h, x_test_h, y_test_h = split_data(high, lookback)
    x_train_l, y_train_l, x_test_l, y_test_l = split_data(low, lookback)
    x_train_c, y_train_c, x_test_c, y_test_c = split_data(close, lookback)
    plt.figure(figsize=(20, 10))

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

    x_train_h = torch.from_numpy(x_train_h).type(torch.Tensor)
    x_test_h = torch.from_numpy(x_test_h).type(torch.Tensor)
    y_train_h = torch.from_numpy(y_train_h).type(torch.Tensor)
    y_test_h = torch.from_numpy(y_test_h).type(torch.Tensor)

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    input_dim = 1     # Number of features/columns used in training/testing. If we plann to use close price and open price for prediction, this would be changed to 2
    hidden_dim = 32   # Hidden dimension
    num_layers = 1    # Number for LSTM/GRU layer(s) used. We are using only 1. If we plan to use bi-directional RNN then this would be changed to 2
    output_dim = 1    # Dimension of the output we are trying to predict (either close price/ open price/ high / low)

    num_epochs = 1000 # we train our LSTM/GRU models for 10000 epochs
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

    net_lstm_h = LSTM(input_dim, hidden_dim, num_layers, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net_lstm_h.parameters(), lr=0.0005)
    hist_lstm = train(net_lstm_h, x_train_h, y_train_h, x_test_h, y_test_h, criterion, optimizer)

    # Plotting
    fig = plt.figure(figsize=(5, 5))
    plt.plot(hist_lstm, color="blue", label="Training MSE for lstm")
    #plt.plot(hist_gru, color="red", linestyle='--', label="Training MSE for gru")
    plt.xlabel('time period')
    plt.ylabel('MSE loss')
    #plt.show()

    #pred_gru_h = net_gru_h(x_test_h.to(device)).detach().cpu()
    pred_lstm_h = net_lstm_h(x_test_h.to(device)).detach().cpu()
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


    return pred_lstm_h


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
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import graphviz

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)

url = 'https://api.binance.com/api/v3/klines'
symbol = 'ADAUSDT'
interval = '12h'
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


RSILB = list(range(41,50,5))
RSIUB = list(range(51,60,5))
day = [14, 31]
MACDLB = list(range(45,50,1))
MACDUB = list(range(50,55,1))
strat_params_list = list(product(RSILB, RSIUB, day, MACDLB, MACDUB))
strat_params_dict_list = {'xa': [40, 50, 60],'xb': [45, 55, 65],'day' : [7, 14, 31]}
surplus = 0
highestsurplus = -100000
spl = len(strat_params_list)
MACDstrategy = 0
RSIstrategy = 0
OBVstrategy = 0
strategyID = 0

for i, spl in enumerate(strat_params_list):
    #print("Strategy %s out of %s..." % (i+1, spl))
    RSIflag = data.ta.rsi(close='close', length=strat_params_list[i][2], append=True, signal_indicators=True, xa=strat_params_list[i][0], xb=strat_params_list[i][1])
    macd_buy_price, macd_sell_price, stoch_macd_signal = implement_stoch_macd_strategy_optimization(data['close'], data['%k'], data['%d'], data['macd'], data['macd_signal'], strat_params_list[i][3], strat_params_list[i][4])
    rsi_buy_price, rsi_sell_price = rsi_buy_sell(data, RSIflag.columns[1], RSIflag.columns[2])
    obv_buy_price, obv_sell_price = OBV_buy_sell(OBVcalculation(data), 'OBV', 'OBV_EMA')
    # for j in range(len(macd_buy_price)):
    #     if not(math.isnan(macd_buy_price[j])) == True and not(math.isnan(rsi_buy_price[j])) == True:
    #         macd_buy_price.remove(macd_buy_price[j])
    #     if not(math.isnan(macd_sell_price[j])) == True and not(math.isnan(rsi_sell_price[j])) == True:
    #         macd_sell_price.remove(macd_sell_price[j])
    newMACDPriceBuy = [item for item in macd_buy_price if not(math.isnan(item)) == True]
    newMACDPriceSell = [item for item in macd_sell_price if not(math.isnan(item)) == True]
    newRSIPriceBuy = [item for item in rsi_buy_price if not(math.isnan(item)) == True]
    newRSIPriceSell = [item for item in rsi_sell_price if not(math.isnan(item)) == True]
    newOBVPriceBuy = [item for item in obv_buy_price if not(math.isnan(item)) == True]
    newOBVPriceSell = [item for item in obv_sell_price if not(math.isnan(item)) == True]
    surplus = sum(newMACDPriceSell) + sum(newRSIPriceSell) + sum(newOBVPriceSell) - sum(newMACDPriceBuy) - sum(newRSIPriceBuy)- sum(newOBVPriceBuy)

    #print("This strategy help you earn",surplus)

    if surplus >= highestsurplus:
        highestsurplus = surplus
        strategyID = i
        MACDstrategy = macd_buy_price, macd_sell_price
        RSIstrategy = rsi_buy_price, rsi_sell_price
        OBVstrategy = obv_buy_price, obv_sell_price

MACDflag = []
RSIflag = []
OBVflag = []
macd_buy_price = MACDstrategy[0]
macd_sell_price = MACDstrategy[1]
rsi_buy_price = RSIstrategy[0]
rsi_sell_price = RSIstrategy[1]
obv_buy_price = OBVstrategy[0]
obv_sell_price = OBVstrategy[1]
for j in range(len(macd_buy_price)):
    if math.isnan(macd_buy_price[j]) == False:
        MACDflag.append(1)
    elif math.isnan(macd_sell_price[j]) == False:
        MACDflag.append(-1)
    else:
        MACDflag.append(0)
for j in range(len(rsi_buy_price)):
    if math.isnan(rsi_buy_price[j]) == False:
        RSIflag.append(1)
    elif math.isnan(rsi_sell_price[j]) == False:
        RSIflag.append(-1)
    else:
        RSIflag.append(0)
for j in range(len(obv_buy_price)):
    if math.isnan(obv_buy_price[j]) == False:
        OBVflag.append(1)
    elif math.isnan(obv_sell_price[j]) == False:
        OBVflag.append(-1)
    else:
        OBVflag.append(0)
strategy = 'sum'
if strategy == 'sum':
    combineflag = []
    for j in range(len(MACDflag)):
        if (MACDflag[j] + RSIflag[j] + OBVflag[j] ) / 3 > 0.1:
            combineflag.append(1)
        elif (MACDflag[j] + RSIflag[j] + OBVflag[j] ) / 3 < -0.1:
            combineflag.append(-1)
        else:
            combineflag.append(0)

elif strategy == 'length':
    combineflag = []
    buycount = 0
    sellcount = 0
    holdcount = 0
    obvcount = 0
    for j in range(len(MACDflag)):
        if MACDflag[j] == 1:
            buycount+=1
        elif MACDflag[j] == -1:
            sellcount+=1
        else:
            holdcount+=1
        if RSIflag[j] == 1:
            buycount+=1
        elif RSIflag[j] == -1:
            sellcount+=1
        else:
            holdcount+=1
        if OBVflag[j] == 1:
            buycount+=1
        elif OBVflag[j] == -1:
            sellcount+=1
        else:
            holdcount+=1
        if buycount > sellcount and buycount >= holdcount:
            combineflag.append(1)
        elif sellcount > buycount and sellcount >= holdcount:
            combineflag.append(-1)
        else:
            combineflag.append(0)

print("the optimized MACD strategy is strategy",strategyID,"Below is the strategy trading details ", MACDstrategy)
print("the optimized RSI strategy is strategy",strategyID,"Below is the strategy trading details ", RSIstrategy)
print("the optimized OBV strategy is strategy",strategyID,"Below is the strategy trading details ", OBVstrategy)
print("The money you will get using the optimized strategy",strategyID,"is", highestsurplus)
print("According to Oscar's method, the buy and sell signal will be ", combineflag)



#-------------decision tree----------------------

clf = DecisionTreeClassifier(criterion='gini', splitter='best',max_depth=3,min_samples_split=2,min_samples_leaf=6,min_weight_fraction_leaf=0.0,
                                    max_features=None,random_state=42,max_leaf_nodes=None,min_impurity_decrease=0.0)

#clf = KNeighborsClassifier(n_neighbors=5)
#clf = svm.SVC()

def calc(data):

    data['macd'] = get_macd(data['close'], 26, 12, 9)[0]
    data['macd_signal'] = get_macd(data['close'], 26, 12, 9)[1]
    data['macd_hist'] = get_macd(data['close'], 26, 12, 9)[2]
    #data = data.dropna()

    length = 14
    data.ta.rsi(close='close', length=length, append=True, signal_indicators=True, xa=50, xb=55)

    df = data
    data["EMA10"] = ta.ema(data["close"], length=10)
    data["EMA30"] = ta.ema(data["close"], length=30)
    #data['ATR'] = df.ta.atr(df['high'].values.any(), df['low'].values.any(), df['close'].values.any(), timeperiod=14)
    #data['ADX'] = df.ta.adx(df['high'].values.any(), df['low'].values.any(), df['close'].values.any(), timeperiod=14)

    
    data = data.fillna(value=np.nan)
    data['ClgtEMA10'] = np.where(data['close'] > data['EMA10'], 1, -1)
    data['EMA10gtEMA30'] = np.where(data['EMA10'] > data['EMA30'], 1, -1)
    data['MACDSIGgtMACD'] = np.where(data['macd_signal'] > data['macd'], 1, -1)

    data['Return'] = data['close'].pct_change(1).shift(-1)
    #data['target_cls'] = np.where(data.Return > 0, 1, 0)
    #data['target_rgs'] = data['Return']

    #data = data.dropna()
    data['RSI_14'].fillna(value=data['RSI_14'].mean(), inplace=True)
    #data['ATR'].fillna(value=data['ATR'].mean(), inplace=True)

    return data

data = calc(data)
predictors_list = ['close', 'RSI_14', 'ClgtEMA10', 'EMA10gtEMA30', 'MACDSIGgtMACD']
X = data[['close']]
X = data[predictors_list]
y = combineflag
# y = [0,-1, 1, -1, 1,-1, 1, -1,1,-1, 1, -1,1,-1, 1, -1,1,-1, 0, -1,1,-1, 0, -1,1,-1, 0, -1,1,-1, 1, -1,1,-1, 0, -1,1,-1, 1, -1,
#      0,-1, 1, -1, 1,-1, 1, -1,1,-1, 1, -1,1,-1, 1, -1,1,-1, 0, -1,1,-1, 0, -1,1,-1, 0, -1,1,-1, 1, -1,1,-1, 0, -1,1,-1, 1, -1,
#      0,-1, 1, -1, 1,-1, 1, -1,1,-1, 1, -1,1,-1, 1, -1,1,-1, 0, -1,1,-1, 0, -1,1,-1, 0, -1,1,-1, 1, -1,1,-1, 0, -1,1,-1, 1, -1,
#      0,-1, 1, -1, 1,-1, 1, -1,1,-1, 1, -1,1,-1, 1, -1,1,-1, 0, -1,1,-1, 0, -1,1,-1, 0, -1,1,-1, 1, -1,1,-1, 0, -1,1,-1, 1, -1,
#      0,-1, 1, -1, 1,-1, 1, -1,1,-1, 1, -1,1,-1, 1, -1,1,-1, 0, -1,1,-1, 0, -1,1,-1, 0, -1,1,-1, 1, -1,1,-1, 0, -1,1,-1, 1, -1,
#      0,-1, 1, -1, 1,-1, 1, -1,1,-1, 1, -1,1,-1, 1, -1,1,-1, 0, -1,1,-1, 0, -1,1,-1, 0, -1,1,-1, 1, -1,1,-1, 0, -1,1,-1, 1, -1,
#      0,-1, 1, -1, 1,-1, 1, -1,1,-1, 1, -1,1,-1, 1, -1,1,-1, 0, -1,1,-1, 0, -1,1,-1, 0, -1,1,-1, 1, -1,1,-1, 0, -1,1,-1, 1, -1,
#      0,-1, 1, -1, 1,-1, 1, -1,1,-1, 1, -1,1,-1, 1, -1,1,-1, 0, -1,1,-1, 0, -1,1,-1, 0, -1,1,-1, 1, -1,1,-1, 0, -1,1,-1, 1, -1,
#      0,-1, 1, -1, 1,-1, 1, -1,1,-1, 1, -1,1,-1, 1, -1,1,-1, 0, -1,1,-1, 0, -1,1,-1, 0, -1,1,-1, 1, -1,1,-1, 0, -1,1,-1, 1, -1,
#      0,-1, 1, -1, 1,-1, 1, -1,1,-1, 1, -1,1,-1, 1, -1,1,-1, 0, -1,1,-1, 0, -1,1,-1, 0, -1,1,-1, 1, -1,1,-1, 0, -1,1,-1, 1, -1,
#      0,-1, 1, -1, 1,-1, 1, -1,1,-1, 1, -1,1,-1, 1, -1,1,-1, 0, -1,1,-1, 0, -1,1,-1, 0, -1,1,-1, 1, -1,1,-1, 0, -1,1,-1, 1, -1,
#      0,-1, 1, -1, 1,-1, 1, -1,1,-1, 1, -1,1,-1, 1, -1,1,-1, 0, -1,1,-1, 0, -1,1,-1, 0, -1,1,-1, 1, -1,1,-1, 0
#      ]

# y = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
#      1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
#      1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
#      1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
#      1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
#      1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
#      1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
#      1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
#      1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
#      1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
#      ]
X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(X, y, test_size=0.3, random_state=432, stratify=y, shuffle=True)
clf = clf.fit(X_cls_train, y_cls_train)
dot_data = tree.export_graphviz(clf, out_file=None,filled=True,feature_names=predictors_list)
#print(graphviz.Source(dot_data).view())
y_cls_pred = clf.predict(X_cls_test)

report = classification_report(y_cls_test, y_cls_pred)
accuracy = accuracy_score(y_cls_test, y_cls_pred)*100
from collections import Counter
print(Counter(y_cls_train))
print(Counter(y_cls_test))

print(y_cls_pred)
# print(len(y_cls_train))
print(report)

#--------------------------------LSTM--------------------------------------------

# x_future = lstmpred()
# x_future = [0.4, 0.7, 0,9, 0.8, 0.3, 0.56, 0.7, 0.8, 0.23, 0.19, 0.78, 
#             0.65, 0.87, 0.42, 0.49, 0.1, 0.3, 0.5, 0.7, 0.2, 0,4, 0.6, 0.8,
#             0.45, 0.63, 0.97, 0.46, 0.55, 0.73, 0.85, 0.09]
# x_future = pd.DataFrame(x_future)
# x_future.columns = ['close']
# x_future = calc(x_future)
# x_future = x_future.loc[:, ['close', 'RSI_14', 'ClgtEMA10', 'EMA10gtEMA30', 'MACDSIGgtMACD']]
# print('predicted price using LSTM is ',x_future)
# y_cls_pred = clf.predict(x_future)
# print('The future price prediceted by LSTM is ', x_future)
# print('The buy/sell signal corresponding future price is ', y_cls_pred)


#--------------------------------ARIMA--------------------------------------------


import statsmodels.api as sm
from statsmodels.tsa.stattools import acf

# Create Training and Test
history, test = train_test_split(data['close'], test_size=0.3,random_state=432, shuffle=False)

signals = np.array([])
#predictions = np.array([])
history = [x for x in history]
predictions = list()

for t in range(len(test)):
    model = sm.tsa.arima.ARIMA(history, order=(9,9,9))
    model_fit = model.fit()
    output = model_fit.forecast()
    # output = pd.DataFrame(output)
    # output.columns = ['close']
    # output = calc(output)
    # output = output.loc[:, ['close', 'RSI_14', 'ClgtEMA10', 'EMA10gtEMA30', 'MACDSIGgtMACD']]
    yhat = output[0]
    #predictions=np.append(predictions, output)
    predictions.append(yhat)
    # signal = clf.predict(pd.DataFrame(predictions))
    # signals = np.append(signals, signal[-1])
    obs = test[t]
    #history=np.append(history,obs)
    history.append(obs)
#error = np.sqrt(np.mean(np.square(predictions-test)))
test = [x for x in test]
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()
#print('RMS error: ',error)
print('predicted price using ARIMA in testing is ', predictions)
#print('predicted signal using ARIMA in testing is ',signals)
print('testing price is ', test)


ARIMAstrategy = 'predict all in once'
#ARIMAstrategy = 'predict one in a time'
signals = np.array([])
predictions = np.array([])
if ARIMAstrategy == 'predict one in a time':

    for t in range(5):
        #model = sm.tsa.arima.ARIMA(history, order=(5,1,0))
        model_fit = model.fit()
        output = model_fit.forecast()
        predictions=np.append(predictions, output)
        signal = clf.predict(pd.DataFrame(predictions))
        signals = np.append(signals, signal[-1])
        history=np.append(history,output)
    print('predicted signal using ARIMA is ',signal)

elif ARIMAstrategy == 'predict all in once':

    x_future = model_fit.forecast(20)
    # x_future = [0.4, 0.7, 0,9, 0.8, 0.3, 0.56, 0.7, 0.8, 0.23, 0.19, 0.78, 
    #             0.65, 0.87, 0.42, 0.49, 0.1, 0.3, 0.5, 0.7, 0.2, 0,4, 0.6, 0.8,
    #             0.45, 0.63, 0.97, 0.46, 0.55, 0.73, 0.85, 0.09]
    x_future = pd.DataFrame(x_future)
    x_future.columns = ['close']
    x_future = calc(x_future)
    x_future = x_future.loc[:, ['close', 'RSI_14', 'ClgtEMA10', 'EMA10gtEMA30', 'MACDSIGgtMACD']]
    print('predicted price using ARIMA is ',x_future)
    future_pred = clf.predict(x_future)
    print('predicted signal using ARIMA is ',future_pred)

