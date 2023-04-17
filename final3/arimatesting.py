import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from calc import calc
from sklearn.model_selection import train_test_split
from bfs import bfs
from numpy import nan
from bruteforce import *
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from datacollection import *
import datetime

def derivation_c(data):

    df = data.to_csv('output.csv')
    #news_w_label = pd.read_csv("Datasets/Combined_News_DJIA.csv")
    df = pd.read_csv('output.csv', names=['close'], header=0)
    # Original Series
    fig, axes = plt.subplots(3, 2, sharex=True)
    axes[0, 0].plot(df.close); axes[0, 0].set_title('Original Series of the closing price')
    plot_acf(df.close, ax=axes[0, 1])
    # 1st Differencing
    axes[1, 0].plot(df.close.diff()); axes[1, 0].set_title('1st Order Differencing')
    plot_acf(df.close.diff().dropna(), ax=axes[1, 1])
    # 2nd Differencing
    axes[2, 0].plot(df.close.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
    plot_acf(df.close.diff().diff().dropna(), ax=axes[2, 1])

    plt.show()

def derivation_v(data):

    df = data.to_csv('output.csv')
    #news_w_label = pd.read_csv("Datasets/Combined_News_DJIA.csv")
    df = pd.read_csv('output.csv', names=['volume'], header=0)
    # Original Series
    fig, axes = plt.subplots(3, 2, sharex=True)
    axes[0, 0].plot(df.volume); axes[0, 0].set_title('Original Series of the volume')
    plot_acf(df.volume, ax=axes[0, 1])
    # 1st Differencing
    axes[1, 0].plot(df.volume.diff()); axes[1, 0].set_title('1st Order Differencing')
    plot_acf(df.volume.diff().dropna(), ax=axes[1, 1])
    # 2nd Differencing
    axes[2, 0].plot(df.volume.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
    plot_acf(df.volume.diff().diff().dropna(), ax=axes[2, 1])

    plt.show()

def derivation_h(data):

    df = data.to_csv('output.csv')
    #news_w_label = pd.read_csv("Datasets/Combined_News_DJIA.csv")
    df = pd.read_csv('output.csv', names=['high'], header=0)
    # Original Series
    fig, axes = plt.subplots(3, 2, sharex=True)
    axes[0, 0].plot(df.high); axes[0, 0].set_title('Original Series of the high price')
    plot_acf(df.high, ax=axes[0, 1])
    # 1st Differencing
    axes[1, 0].plot(df.high.diff()); axes[1, 0].set_title('1st Order Differencing')
    plot_acf(df.high.diff().dropna(), ax=axes[1, 1])
    # 2nd Differencing
    axes[2, 0].plot(df.high.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
    plot_acf(df.high.diff().diff().dropna(), ax=axes[2, 1])

    plt.show()

def derivation_l(data):

    df = data.to_csv('output.csv')
    #news_w_label = pd.read_csv("Datasets/Combined_News_DJIA.csv")
    df = pd.read_csv('output.csv', names=['low'], header=0)
    # Original Series
    fig, axes = plt.subplots(3, 2, sharex=True)
    axes[0, 0].plot(df.low); axes[0, 0].set_title('Original Series of the low price')
    plot_acf(df.low, ax=axes[0, 1])
    # 1st Differencing
    axes[1, 0].plot(df.low.diff()); axes[1, 0].set_title('1st Order Differencing')
    plot_acf(df.low.diff().dropna(), ax=axes[1, 1])
    # 2nd Differencing
    axes[2, 0].plot(df.low.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
    plot_acf(df.low.diff().diff().dropna(), ax=axes[2, 1])

    plt.show()

def arimatesting1(data, history, test, plotderivationgraph):

    if plotderivationgraph == True:
        derivation_c(history)
        derivation_h(history)
        derivation_l(history)
        derivation_v(history)

    signals = np.array([])
    #predictions = np.array([])
    history = [x for x in history]
    predictions = list()

    for t in range(len(test)):
        model = sm.tsa.arima.ARIMA(history, order=(5,1,0))
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
    error = np.sqrt(np.mean(np.square(predictions-test)))
    test = [x for x in test]
    fig = plt.figure(figsize=(15, 10))
    plt.cla()
    pandafuture = pd.DataFrame()
    data.index = pd.to_datetime(data.index)
    pandafuture['datetime'] = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=len(test))
    pandafuture = pandafuture.set_index('datetime')
    plt.plot(pandafuture.index,predictions, label = 'prediction by ARIMA testing of one-time-one-day mode')
    plt.plot(pandafuture.index,test, label = 'testing of ARIMA of one-time-one-day mode')
    plt.legend()
    plt.show()
    print('RMS error: ',error)
    #print('predicted price using ARIMA in testing is ', predictions)
    #print('predicted signal using ARIMA in testing is ',signals)
    #print('testing price is ', test)


def arimatestingall(data, history, test, plotderivationgraph):
    model = sm.tsa.arima.ARIMA(history, order=(20,1,0))
    model_fit = model.fit()
    output = model_fit.forecast(len(test))
    output = [x for x in output]
    error = np.sqrt(np.mean(np.square(output-test)))
    test = [x for x in test]
    plt.cla()
    pandafuture = pd.DataFrame()
    data.index = pd.to_datetime(data.index)
    pandafuture['datetime'] = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=len(test))
    pandafuture = pandafuture.set_index('datetime')
    plt.plot(pandafuture.index,output, label = 'prediction by ARIMA testing of all-in-once mode')
    plt.plot(pandafuture.index,test, label = 'testing of ARIMA of all-in-once mode')
    plt.legend()
    plt.show()
    print('RMS error: ',error)
    #print('predicted price using ARIMA in testing is ', output)
    #print('predicted signal using ARIMA in testing is ',signals)
    #print('testing price is ', test)
    if plotderivationgraph == True:
        derivation_c(history)
        derivation_h(history)
        derivation_l(history)
        derivation_v(history)


def myarima(data, clf, pred_num, ARIMAstrategy, plotderivationgraph, plotbfsgraph):

    import statsmodels.api as sm
    from statsmodels.tsa.stattools import acf

    #data.index = pd.DatetimeIndex(data.index).to_period('D')

    # Create Training and Test
    history_c, test_c = train_test_split(data['close'], test_size=0.3,random_state=432, shuffle=True)
    history_h, test_h = train_test_split(data['high'], test_size=0.3,random_state=432, shuffle=True)
    history_l, test_l = train_test_split(data['low'], test_size=0.3,random_state=432, shuffle=True)
    history_v, test_v = train_test_split(data['volume'], test_size=0.3,random_state=432, shuffle=True)

    # X_train_date=data['datetime'][:int(0.7*len(X))]
    # X_test_date=data['datetime'][int(0.7*len(X)):]

    arimatesting1(data, history_c, test_c, plotderivationgraph)
    arimatesting1(data, history_h, test_h, plotderivationgraph)
    arimatesting1(data, history_l, test_l, plotderivationgraph)
    arimatesting1(data, history_v, test_v, plotderivationgraph)
        
    arimatestingall(data, history_c, test_c, plotderivationgraph)
    arimatestingall(data, history_h, test_h, plotderivationgraph)
    arimatestingall(data,history_l, test_l, plotderivationgraph)
    arimatestingall(data,history_v, test_v, plotderivationgraph)

    if ARIMAstrategy == 'predict one in a time':
        signals = np.array([])
        predictions = np.array([])
        clist = []
        hlist = []
        llist = []
        vlist = []
        future = pd.DataFrame()
        for t in range(pred_num):
            model = sm.tsa.arima.ARIMA(history_c, order=(20,1,0))
            model_fit = model.fit()
            output = model_fit.forecast()
            output = [x for x in output]
            clist.append(output)
            clist = [x for x in clist]
            # clist = pd.DataFrame(clist)
            # clist.columns = ['close']
            history_c=np.append(history_c,output)
            
            model = sm.tsa.arima.ARIMA(history_h, order=(20,1,0))
            model_fit = model.fit()
            output = model_fit.forecast()
            output = [x for x in output]
            hlist.append(output)
            # hlist = pd.DataFrame(hlist)
            # hlist.columns = ['high']
            history_h=np.append(history_h,output)

            model = sm.tsa.arima.ARIMA(history_l, order=(20,1,0))
            model_fit = model.fit()
            output = model_fit.forecast()
            output = [x for x in output]
            llist.append(output)
            # llist = pd.DataFrame(llist)
            # llist.columns = ['low']
            history_l=np.append(history_l,output)

            model = sm.tsa.arima.ARIMA(history_v, order=(20,1,0))
            model_fit = model.fit()
            output = model_fit.forecast()
            output = [x for x in output]
            vlist.append(output)
            # vlist = pd.DataFrame(vlist)
            # vlist.columns = ['volume']
            history_v=np.append(history_v,output)

            # ax = plt.gca()

            future = pd.DataFrame()
            future['close'] = pd.DataFrame(clist)
            future['high'] = pd.DataFrame(hlist)
            future['low'] = pd.DataFrame(llist)
            future['volume'] = pd.DataFrame(vlist)

            # ax = plt.gca()
            # future.plot(kind='line',y='close',color='yellow', ax=ax)
            # future.plot(kind='line',y='high',color='black', ax=ax)
            # future.plot(kind='line',y='low',color='orange', ax=ax)
            # plt.title('the trend of the predicted price using ARIMA one-in-a-time')
            # plt.show()

            # h = pd.DataFrame()
            # h['close'] = pd.DataFrame(history_c)
            # h['high'] = pd.DataFrame(history_h)
            # h['low'] = pd.DataFrame(history_l)
            # h['volume'] = pd.DataFrame(history_v)

            # h.plot(kind='line',y='close',color='yellow', ax=ax)
            # h.plot(kind='line',y='high',color='black', ax=ax)
            # h.plot(kind='line', y='low',color='orange', ax=ax)
            # plt.title('the increase of data used in prediction using ARIMA one-in-a-time')
            # plt.show()

            future = calc(future)
            future = future.loc[:, ['close', 'RSI_14', 'EMA10', 'EMA30', 'macd','OBV', 'ATR','ClgtEMA10', 'EMA10gtEMA30', 'MACDSIGgtMACD', '%k', '%d']]

            future = future.fillna(value=future['RSI_14'].mean())

            signal = clf.predict(future)
            signals = np.append(signals, signal[-1])

        sell = []
        buy = []
        for i in range(pred_num):
            if signals[i] == 1:
                buy.append(future['close'][i])
                sell.append(nan)
            elif signals[i] == -1:
                buy.append(nan)
                sell.append(future['close'][i])
            else:
                buy.append(nan)
                sell.append(nan)
        future['buy price signal'] = pd.DataFrame(buy)
        future['sell price signal'] = pd.DataFrame(sell)

        plt.cla()
        data.index = pd.to_datetime(data.index)
        future['datetime'] = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=pred_num)
        future = future.set_index('datetime')

        plt.figure(figsize=(15,8))
        plt.plot(future.index, future['close'])
        plt.scatter(future.index, future['buy price signal'], color = 'red')
        plt.scatter(future.index, future['sell price signal'], color = 'green')
        #plt.scatter(x_future.index[x_future['signal' == -1.0]], x_future['signal' == -1.0], color = 'blue', label = 'sell')
        plt.legend()
        plt.show()
        print('predicted signal using ARIMA is ',signals)

        return signals, future, future

    elif ARIMAstrategy == 'predict all in once':

        future = pd.DataFrame()
        model = sm.tsa.arima.ARIMA(history_c, order=(20,1,0))
        model_fit = model.fit()
        if plotderivationgraph == True:
            residuals = pd.DataFrame(model_fit.resid)
            fig, ax = plt.subplots(1,2)
            residuals.plot(title="Residuals", ax=ax[0])
            residuals.plot(kind='kde', title='Density', ax=ax[1])
            plt.show()
        x_future = model_fit.forecast(pred_num)
        x_future = [x for x in x_future]
        if pred_num <= 22:
            bfsfuture, bfsfinal, bfsfrontier = bfs('b',x_future, plotbfsgraph=plotbfsgraph)
            print('predicted signal using bfs is ', bfsfuture[1])
            #print('the final frontier is ', bfsfrontier)
            bfssignal = bfsfuture[1]
        x_future = pd.DataFrame(x_future)
        x_future.columns = ['close']
        future['close'] = x_future

        model = sm.tsa.arima.ARIMA(history_h, order=(20,1,0))
        model_fit = model.fit()
        x_future = model_fit.forecast(pred_num)
        x_future = [x for x in x_future]
        x_future = pd.DataFrame(x_future)
        x_future.columns = ['high']
        future['high'] = x_future
        if plotderivationgraph == True:
            residuals = pd.DataFrame(model_fit.resid)
            fig, ax = plt.subplots(1,2)
            residuals.plot(title="Residuals", ax=ax[0])
            residuals.plot(kind='kde', title='Density', ax=ax[1])
            plt.show()

        model = sm.tsa.arima.ARIMA(history_l, order=(20,1,0))
        model_fit = model.fit()
        x_future = model_fit.forecast(pred_num)
        x_future = [x for x in x_future]
        x_future = pd.DataFrame(x_future)
        x_future.columns = ['low']
        future['low'] = x_future
        if plotderivationgraph == True:
            residuals = pd.DataFrame(model_fit.resid)
            fig, ax = plt.subplots(1,2)
            residuals.plot(title="Residuals", ax=ax[0])
            residuals.plot(kind='kde', title='Density', ax=ax[1])
            plt.show()

        model = sm.tsa.arima.ARIMA(history_v, order=(20,1,0))
        model_fit = model.fit()
        x_future = model_fit.forecast(pred_num)
        x_future = [x for x in x_future]
        x_future = pd.DataFrame(x_future)
        x_future.columns = ['volume']
        future['volume'] = x_future
        if plotderivationgraph == True:
            residuals = pd.DataFrame(model_fit.resid)
            fig, ax = plt.subplots(1,2)
            residuals.plot(title="Residuals", ax=ax[0])
            residuals.plot(kind='kde', title='Density', ax=ax[1])
            plt.show()

        ax = plt.gca()
        plt.cla()
        data.index = pd.to_datetime(data.index)
        future['datetime'] = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=pred_num)
        future.plot(kind='line',x='datetime',y='close',color='yellow', ax=ax)
        future.plot(kind='line',x='datetime',y='high',color='black', ax=ax)
        future.plot(kind='line', x='datetime',y='low',color='orange', ax=ax)
        plt.title('the trend of the predicted price using ARIMA')
        plt.show()
        
        x_future = calc(future)
        print("here is the strategies suggested by bruteforce")
        x_future_2 = x_future
        x_future_2, combineflag, macdlb, macdub, rsilb, rsiub, newsurplus, highestsurplus, strategyID, brute = bruteforce(x_future_2,plotgraph=False,
                                                                                       plotmacdgraph=False,
                                                                                       plotrsigraph=False,
                                                                                       plotobvgraph=False,
                                                                                       plotemagraph=False)
        print("here is the strategies suggested by bruteforce: \n MACD:", macdlb, macdub, '\n, RSI:', rsilb, rsiub)
        x_future = x_future.loc[:, ['close', 'RSI_14', 'EMA10', 'EMA30', 'macd','OBV', 'ATR','ClgtEMA10', 'EMA10gtEMA30', 'MACDSIGgtMACD', '%k', '%d']]
        x_future = x_future.fillna(value=x_future['RSI_14'].mean())
        #print('predicted price using ARIMA is ',x_future)
        print("here is the strategies suggested by AI")
        future_pred = clf.predict(x_future)
        print('predicted possible signal using ARIMA is ',future_pred)

        # if pred_num <= 22:
        #     x_future['signal'] = pd.DataFrame(bfssignal)
        # else:
        #     x_future['signal'] = pd.DataFrame(future_pred)

        sell = []
        buy = []
        if pred_num <= 22:
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
            for i in range(len(future_pred)):
                if future_pred[i] == 1:
                    buy.append(x_future['close'][i])
                    sell.append(nan)
                elif future_pred[i] == -1:
                    buy.append(nan)
                    sell.append(x_future['close'][i])
                else:
                    buy.append(nan)
                    sell.append(nan)
            x_future['buy price signal'] = pd.DataFrame(buy)
            x_future['sell price signal'] = pd.DataFrame(sell)

        #x_future['datetime'] = pd.to_datetime(x_future['datetime'])


        plt.cla()
        data.index = pd.to_datetime(data.index)
        x_future['datetime'] = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=pred_num)
        x_future = x_future.set_index('datetime')

        plt.figure(figsize=(15,8))
        plt.plot(x_future.index, x_future['close'])
        plt.scatter(x_future.index, x_future['buy price signal'], color = 'red')
        plt.scatter(x_future.index, x_future['sell price signal'], color = 'green')
        plt.legend()
        plt.show()

        if pred_num <= 22:
            return (bfsfuture, bfsfinal, bfsfrontier), x_future, x_future_2
        else:
            return future_pred, x_future, x_future_2