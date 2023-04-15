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

def arimatesting1(history, test):

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
    # plt.plot(test)
    # plt.plot(predictions, color='red')
    # plt.show()
    print('RMS error: ',error)
    #print('predicted price using ARIMA in testing is ', predictions)
    #print('predicted signal using ARIMA in testing is ',signals)
    #print('testing price is ', test)


def arimatestingall(history, test):
    model = sm.tsa.arima.ARIMA(history, order=(20,1,0))
    model_fit = model.fit()
    output = model_fit.forecast(len(test))
    output = [x for x in output]
    error = np.sqrt(np.mean(np.square(output-test)))
    test = [x for x in test]
    plt.plot(test)
    plt.plot(output, color='red')
    plt.show()
    print('RMS error: ',error)
    #print('predicted price using ARIMA in testing is ', output)
    #print('predicted signal using ARIMA in testing is ',signals)
    #print('testing price is ', test)


def myarima(data, clf, pred_num, ARIMAstrategy):

    import statsmodels.api as sm
    from statsmodels.tsa.stattools import acf

    data.index = pd.DatetimeIndex(data.index).to_period('D')

    # Create Training and Test
    history_c, test_c = train_test_split(data['close'], test_size=0.3,random_state=432, shuffle=True)
    history_h, test_h = train_test_split(data['high'], test_size=0.3,random_state=432, shuffle=True)
    history_l, test_l = train_test_split(data['low'], test_size=0.3,random_state=432, shuffle=True)
    history_v, test_v = train_test_split(data['volume'], test_size=0.3,random_state=432, shuffle=True)

    # X_train_date=data['datetime'][:int(0.7*len(X))]
    # X_test_date=data['datetime'][int(0.7*len(X)):]

    # arimatesting1(history_c, test_c)
    # arimatesting1(history_h, test_h)
    # arimatesting1(history_l, test_l)
    # arimatesting1(history_v, test_v)
        
    # arimatestingall(history_c, test_c)
    # arimatestingall(history_h, test_h)
    # arimatestingall(history_l, test_l)
    # arimatestingall(history_v, test_v)

    ARIMAstrategy = 'predict all in once'
    #ARIMAstrategy = 'predict one in a time'

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

            future = pd.DataFrame()
            future['close'] = pd.DataFrame(clist)
            future['high'] = pd.DataFrame(hlist)
            future['low'] = pd.DataFrame(llist)
            future['volume'] = pd.DataFrame(vlist)

            future = calc(future)
            future = future.loc[:, ['close', 'ClgtEMA10', 'EMA10gtEMA30', 'MACDSIGgtMACD','OBV', 'RSI_14', 'ATR']]

            #print('future', future)

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


        plt.figure(figsize=(15,8))
        plt.plot(future.index, future['close'])
        plt.scatter(future.index, future['buy price signal'], color = 'red')
        plt.scatter(future.index, future['sell price signal'], color = 'green')
        #plt.scatter(x_future.index[x_future['signal' == -1.0]], x_future['signal' == -1.0], color = 'blue', label = 'sell')
        plt.legend()
        plt.show()
        print('predicted signal using ARIMA is ',signal)

    elif ARIMAstrategy == 'predict all in once':

        future = pd.DataFrame()
        model = sm.tsa.arima.ARIMA(history_c, order=(20,1,0))
        model_fit = model.fit()
        x_future = model_fit.forecast(pred_num)
        x_future = [x for x in x_future]
        print('pred_num is ', pred_num)
        if pred_num <= 22:
            bfsfuture = bfs('b',x_future)
            print('predicted signal using bfs is ', bfsfuture[1])
            bfssignal = bfsfuture[1]
        # x_future = [0.4, 0.7, 0,9, 0.8, 0.3, 0.56, 0.7, 0.8, 0.23, 0.19, 0.78, 
        #             0.65, 0.87, 0.42, 0.49, 0.1, 0.3, 0.5, 0.7, 0.2, 0,4, 0.6, 0.8,
        #             0.45, 0.63, 0.97, 0.46, 0.55, 0.73, 0.85, 0.09]
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

        model = sm.tsa.arima.ARIMA(history_l, order=(20,1,0))
        model_fit = model.fit()
        x_future = model_fit.forecast(pred_num)
        x_future = [x for x in x_future]
        x_future = pd.DataFrame(x_future)
        x_future.columns = ['low']
        future['low'] = x_future

        model = sm.tsa.arima.ARIMA(history_v, order=(20,1,0))
        model_fit = model.fit()
        x_future = model_fit.forecast(pred_num)
        x_future = [x for x in x_future]
        x_future = pd.DataFrame(x_future)
        x_future.columns = ['volume']
        future['volume'] = x_future
        
        x_future = calc(future)
        print("here is the strategies suggested by AI")
        x_future_2 = x_future
        x_future_2, combineflag, macdlb, macdub, rsilb, rsiub, newsurplus = bruteforce(x_future_2)
        x_future = x_future.loc[:, ['close', 'RSI_14', 'EMA10', 'EMA30', 'macd','OBV', 'ATR','ClgtEMA10', 'EMA10gtEMA30', 'MACDSIGgtMACD', '%k', '%d']]
        x_future = x_future.fillna(value=x_future['RSI_14'].mean())
        #print('predicted price using ARIMA is ',x_future)
        future_pred = clf.predict(x_future)
        print('predicted signal using ARIMA is ',future_pred)

        # if pred_num <= 22:
        #     x_future['signal'] = pd.DataFrame(bfssignal)
        # else:
        #     x_future['signal'] = pd.DataFrame(future_pred)

        sell = []
        buy = []
        if pred_num <= 22:
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


        plt.figure(figsize=(15,8))
        plt.plot(x_future.index, x_future['close'])
        plt.scatter(x_future.index, x_future['buy price signal'], color = 'red')
        plt.scatter(x_future.index, x_future['sell price signal'], color = 'green')
        plt.legend()
        plt.show()

        if pred_num <= 22:
            return bfssignal
        else:
            return future_pred