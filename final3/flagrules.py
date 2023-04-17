import math
import pandas as pd
import matplotlib.pyplot as plt

def flagrules(data, MACDstrategy, RSIstrategy, OBVstrategy, EMAstrategy):
    surplus = 0
    MACDflag = []
    RSIflag = []
    OBVflag = []
    EMAflag = []
    macd_buy_price = MACDstrategy[0]
    macd_sell_price = MACDstrategy[1]
    rsi_buy_price = RSIstrategy[0]
    rsi_sell_price = RSIstrategy[1]
    obv_buy_price = OBVstrategy[0]
    obv_sell_price = OBVstrategy[1]
    ema_buy_price = EMAstrategy[0]
    ema_sell_price = EMAstrategy[1]
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
    for j in range(len(ema_buy_price)):
        if math.isnan(ema_buy_price[j]) == False:
            EMAflag.append(1)
        elif math.isnan(ema_sell_price[j]) == False:
            EMAflag.append(-1)
        else:
            EMAflag.append(0)

    df1 = pd.DataFrame({'macd flag':MACDflag,'rsi flag':RSIflag,'obv flag':OBVflag,'ema flag':EMAflag})
    df1json = df1.to_json(orient = 'columns')
    ax = plt.gca()
    df1.plot(kind='line',y='macd flag',color='yellow', ax=ax)
    df1.plot(kind='line',y='rsi flag',color='blue', ax=ax)
    df1.plot(kind='line',y='obv flag',color='red', ax=ax)
    df1.plot(kind='line',y='ema flag',color='green', ax=ax)
    plt.title('the flag of each indicator according to flag rules')

    print('the flag summary', df1)

    strategy = 'sum'
    if strategy == 'sum':
        combineflag = []
        for j in range(len(EMAflag)):
            if (MACDflag[j] + RSIflag[j] + OBVflag[j] + EMAflag[j] ) / 4 > 0.1:
                combineflag.append(1)
            elif (MACDflag[j] + RSIflag[j] + OBVflag[j] + EMAflag[j]) / 4 < -0.1:
                combineflag.append(-1)
            else:
                combineflag.append(0)

    elif strategy == 'length':
        combineflag = []
        buycount = 0
        sellcount = 0
        holdcount = 0
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
            if EMAflag[j] == 1:
                buycount+=1
            elif EMAflag[j] == -1:
                sellcount+=1
            else:
                holdcount+=1
            if buycount > sellcount and buycount >= holdcount:
                combineflag.append(1)
            elif sellcount > buycount and sellcount >= holdcount:
                combineflag.append(-1)
            else:
                combineflag.append(0)
    
    for i in range(len(combineflag)):
        if combineflag[i] == 1:
            surplus+=data['close'][i]
        elif combineflag[i] == -1:
            surplus-=data['close'][i]
        elif combineflag[i] ==0:
            surplus = surplus

    print("According to Oscar's method, the buy and sell signal will be ", combineflag)
    print("According to Oscar's method, the surplus will be ", surplus)
    return MACDstrategy, RSIstrategy, OBVstrategy, EMAstrategy, combineflag, data, surplus