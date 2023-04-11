import math

def flagrules(MACDstrategy, RSIstrategy, OBVstrategy, strategyID, highestsurplus):
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
    return MACDstrategy, RSIstrategy, OBVstrategy, highestsurplus, combineflag