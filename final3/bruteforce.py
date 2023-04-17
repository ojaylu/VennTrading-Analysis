from implement_stoch_macd_strategy_optimization import *
from rsi import rsi_buy_sell
from obv import *
from ema import *
from flagrules import *
from itertools import product
import pandas as pd
import pandas_ta as ta
from pandas.plotting import table
from IPython.display import display


def bruteforce(data, plotgraph,plotmacdgraph,plotrsigraph,plotobvgraph,plotemagraph):

    RSILB = list(range(41,50,5))
    RSIUB = list(range(51,60,5))
    day = [14, 31]
    MACDLB = list(range(45,50,1))
    MACDUB = list(range(50,55,1))
    strat_params_list = list(product(RSILB, RSIUB, day, MACDLB, MACDUB))
    #strat_params_dict_list = {'xa': [40, 50, 60],'xb': [45, 55, 65],'day' : [7, 14, 31]}
    surplus = 0
    highestsurplus = -100000
    spl = len(strat_params_list)
    MACDstrategy = 0
    RSIstrategy = 0
    OBVstrategy = 0
    EMAstrategy = 0
    strategyID = 0
    brute = pd.DataFrame(columns = ['macd buying price', 'macd selling price', 'rsi buying price', 'rsi selling price','obv buying price',
                                    'obv seling price', 'ema buying price','ema selling price' ])

    for i, spl in enumerate(strat_params_list):
        #print("Strategy %s out of %s..." % (i+1, spl))
        RSIflag = data.ta.rsi(close='close', length=strat_params_list[i][2], append=True, signal_indicators=True, xa=strat_params_list[i][0], xb=strat_params_list[i][1])
        macd_buy_price, macd_sell_price, stoch_macd_signal = implement_stoch_macd_strategy_optimization(data['close'], data['%k'], data['%d'], data['macd'], data['macd_signal'], strat_params_list[i][3], strat_params_list[i][4], plotmacdgraph=plotmacdgraph)
        rsi_buy_price, rsi_sell_price = rsi_buy_sell(data, RSIflag.columns[1], RSIflag.columns[2], plotrsigraph=plotrsigraph)
        obv_buy_price, obv_sell_price = OBV_buy_sell(OBVcalculation(data), 'OBV', 'OBV_EMA', plotobvgraph=plotobvgraph)
        data1, signal, ema_buy_price, ema_sell_price = calc_signal(data, 10, 30, plotemagraph=plotemagraph)
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
        newEMAPriceBuy = [item for item in ema_buy_price if not(math.isnan(item)) == True]
        newEMAPriceSell = [item for item in ema_sell_price if not(math.isnan(item)) == True]

        if plotgraph == True:

            plotMACDPriceBuy = [0 if math.isnan(x) else x for x in macd_buy_price]
            plotMACDPriceSell = [0 if math.isnan(x) else x for x in macd_sell_price]
            plotRSIPriceBuy = [0 if math.isnan(x) else x for x in rsi_buy_price]
            plotRSIPriceSell = [0 if math.isnan(x) else x for x in rsi_sell_price]
            plotOBVPriceBuy = [0 if math.isnan(x) else x for x in obv_buy_price]
            plotOBVPriceSell = [0 if math.isnan(x) else x for x in obv_sell_price]
            plotEMAPriceBuy = [0 if math.isnan(x) else x for x in ema_buy_price]
            plotEMAPriceSell = [0 if math.isnan(x) else x for x in ema_sell_price]

            plt.subplot(2, 4, 1)
            plt.plot(plotMACDPriceBuy,label = "macd buying price")
            plt.subplot(2, 4, 2)
            plt.plot(plotMACDPriceSell, label = "macd selling price")
            plt.subplot(2, 4, 3)
            plt.plot(plotRSIPriceBuy, label = "rsi buying price")
            plt.subplot(2, 4, 4)
            plt.plot(plotRSIPriceSell, label = "rsi selling price")
            plt.subplot(2, 4, 5)
            plt.plot(plotOBVPriceBuy, label = "obv buying price")
            plt.subplot(2, 4, 6)
            plt.plot(plotOBVPriceSell, label = "obv selling price")
            plt.subplot(2, 4, 7)
            plt.plot(plotEMAPriceBuy, label = "ema buying price")
            plt.subplot(2, 4, 8)
            plt.plot(plotEMAPriceSell, label = "ema selling price")
            plt.xlabel('time')
            plt.ylabel('price')
            #plt.text(-10, 100, 'after perform series of condition, buy/sell signal is generated', fontsize = 22, bbox = dict(facecolor = 'red', alpha = 0.5))
            plt.title('the selling price and buying price corresponding to different indicator of strategy')
            plt.legend()
            plt.show()
        surplus = sum(newMACDPriceSell) + sum(newRSIPriceSell) + sum(newOBVPriceSell) + sum(newEMAPriceBuy)
        - sum(newMACDPriceBuy) - sum(newRSIPriceBuy) - sum(newOBVPriceBuy) - sum(newEMAPriceSell)

        df1 = pd.DataFrame({'macd buying price':[macd_buy_price],
                              'macd selling price':[macd_sell_price],
                              'rsi buying price':[rsi_buy_price],
                              'rsi selling price':[rsi_sell_price],
                              'obv buying price':[obv_buy_price],
                              'obv seling price': [obv_sell_price],
                              'ema buying price':[ema_buy_price],
                              'ema selling price':[ema_sell_price],
                              'surplus':surplus})
        
        brute = brute.append(df1, ignore_index = True)


        #print("This strategy help you earn",surplus)

        if surplus >= highestsurplus:
            macdlb = strat_params_list[i][3]; macdub = strat_params_list[i][4]
            rsilb = strat_params_list[i][0]; rsiub = strat_params_list[i][1]
            highestsurplus = surplus
            strategyID = i
            MACDstrategy = macd_buy_price, macd_sell_price
            RSIstrategy = rsi_buy_price, rsi_sell_price
            OBVstrategy = obv_buy_price, obv_sell_price
            EMAstrategy = ema_buy_price, ema_sell_price

    
    ax = plt.gca()
    plt.cla()
    brute.plot(kind='line',y='surplus',color='black', ax=ax)
    plt.title('the distribution of the surplus of different combination')
    plt.show()

    print("the optimized MACD strategy is strategy",strategyID,"Below is the strategy trading details ", MACDstrategy)
    print("the optimized RSI strategy is strategy",strategyID,"Below is the strategy trading details ", RSIstrategy)
    print("the optimized OBV strategy is strategy",strategyID,"Below is the strategy trading details ", OBVstrategy)
    print("the optimized EMA strategy is strategy",strategyID,"Below is the strategy trading details ", EMAstrategy)
    print("The money you will get using the optimized strategy ",strategyID,"is", highestsurplus)
    print("However, there may be some cases where the indicators generate contradictory buy/sell signals")
    print("We need a rules to determine the buy/sell signal combining all indicators, the rule is performed below")

    flagstrategy = 'sum'

    MACDstrategy, RSIstrategy, OBVstrategy, EMAstrategy, combineflag, data, newsurplus, flagsummary = flagrules(
        data, MACDstrategy, RSIstrategy, OBVstrategy, EMAstrategy, flagstrategy)
    

    print('top 20 is', brute['surplus'].idxmax())
    print('the final dataframe is', brute)

    # ax = plt.gca()
    # brute.plot(kind='line',y='macd buying price',color='yellow', ax=ax)
    # brute.plot(kind='line',y='rsi buying price',color='black', ax=ax)
    # brute.plot(kind='line', y='obv buying price',color='orange', ax=ax)
    # plt.title('the trend of brute force')
    # plt.show()

    brutejson = brute.to_json(orient = 'columns')

    #combineflag = bfs('b', data['close'])

    data['prediction'] = combineflag
    data['RSI_14'].fillna(value=data['RSI_14'].mean(), inplace=True)

    return data, combineflag, macdlb, macdub, rsilb, rsiub, newsurplus, highestsurplus, strategyID, brute