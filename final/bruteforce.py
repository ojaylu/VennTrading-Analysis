from implement_stoch_macd_strategy_optimization import *
from rsi import rsi_buy_sell
from obv import *
from ema import *
from flagrules import *
from itertools import product


def bruteforce(data):

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

    for i, spl in enumerate(strat_params_list):
        #print("Strategy %s out of %s..." % (i+1, spl))
        RSIflag = data.ta.rsi(close='close', length=strat_params_list[i][2], append=True, signal_indicators=True, xa=strat_params_list[i][0], xb=strat_params_list[i][1])
        macd_buy_price, macd_sell_price, stoch_macd_signal = implement_stoch_macd_strategy_optimization(data['close'], data['%k'], data['%d'], data['macd'], data['macd_signal'], strat_params_list[i][3], strat_params_list[i][4])
        rsi_buy_price, rsi_sell_price = rsi_buy_sell(data, RSIflag.columns[1], RSIflag.columns[2])
        obv_buy_price, obv_sell_price = OBV_buy_sell(OBVcalculation(data), 'OBV', 'OBV_EMA')
        data1, signal, ema_buy_price, ema_sell_price = calc_signal(data, 10, 30)
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
        surplus = sum(newMACDPriceSell) + sum(newRSIPriceSell) + sum(newOBVPriceSell) + sum(newEMAPriceBuy)
        - sum(newMACDPriceBuy) - sum(newRSIPriceBuy) - sum(newOBVPriceBuy) - sum(newEMAPriceSell)

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

    print("the optimized MACD strategy is strategy",strategyID,"Below is the strategy trading details ", MACDstrategy)
    print("the optimized RSI strategy is strategy",strategyID,"Below is the strategy trading details ", RSIstrategy)
    print("the optimized OBV strategy is strategy",strategyID,"Below is the strategy trading details ", OBVstrategy)
    print("the optimized EMA strategy is strategy",strategyID,"Below is the strategy trading details ", EMAstrategy)
    print("The money you will get using the optimized strategy ",strategyID,"is", highestsurplus)
    print("However, there may be some cases where the indicators generate contradictory buy/sell signals")
    print("We need a rules to determine the buy/sell signal combining all indicators, the rule is performed below")

    MACDstrategy, RSIstrategy, OBVstrategy, EMAstrategy, combineflag, data, newsurplus = flagrules(
        data, MACDstrategy, RSIstrategy, OBVstrategy, EMAstrategy)
    

    #combineflag = bfs('b', data['close'])

    data['prediction'] = combineflag
    data['RSI_14'].fillna(value=data['RSI_14'].mean(), inplace=True)

    return data, combineflag, macdlb, macdub, rsilb, rsiub, newsurplus