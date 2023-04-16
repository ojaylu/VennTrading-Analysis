import numpy as np
import math

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