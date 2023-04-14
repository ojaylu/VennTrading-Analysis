import math
import numpy as np

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