import numpy as np


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
