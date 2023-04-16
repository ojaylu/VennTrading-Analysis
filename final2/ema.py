import matplotlib.pyplot as plt
import numpy as np
import math

#Function to calculate the signal (1: True, 0: False -> Both 0 indicate 'Hold')
def calc_signal(df, short_days, long_days):
    signal = []
    ema_buy_price = [0]
    ema_sell_price = [0]
    short_term = 'EMA'+ str(short_days)
    long_term = 'EMA'+ str(long_days)
    df['Buy'] = 0
    df['Sell'] = 0
    for i in range(1, len(df['close'])):
        if (df[short_term][i] > df[long_term][i]) and (df[short_term][i-1] <= df[long_term][i-1]):
            df['Buy'][i] = 1 #Buy
        elif (df[short_term][i] < df[long_term][i]) and (df[short_term][i-1] >= df[long_term][i-1]):
            df['Sell'][i] = 1 #Sell
    for i in range(1, len(df['close'])):
        if (df[short_term][i] > df[long_term][i]) and (df[short_term][i-1] <= df[long_term][i-1]):
            ema_buy_price.append(df['close'][i])
            ema_sell_price.append(np.nan)
        elif (df[short_term][i] < df[long_term][i]) and (df[short_term][i-1] >= df[long_term][i-1]):
            ema_buy_price.append(np.nan)
            ema_sell_price.append(df['close'][i])
        else:
            ema_buy_price.append(np.nan)
            ema_sell_price.append(np.nan)

    # plotEMAPriceBuy = [0 if math.isnan(x) else x for x in ema_buy_price]
    # plotEMAPriceSell = [0 if math.isnan(x) else x for x in ema_sell_price]
    # plt.plot(plotEMAPriceBuy, label = 'ema sell price')
    # plt.plot(plotEMAPriceSell, label = 'ema sell price')
    # plt.title('EMA buying/selling price after strategy: \n if the ema10 is greater than ema30 while on the next day ema10 is smaller than ema30, a buy signal will be generated. \n if the ema10 is smaller than ema30 while on the next day ema10 is greater than ema30, a sell signal will be generated.')
    # plt.legend()
    # plt.show()


    return df, signal, ema_buy_price, ema_sell_price

def show_graph(df, short_days, long_days):
    short_term = 'EMA'+ str(short_days)
    long_term = 'EMA'+ str(long_days)
    fig = plt.figure(figsize=(15, 15))
    plt.plot(df['Close'][long_days:], color="black", label="Closing Price")
    plt.plot(df[short_term][long_days:], color="magenta", linestyle='--', label=str(short_days)+" Days Moving Average")
    plt.plot(df[long_term][long_days:], color="blue", linestyle='--', label=str(long_days)+" Days Moving Average")
    plt.plot(df[long_days:][df['Buy'] == 1][short_term], marker='o', color="green", linestyle='None', label="Golden Cross (Buy)")
    plt.plot(df[long_days:][df['Sell'] == 1][short_term], marker='o', color="red", linestyle='None', label="Death Cross (Sell)")
    plt.legend()
    datelist = [df['Date'][i] for i in range(long_days, 2000, 100)]
    plt.xticks(ticks=list(range(long_days, 2000, 100)), labels=datelist, rotation = 45)
    plt.show()