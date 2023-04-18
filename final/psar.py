from collections import deque
from datacollection import datacollection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class PSAR:

  def __init__(self, init_af=0.02, max_af=0.2, af_step=0.02):
    self.max_af = max_af
    self.init_af = init_af
    self.af = init_af
    self.af_step = af_step
    self.extreme_point = None
    self.high_price_trend = []
    self.low_price_trend = []
    self.high_price_window = deque(maxlen=2)
    self.low_price_window = deque(maxlen=2)

    # Lists to track results
    self.psar_list = []
    self.af_list = []
    self.ep_list = []
    self.high_list = []
    self.low_list = []
    self.trend_list = []
    self._num_days = 0

  def calcPSAR(self, high, low):
    if self._num_days >= 3:
      psar = self._calcPSAR()
    else:
      psar = self._initPSARVals(high, low)

    psar = self._updateCurrentVals(psar, high, low)
    self._num_days += 1

    return psar

  def _initPSARVals(self, high, low):
    if len(self.low_price_window) <= 1:
      self.trend = None
      self.extreme_point = high
      return None

    if self.high_price_window[0] < self.high_price_window[1]:
      self.trend = 1
      psar = min(self.low_price_window)
      self.extreme_point = max(self.high_price_window)
    else: 
      self.trend = 0
      psar = max(self.high_price_window)
      self.extreme_point = min(self.low_price_window)

    return psar

  def _calcPSAR(self):
    prev_psar = self.psar_list[-1]
    if self.trend == 1: # Up
      psar = prev_psar + self.af * (self.extreme_point - prev_psar)
      psar = min(psar, min(self.low_price_window))
    else:
      psar = prev_psar - self.af * (prev_psar - self.extreme_point)
      psar = max(psar, max(self.high_price_window))

    return psar

  def _updateCurrentVals(self, psar, high, low):
    if self.trend == 1:
      self.high_price_trend.append(high)
    elif self.trend == 0:
      self.low_price_trend.append(low)

    psar = self._trendReversal(psar, high, low)

    self.psar_list.append(psar)
    self.af_list.append(self.af)
    self.ep_list.append(self.extreme_point)
    self.high_list.append(high)
    self.low_list.append(low)
    self.high_price_window.append(high)
    self.low_price_window.append(low)
    self.trend_list.append(self.trend)

    return psar

  def _trendReversal(self, psar, high, low):
    # Checks for reversals
    reversal = False
    if self.trend == 1 and psar > low:
      self.trend = 0
      psar = max(self.high_price_trend)
      self.extreme_point = low
      reversal = True
    elif self.trend == 0 and psar < high:
      self.trend = 1
      psar = min(self.low_price_trend)
      self.extreme_point = high
      reversal = True

    if reversal:
      self.af = self.init_af
      self.high_price_trend.clear()
      self.low_price_trend.clear()
    else:
        if high > self.extreme_point and self.trend == 1:
          self.af = min(self.af + self.af_step, self.max_af)
          self.extreme_point = high
        elif low < self.extreme_point and self.trend == 0:
          self.af = min(self.af + self.af_step, self.max_af)
          self.extreme_point = low

    return psar
  
def psar(data, plotpsargraph):

    indic = PSAR()
    data['PSAR'] = data.apply(lambda x: indic.calcPSAR(x['high'], x['low']), axis=1)
    # Add supporting data
    data['EP'] = indic.ep_list
    data['Trend'] = indic.trend_list
    data['AF'] = indic.af_list
    data.head()

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    psar_bull = data.loc[data['Trend']==1]['PSAR']
    psar_bear = data.loc[data['Trend']==0]['PSAR']
    ticker = 'BTC'

    if (len(data['Trend'])>2):
        buy_sigs = data.loc[data['Trend'].diff()==1]['close']
        short_sigs = data.loc[data['Trend'].diff()==-1]['close']
    else:
        buy_sigs = data['close']
        short_sigs = data['close']

    #y = df['close'].fillna(method='ffill')
    data['PSAR'].fillna(value=data['PSAR'].mean(), inplace=True)
    data['EP'].fillna(value=data['EP'].mean(), inplace=True)
    data['Trend'].fillna(value=data['Trend'].mean(), inplace=True)
    data['AF'].fillna(value=data['AF'].mean(), inplace=True)

    print(len(short_sigs))

    signPriceBuy = []
    signPriceSell = []
    buycount = 0
    sellcount = 0
    for i in range(len(data['close'])):
        if (not buy_sigs.empty) and (data['close'][i] == buy_sigs.iloc[buycount]):
            signPriceBuy.append(data['close'][i])
            signPriceSell.append(np.nan)
            if buycount < len(buy_sigs) -1:
                buycount+=1
        elif (not short_sigs.empty) and (data['close'][i] == short_sigs.iloc[sellcount]):
            signPriceSell.append(data['close'][i])
            signPriceBuy.append(np.nan)
            if sellcount < len(short_sigs)-1:
                sellcount+=1
        else:
            signPriceSell.append(np.nan)
            signPriceBuy.append(np.nan)
    
    if plotpsargraph == True:
       
        plt.figure(figsize=(12, 8))
        plt.plot(data['close'], label='close', linewidth=1, zorder=0)
        plt.scatter(buy_sigs.index, buy_sigs, color=colors[2], 
                    label='Buy', marker='^', s=100)
        plt.scatter(short_sigs.index, short_sigs, color=colors[4], 
                    label='Short', marker='v', s=100)
        plt.scatter(psar_bull.index, psar_bull, color=colors[1], label='Up Trend')
        plt.scatter(psar_bear.index, psar_bear, color=colors[3], label='Down Trend')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.title(f'{ticker} Price and Parabolic SAR')
        plt.legend()
        plt.show()

    return signPriceBuy, signPriceSell