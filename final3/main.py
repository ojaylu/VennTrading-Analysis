import pandas as pd
import pandas_ta as ta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import requests
import matplotlib.pyplot as plt
from math import floor
from termcolor import colored as cl
from binance.client import Client
import datetime as dt
import json
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import *
import graphviz
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from itertools import product
import math
from numpy import nan
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import graphviz
import shap
from lime.lime_tabular import LimeTabularExplainer
from bfs import bfs
from flagrules import flagrules
from datacollection import datacollection
from LSTMpred import *
from obv import *
from rsi import rsi_buy_sell
from arimatesting import *
from macd import get_macd
from stochastic import get_stoch_osc
from calc import calc
from implement_stoch_macd_strategy_optimization import *
from ema import *
from bruteforce import *
from decisiontree import decisiontree

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)

def optimize_second():
    finImgList = {}
    data = datacollection()

    # MACD CALCULATION
    # data['macd'] = get_macd(data['close'], 26, 12, 9)[0]
    # data['macd_signal'] = get_macd(data['close'], 26, 12, 9)[1]
    # data['macd_hist'] = get_macd(data['close'], 26, 12, 9)[2]
    # data = data.dropna()

    data = calc(data)

    print("data before bruteforce", data.columns)
    data, combineflag, macdlb, macdub, rsilb, rsiub, newsurplus = bruteforce(data,plotgraph=False,plotmacdgraph=False,
                                                                             plotobvgraph=False, plotrsigraph=False, plotemagraph=False)
    print('rsi should be ',rsilb, 'to', rsiub)

    print("data before AI", data.columns[data.isna().any()])
    data, clf, DTimgList = decisiontree(data)
    finImgList.update(DTimgList)

    xx, LSTMImgList = mylstm(data, clf, plotbfsgraph=False)
    finImgList.update(LSTMImgList)

    yy, ARIMAImgList = myarima(data, clf, 20, 'predict all in once', plotderivationgraph=False, plotbfsgraph=False)
    finImgList.update(ARIMAImgList)

    return {"optimized": {"simple_MACD": { "lower": macdlb , "upper": macdub}, 
                          "RSI": { "lower": rsilb, "upper": rsiub, "period": 14}},
            "images": finImgList}
