# https://www.kaggle.com/code/mgabrielkerr/visualizing-knn-svm-and-xgboost-on-iris-dataset


import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot
from matplotlib.pylab import rcParams
from pandas import Series, datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
import random
from xgboost import XGBClassifier
import seaborn as sns
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas_ta as ta
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from datetime import date
#separate line

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import requests
from math import floor
from termcolor import colored as cl
from binance.client import Client
import datetime as dt
import json
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from itertools import product
import math
from numpy import nan
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, neighbors
from mlxtend.plotting import plot_decision_regions
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)


def knn_comparison(data, k):
    x = data[['Top1','Top2']].values
    y = data['Label'].astype(int).values
    clf = neighbors.KNeighborsClassifier(n_neighbors=k)
    clf.fit(x, y)
    # Plotting decision region
    plot_decision_regions(x, y, clf=clf, legend=2)
    # Adding axes annotations
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Knn with K='+ str(k))
    plt.show()






#NLP--------------------------------------------------------------------------------NLP


# read data
data = pd.read_csv("Datasets/Combined_News_DJIA.csv")

# concatenate all news into one
data["combined_news"] = data.filter(regex=("Top.*")).apply(lambda x: ''.join(str(x.values)), axis=1)

# convert to feature vector
feature_extraction = TfidfVectorizer()
X = feature_extraction.fit_transform(data["combined_news"].values)

# split into training- and test set
num_training = data[data['Date'] < '20150101'].shape[0]
X_train = X[:num_training]
X_test = X[num_training:]
y_train = data["Label"].values[:num_training]
y_test = data["Label"].values[num_training:]

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler(with_mean=False)
# sc.fit(X_train)
# X_train_std = sc.transform(X_train)
# X_test_std = sc.transform(X_test)

# train classifier
clf = SVC(probability=True, kernel='rbf')
clf = KNeighborsClassifier(n_neighbors=1, p=2)
# clf = DecisionTreeClassifier(criterion='gini', splitter='best',max_depth=3,min_samples_split=2,min_samples_leaf=6,min_weight_fraction_leaf=0.0,
#                                     max_features=None,random_state=42,max_leaf_nodes=None,min_impurity_decrease=0.0)
clf.fit(X_train, y_train)

# predict and evaluate predictions
predictions = clf.predict_proba(X_test)
print('ROC-AUC score is ' + str(roc_auc_score(y_test, predictions[:,1])))

predicted = clf.predict(X_test)

from sklearn.metrics import accuracy_score

print('Accuracy using Support Vector Classifier with TF-IDF ' + str(accuracy_score(y_test, predicted)))

print(y_test)
print(predicted)

for i in [1, 5,20,30,40,60]:
    knn_comparison(data, i)