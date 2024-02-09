import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os,tqdm,random,re,sys

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

#Load train file
df_train = pd.DataFrame()
df = pd.read_csv('winter_train.csv')
df_train = df[['time', 'at','ah','gt','gh']].copy()
df_train = df_train.dropna(axis=0)

#Load test file
df_test = pd.DataFrame()
df = pd.read_csv('winter_test.csv')
df_test = df[['time', 'at','ah','gh']].copy()
df_test = df_test.dropna(axis=0)

print('df_train')
print(df_train)

#Split train data,validation
model_train_y = df_train.pop('gt')
model_train_X = df_train
model_test = df_test

print('model_train_y')
print(model_train_y)

print('model_train_X')
print(model_train_X)


train_X,test_X,train_y,test_y = train_test_split(model_train_X,model_train_y,test_size = 0.2 ,random_state = 3)

print('train_X')
print(train_X)

print('test_X')
print(test_X)

print('train_y')
print(train_y)

print('test_y')
print(test_y)

#ML4 randomforest depth=10
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=10, random_state=0,n_estimators=100)
regr.fit(train_X,train_y)

predict_validate = regr.predict(test_X)
print('ML4', np.mean((predict_validate-test_y)**2))

predict_test = regr.predict(model_test)
print('Actual Ground Temperature = 29.11 Machine Learning predict = ',predict_test)
#df_ans4 = pd.DataFrame()
#df_ans4['gt'] = predict_test
#df_ans4.to_csv('summer_answer.csv',index = None)

