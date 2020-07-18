# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 14:53:08 2020

@author: rksha
"""
#MAR_ACTIVITY
#MAKAUT
#DEEP_LEARNING
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_train = pd.read_csv("Google_Stock_Price_Train.csv")
train_set = df_train.iloc[:,1:2].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
train_set_sc = sc.fit_transform(train_set)

X_train = []
y_train = []
for i in range (60,1258):
    X_train.append(train_set_sc[i-60:i,0])
    y_train.append(train_set_sc[i,0])
X_train,y_train = np.array(X_train),np.array(y_train)    
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()
regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (X_train.shape[1],1)))
regressor.add(Dropout(p=0.2))

regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(p=0.2))


regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(p=0.2))


regressor.add(LSTM(units = 50,return_sequences = False))
regressor.add(Dropout(p=0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

regressor.fit(X_train,y_train,batch_size = 32,epochs = 100)

df_test = pd.read_csv("Google_Stock_Price_Test.csv")
test_set = df_test.iloc[:,1:2].values

df_total = pd.concat((df_train["Open"],df_test["Open"]),axis = 0)
inputs = df_total[len(df_total)-len(df_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range (60,80):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)   
X_test1 = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
y_pred = regressor.predict(X_test1)
y_pred = sc.inverse_transform(y_pred)

plt.plot(test_set,color = 'red',label = 'real_price')
plt.plot(y_pred,color = 'blue',label = 'predicted_price')
plt.title('stocks')
plt.xlabel("time")
plt.ylabel("price")
plt.legend()
plt.show()

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(test_set, y_pred))
