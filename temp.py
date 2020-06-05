# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("Churn_Modelling.csv")
X =  df.iloc[:,3:-1].values
y =  df .iloc[:,-1].values
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le1 = LabelEncoder()
X[:,2] = le1.fit_transform(X[:,2])
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [("encoder",OneHotEncoder(),[1])],remainder = 'passthrough')
X = np.array(ct.fit_transform(X))
X = X[:,1:12]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
clf = Sequential()
clf.add(Dense(output_dim = 6, init = 'uniform',activation = 'relu',input_dim = 11))
clf.add(Dropout(p = 0.1))
clf.add(Dense(output_dim = 6,init = 'uniform',activation = "relu"))
clf.add(Dropout(p = 0.1))
clf.add(Dense(output_dim = 1, init = 'uniform',activation = "sigmoid"))
clf.compile(optimizer = "adam",loss = 'binary_crossentropy',metrics = ["accuracy"])
clf.fit(X_train,y_train,batch_size = 10,epochs = 100)
y_pred = clf.predict(X_test)
y_pred = (y_pred>0.5)
new_predict = clf.predict(sc.transform([[0,0,600,1,40,3,60000,2,1,1,50000]]))
new_predict = (new_predict>0.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_clf():
    clf = Sequential()
    clf.add(Dense(output_dim = 6, init = 'uniform',activation = 'relu',input_dim = 11))
    clf.add(Dense(output_dim = 6,init = 'uniform',activation = "relu"))
    clf.add(Dense(output_dim = 1, init = 'uniform',activation = "sigmoid"))
    clf.compile(optimizer = "adam",loss = 'binary_crossentropy',metrics = ["accuracy"])
    return clf
clf = KerasClassifier(build_fn = build_clf,batch_size = 10,epochs = 100)
accuracy = cross_val_score( estimator = clf, X = X_train, y = y_train,cv = 10,n_jobs = -1)
mean = accuracy.mean()

variance = accuracy.std()

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def build_clf(optimizer):
    clf = Sequential()
    clf.add(Dense(output_dim = 6, init = 'uniform',activation = 'relu',input_dim = 11))
    clf.add(Dense(output_dim = 6,init = 'uniform',activation = "relu"))
    clf.add(Dense(output_dim = 1, init = 'uniform',activation = "sigmoid"))
    clf.compile(optimizer = optimizer,loss = 'binary_crossentropy',metrics = ["accuracy"])
    return clf
clf = KerasClassifier(build_fn = build_clf)
parameters = {'batch_size':[25,32],'epochs':[100,500],'optimizer':['adam','rmsprop']}
grid = GridSearchCV(estimator = clf, param_grid = parameters,scoring = "accuracy",cv = 10)
grid = grid.fit(X_train,y_train)
best_para = grid.best_params_
best_accuracy = grid.best_score_ 