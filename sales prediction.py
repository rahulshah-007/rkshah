# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 12:28:26 2020

@author: rksha
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")


df.drop(df.columns[[6,72,73,74]],axis = 1,inplace = True)

df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())

for col in ('MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinType2'
            ,'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond'):
    df[col] = df[col].fillna('None')

df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    
df['Electrical'] = df['Electrical'].fillna('SBrkr')

df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)



categorical = []
for col in df.columns:
    if df[col].dtype == 'O':
       categorical.append(col)
       
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

label_encoder = {}

for col in categorical:
    label_encoder[col] = LabelEncoder()
    df[col] = label_encoder[col].fit_transform(df[col])
    
X = df.iloc[:,1:76].values
y = df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
 
from sklearn.ensemble import GradientBoostingRegressor
regr1 = GradientBoostingRegressor(random_state = 0)
regr1.fit(x_train,y_train)
y_pred1 = regr1.predict(x_test)
plt.scatter(y_pred1,y_test,color = 'blue')
from sklearn.metrics import r2_score
acc1 = r2_score(y_test,y_pred1)

df_test = pd.read_csv("test.csv")

df_test.drop(df_test.columns[[6,72,73,74]],axis = 1,inplace = True)

df_test['LotFrontage'] = df_test['LotFrontage'].fillna(df_test['LotFrontage'].mean())

for col in ('MasVnrType','MSZoning','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
            'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond'
            ,'Exterior1st','Exterior2nd','KitchenQual','Functional','SaleType'):
    df_test[col] = df_test[col].fillna('None')

for col in ('MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtHalfBath',
            'BsmtFullBath','GarageCars','GarageArea'):
    df_test[col] = df_test[col].fillna(0)
    
df_test['Electrical'] = df_test['Electrical'].fillna('SBrkr')

df_test['Utilities'] = df_test['Utilities'].fillna('AllPub')

df_test['GarageYrBlt'] = df_test['GarageYrBlt'].fillna(0)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

label_encoder = {}

for col in categorical:
    label_encoder[col] = LabelEncoder()
    df_test[col] = label_encoder[col].fit_transform(df_test[col])
    
X_test = df_test.iloc[:,1:77].values

X_test = sc.transform(X_test)

y_pred2 = regr1.predict(X_test)

df_test['SalePrice'] = y_pred2

output = pd.DataFrame({'Id': df_test.Id,'SalePrice':y_pred2})
output.to_csv('submission.csv', index=False)




