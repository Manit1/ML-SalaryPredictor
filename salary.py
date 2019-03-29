# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 11:51:04 2018

@author: Manit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("Salary_Data.csv")

data.head()

#Collecting X and Y
X=data.iloc[:,:-1].values
Y=data.iloc[:,1].values

#split dataset into two parts
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=1/3)

#import models

from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train,y_train)

y_predict=regressor.predict(X_test)

plt.plot(X_test,y_predict)
plt.scatter(X_test,y_test,marker='o',color='red')
plt.show()