# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 13:44:49 2020

@author: gnanam.natarajan
"""


# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('city_day.csv')
dataset.dropna(inplace = True)
dataset.drop("Date" , axis = 1, inplace = True)
X = dataset.iloc[:, :-2].values
y = dataset['AQI'].values

# Encoding categorical data
# # Encoding categorical data
# # Encoding the Independent Variable
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
from sklearn.compose import ColumnTransformer
columnTransformer = ColumnTransformer([('encoder',
OneHotEncoder(),
[0])],
remainder='passthrough')

onehotencoder = OneHotEncoder()
X = np.array(columnTransformer.fit_transform(X), dtype = np.int)
labelencoder_y = LabelEncoder()
#X=X.reshape(-1,1)
#y=y.reshape(-1,1)
# Avoiding the Dummy Variable Trap
#X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Find Score
#accuracy = regressor.score(y_train,y_pred)
#accuracy = regressor.score([y_test]],y_pred)

from sklearn.metrics import mean_squared_error, r2_score
# model evaluation
#rmse = mean_squared_error(y_train, y_pred)
r2 = r2_score(X_train, y_pred)

# printing values
print('Slope:' ,regressor.coef_)
print('Intercept:', regressor.intercept_)
#print('Root mean squared error: ', rmse)
print('R2 score: ', r2)

#from sklearn.metrics import accuracy_score
#accuracy = accuracy_score(y_test, y_pred)

# from sklearn.model_selection import cross_val_score
# accuracy=cross_val_score(regressor,X_test,y_pred,cv=2)

#from sklearn.metrics import r2_score, mean_squared_error ,confusion_matrix,classification_report,matthews_corrcoef,accuracy_score
#accuracy = mean_squared_error(y_test,y_pred )