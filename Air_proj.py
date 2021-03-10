# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 20:48:28 2020

@author: gnanam.natarajan
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings; warnings.simplefilter('ignore')
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_squared_error ,confusion_matrix,classification_report,matthews_corrcoef,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split    
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pickle
#Load DataSet
dataset=pd.read_csv('city_day.csv',encoding="ISO-8859-1")
print(dataset.describe())

print(dataset.head(10))
df=dataset.dropna()
print(df.head(10))

print(df.shape)

#print(df.columns)

df.describe()
print(df.info())

#checking the duplicate data in dataset
df.duplicated()
print(sum(df.duplicated()))
print(df.isnull().sum())

#print("Minimum value of average pollution is : ", df.AQI.min())
#print("Maximum value of average pollution is : ", df.AQI.max())

#df.AQI_Bucket.unique()

"""pd.crosstab(df.City,df.AQI)

fig,ax=plt.subplots(figsize=(16,10))
ax=sns.barplot(x='PM2.5',y='AQI',data=df)
ax.set(ylabel="pollutants details", title='Pollution Values')
plt.show()

#import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(16,10))
ax=sns.barplot(x='AQI',y='AQI_Bucket',data=df)
ax.set(ylabel="pollutants details", title='Pollution Values')
plt.show()"""

"""var_mod=['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2',
       'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI',]
le=LabelEncoder()
for i in var_mod: 
    df[i]=le.fit_transform(df[i]).astype(str)

df.head()"""

#df['class']=df.AQI_Bucket.map({'Moderate':3 ,'Good':1,'Poor':0 ,'Satisfactory':4, 'Severe':5 ,'Very Poor':0}) 
#X=df.drop(labels='class',axis=1)
#X=df.drop(['Date' ,'Xylene','Toluene','Benzene','PM2.5','City','PM10','NO','NO2','NOx','NH3','CO','SO2','O3'],axis=1)

#df['AOI_PD']=df.AQI_Bucket.map({'Moderate':3 ,'Good':1,'Poor':4 ,'Satisfactory':2, 'Severe':5 ,'Very Poor':4})
#X=df.drop(['Date' ,'Xylene','Toluene','Benzene','PM2.5','City','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','AQI_Bucket','AOI_PD'],axis=1)
#print("X:",X)
#y=df.loc[:,'AOI_PD']
#print("y::",y)
df=dataset.dropna()

df = df.dropna(subset=['City'])
#df['City'] = df['City'].astype(int)
dataset.drop("City" , axis = 1, inplace = True)
dataset.drop("Date" , axis = 1, inplace = True)
X = dataset.iloc[:, :-2].values
y = dataset['AQI'].values

from sklearn.preprocessing import OneHotEncoder , LabelEncoder
from sklearn.compose import ColumnTransformer
"""columnTransformer = ColumnTransformer([('encoder',OneHotEncoder(),[0])],remainder='passthrough')

onehotencoder = OneHotEncoder()
X = np.array(columnTransformer.fit_transform(X), dtype = np.int)
labelencoder_y = LabelEncoder()"""



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)
print("Num of training dataset :",len(X_train))
print("Num of test dataset :",len(X_test))
print("Total num of dataset :",len(X_train)+len(X_test)) 

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

  

#X=data.drop(labels='class',axis=1)
#Response variable
#y=df.loc[:,'class']

#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1 ,stratify=y)
    
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import LogisticRegression

log_reg=LogisticRegression()

log_reg.fit(X_train,y_train)
predictor=log_reg.predict(X_test)
# Saving model to disk
pickle.dump(log_reg, open('model.pkl','wb'))
print("Start")

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[9]]))

print('Classification report of Logistic Regression Results:')

print(classification_report(y_test,predictor))

accuracy=cross_val_score(log_reg,X,y,cv=50)
print('Cross validation test results of accuracy :')
print(accuracy)

print("Accuracy result of Logistic Regression is :",accuracy.mean()*100)

confusion_mex=confusion_matrix(y_test,predictor)
print("confusion_matrix result of Logistic Regression ", confusion_mex)
sensitivity=confusion_mex[0,0]/(confusion_mex[0,0]+confusion_mex[0,1])
print("sensitivity:",sensitivity)

spec=confusion_mex[1,1]/(confusion_mex[0,1]+confusion_mex[1,1])
print("Specificity:",spec)
#from sklearn.preprocessing import StandardScaler



#-- RandomForest Regression---#
"""regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print("Start")
print('Classification report of Rnadom  Results:')

print(classification_report(y_test,predictor))

accuracy=cross_val_score(regressor,X,y,cv=50)
print('Cross validation test results of accuracy :')
print(accuracy)

print("Accuracy result of Rnadom is :",accuracy.mean()*100)

confusion_mex=confusion_matrix(y_test,predictor)
print("confusion_matrix result of Rnadom", confusion_mex)
sensitivity=confusion_mex[0,0]/(confusion_mex[0,0]+confusion_mex[0,1])
print("sensitivity:",sensitivity)

spec=confusion_mex[1,1]/(confusion_mex[0,1]+confusion_mex[1,1])
print("Specificity:",spec)"""