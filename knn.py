# -*- coding: utf-8 -*-
"""
Created on Tue May  5 16:39:20 2020

@author: Dell
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn import datasets
from matplotlib import pyplot as plt
import sklearn.metrics as metrics
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
data=pd.read_csv("C:/Users/Dell/Desktop/final_clean30April(delivery)Part2.csv")
#Index(['clientid', 'income', 'age', 'loan', 'LTI', 'default'], dtype='object')
#print(data.keys())
data.features=data[["text"]]
df=pd.DataFrame(data.features)
data.features=pd.get_dummies(data["text"])
#df['text']=pd.to_numeric(df["text"],errors="coerce")
data.target=data.Label
#print(dtypes)
#data.features = SimpleImputer(missing_values=np.nan, strategy='mean')

data.features=preprocessing.MinMaxScaler().fit_transform(data.features)


feature_train,feature_test,target_train,target_test=train_test_split(data.features,data.target,test_size=0.2)

model=KNeighborsClassifier(n_neighbors=52)
fittedModel=model.fit(feature_train,target_train)
predictions=fittedModel.predict(feature_test)
print(accuracy_score(target_test,predictions))

