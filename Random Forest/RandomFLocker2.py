# -*- coding: utf-8 -*-
"""
Created on Thu May 14 11:13:01 2020

@author: Dell
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import math

data=pd.read_csv("C:/Users/Dell/Desktop/final_clean5MayLocker.csv")
data.features=["text"]
data.target=data["Label 2"]
data.features=pd.get_dummies(data["text"])


features_train,features_test,target_train,target_test=train_test_split(data.features,data.target,test_size=0.25)

rf = RandomForestRegressor(n_estimators = 10)
rf.fit(features_train,target_train)
predictions=rf.predict(features_test)
#errors=abs(predictions-target_test)
#print("Mean absolute Error:-",round(np.mean(errors),2),'degrees.')
#mape = 100 * (errors / target_test)
# Calculate and display accuracy
print("Accuracy:-",metrics.accuracy_score(target_test,predictions))