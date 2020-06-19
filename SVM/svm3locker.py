# -*- coding: utf-8 -*-
"""
Created on Thu May 14 00:59:29 2020

@author: Dell
"""


import pandas as pd
import numpy as np
from sklearn import svm,metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing
import nltk
nltk.download('stopwords')
data=pd.read_csv("C:/Users/Dell/Desktop/final_clean5MayLocker.csv")
data.features=["text"]
df=pd.DataFrame(data.features)
data.features=pd.get_dummies(data["text"])
data.target=data["Label 2"]
data.features=preprocessing.MinMaxScaler().fit_transform(data.features)
feature_train,feature_test,target_train,target_test=train_test_split(data.features,data.target,test_size=0.25)
clf=svm.SVC(kernel="linear")
clf.fit(feature_train,target_train)
predictions=clf.predict(feature_test)
print("Accuracy:-",metrics.accuracy_score(target_test,predictions))
print("Precision:-",metrics.precision_score(target_test,predictions))
print("Recall:-",metrics.recall_score(target_test,predictions))