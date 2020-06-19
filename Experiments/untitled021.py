# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:40:21 2020

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
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
import numpy as np
import string
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
def remove_puncs(text):
    no_punc=" ".join([i for i in text if not i in string.punctuation])
    return no_punc
"""
def remove_stopwords(text):
    words=[i for i in text if i not in stopwords.words('english')]
    return words
"""

stop=stopwords.words('english')
data=pd.read_csv("C:/Users/Dell/Desktop/final_clean30April(delivery)Part2.csv")
data.features=data[["text"]]
df=pd.DataFrame(data.features)
data.features=data["text"].apply(lambda x:remove_puncs(x))
data.features=sent_tokenize(str(data.features))
data.features=word_tokenize(str(data.features))
data.features=pd.get_dummies(df["text"])
#df['text']=pd.to_numeric(df["text"],errors="coerce")
data.target=data.Label
#print(dtypes)
#data.features = SimpleImputer(missing_values=np.nan, strategy='mean')


data.features=preprocessing.MinMaxScaler().fit_transform(data.features)


#feature_train,feature_test,target_train,target_test=train_test_split(data.features,data.target,test_size=0.25)
cross_validate_scores=[]

for k in range(1,100):
     knn=KNeighborsClassifier(n_neighbors=k)
     scores=cross_val_score(knn,data.features,data.target,cv=10,scoring="accuracy")
     cross_validate_scores.append(scores.mean())
print("optimal k with cross validation:-",np.argmax(cross_validate_scores))  