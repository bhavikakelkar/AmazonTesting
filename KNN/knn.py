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
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
import numpy 
import string
stemming=PorterStemmer()
nltk.download('stopwords')
nltk.download('punkt')


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
#data.features=[word for word in data.features if word.isalpha()]

#data.features=nltk.word_tokenize(data.features)

#data.features=data.features.apply(lambda x: ' '.join([word for word in x if word not in stopwords.words()]))
#df['text']=pd.to_numeric(df["text"],errors="coerce")
#data.features=data["text"].apply(lambda x:remove_stopwords(x))
data.target=data.Label
#print(dtypes)
#data.features = SimpleImputer(missing_values=np.nan, strategy='mean')
print(data.features)
data.features=pd.get_dummies(data["text"])


data.features=preprocessing.MinMaxScaler().fit_transform(data.features)


feature_train,feature_test,target_train,target_test=train_test_split(data.features,data.target,test_size=0.25)

model=KNeighborsClassifier(n_neighbors=52)

fittedModel=model.fit(feature_train,target_train)
predictions=fittedModel.predict(feature_test)
predTrain=fittedModel.predict(feature_train)
print("Test:-",accuracy_score(target_test,predictions))
print("Training:-",accuracy_score(target_train,predTrain))
print(feature_train)