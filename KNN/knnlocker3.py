# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:24:31 2020

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
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import string
stemming=PorterStemmer()
nltk.download('stopwords')
nltk.download('punkt')


def remove_puncs(text):
    no_punc=" ".join([i for i in text if not i in string.punctuation])
    return no_punc

def remove_stopwords(text):
    words=[i for i in text if i not in stopwords.words('english')]
    return words

def extract_ngrams(data, num):
    n_grams = TextBlob(data).ngrams(num)
    grams1=" ".join([grams for grams in n_grams])
    return grams1

stop=stopwords.words('english')
data=pd.read_csv("C:/Users/Dell/Desktop/final_clean5MayLocker.csv")
data.features=data["text"]
df=pd.DataFrame(data.features)
#data.features=data.features.apply(lambda x:remove_puncs(x))
data.features=sent_tokenize(str(data["text"]))
data.features=word_tokenize(str(data["text"]))
#data.features=[word for word in data.features if word.isalpha()]

#data.features=nltk.word_tokenize(data.features)
data.features=df["text"].apply(lambda x:remove_stopwords(x))
#data.features=data["text"].apply(lambda x:extract_ngrams(x,2))

data.features=pd.get_dummies(df["text"])
#df['text']=pd.to_numeric(df["text"],errors="coerce")
data.target=data["Label 2"]
#print(dtypes)
#data.features = SimpleImputer(missing_values=np.nan, strategy='mean')


data.features=preprocessing.MinMaxScaler().fit_transform(data.features)


feature_train,feature_test,target_train,target_test=train_test_split(data.features,data.target,test_size=0.25)
model=KNeighborsClassifier(n_neighbors=42)
fittedModel=model.fit(feature_train,target_train)
predictions=fittedModel.predict(feature_test)
predTrain=fittedModel.predict(feature_train)
print("Test:-",accuracy_score(target_test,predictions))

print(classification_report(target_test, predictions))