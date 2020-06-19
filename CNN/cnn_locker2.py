# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:53:10 2020

@author: Dell



import io
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import numpy 
import string
import csv
stemming=PorterStemmer()
nltk.download('stopwords')
nltk.download('punkt')
import pandas as pd

def remove_punction_and_stopwords(msg):
   stop_words = set(stopwords.words('english'))
   word_tokens = word_tokenize(msg)
   filtered_words = [w for w in msg if w not in word_tokens and w not in string.punctuation]
   new_sentence = ''.join(filtered_words)
   return new_sentence

data=pd.read_csv("C:/Users/Dell/Desktop/final_clean26April_remove(locker)N.csv")
data.features=data["text"]
df=pd.DataFrame(data.features)

data.features=sent_tokenize(str(data.features))
data.features=word_tokenize(str(data.features))
data.features=df["text"].apply(lambda x:remove_punction_and_stopwords(x))
data["text"]=data.features
     
data.to_csv("lockscleanfile1.csv",encoding='utf-8')
print(data["text"])  



import io
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
file1 = open("C:/Users/Dell/Desktop/final_clean26April_remove(locker)N.csv")
line = file1.read()
words = line.split()
for r in words:
    if not r in stop_words:
        appendFile = open('test1.csv','a')
        appendFile.write(" "+r)
        
        """
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv1D,MaxPooling1D
from keras.preprocessing import sequence,text
from sklearn.model_selection import train_test_split
from keras.layers import Embedding
from sklearn import preprocessing
from keras.utils import plot_model
from textblob import TextBlob
import numpy as np
import pandas as pd
data=pd.read_csv("C:/Users/Dell/Desktop/final_clean5MayLocker.csv")
maxlen=100
X=data["text"]
#X=X.apply(lambda x:extract_ngrams(x,2))
X=pd.get_dummies(X)
y=data["Label 2"]
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.25)


X_train=sequence.pad_sequences(X_train,maxlen=800)
X_test=sequence.pad_sequences(X_test,maxlen=800)

model=Sequential()
#model.add(BatchNormalization())
model.add(Embedding(500,10,input_length=800))
model.add(Dropout(0.5))
model.add(Conv1D(32,8,padding='valid',activation='relu'))

model.add(MaxPooling1D())
model.add(Conv1D(32,8,padding='valid',activation='relu'))

model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(250,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1,activation="sigmoid"))

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])

history=model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=128,epochs=15,verbose=2)
testacc=model.evaluate(X_test,y_test,verbose=0)
print("accuracy:%.1f%%" %(testacc[1]*100))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


