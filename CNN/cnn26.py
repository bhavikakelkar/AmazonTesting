# -*- coding: utf-8 -*-
"""
Created on Sat May  9 02:38:17 2020

@author: Dell
"""


from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
#from keras.layers.normalisation import BatchNormalization
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
from matplotlib import pyplot as plt
"""


vocab_size=1000
maxlen=1000
batch_size=32
embedding_dims=10
filters=16
kernel_size=8
hidden_dims=250
epochs=10
"""

#np.random.seed(1)
data=pd.read_csv("C:/Users/Dell/Desktop/final_clean26April_remove(locker)N.csv")
maxlen=100
X=data["text"]
#X=X.apply(lambda x:extract_ngrams(x,2))
X=pd.get_dummies(X)
y=data["Label"]
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.20)


X_train=sequence.pad_sequences(X_train,maxlen=400)
X_test=sequence.pad_sequences(X_test,maxlen=400)

model=Sequential()
model.add(Embedding(500,10,input_length=400))

model.add(Dropout(0.5))
model.add(Conv1D(16,3,padding='valid',activation='relu'))
model.add(MaxPooling1D())
model.add(Conv1D(16,3,padding='valid',activation='relu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(250,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1,activation="sigmoid"))
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
history=model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=16,epochs=15,verbose=2)
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

#plot_model(model,show_shapes=True,show_layer_names=True)


#X_train=pd.get_dummies(X_train)
#X_test=pd.get_dummies(X_test)

"""
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
text=data["text"].copy()
data.drop(["text"],axis=1,inplace=True)
#x=pd.get_dummies(data.values)
#y=pd.get_dummies(text.values)
#X_train=sequence.pad_sequences(X_train,maxlen=200)
#X_test = sequence.pad_sequences(X_test,maxlen=200)

model=Sequential()
#model.add(Embedding(5000, 32, input_l))
model.add(Conv1D(32,8,input_shape=(9984,1)))
#input_shape=(input_dim,1)
model.add(Activation('relu'))
model.add(MaxPooling1D(2))

model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(250,activation='relu'))
#model.add(Flatten())
#
model.add(Dense(1,activation="sigmoid"))
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
#model.summary()
#model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=2,batch_size=128,verbose=2)

#model.add(Flatten())
model.fit(X_train, y_train,epochs=10,verbose=2)
score=model.evaluate(X_test,y_test,verbose=0)
print("accuracy:%.2f%%" %(score[1]*100))
"""