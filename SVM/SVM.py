# -*- coding: utf-8 -*-
"""
Created on Sun May  3 15:01:28 2020

@author: Dell
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 22:08:42 2020

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
data=pd.read_csv("C:/Users/Dell/Desktop/final_clean30April(delivery)Part2.csv")
data.features=["text","Neutral"]
df=pd.DataFrame(data.features)
data.features=pd.get_dummies(data["text"])
data.target=data.Label
data.features=preprocessing.MinMaxScaler().fit_transform(data.features)
feature_train,feature_test,target_train,target_test=train_test_split(data.features,data.target,test_size=0.25)
clf=svm.SVC(kernel="linear")
clf.fit(feature_train,target_train)
predictions=clf.predict(feature_test)
print("Accuracy:-",metrics.accuracy_score(target_test,predictions))
print("Precision:-",metrics.precision_score(target_test,predictions))
print("Recall:-",metrics.recall_score(target_test,predictions))
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
from sklearn.model_selection import cross_val_score
from sklearn.impute import MissingIndicator
from sklearn.impute import SimpleImputer
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
"""
"""
le=preprocessing.LabelEncoder()
le.fit(["text"])


data.features=preprocessing.MinMaxScaler().fit_transform(data.features)


feature_train,feature_test,target_train,target_test=train_test_split(data.features,data.target,test_size=0.2)

model=KNeighborsClassifier(n_neighbors=52)
fittedModel=model.fit(feature_train,target_train)
predictions=fittedModel.predict(feature_test)
print(accuracy_score(target_test,predictions))
"""
"""
cross_validate_scores=[]

for k in range(1,100):
     knn=KNeighborsClassifier(n_neighbors=k)
     scores=cross_val_score(knn,data.features,data.target,cv=10,scoring="accuracy")
     cross_validate_scores.append(scores.mean())
print("optimal k with cross validation:-",np.argmax(cross_validate_scores))     
#print(confusion_matrix(target_test,predictions))
print(accuracy_score(target_test,predictions))
"""
"""
diabetes=datasets.load_diabetes()
#'data', 'target', 'DESCR', 'feature_names', 'data_filename', 'target_filename
#print(diabetes.keys())
#print(diabetes.data)
diabetes_X=diabetes.data
diabetes_X=preprocessing.MinMaxScaler().fit_transform(diabetes_X)
#print(diabetes_X)
train_test_split(diabetes_X,)

diabetes_X_train=diabetes_X[:-30] #taking last 30
diabetes_X_test=diabetes_X[:-30] #first 30
#print(diabetes_X_train)
diabetes_Y_train=diabetes.target[:-30]#same as diaxtrain
diabetes_Y_test=diabetes.target[:-30]
#x axis:- feature
#y axis:-label

model=KNeighborsClassifier(n_neighbors=2)
fitted_model=model.fit(diabetes_X_test,diabetes_Y_train)
diabetes_y_predict=fitted_model.predict(diabetes_X_test)
print("Mean squared error is:-",mean_squared_error(diabetes_Y_test,diabetes_y_predict))
#print("weights:-",model.coef_)
#print("intercept",model.intercept_)

#plt.scatter(diabetes_X[sex ],diabetes_X[age])
#plt.show()
print(confusion_matrix(diabetes_Y_test,diabetes_y_predict))
#plt.plot(diabetes_X_test,diabetes_y_predict,'ro')
print("accuracy score is:-",metrics.accuracy_score(diabetes_Y_test,diabetes_y_predict))
"""

"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import sklearn.metrics as metrics
import string
from sklearn.model_selection import cross_val_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

data=pd.read_csv("C:/Users/Dell/Desktop/spyder/MLCodes/final_clean30April(delivery)Part2.csv")
#print(data.keys())

data_features=data[["text"]]

df=pd.DataFrame(data_features)
#print(df)
#print(df.dtypes)
data.target=data.Label
#df['text']=pd.to_numeric(df["text"],errors="coerce")
#df['Neutral']=df['Neutral'].astype(float)
#df['text']=df['text'].astype(float)



#print(df.dtypes)
#print(df['text'])
data_features=preprocessing.MinMaxScaler().fit_transform(data_features)
feature_train,feature_test,target_train,target_test=train_test_split(data_features,data.target,test_size=0.25)
#feature_train.fillna(feature_train.mean(),inplace=True)
#target_train.fillna(target_train.mean(),inplace=True)
#np.nan_to_num(feature_train)
#np.nan_to_num(target_train)

model=KNeighborsClassifier(n_neighbors=32)
fittedModel=model.fit(feature_train,target_train)
predictions=fittedModel.predict(feature_test)
print(confusion_matrix(target_test,predictions))
print(accuracy_score(target_test,predictions))
"""
"""
cross_validate_scores=[]

for k in range(1,100):
     knn=KNeighborsClassifier(n_neighbors=k)
     scores=cross_val_score(knn,data.features,data.target,cv=10,scoring="accuracy")
     cross_validate_scores.append(scores.mean())
print("optimal k with cross validation:-",np.argmax(cross_validate_scores))     
print(confusion_matrix(target_test,predictions))
print(accuracy_score(target_test,predictions))
"""


"""
print(df.dtypes)
#data.replace([np.inf, -np.inf], np.nan, inplace=True)
feature_train,feature_test,target_train,target_test=train_test_split(data_features,data.target,test_size=0.25)

countvec=CountVectorizer()
xtraincounts=countvec.fit_transform(feature_train)

tfidfTransformer=TfidfTransformer()
xtrainTfidf=tfidfTransformer.fit_transform(xtraincounts)


model=MultinomialNB().fit(xtrainTfidf,data.target)
fittedModel=model.fit(feature_train,target_train)
predictions=fittedModel.predict(feature_test)
print(accuracy_score(target_test,predictions))



"""