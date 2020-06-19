# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 20:31:05 2020

@author: Dell
"""
import nltk
import csv
from textblob import TextBlob
pos=0
neg=0


def sentiment2(review,pos,neg):
    blob=TextBlob(review)
    if blob.sentiment.polarity>0:
        print("Positive:-")
        print(review)
        pos=pos+1
    elif blob.sentiment.polarity<0:
        print("Negative:-")
        print(review)
        neg=neg+1
    return pos,neg
   # return sentimentpercent(pos,neg)

def sentimentpercent(pos,neg):
    global result
    total=pos+neg
    if pos>neg:
        sent="Positive"
        result=pos/total*100
        print("\n\nPOSITIVE PERCENTAGE IS:-",result)
    else:
        sent="Negative"
        result=neg/total*100
        print("\n\nNEGATIVE PERCENTAGE IS:-",result)
#    return{sent:result}


with open('delivery_clean2.csv','rt',encoding='utf8') as reviews:
    reader=csv.reader(reviews)
      
    for row in reader:
        
        row=str([cell.encode('utf-8') for cell in row])
        #print(row)
        ans1,ans2=sentiment2(row,pos,neg)
        pos=ans1
        neg=ans2
print(pos,neg)
sentimentpercent(pos,neg)
        

"""

    

            



url ='amazondelivery.txt'
file=open(url,encoding="utf8")



for reviews in file:
    value=reviews.split()
    #print(value)
    sentiment(value)
    


#t=file.read()
#print(type(t))

#bobo = TextBlob(t)
#print(format(bobo.sentiment))

"""