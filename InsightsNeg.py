from twitterscraper import query_tweets
import datetime as dt
import pandas as pd
import sys
import re,string
import nltk 
from textblob import TextBlob
from textblob import Word
import matplotlib.pyplot as plt; 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import csv
def dates(startYear,startMonth,startDate,endYear,endMonth,endDate):
    bd=dt.date(startYear,startMonth,startDate)
    ed=dt.date(endYear,endMonth,endDate)
    ln = 'english'
    limit =100000
    tweets= query_tweets("amazon locker",begindate=bd,enddate=ed,limit=limit,lang=ln)
    data= pd.DataFrame(t.__dict__ for t in tweets)
    data.drop_duplicates(subset="text",keep='first',inplace=True)
    data.drop(['has_media', 'hashtags', 'img_urls', 'is_replied','is_reply_to','likes','links','parent_tweet_id','replies','reply_to_users','screen_name','text_html','timestamp','timestamp_epochs','user_id','tweet_url','video_url','retweets','username','tweet_id'],axis=1,inplace=True)
    data.to_csv("uncleanedPART.csv")
 
    dataset= pd.read_csv("uncleanedPART.csv",encoding='latin1')
    length= len((dataset))

    print("length=",length)
    
    def strip_links(text):
        link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
        links         = re.findall(link_regex, text)
        for link in links:
            text = text.replace(link[0], ', ')    
        return text
    
    def strip_all_entities(text):
        entity_prefixes = ['@','#']
        for separator in  string.punctuation:
            if separator not in entity_prefixes :
               text = text.replace(separator,' ')
        words = []
        for word in text.split():
            word = word.strip()
            if word:
               if word[0] not in entity_prefixes:
                words.append(word)
        return ' '.join(words)


    def removeNonAscii(s): 
        return "".join(filter(lambda x: ord(x)<128, s))


    header = ['text', 'categories']
    with open('cleanedPART.csv','w',encoding="utf-8") as file:
      writer = csv.writer(file, delimiter=',')
      writer.writerow(i for i in header)
      for i in range(0,len(dataset)):
          text=dataset['text'][i]
          review= strip_all_entities(strip_links(text))
          review1=removeNonAscii(review)
          file.write(review1)
          file.write('\n')

   
        

    data= pd.read_csv("cleanedPART.csv",encoding="utf-8")
    
    positive=0
    negative=0
    neutral=0
    with open('cleanedPART1ForCNN.csv','w',encoding="utf-8", newline='') as f:
     writer = csv.writer(f, delimiter=',')
     writer.writerow(i for i in header)
     
     for i in range(0,len(data)):
            #print(data["text"][i])
            blob= TextBlob(data["text"][i])
            x = blob.sentiment.polarity
            if(x<0):
                    writer.writerow([data['text'][i],'0'])
                    negative+=1
            elif(x==0):
                    writer.writerow([data['text'][i],'2'])
                    neutral+=1
            else:
                    writer.writerow([data['text'][i],'1'])
                    positive+=1
    print("Positive review=", positive)
    print("negative reviews=",negative)
    print("neutral reviews=", neutral)
    
    dataset=pd.read_csv('cleanedPART1ForCNN.csv')
   
    
    
    
    
    Item= dict()
    Item['fit']=0
    Item['fraud']=0
    Item['ship']=0
    Item['shipping']=0
    Item['shipped']=0
    Item['deliver']=0
    Item['delivered']=0
    Item['delivery']=0
    Item['storage']=0
    Item['space']=0
    Item['delivered']=0
    Item['full']=0
    Item['tight']=0 
    Item['big']=0
    Item['filled']=0
    Item['available']=0
    Item['time']=0
    Item['ago']=0
    Item['closed']=0
    Item['late']=0
    Item['tomorrow']=0
    Item['long']=0
    Item['location']=0
    Item['remotely']=0
    Item['walk']=0
    Item['far']=0
    Item['located']=0
    Item['broken']=0
    Item['lost']=0
    Item['fraud']=0
    Item['secure']=0
    Item['stolen']=0
    Item['steal']=0
    Item['missing']=0
    Item['fake']=0
    Item['empty']=0
    Item['working']=0
    Item['opening']=0
    Item['code']=0
    Item['work']=0
    Item['secret']=0
    Item['confused']=0
    Item['functional']=0
    Item['address']=0
    Item['expensive']=0
    Item['charge']=0
    Item['cost']=0
    Item['price']=0
    Item['paid']=0
    Item['pay']=0
    Item['fee']=0
    Item['free']=0
    Item['worth']=0


   
    for i in range(0,len(dataset)):
        T= TextBlob(dataset['text'][i])
        w= T.lower()
        #print(w)
        if(w.sentiment.polarity<=0):
           for word in w.words:
               review= Word(word)
               r= review.lemmatize()
               if(r!='amazon' and r!='locker'):
                 #print(r)
                 if(r in Item):
                    Item[r]+=1
                 else:
                    Item[r]=1
    shipping =0;
    shipping += Item['ship']+Item['shipping']+ Item['shipped']

    delivery=0;
    delivery+= Item['deliver']+ Item['delivered']+Item['delivery']

#7
    space=0
    space+= Item['fit']+Item['storage']+Item['space']+Item['delivered']+Item['full']+Item['tight']+ Item['big']+Item['filled']+Item['available']

#2
    time=0;
    time+= Item['time']+Item['ago']+Item['closed']+Item['late']+Item['tomorrow']+Item['long']

#4
    charge=0;
    charge+= Item['expensive']+Item['charge']+Item['cost']+Item['price']+Item['paid']+Item['pay']+ Item['fee']+ Item['free']+ Item['worth']

#5
    location=0;
    location+=Item['location']+ Item['remotely']+Item['walk']+Item['far']+Item['located']

#3
    secure =0;
    secure+=Item['broken']+Item['lost']+Item['fraud']+Item['secure']+Item['stolen']+Item['steal']+Item['missing']+Item['fake']+Item['empty']

#1
    working=0;
    working+= Item['working']+Item['opening']+Item['code']+Item['work']+Item['secret']+ Item['confused']+Item['functional']

#6
    address=0;
    address+= Item['address']

    print("working =",working)
    print("secure =",secure)
    print("charge =",charge)
    print("time =",time)
    print("shipping =",shipping)
    print("delivery =",delivery)
    print("space =",space)
    print("location =",location)
    print("address= ",address)

     
    x=input("Enter the category for which you want to see insights:-")
    
    while(x=='space' or x=='working' or x=='secure' or x=='charge' or x=='time' or x=='shipping' or x=='delivery' or x=='address' or x=='location'):
        if(x=='space'):      
            objects = ('fit', 'storage', 'full', 'tight', 'big', 'filled      ','available       ','delivered  ','space')
            y_pos = np.arange(len(objects))
            performance = [Item['fit'],Item['storage'],Item['full'],Item['tight'],Item['big'],Item['filled'],Item['available'],Item['delivered'],Item['space']]
            plt.bar(y_pos, performance, align='center', alpha=1.0)
            plt.xticks(y_pos, objects)
            plt.ylabel('Tweets')
            plt.title('Insights')
            plt.show()
        elif(x=='working'):
            objects=('working','opening','code','work','secret','confused','functional')
            y_pos = np.arange(len(objects))
            performance = [Item['working'],Item['opening'],Item['code'],Item['work'],Item['secret'],Item['confused'],Item['functional']]
            plt.bar(y_pos, performance, align='center', alpha=1.0)
            plt.xticks(y_pos, objects)
            plt.ylabel('Tweets')
            plt.title('Insights')
            plt.show()
        elif(x=='secure'):
            objects=('broken','lost','fraud','secure','stolen','steal','missing','fake','empty')
            y_pos=np.arange(len(objects))
            performance=[Item['broken'],Item['lost'],Item['fraud'],Item['secure'],Item['stolen'],Item['steal'],Item['missing'],Item['fake'],Item['empty']]
            plt.bar(y_pos, performance, align='center', alpha=1.0)
            plt.xticks(y_pos, objects)
            plt.ylabel('Tweets')
            plt.title('Insights')
            plt.show()        
        elif(x=='charge'):
            objects=('expensive','charge','cost','price','paid','pay','fee','free','worth')
            y_pos=np.arange(len(objects))
            performance=[Item['expensive'],Item['charge'],Item['cost'],Item['price'],Item['paid'],Item['pay'],Item['fee'],Item['free'],Item['worth']]
            plt.bar(y_pos, performance, align='center', alpha=1.0)
            plt.xticks(y_pos, objects)
            plt.ylabel('Tweets')
            plt.title('Insights')
            plt.show()  
        elif(x=='time'):
            objects=('time','ago','closed','late','tomorrow','long')
            y_pos=np.arange(len(objects))
            performance=[Item['time'],Item['ago'],Item['closed'],Item['late'],Item['tomorrow'],Item['long']]
            plt.bar(y_pos, performance, align='center', alpha=1.0)
            plt.xticks(y_pos, objects)
            plt.ylabel('Tweets')
            plt.title('Insights')
            plt.show()
        elif(x=='shipping'):
            objects=('ship','shipping','shipped')
            y_pos=np.arange(len(objects))
            performance=[Item['ship'],Item['shipping'],Item['shipped']]
            plt.bar(y_pos, performance, align='center', alpha=1.0)
            plt.xticks(y_pos, objects)
            plt.ylabel('Tweets')
            plt.title('Insights')
            plt.show()
        elif(x=='delivery'):
            objects=('deliver','delivered','delivery')
            y_pos=np.arange(len(objects))
            performance=[Item['deliver'],Item['delivered'],Item['delivery']]
            plt.bar(y_pos, performance, align='center', alpha=1.0)
            plt.xticks(y_pos, objects)
            plt.ylabel('Tweets')
            plt.title('Insights')
            plt.show()
        elif(x=='location'):
            objects=('location','remotely','walk','far','located')
            y_pos=np.arange(len(objects))
            performance=[Item['location'],Item['remotely'],Item['walk'],Item['far'],Item['located']]
            plt.bar(y_pos, performance, align='center', alpha=1.0)
            plt.xticks(y_pos, objects)
            plt.ylabel('Tweets')
            plt.title('Insights')
            plt.show()
        elif(x=='address'):
            objects=('address')
            y_pos=np.arange(len(objects))
            performance=[Item['address']]
            plt.bar(y_pos, performance, align='center', alpha=1.0)
            plt.xticks(y_pos, objects)
            plt.ylabel('Tweets')
            plt.title('Insights')
            plt.show()
        else:
            print("ENDING...")
            break
        x=input('Insights? ')
     
if __name__ == "__main__":
    startYear=int(input("Enter your startyear:-"))
    startMonth=int(input("Enter start month:-"))
    startDate =int(input("Enter start date:-"))
    endYear   =int(input("Enter end year:-"))
    endMonth  =int(input("Enter end month:-"))
    endDate   =int(input("Enter end date:-"))
    dates(startYear,startMonth,startDate,endYear,endMonth,endDate)
    
   
