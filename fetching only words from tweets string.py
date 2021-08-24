#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tweepy
import csv
import pandas as pd
import re #regular expression
import string 

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer

#paste your secret keys obtained from your twitter account in these quotes
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

csvFile = open('fetchonlywords.csv', 'a', newline='')

csvWriter = csv.writer(csvFile)

count=0

for tweet_info in tweepy.Cursor(api.search, q="#LokSabhaElections2019" + " -filter:retweets", 
                                tweet_mode="extended", #full text 
                                lang = "en").items(4000):
    if "retweeted_status" in dir(tweet_info):
        tweet=tweet_info.retweeted_status.full_text
    else:
        tweet=tweet_info.full_text
    
    print(count, "------------------")
    count=count+1
    print(tweet)
    print("-----------------///////////")

    #removing links (is this reqd now? since we are later removing everything other than words and numbers later on anyways?? so wont these be removed automatically?)
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, tweet)
    for link in links:
        tweet = tweet.replace(link[0], ', ')    
    
    res = re.findall(r'[a-zA-Z0-9]+', tweet)
    #print (str(res)) 
    final_text = " ".join(res)
    #final_text = strip_all_entities(final_text)
    print(final_text)
    csvWriter.writerow([final_text])

csvFile.close()


# In[ ]:




