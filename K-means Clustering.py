#!/usr/bin/env python
# coding: utf-8

# In[2]:


import re,string
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer


# In[3]:


data = pd.read_csv("fetchonlywords.csv",error_bad_lines=False,usecols =["headline_text"])
data.head(15)


# In[4]:


data.info()


# In[5]:

#displays only duplicated data
data[data['headline_text'].duplicated(keep=False)].sort_values('headline_text').head(8)

data = data.drop_duplicates('headline_text')


# In[6]:


data.head(15)


# In[7]:


punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')','s','I','m', '[', ']',"whi","t", '{', '}',"amp" ,"allow",'error', 'feels', 'finally', 'flagged', 'haters', 'hq',   'ji', 'know', 'latest', 'leader', 'let', 'man', 'mission', 'mod', 'model',  'new', 'nmy', 'nthe', 'nthis', 'people', 'proof', 'proud', 'qcmricwk5p', 'question',  'rath', 'ready', 'respect', 'role', 'rt', 'shri', 'sir', 'sketch', 'svzosxdpuv', 'trial', 'unique',"https://t.co/qCmricwk5P","b'rt","Ca\\xe2\\x80\\xa6","b'RT","b'i","\\n\\","Mod\\xe2\\x80\\xa6","https://t.co/svZOsxdpUv","@n\\xe2\\x80\\xa6","Let\'s","\\xe2\\x80\\xa6","\\xf0\\x9f\\x98\\x85\\n", 'weapon', 'x85', "biggest","ca","craft","buy","basic","ji","x85","%","x82", "b'RT",'@','#',"\\n","xf0","x9f","x98","x8d","xf0",'x80', 'x92', 'x9e', 'xa6', 'xe2',"https","http"]
stop_words = text.ENGLISH_STOP_WORDS.union(punc)
#print(text.ENGLISH_STOP_WORDS.__all__)
desc = data['headline_text'].values
print(desc)

vectorizer = TfidfVectorizer(stop_words = stop_words)
X = vectorizer.fit_transform(desc) 
print(X)


# In[8]:


word_features = vectorizer.get_feature_names()
print(len(word_features))
print(word_features)


# In[9]:


# Stemming


# In[10]:


# Tokenizing


# In[11]:


stemmer = SnowballStemmer('english')
tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')

def tokenize(text):
    return [stemmer.stem(word) for word in tokenizer.tokenize(text.lower())]
#tokenizer.tokenize()

# In[12]:


# Vectorization with stop words(words irrelevant to the model), stemming and tokenizing


# In[13]:


vectorizer2 = TfidfVectorizer(stop_words = stop_words, tokenizer = tokenize)
X2 = vectorizer2.fit_transform(desc)
word_features2 = vectorizer2.get_feature_names()
print(len(word_features2))
print(word_features2) 


# In[14]:


vectorizer3 = TfidfVectorizer(stop_words = stop_words, tokenizer = tokenize, max_features = 1000)
X3 = vectorizer3.fit_transform(desc)
words = vectorizer3.get_feature_names()
print(len(words))


# In[15]:


# K-means clustering
# Elbow method to select number of clusters


# In[16]:


# number of clusters=point that looks like elbow in graph


# In[17]:


from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X3)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')	#within cluster sum of squares
plt.savefig('elbow.png')
plt.show()


# In[18]:


print(words)


# In[19]:


kmeans = KMeans(n_clusters = 2, n_init = 20, n_jobs = 1)
kmeans.fit(X3)
y_means = kmeans.fit_predict(X3)
print (y_means)
print(len(y_means))

# 2 clusters
common_words = kmeans.cluster_centers_.argsort()[:,-1:-26:-1]
for num, centroid in enumerate(common_words):
    print('\n'+str(num) + ' : ' + ', '.join(words[word] for word in centroid))


# In[20]:


kmeans = KMeans(n_clusters = 3, n_init = 20, n_jobs = 1)
kmeans.fit(X3)
# 3 clusters
common_words = kmeans.cluster_centers_.argsort()[:,-1:-26:-1]
for num, centroid in enumerate(common_words):
    print('\n'+str(num) + ' : ' + ', '.join(words[word] for word in centroid))


# In[21]:


kmeans = KMeans(n_clusters = 5, n_init = 20, n_jobs = 1)
kmeans.fit(X3)
# 5 clusters
common_words = kmeans.cluster_centers_.argsort()[:,-1:-26:-1]
for num, centroid in enumerate(common_words):
    print('\n'+str(num) + ' : ' + ', '.join(words[word] for word in centroid))


# In[22]:


kmeans = KMeans(n_clusters = 7, n_init = 20, n_jobs = 1)
kmeans.fit(X3)
# 7 clusters
common_words = kmeans.cluster_centers_.argsort()[:,-1:-26:-1]
for num, centroid in enumerate(common_words):
    print('\n'+str(num) + ' : ' + ', '.join(words[word] for word in centroid))


# In[23]:


kmeans = KMeans(n_clusters = 8, n_init = 20, n_jobs = 1)
kmeans.fit(X3)
# 8 clusters
common_words = kmeans.cluster_centers_.argsort()[:,-1:-26:-1]
for num, centroid in enumerate(common_words):
    print('\n'+str(num) + ' : ' + ', '.join(words[word] for word in centroid))


# In[ ]:




