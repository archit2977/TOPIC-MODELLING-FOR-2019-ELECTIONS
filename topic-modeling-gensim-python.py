#!/usr/bin/env python
# coding: utf-8

# In[3]:

#nltk for stopwords
#gensim simple preprocess fn for stemming etc
import nltk
nltk.download('stopwords')


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


from sklearn.feature_extraction import text


# In[6]:


import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# In[7]:


# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use' ,'get'])
print(len(stop_words))
print(stop_words)


# In[8]:


#Scikit stopwords
punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')','s','I','m', '[', ']',"whi","t", '{', '}',"amp" ,"allow",'error', 'feels', 'finally', 'flagged', 'haters', 'hq',   'ji', 'know', 'latest', 'leader', 'let', 'man', 'mission', 'mod', 'model',  'new', 'nmy', 'nthe', 'nthis', 'people', 'proof', 'proud', 'qcmricwk5p', 'question',  'rath', 'ready', 'respect', 'role', 'rt', 'shri', 'sir', 'sketch', 'svzosxdpuv', 'trial', 'unique',"https://t.co/qCmricwk5P","b'rt","Ca\\xe2\\x80\\xa6","b'RT","b'i","\\n\\","Mod\\xe2\\x80\\xa6","https://t.co/svZOsxdpUv","@n\\xe2\\x80\\xa6","Let\'s","\\xe2\\x80\\xa6","\\xf0\\x9f\\x98\\x85\\n", 'weapon', 'x85', "biggest","ca","craft","buy","basic","ji","x85","%","x82", "b'RT",'@','#',"\\n","xf0","x9f","x98","x8d","xf0",'x80', 'x92', 'x9e', 'xa6', 'xe2',"https","http"]
stop_words = text.ENGLISH_STOP_WORDS.union(punc)
print(len(stop_words))
print(stop_words)


# In[9]:


#Converting .csv into .json format

import csv
import json

csvfile = open('fetchonlywords.csv', 'r')
jsonfile = open('fetchonlywords.json', 'w')

# fieldnames = ("headline_text")
fieldnames = ('h')
reader = csv.DictReader( csvfile, fieldnames)
#count=50000
for row in reader:
    json.dump(row, jsonfile)
    jsonfile.write('\n')
csvfile.close()
jsonfile.close()


# In[10]:


df = pd.read_json('fetchonlywords.json', lines=True)
# df.info()
print(df.h.unique())
df.head()


# In[11]:


# Convert to list
data = df.h.values.tolist()


# In[12]:


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))

print(data_words[0:])


# In[13]:


# bigram trigram model
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

print(trigram_mod[bigram_mod[data_words[0]]])


# In[14]:


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[15]:


# remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# make Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

nlp = spacy.load('en', disable=['parser', 'ner'])

data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[0:])


# In[16]:


id2word = corpora.Dictionary(data_lemmatized)
texts = data_lemmatized
# use of tfidf
corpus = [id2word.doc2bow(text) for text in texts]

print(corpus[0:])


# In[17]:


id2word[3]


# In[18]:


[[(id2word[id], freq) for id, freq in cp] for cp in corpus[0:]]


# In[19]:


# build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=8, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


# In[20]:


pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


# In[21]:


# Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# In[22]:


# Visualize topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis


# In[23]:


import os
from gensim.models.wrappers import LdaMallet

os.environ['MALLET_HOME'] = 'C:/Users/dell/mallet-2.0.8/'
mallet_path = 'C:/Users/dell/mallet-2.0.8/bin/mallet' 
# mallet_path = "C:\\Users\\dell\\mallet-2.0.8\\bin\\mallet"
# mallet_path = "mallet-2.0.8\\bin\\mallet" 
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=20, id2word=id2word)


# In[24]:


pprint(ldamallet.show_topics(formatted=False))

coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score: ', coherence_ldamallet)


# In[25]:


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):

    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values







# In[26]:


model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=40, step=6)


# In[27]:


# limit=40; start=5; step=2;

limit=40; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


# In[28]:


for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
    


# In[29]:



def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data):

    sent_topics_df = pd.DataFrame()

    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


# In[30]:


# model_list[0] = 2 topics
optimal_model = model_list[0]
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=50))    


# In[31]:


df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data)


df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

df_dominant_topic.head()


# In[32]:


sent_topics_sorteddf_mallet = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                            axis=0)

sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

sent_topics_sorteddf_mallet.head(20)



# In[33]:


# model_list[1] = 8 topics
optimal_model = model_list[1]
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=50))    


# In[34]:


df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data)

df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

df_dominant_topic.head()


# In[35]:


sent_topics_sorteddf_mallet = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                            axis=0)

sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

sent_topics_sorteddf_mallet.head(20)



# In[40]:


# model_list[2] = 14 topics
optimal_model = model_list[2]
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=50))    


# In[37]:


df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data)

df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

df_dominant_topic.head()


# In[38]:


sent_topics_sorteddf_mallet = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                            axis=0)

sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

sent_topics_sorteddf_mallet.head(20)



# In[39]:


topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

topic_contribution = round(topic_counts/topic_counts.sum(), 4)

print()

topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]

df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

df_dominant_topics.columns = ['Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

df_dominant_topics


# In[ ]:




