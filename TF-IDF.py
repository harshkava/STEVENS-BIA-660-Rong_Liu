# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 20:20:50 2018

@author: Harsh Kava


 TF-IDF: is composed by two terms: 
      - TF(Term Frequency): which measures how frequently a term, say w, occurs in a document. 
      - IDF (Inverse Document Frequency): measures how important a term is within the corpus. 
"""

import nltk, re, string
from nltk.corpus import stopwords

# library for normalization
from sklearn.preprocessing import normalize

# numpy is the package for matrix caculation
import numpy as np  

stop_words = stopwords.words('english')

docs=["Oil prices soar to all-time record", 
"Stocks end up near year end", 
"Money funds rose in latest week",
"Stocks up; traders eye crude oil prices",
"Dollar rising broadly on record trade gain"]   

# Step 1. get tokens of each document as list

def get_doc_tokens(doc):
    tokens=[token.strip() \
            for token in nltk.word_tokenize(doc.lower()) \
            if token.strip() not in stop_words and\
               token.strip() not in string.punctuation]
    
    # you can add bigrams, collocations, or lemmatization here
    
    # create token count dictionary
    token_count=nltk.FreqDist(tokens)
    
    # or you can create dictionary by yourself
    #token_count={token:tokens.count(token) for token in set(tokens)}
    return token_count

# step 2. process all documents to 
# a dictionary of dictionaries
docs_tokens={idx:get_doc_tokens(doc) \
             for idx,doc in enumerate(docs)}
print(docs_tokens)

# step 3. get document-term matrix
# contruct a document-term matrix where 
# each row is a doc 
# each column is a token
# and the value is the frequency of the token

import pandas as pd

# since we have a small corpus, we can use dataframe 
# to get document-term matrix
# but don't use this when you have a large corpus

dtm=pd.DataFrame.from_dict(docs_tokens, orient="index" )
dtm=dtm.fillna(0)
print(dtm)


# step 4. get normalized term frequency (tf) matrix

# convert dtm to numpy arrays
tf=dtm.values

# sum the value of each row
doc_len=tf.sum(axis=1)
print(doc_len)

# divide dtm matrix by the doc length matrix
tf=np.divide(tf.T, doc_len).T
print(tf)

# step 5. get idf

# get document freqent
df=np.where(tf>0,1,0)
#df

# get idf
idf=np.log(np.divide(len(docs), \
        np.sum(df, axis=0)))+1
print("\nIDF Matrix")
print (idf)


smoothed_idf=np.log(np.divide(len(docs)+1, np.sum(df, axis=0)+1))+1
print("\nSmoothed IDF Matrix")
print(smoothed_idf)

# step 6. get tf-idf
print("TF-IDF Matrix")
tf_idf=normalize(tf*idf)
print(tf_idf)

print("\nSmoothed TF-IDF Matrix")
smoothed_tf_idf=normalize(tf*smoothed_idf)
print(smoothed_tf_idf)



## Document similarity

# package to calculate distance
from scipy.spatial import distance

# calculate cosince distance of every pair of documents 
# convert the distance object into a square matrix form
# similarity is 1-distance
similarity=1-distance.squareform\
(distance.pdist(tf_idf, 'cosine'))
similarity

# find top doc similar to first one
np.argsort(similarity)[:,::-1][0,0:2]

for idx, doc in enumerate(docs):
    print(idx,doc)