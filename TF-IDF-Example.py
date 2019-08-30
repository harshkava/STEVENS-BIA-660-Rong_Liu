# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 20:32:05 2018

@author: Harsh Kava
"""

import nltk, re, string
from sklearn.preprocessing import normalize
from nltk.corpus import stopwords
# numpy is the package for matrix cacluation
import numpy as np  
import pandas as pd

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
    
    # you can add bigrams, collocations, stemming, 
    # or lemmatization here
    
    token_count={token:tokens.count(token) for token in set(tokens)}
    return token_count

def tfidf(docs):
    # step 2. process all documents to get list of token list
    docs_tokens={idx:get_doc_tokens(doc) \
             for idx,doc in enumerate(docs)}

    # step 3. get document-term matrix
    dtm=pd.DataFrame.from_dict(docs_tokens, orient="index" )
    dtm=dtm.fillna(0)
      
    # step 4. get normalized term frequency (tf) matrix        
    tf=dtm.values
    doc_len=tf.sum(axis=1)
    tf=np.divide(tf.T, doc_len).T
    
    # step 5. get idf
    df=np.where(tf>0,1,0)
    #idf=np.log(np.divide(len(docs), \
    #    np.sum(df, axis=0)))+1

    smoothed_idf=np.log(np.divide(len(docs)+1, np.sum(df, axis=0)+1))+1    
    smoothed_tf_idf=tf*smoothed_idf
    
    return smoothed_tf_idf

smoothTfIdf = tfidf(docs)
print(smoothTfIdf)