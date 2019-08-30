# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 21:56:19 2018

@author: Harsh Kava

Document search by TF-IDF

1. Modify tfidf and get_doc_tokens functions in Section 7.5 of your lecture notes to add “normalize” as a parameter. This parameter can take two possible values: None, "stem". The default value is None; if this parameter is set to "stem", stem each token. 
2. In the main block, do the following:
    1. Read the dataset “amazon_review_300.csv”. This dataset has 3 columns: label, title, review. We’ll use “review” column only in this example.
    2. Calculate the tf-idf matrix for all the reviews using the modified functions tfidf function, each time with a different “normalize” value 
    3. Take any review from your dataset, for each "normalize" option, find the top 5 documents most similar to the selected review, and print out these reviews
    4. Check if the top 5 reviews change under different "normalize" options. Which option do you think works better for the search? Write down your analysis as a print-out, or attach a txt file if you wish.
    5. (**bouns**) For each pair of similar reviews you find in (C),
    e.g. review x is similar to review y, find matched words under each "normalize" option.
    Print out top 10 words contributing most to their cosine similarity.
    (Hint: you need to modify the tfidf function to return the set of words as a vocabulary)
"""
import csv
import nltk
#from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from scipy.spatial import distance

# modify these two functions
def get_doc_tokens(doc, stemming):
    #print(doc)
    
    tokens = nltk.word_tokenize(doc.lower())
    clean_tokens =[]
    stopwords = nltk.corpus.stopwords.words('english')
    for token in tokens:
        if token.strip() not in stopwords:
            if token.strip() not in string.punctuation:
                clean_tokens.append(token.strip())
    
    
    if stemming:
        porter_stemmer = PorterStemmer()
        stemmed_tokens = []
        for token in clean_tokens:
            stemmed_tokens.append(porter_stemmer.stem(token))
        
        token_count={token:stemmed_tokens.count(token) for token in set(stemmed_tokens)}
        return token_count
   
    else:
        token_count={token:clean_tokens.count(token) for token in set(clean_tokens)}
        return token_count

def tfidf(docs,stemming):
    
    print('Length of Docs: ',len(docs))
    for doc in docs:
        get_doc_tokens(doc,stemming)
    
    # step 2. process all documents to get list of token list
    docs_tokens={idx:get_doc_tokens(doc,stemming) for idx,doc in enumerate(docs)}

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
    smoothed_tf_idf= normalize(tf*smoothed_idf)
    
    similarity=1-distance.squareform(distance.pdist(smoothed_tf_idf, 'cosine'))
    top_5=np.argsort(similarity)[:,::-1][0,0:5]
    
    for idx, doc in enumerate(docs):
        if idx in top_5:
            print(idx,doc)
            print("\n")
    
    return top_5

if __name__ == "__main__":  
      
    # load data
    docs=[]
    with open("amazon_review_300.csv","r") as f:
        reader=csv.reader(f)
        for line in reader:
            docs.append(line[2])

    # Find similar documents -- No STEMMING
    print(tfidf(docs, False))
    # Find similar documents -- STEMMING  
    print(tfidf(docs, True))