# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 20:23:44 2018

@author: Harsh Kava

- Defines a function top_collocation(tokens, K) to find top-K collocations in specific patterns in a document as follows:
  - takes a list of tokens and K as inputs
  - uses the following steps to find collocations:
    - POS tag each token
    - create bigrams
    - get frequency of each bigram (you can use nltk.FreqDist)
    - keep only bigrams matching the following patterns:
       - Adj + Noun: e.g. linear function
       - Noun + Noun: e.g. regression coefficient
  - returns top K collocations by frequency
"""

# add import statement
import nltk
import string

def top_collocation(tokens, K):
    result=[]
    
    stop_words = nltk.corpus.stopwords.words('english')  + list(string.punctuation)
    clean_tokens= []
    for i in tokens:
        if i not in stop_words: clean_tokens.append(i)
    #print(clean_tokens)
    # POS tag each tokenized word
    tagged_tokens= nltk.pos_tag(clean_tokens)
    #print(tagged_tokens)
    bigrams=list(nltk.bigrams(tagged_tokens))
    #print(bigrams)
    for (x,y) in bigrams :
        if ((x[1].startswith('JJ') and y[1].startswith('NN')) or (x[1].startswith('NN') and y[1].startswith('NN')) ):
            result.append((x[0],y[0]))

    fdist = nltk.FreqDist(result)
    #print(fdist)
    
    result = fdist.most_common(K)
    return result

if __name__ == "__main__":  
    
    # test collocation
    text=nltk.corpus.reuters.raw('test/14826')
    tokens=nltk.word_tokenize(text.lower())
    print(top_collocation(tokens, 10))
    
