# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 15:20:40 2018

@author: Harsh Kava
"""
"""
1. Defines a function "**tokenize**" as follows:
   - takes a string as an input
   - converts the string into lowercase
   - tokenizes the lowercased string into tokens. Each token has at least two characters. A token **only contains letters (i.e. a-z or A-Z), "-" (hyphen), or "_" (underscore)**. Moreover, ** a token cannot starts or ends with "-" or "_" **. 
   - removes stop words from the tokens (use English stop words list from NLTK)
   - returns the resulting token list as the output
   
2. Defines a function "**sentiment_analysis**" as follows:
   - takes a string, a list of positive words, and a list of negative words as inputs. Assume the lists are read from positive-words.txt and negative-words.txt outside of this function.
   - tokenize the string using NLTK word tokenizer
   - counts positive words and negative words in the tokens using the positive/negative words lists. The final positive/negative words are defined as follows:
     - Positive words:
       * a positive word not preceded by a negation word (i.e. not, n't, no, cannot, neither, nor, too)
       * a negative word preceded by a negation word
     - Negative words:
       * a negative word not preceded by a negation word
       * a positive word preceded by a negation word
   - determines the sentiment of the string as follows:
     - 2: number of positive words > number of negative words
     - 1: number of positive words <= number of negative words
   - returns the sentiment 

3. Defines a function called **performance_evaluate** to evaluate the accuracy of the sentiment analysis in (2) as follows: 
   - takes an input file ("amazon_review_300.csv"), a list of positive words, and a list of negative words as inputs. The input file has a list of reviews in the format of (label, title, review). Use label (either '2' or '1') and review columns (i.e. columns 1 and 3 only) here.
   - reads the input file to get reviews as a list of (label, reviews) tuples
   - for each review, predicts its sentiment using the function defined in (2), and compare the prediction with its label
   - returns the accuracy as the number of correct sentiment predictions/total reviews

"""

import nltk
from nltk.util import ngrams
import re
from nltk.tokenize import sent_tokenize
from nltk import load
import csv

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def tokenize(text):
    
    tokens=[]
    
    text=re.sub('[^a-zA-Z-_\d]',' ',text.lower())#replace chars that are not letters or numbers with a spac
    text=re.sub(' +',' ',text).strip()#remove duplicate spaces

    #creating set of Stop words    
    stop_words = set(stopwords.words('english'))
    #print(stop_words)

    #tokenize the sentence 
    word_tokens = word_tokenize(text)
 
    #filter the words based on conditions
    for w in word_tokens:
        if w not in stop_words:
            if(len(w) >2):
                if not w.startswith("_") and not w.startswith("-"):
                    tokens.append(w)
            
    #print(tokens)
    return tokens

def sentiment_analysis(text, positive_words, negative_words):
    
    sentiment=None
    # write your code here
    
    posWordCount = 0
    negWordCount = 0
    
    #print('text: ',text)
    tokens = tokenize(text)
    
    twograms = ngrams(tokens,2) #compute 2-grams   
    
      # Rules:
    #    - Positive words:
     #  * a positive word not preceded by a negation word (i.e. not, n't, no, cannot, neither, nor, too)
      # * a negative word preceded by a negation word (ex -not bad)
     #- Negative words:
      # * a negative word not preceded by a negation word
      # * a positive word preceded by a negation word
    #print('Positive Words: ',positive_words)
    
    for tg in twograms:  
        if tg[0] in negative_words and tg[1] in negative_words: # a negative word preceded by a negation word (ex -not bad)
            posWordCount += 1
        elif tg[0] not in negative_words and tg[1] in positive_words: #a positive word not preceded by a negation word (i.e. not, n't, no, cannot, neither, nor, too)
            posWordCount += 1
        elif tg[0] not in negative_words and tg[1] in negative_words: # a negative word not preceded by a negation word
            negWordCount += 1
        elif tg[0] in negative_words and tg[1] in positive_words: # a positive word preceded by a negation word
            negWordCount += 1
    
    
    if(posWordCount > negWordCount):
        sentiment = 2
    elif(posWordCount <= negWordCount):
        sentiment = 1
        
    return sentiment

# return all the 'adv adj' twograms
def getAdvAdjTwograms(terms,adj,adv):
    result=[]
    twograms = ngrams(terms,2) #compute 2-grams    
   	 #for each 2gram
    for tg in twograms:  
        if tg[0] in adv and tg[1] in adj: # if the 2gram is a an adverb followed by an adjective
            result.append(tg)

    return result


def performance_evaluate(input_file, positive_words, negative_words):
    
    accuracy=0
    
    reviews = []
    # write your code here
    with open(input_file) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            label = row[0]
            rev =row[2]
            tup =(label,rev)
            reviews.append(tup)
        
    #print(reviews)
    #print(len(reviews))

    
    for review in reviews:
        
        #print(review[0])    
        myprediction = sentiment_analysis(review[1], positive_words, negative_words)
        
        if(int(myprediction) == int(review[0])):
           #print('Match')
           accuracy +=1

    return accuracy


if __name__ == "__main__":  
    
    text="this is a breath-taking ambitious movie; test text: abc_dcd abc_ dvr89w, abc-dcd -abc"

    tokens=tokenize(text)
    print("tokens:")
    print(tokens)
    
    
    with open("positive-words.txt",'r') as f:
        positive_words=[line.strip() for line in f]
        
    with open("negative-words.txt",'r') as f:
        negative_words=[line.strip() for line in f]
        
    print("\nsentiment")
    sentiment=sentiment_analysis(text, positive_words, negative_words)
    print(sentiment)
    
    accuracy=performance_evaluate("amazon_review_300.csv", positive_words, negative_words)
    print("\naccuracy")
    print(accuracy)