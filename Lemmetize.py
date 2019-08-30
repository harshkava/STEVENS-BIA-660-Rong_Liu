# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 18:10:54 2018

@author: Harsh Kava

Program: Lemmatizing the tokens from text 
"""
        
import string
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

import nltk

news=["Oil prices soar to all-time record", 
"Stocks end up near year end", 
"Money funds rose in latest week",
"Stocks up; traders eye crude oil prices",
"Dollar rising broadly on record trade gain"]

text=". ".join(news).lower() # join list of words with "." to create a paragraph
print('Text:',text,'\n')

# first tokenize the text
tokens=nltk.word_tokenize(text)

# then find the POS tag of each word
# tagged_token is a list of (word, pos_tag)
tagged_tokens= nltk.pos_tag(tokens)
print('POS tag of each word: ',tagged_tokens ,'\n')

# wordnet and treebank have different tagging systems
# define a mapping between wordnet tags and POS tags as a function
def get_wordnet_pos(pos_tag):
    
    # if pos tag starts with 'J'
    if pos_tag.startswith('J'):
        # return wordnet tag "ADJ"
        return wordnet.ADJ
    
    # if pos tag starts with 'V'
    elif pos_tag.startswith('V'):
        # return wordnet tag "VERB"
        return wordnet.VERB
    
    # if pos tag starts with 'N'
    elif pos_tag.startswith('N'):
        # return wordnet tag "NOUN"
        return wordnet.NOUN
    
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        # be default, return wordnet tag "NOUN"
        return wordnet.NOUN

# get lemmatized tokens
# lemmatize every word in tagged_tokens
le_words=[wordnet_lemmatizer.lemmatize(word, get_wordnet_pos(tag)) \
          # tagged_tokens is a list of tuples (word, tag)
          for (word, tag) in tagged_tokens \
          # remove stop words
          if word not in stop_words and \
          # remove punctuations
          word not in string.punctuation]

print ('Lemetized words :', le_words)

# nltk.FreqDist gives you the freq distribution 
# of items in a list 
# it's similar to a dictionary
word_dist=nltk.FreqDist(le_words)
print(word_dist)