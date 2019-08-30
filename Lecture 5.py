# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 17:35:00 2018

@author: Harsh Kava
"""

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import nltk

news=["Oil prices soar to all-time record", 
"Stocks end up near year end", 
"Money funds rose in latest week",
"Stocks up; traders eye crude oil prices",
"Dollar rising broadly on record trade gain"]

text=". ".join(news).lower() # join list of words with "." to create a paragraph
print(text)

sentences = nltk.sent_tokenize(text)    # Using NLTK.sent_tokenize() To tokenize the sentences
print(len(sentences))
print(sentences)

# to build bigrams or tri-grams

tokens=nltk.word_tokenize(text) # get unigrams

# bigrams are formed from unigrams
# nltk.bigram returns an iterator
bigrams=list(nltk.bigrams(tokens))
print(bigrams)

# trigrams
trigrams= list(nltk.trigrams(tokens))
print(trigrams)


# - **Collocation**: an expression consisting of two or more words that correspond to some conventional way of saying things, e.g. red wine, United States, graduate students etc.
#    - Collocations are not fully compositional in that there is usually an element of meaning added to the combination.

from nltk.collocations import *
# bigram association measures
bigram_measures = nltk.collocations.BigramAssocMeasures()

# construct bigrams using words from our example
finder = BigramCollocationFinder.from_words(nltk.word_tokenize(text))
# the corpus is too small
print(finder.nbest(bigram_measures.raw_freq, 10)  )

# construct bigrams using words from a NLTK corpus
finder = BigramCollocationFinder.from_words(nltk.corpus.genesis.words('english-web.txt'))
#finder.nbest(bigram_measures.pmi, 10)  
print(finder.nbest(bigram_measures.raw_freq, 10) )



# Find collocation by filter

import string
# construct bigrams using words from a NLTK corpus

stop_words = nltk.corpus.stopwords.words('english')

finder = BigramCollocationFinder.from_words(nltk.corpus.genesis.words('english-web.txt'))

finder.apply_word_filter(lambda w: w.lower() in stop_words or w.strip(string.punctuation)=='')

print(finder.nbest(bigram_measures.raw_freq, 10) )

# better?
# how to get rid of "xxx said"?


# Exercise 3.4.1. Metrics for Collocations

from nltk.collocations import *

# construct bigrams using words from a NLTK corpus
finder = BigramCollocationFinder.from_words(nltk.corpus.genesis.words('english-web.txt'))
# find top-n bigrams by pmi
print(finder.nbest(bigram_measures.pmi, 10) )

# filter bigrams by frequency
finder.apply_freq_filter(5)
print(finder.nbest(bigram_measures.pmi, 10) )




# NLTK POS Tagging

# The input to the tagging function is a list of words

# tokenize the text
tokens=nltk.word_tokenize(text)

# tag each tokenized word
tagged_tokens= nltk.pos_tag(tokens)

print(tagged_tokens)


# Extract Phrases by POS

# Extract phrases in pattern of adjective + noun
# i.e. nice house, growing market

bigrams=list(nltk.bigrams(tagged_tokens))
print(bigrams)

phrases=[ (x[0],y[0]) for (x,y) in bigrams if x[1].startswith('JJ') and y[1].startswith('NN')]
print(phrases)

# Extract Noun+Verb, 
# i.e. prices soar
phrases=[ (x[0],y[0]) for (x,y) in bigrams if x[1].startswith('NN') and y[1].startswith('VB')]
print(phrases)



#Stemming 
#* **Stemming**: reducing inflected (or sometimes derived) words to their **stem, base or root** form. 


# Stemming Using Porter Stemmer
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()

print("Stem of organizing/organized/organizes/organization")
print(porter_stemmer.stem('organizing'))
print(porter_stemmer.stem('organized'))
print(porter_stemmer.stem('organizes'))
print(porter_stemmer.stem('organization'))

print("\nStem of crying")
print(porter_stemmer.stem('crying'))


#Lemmatization
# **Lemmatization**: determining the lemma for a given word, 
#   * A lemma is a word which stands at the head of a definition in a dictionary, e.g. run (lemma),  runs, ran and running (inflections) 
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet

wordnet_lemmatizer = WordNetLemmatizer()

print("organizing (verb) :", \
      wordnet_lemmatizer.lemmatize\
      ('organizing', wordnet.VERB))
print('organized (verb) :', \
      wordnet_lemmatizer.lemmatize\
      ('organized', wordnet.VERB))
print('organized (adjective) :',\
      wordnet_lemmatizer.lemmatize('organized', wordnet.ADJ))
print('organization (noun) :',\
      wordnet_lemmatizer.lemmatize('organization'))
print('crying (adjective) :',\
      wordnet_lemmatizer.lemmatize('crying', wordnet.ADJ))
print('crying (verb) :', \
      wordnet_lemmatizer.lemmatize('crying', wordnet.VERB))
#Note:* **Difference** between stemming and lemmatization: 
 #  * a stemmer operates on a single word **without knowledge of the context**, and therefore cannot discriminate between words which have different meanings depending on part of speech. While, lemmatization **requires context and POS tags**. 
  # * Stemming may not generate a real word, but lemmization always generates real words.
   #*  However, stemmers are typically easier to implement and run faster with reduced accuracy.
   