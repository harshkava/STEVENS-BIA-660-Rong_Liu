# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 20:09:20 2018

@author: Harsh Kava
"""
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

# notice rising/rose -> rise, prices -> price
# notice the two "end" tokens are tagged as VBP and NN perspectively

# A little more NLTK.FreqDist()

word_dist=nltk.FreqDist(tokens)

# find the top 10
print('the top 10: ', word_dist.most_common(10))

# print frequency of each word
for word in word_dist:
    print(word,' : ', word_dist[word],' : ',word_dist.freq(word) )


# get frequency of bigrams (or any list of items)
bigrams=nltk.bigrams(tokens)
bigram_dist=nltk.FreqDist(bigrams)

print('frequency of bigrams:',bigram_dist)