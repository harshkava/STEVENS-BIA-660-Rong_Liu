# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 20:12:35 2018

@author: Harsh Kava

#Named Entity Recognition (NER)
- Definition: find and classify real word entities (Person, Organization, Event etc.) in text
- Example: sentence "Jim bought 300 shares of Acme Corp. in 2006" can be annotated as [Jim](Person) bought 300 shares of [Acme Corp.](Organization) in 2006"
"""
from nltk import word_tokenize, pos_tag, ne_chunk
 
sentence = "Jim bought 300 shares of Acme Corp. in 2006."

# the input to ne_chunk is list of (token, pos tag) tuples
ner_tree=ne_chunk(pos_tag(word_tokenize(sentence)))

# ne_chunk returns a tree
print(ner_tree)

# get PERSON out of the tree
person=[]
for t in ner_tree.subtrees():
    if t.label() == 'PERSON':
        person.append(t.leaves())
print("PERSON",person)