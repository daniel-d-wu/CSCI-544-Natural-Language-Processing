#!/usr/bin/env python
# coding: utf-8

# In[161]:


"""
Author: Daniel Wu
Purpose: PS4 - Train a perceptron model
         on hotel reviews
         
output: vanillamodel.txt
        averagedmodel.txt
"""

import os
import sys
import math
import re
import string
import glob
import numpy as np
import random

def pause():
    programPause = input("Press the <ENTER> key to continue...")
    print("Paused Program")
    
# root_dir = sys.argv[1]
root_dir = "/Users/user/Desktop/Fall_2020/CSCI_544/Coding_Assignments/PA4/op_spam_training_data"

# filepath dictionary
p = {}

p['nd'] = "/negative_polarity/deceptive_from_MTurk"
p['nt'] = "/negative_polarity/truthful_from_Web"
p['pd'] = "/positive_polarity/deceptive_from_MTurk"
p['pt'] = "/positive_polarity/truthful_from_TripAdvisor"

def store_reviews(sub_dir):
    
    review_list = []
    
    for path in list(os.walk(root_dir + sub_dir))[1:]:    
        for text in path[2]:        
            file_path = path[0] + "/" + text
        
            with open(file_path) as doc:                
                review_list.append(''.join(doc.readlines()))
        
    return review_list

# review dictionary 
reviews = {}

for sub_dir in ['nd', 'nt', 'pd', 'pt']:                
        reviews[sub_dir] = store_reviews(p[sub_dir])

        
# set stop words - hotel context
#                  first / second person pronouns
#                  common filler words
# some of these are also retroactively added in
# based on joint probabilities being too high

stop_words = ['hotel', 'hotels', 'stay', 'stayed',
              'book', 'booked', 'reserve', 'reserved',
              'room', 'rooms',
              'reservation', 'here',
              'i', 'me', 'my', 'mine',
              'the', 'we', 'our', 'ours',
              'it', 'its', 'they', 'them',
              'he', 'she', 'him', 'her', 'his',
              'they', 'them', 'theirs', 'who', 'what', 'where',
              'when', 'am', 'are', 'about',
              'to', 'in', 'out', 'up', 'down',
              'a', 'an', 'how', 'if', 'as', 'on',
              'some', 'can', 'is', 'be', 'any', 
              'through', 'of', 'off',
              'these', 'those', 'that',              
              'one', 'ha', 'would', 'from', 'by', 'thing',
              'this', 'and', 'for', ' ', 'during', 'before',
              'after', 'very'
              "i'll", "we'll", "it's",
              "i'm"
             ]

puncs1 = string.punctuation.replace("'", '')
puncs2 = puncs1.replace("-", '')
puncs = list(puncs2)

#get list of tokens - seperate by space " "
#generalize time,
#generalize amount,
#separate punctuation

token_bag = {}
clean_reviews = {}

for cls in ['nd', 'nt', 'pd', 'pt']:
    
    word_list = ""    
    clean_reviews[cls] = []
    
    for review in reviews[cls]:        
                        
        review = re.sub(r"(?:[0-2]?[0-9])(?:(?:am|pm)|(?::[0-5][0-9]?)(?:am|pm)?)", "timetok", review)        
        review = re.sub(r"\$\d+(?:\.\d?\d)?", "amttok", review)        
        review = review.translate(str.maketrans({punc: " {0} ".format(punc) for punc in puncs}))                         
        word_list = word_list + review.lower()        
        clean_reviews[cls].append(review.lower())                    
    token_bag[cls] = set(word_list.split(' '))
    
    # remove stop words and punctuations
    token_bag[cls] = [tok for tok in token_bag[cls] if tok not in stop_words]    
    #get rid of letters and 2-letter words, but keep a few punctuations 
    token_bag[cls] = [tok for tok in token_bag[cls] if (len(tok) > 2 or tok in ('?', '!'))]
        
# Create total bag of words
token_bag['total'] = set(token_bag['nd'] + token_bag['nt'] + token_bag['pd'] + token_bag['pt'])

# get counts in the 4-class classifier    
token_count = {}

for cls in ['nd', 'nt', 'pd', 'pt']:    
    token_count[cls] = {}    
            
    for tok in token_bag['total']:            
        tok_count = len([1 for review in clean_reviews[cls] if tok in review])                                        
        token_count[cls][tok] = {}
        token_count[cls][tok] = tok_count  
        
# Use joint probabilities for feature selection
joint_prob_pn = {} #for pos/neg
joint_prob_td = {} #for true/deceptive

for tok in token_bag['total']:    
    joint_prob_pn[tok] = [ (token_count['pd'][tok] + token_count['pt'][tok] + 1 ) / 
                           (len(clean_reviews['pd']) + len(clean_reviews['pt']) + 1),
                           (token_count['nd'][tok] + token_count['nt'][tok] + 1 ) / 
                           (len(clean_reviews['nd']) + len(clean_reviews['nt']) + 1) ]              
    joint_prob_td[tok] = [ (token_count['pt'][tok] + token_count['nt'][tok] + 1 ) / 
                           (len(clean_reviews['pt']) + len(clean_reviews['nt']) + 1),
                           (token_count['pd'][tok] + token_count['nd'][tok] + 1 ) / 
                           (len(clean_reviews['pd']) + len(clean_reviews['nd']) + 1) ]

joint_prob_pn = list(joint_prob_pn.items())
joint_prob_pn.sort(key= lambda x: x[1][0] + x[1][1], reverse=True)
joint_prob_pn = joint_prob_pn[0:1000]

joint_prob_td = list(joint_prob_td.items())
joint_prob_td.sort(key= lambda x: x[1][0] + x[1][1], reverse=True)
joint_prob_td = joint_prob_td[0:1000]

pn_features = []
td_features = []
for i in range(len(joint_prob_pn)):
    pn_features.append(joint_prob_pn[i][0])
    
for i in range(len(joint_prob_td)):    
    td_features.append(joint_prob_td[i][0])

# convert dictionary to (indexed) list
doc_list = []
y_pn = []
y_td = []

for cls in ['nd', 'nt', 'pd', 'pt']:    
    if cls == "nd" or cls == "nt":
        pn_label = -1
    else:
        pn_label = 1        
    if cls == "nd" or cls == "pd":
        td_label = -1
    else:
        td_label = 1
    
    for i in range(len(clean_reviews[cls])):
        doc_list.append(clean_reviews[cls][i])        
        y_pn.append(pn_label)
        y_td.append(td_label)           
    
# build feature dictionary (efficient matrix)
pn_feature_count = {}
td_feature_count = {}

for i in range(len(doc_list)):    
    pn_feature_count[i] = {}
    td_feature_count[i] = {}
    
    for word in pn_features:
        cnt = doc_list[i].count(word)
        if cnt == 0:
            continue
        pn_feature_count[i][word] = {}
        pn_feature_count[i][word] = cnt
        
    for word in td_features:
        cnt = doc_list[i].count(word)
        if cnt == 0:
            continue
        td_feature_count[i][word] = {}
        td_feature_count[i][word] = cnt
        
# pn_feature_count stores feature:count for pn_features
    
    
# PERCEPTRON
#args: labels           [y_pn, y_td]
#      doc features     [pn_feature_count, td_feature_count]
#      features         [pn_features, td_features]
#      it - iteration

def percep(label, doc_feature_count, features, it):

    w = np.zeros(len(features), dtype = int)
    b = 0    
        
    c = 1
    mu = np.zeros(len(features), dtype = int)
    beta = 0        

    for k in range(it):    
        if k == it - 1:
            count = 0

        #randomize here
        r = list(range(len(pn_feature_count)))
        random.shuffle(r)

        # for each document
        for i in r:      
            
            y = label[i]             
            w_x = 0     

            w_rep = np.zeros(len(features), dtype = int) # potential replacer

            #iterate features and update weights        
            for token in doc_feature_count[i].keys():            
                dex = features.index(token)              
                w_x = w_x + (w[dex] * doc_feature_count[i][token])            
                
                w_rep[dex] = y * doc_feature_count[i][token]           
                
            if (w_x + b) * y <= 0:   
                w = w + w_rep            
                b = b + y
                
                mu = mu + w_rep * c                
                beta = beta + y * c
                
            else:
                if k == it-1:
                    count += 1
                pass
            
            c += 1
            
    print(f'{count} correctly classified')
    
    return w, b, w - (1/c) * mu, b - (1/c) * beta
        
van_weight_pn, van_bias_pn, avg_weight_pn, avg_bias_pn = percep(y_pn, pn_feature_count, pn_features, 10)
van_weight_td, van_bias_td, avg_weight_td, avg_bias_td = percep(y_td, td_feature_count, td_features, 10)
    
# outfile
outfile = ""

for i in range(len(van_weight_pn)):
    outfile += f"pn {pn_features[i]} {van_weight_pn[i]} {van_bias_pn} \n"    
for i in range(len(van_weight_td)):
    outfile += f"td {td_features[i]} {van_weight_td[i]} {van_bias_td} \n"

file = open("./vanillamodel.txt", "w")
file.writelines(outfile[:-2])
file.close()

outfile = ""

for i in range(len(avg_weight_pn)):
    outfile += f"pn {pn_features[i]} {avg_weight_pn[i]} {avg_bias_pn} \n"    
for i in range(len(avg_weight_td)):
    outfile += f"td {td_features[i]} {avg_weight_td[i]} {avg_bias_td} \n"

file = open("./averagedmodel.txt", "w")
file.writelines(outfile[:-2])
file.close()

