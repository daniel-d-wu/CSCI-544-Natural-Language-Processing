#!/usr/bin/env python
# coding: utf-8

# In[6]:


"""
Author: Daniel Wu
Purpose: PS4 - classify test data
"""

import os
import sys
import math
import re
import string
import glob
import random

def pause():
    programPause = input("Press the <ENTER> key to continue...")
    print("Paused Program")
    
# bring in model parameters

#modelfile = sys.argv[1]
modelfile = "./vanillamodel.txt" 

with open(modelfile) as file:
    lines = []
    for line in file:
        lines.append(line[0:-2])
        
        
# Store parameters in dictionaries 
pn_params = {}        
td_params = {}

pn_cnt = 0
td_cnt = 0

for line in lines:
    line_break = line.split(' ')        
        
    if line_break[0] == 'pn':                
        pn_cnt +=1
        if pn_cnt == 1:        
            pn_params['BIAS'] = float(line_break[3])
        pn_params[line_break[1]] = float(line_break[2])
                        
    elif line_break[0] == 'td':
        td_cnt += 1
        if td_cnt == 1:            
            td_params['BIAS'] = float(line_break[3])
        td_params[line_break[1]] = float(line_break[2])
        
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

# root_dir = sys.argv[2]
root_dir = "/Users/user/Desktop/Fall_2020/CSCI_544/Coding_Assignments/PA4/dev_dataset"
file_path = glob.glob(os.path.join(root_dir, '*/*/*/*.txt'))


pn_bias = pn_params['BIAS']
td_bias = td_params['BIAS']

outfile = ''

for review in file_path:
    
    with open(review) as doc:                        
        test_obs = ''.join(doc.readlines())        
            
    test_obs = re.sub(r"(?:[0-2]?[0-9])(?:(?:am|pm)|(?::[0-5][0-9]?)(?:am|pm)?)", "timetok", test_obs)
    test_obs = re.sub(r"\$\d+(?:\.\d?\d)?", "amttok", test_obs)
    test_obs = test_obs.translate(str.maketrans({punc: " {0} ".format(punc) for punc in puncs}))  
    test_obs = test_obs.lower()        
    
    pn_tot = pn_bias
    td_tot = td_bias    
    
    for word in stop_words:
        stop_word = ' ' + word + ' '
        test_obs = test_obs.replace(stop_word, ' ')        
        
    test_obs = test_obs.split(' ')   
    test_obs = [tok for tok in test_obs if tok not in stop_words]
    test_obs = [tok for tok in test_obs if (len(tok) > 2 or tok in ('?', '!'))]

    for token in test_obs:
        
        if token in pn_params.keys():
            pn_tot = pn_tot + pn_params[token]
            
        if token in td_params.keys():
            td_tot = td_tot + td_params[token]
    
    
    if pn_tot >= 0:
        pn_class = 'positive'
    elif pn_tot < 0:
        pn_class = 'negative'
                
    if td_tot >= 0:
        td_class = 'positive'
    elif td_tot < 0:
        td_class = 'negative'        

    outfile += f'{td_class} {pn_class} {review}\n'

file = open("./percepoutput.txt", "w")
file.writelines(outfile[:-1])
file.close()
    


# In[4]:


pn_params


# In[5]:


pn_params['BIAS']


# In[ ]:




