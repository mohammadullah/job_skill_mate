#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script works on the cleaned resume files generated from "cleaning.py"
It reads all the files in that folder and extract skill from the "skill" column
Clean the skills and save in a text file  
"""

import glob
import pandas as pd
import operator
from functools import reduce

all_files = glob.glob('cleaned/*.json')

df_list = []

# Read and append the json files in dataframe
for filename in all_files:
    df = pd.read_json(filename, lines=True)
    df_list.append(df)

# Concate in one single dataframe
frame = pd.concat(df_list, axis=0, ignore_index=True)

# Get the skill column and skills
frame1 = frame.skill
frame1 = frame1.values
frame1 = frame1.tolist()
frame1 = reduce(operator.iconcat, frame1, [])

# Convert list to string
str1 = ",".join([str(x) for x in frame1])
# Lower case
str1 = str1.lower()
# keep all letters, #, +, and space 
str1 = "".join([c if (ord(c) > 96 and ord(c) < 123 or ord(c) == 32 or 
                      ord(c) == 35 or ord(c) == 43) else "," for c in str1])
# replace "and" and "tab"
str1 = str1.replace(" and ", ",")
str1 = str1.replace("\t", ",")

# Split skills at "," and remove whitespace and empty list content
voc_list = str1.split(",")
voc_list = [x.strip() for x in voc_list]
voc_list = [x for x in voc_list if x] 

# Remove duplicate skills
voc_uniq = set(voc_list)
voc_uniq = list(voc_uniq)

# remove skills that contains more than three strings
ngram = [x.split(" ") for x in voc_uniq]
ngram = [x for x in ngram if len(x) <= 3]
voc_final = [" ".join(x) for x in ngram]


# Write in file
with open('vocabulary.txt', 'w') as f:
    for item in voc_final:
        f.write("%s\n" % item)