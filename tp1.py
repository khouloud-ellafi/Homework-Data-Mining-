# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 12:31:44 2020

@author: Multi media
"""

import nltk,sys
import numpy as np,pandas as pd
import matplotlib.pyplot as plt 
import nltk.data
from nltk.stem.porter import *
pd.options.mode.chained_assignment = None

# Dataset coversion to dataframe
df = pd.read_csv('Reviews.csv',encoding='latin-1')

#The data basic informations
print('--- Print the Basic Info of the data ----')
print(df.info())
print(df.shape)

#Showing the head and tail of the dataset
print('--- Print the Head/Tail of the data -----')
print(df.head())
print('------------------------')
print(df.tail())

df['Score'].plot(kind='hist')
plt.show()

#Removing RT word from the dataframe
df.loc[:, "Text"] = df.loc[:, "Text"].replace('RT+\s+', '', regex=True)

#working on the first five lines of our dataset
short_data = df.head()

#Step_1
# Removing stop words
from nltk.corpus import stopwords
stop = stopwords.words("english")

print(short_data['Text'])
print('-------Stop words removed--------')
short_data['Step_1'] = short_data['Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
print(short_data['Step_1'])

#Step_2
# Function to remove emoji.
import re 
def emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

print(short_data['Step_1'])
print('-------remove emoji--------')
short_data['Step_2'] = short_data['Step_1'].apply(lambda x:  emoji(x)  ) 
print(short_data['Step_2'])
#Step3
#Removing @ mentions
import re 
def remove_mentions(Text):
    return re.sub(r'@\w+', '', Text)

print(short_data['Step_2'])
print('-------remove mentions--------')
short_data['Step_3'] = short_data['Step_2'].apply(lambda x:  remove_mentions(x)  ) 
print(short_data['Step_3'])
#Step4
#Removing urls from our text
def remove_urls(Text):
    return re.sub(r'http.?://[^\s]+[\s]?', '', Text)

print(short_data['Step_3'])
print('------- remove urls--------')
short_data['Step_4'] = short_data['Step_3'].apply(lambda x:  remove_urls(x)  ) 
print(short_data['Step_4'])

# step 5 stemming 
ps = PorterStemmer()
print(short_data['Step_4'])
print('-------Stemming--------')
short_data['Step_5'] = short_data['Step_4'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split() ]))
print(short_data['Step_5'])



# step 6 Lemmazation
print(short_data['Step_5'])
print('-------Part of Speech Tagging--------')
short_data['Step_6'] = short_data['Step_5'].apply(lambda x: nltk.pos_tag(nltk.word_tokenize(x)))
print(short_data['Step_6'])



# step 7 Capitalization
print(short_data['Step_4'])
print('-------Capitalization--------')
short_data['Step_7'] = short_data['Step_4'].apply(  lambda x: ' '.join( [ word.upper() for word in x.split() ] ) )
print(short_data['Step_7'])

# - end code -