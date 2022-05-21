# -*- coding: utf-8 -*-

import nltk.data
import pandas as pd
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

#working on the first five lines of our dataset
short_data = df.head()

# Removing stop words
from nltk.corpus import stopwords
stop = stopwords.words("english")

print(short_data['Text'])
print('-------Stop words removed--------')
short_data['Step_1'] = short_data['Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
print(short_data['Step_1'])


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

#Calcul  frequence mots
import nltk
nltk.download('punkt')
import nltk 
import matplotlib.pyplot as plt
from wordcloud import WordCloud 
from nltk.tokenize import word_tokenize
word=short_data['Step_4'].str.cat(sep=' ')
tokens=word_tokenize(word)
# hna tokenization ili hiya séparartion les mots w t7othom fi data rit hna sep=espace heka sépérateur akid

from nltk.probability import FreqDist
frekDist = FreqDist(tokens)

#Afficher les mots les plus fréquentes 
wordcloud = WordCloud().generate_from_frequencies(frekDist)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
# word cloud les mots plus fréquentes 

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
tv = TfidfVectorizer(min_df = 0.05,use_idf=True, max_df = 0.5,lowercase=True, max_features=1500, stop_words = 'english')
X = tv.fit_transform(short_data['Step_4'])
 



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,short_data['Step_4'], random_state = 0)




#Apprentissage _SVM
  
from skearn.svm import SVC
svclassifier=SVC(kernel='linear')
svclassifier.fit(x_train,y_train)

from sklearn.metrics import accuracy_score
y_pred=svclassifier.predict(x_test)


#Evaluation
from sklearn.metrics import classification_report, confusion_matrix
results = confusion_matrix(y_test, y_pred)
print ('Confusion Matrix :')
print(results)
print ('Accuracy Score :',accuracy_score(y_test, y_pred) )
print ('Report : ')
print (classification_report(y_test, y_pred))
plt.show()








