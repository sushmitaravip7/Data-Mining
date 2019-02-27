# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 11:53:52 2019

@author: sush6
"""

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')

from nltk.corpus import stopwords

import pandas as pd

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

query = "love story"

    
#stop_words = ["and","the","as","a","i","i'm","to","this","of","I","that","but","who","has","is",]

#read from csv

stop_words = set(stopwords.words('english'))

df = pd.read_csv('example.csv',low_memory=False,dtype={"bookID": int, "title": str, "author":str, "rating": float, "ratingsCount": int, "reviewsCount": int,
                                 "reviewerName": str, "reviewerRatings": int, "review": str})

df.dropna(inplace = True)

#df["review"]= df["review"].str.lower().str.split(" ", n = -1, expand = False)



word_vec = df["review"].str.lower().str.replace(r'[^\w\s]+', ' ').apply(str.split).apply(lambda x: [item for item in x if item not in stop_words]).apply(pd.value_counts).fillna(0)

# Compute term frequencies


query_df = pd.Series(0, index=word_vec.columns)
for word in query.split():
    try:
        query_df[word] +=1
    except:
        pass


tf = word_vec.divide(np.sum(word_vec, axis=1), axis=0)
query_tf = query_df.apply(lambda x: x / np.sum(query_df))


# Compute inverse document frequencies


idf = 1 + np.log10(len(word_vec) / word_vec[word_vec > 0].count())



# Compute TF-IDF vectors

tfidf = np.multiply(tf, idf.to_frame().T)

query_tfidf = np.multiply(query_tf, idf.T)



#cosine = tfidf.dot(query_tfidf)/
cosine_similarities = cosine_similarity(tfidf, query_tf.to_frame().T).flatten()
print(cosine_similarities)
df['Similarity'] = cosine_similarities

df.sort_values("Similarity", inplace=True, ascending=False)

display=df.head(n=5)
print(display)
print (display[['title', 'author']])
s=display.to_string(columns=['title', 'author'])
print(s)