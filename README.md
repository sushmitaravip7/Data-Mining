# Read and Review Search Engine:

### Introduction

Read and Review is a search engine which takes input from the user and provides the top 5 books or novels related to the input.
I have used Goodreads dataset for this search engine from Kaggle.com. 
The Search engine searches for reviews that relate to the keywords or the query the user inputs and then puts the results found in the order of relevance 
to the topic that was searched for.

### Implementation

Processing the data by removing punctuation and symbols, stop words and tokenizing them and adding them in an index.

![alt text](https://raw.githubusercontent.com/username/projectname/branch/path/to/implementation_1.png)

#### Calculate the tf and idf and tfidf.

![alt text](https://raw.githubusercontent.com/username/projectname/branch/path/to/implementation_1.png)

#### Calculate the Cosine similarity between the query and the tfidf.

![alt text](https://raw.githubusercontent.com/username/projectname/branch/path/to/implementation_1.png)

#### Rank the documents

![alt text](https://raw.githubusercontent.com/username/projectname/branch/path/to/implementation_1.png)


Wrapped it around with Django and hosted it on PythonAnywhere. 
The link: http://sushmitaravip.pythonanywhere.com/Search


### References:
https://www.geeksforgeeks.org/counting-the-frequencies-in-a-list-using-dictionary-in-python/
https://www.sharpsightlabs.com/blog/numpy-sum/
https://github.com/williamscott701/Information-Retrieval/blob/master/2.%20TF-IDF%20Ranking%20-%20Cosine%20Similarity%2C%20Matching%20Score/TF-IDF.ipynb
https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
http://www.sfs.uni-tuebingen.de/~ddekok/ir/lectures/tf-idf-dump.html
https://stackoverflow.com/questions/12118720/python-tf-idf-cosine-to-find-document-similarity
https://github.com/williamscott701/Information-Retrieval/blob/master/2.%20TF-IDF%20Ranking%20-%20Cosine%20Similarity%2C%20Matching%20Score/TF-IDF.ipynb
https://www.saltycrane.com/blog/2007/10/using-pythons-finditer-to-highlight/
https://stackoverflow.com/questions/50573053/similarity-between-2-columns-of-a-dataframe
https://docs.djangoproject.com/en/2.1/ref/templates/builtins/
