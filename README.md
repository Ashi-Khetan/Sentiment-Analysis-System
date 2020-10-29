# Sentiment-Analysis-System
Sentiment Analysis aims to determine the attitude of a speaker or a writer with respect to some topic or the overall contextual polarity of a document .The attitude may be his or her judgment or evaluation affective state, or the intended emotional communication. 

Following are the steps performed in the project:

a. Extraction of data :- This is done using the Numpy and Pandas library. In our project , we import both the libraries and load our dataset through it.
Part of the project code which applies it:
import numpy
import pandas as pd
df=pd.read_csv('C:\\Users\\Ashi_Khetan\\Downloads\\Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

b. Cleaning of data:- This is done by using the re module in python. The module re provides full support for Perl-like regular expressions in Python. A regular expression is a special sequence of characters that helps you match or find other strings or sets of strings, using a specialized syntax held in a pattern.
In our project, we have used re module to remove all the digits from the review in return of
white spaces.
Part of the project code which applies it:
import re
for i in range(0,1000):
review=re.sub('[^a-zA-Z]',' ',df['Review'][i])

c. Removing stop words :- In our project, we have used the nltk to remove the stopwords from our dataset. This is done by importing stopwords from nltk.corpus.
Part of the project code which applies it:
import nltk
from nltk.corpus import stopwords
review=[word for word in review if not word in set(stopwords.words('english'))]

d. Stemming of words(finding root word) :- In our project, we have used the nltk to find the root word in the text. This is done by importing PorterStemmer from nltk.stem .
Part of the project code which applies it:
import nltk
from nltk.stem import PorterStemmer
ps=PorterStemmer()
review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
review=' '.join(review)

e. Tokenization(building of sparse matrix) :- Tokenization is the process of breaking a stream of text up into words, phrases, symbols, or other meaningful elements called tokens.In our project, text document is broken into sentences and added to a list.

f. Vectorization(maximum count of the root word):- Using CountVectorizer from scikit.feature_extraction.text , we can convert a collection of text documents to a matrix of
token counts. The fit_transform method tokenize the strings and give you a vector for each string, each dimension of which corresponds to the number of times a token is found in the corresponding string. Most of the entries in all of the vectors will be zero, since only the entries which correspond to tokens found in that specific string will have positive values, but the vector is as long as the total number of tokens for the whole corpus. It also gives you the count vectors for the training data.
Part of the project code which applies it:
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x=cv.fit_transform(L).toarray()

g. Building up of model using Python(classification model) :- We use GaussianNB which is a special type of NB(Naive Bayes) algorithm for classification and is used when the features have continuous values.
Part of the project code which applies it:
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)

h. Training of model :- Using train_test_split from sklearn.cross_validation we can split arrays or matrices into random train and test subsets. The training set contains a known output and the model learns on this data in order to be generalized to other data later on.
Part of the project code which applies it:
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

i.Testing of model :- We have the test dataset (or subset) in order to test our model’s prediction on this subset.
Part of the project code which applies it:
y_pred=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

j. Accuracy check :- We find the accuracy check to the efficiency of our model.
Accuracy= correct outcomes/ total outcomes *100
Part of the project code which applies it:
r=(67+113)/(67+113+70)*100

k. Data visualization :- Using Matplot and Seaborn library we have applied Heatmap to visualize our confusion matrix and find the overall sentiment of the text. A heatmap is a graphical representation of data where the individual values contained in a matrix are represented as colors.
Part of the project code which applies it:
from matplotlib import pyplot as plt
from matplotlib import style
import seaborn as sns
sns.heatmap(cm)
plt.xlabel("True values")
plt.ylabel("Predictions")
plt.show()
