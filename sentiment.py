import nltk
from nltk.data import load
import numpy as np  

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup

tokenizer = load('file:C:/english.pickle')
wordnet_lemmatizer = WordNetLemmatizer()
#stopwords = set(w.rstrip() for w in open('stopwords.txt'))

positive_reviews = BeautifulSoup(open('domain_sentiment_data.tar/sorted_data_acl/electronics/positive.review').read(), "lxml")
positive_reviews = positive_reviews.findAll('review_text')

negative_reviews = BeautifulSoup(open('domain_sentiment_data.tar/sorted_data_acl/electronics/negative.review').read(), "lxml")
negative_reviews = negative_reviews.findAll('review_text')

np.random.shuffle(positive_reviews)
positive_reviews = positive_reviews[:len(negative_reviews)]


def my_tokenizer(s):
	s = s.lower()
	tokens = nltk.tokenize.word_tokenize(s)
	tokens = [t for t in tokens if len(t) >2]
	tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
	#tokens = [t for t in tokens if t not in stopwords]
	return tokens

word_index_map = {}
current_index = 0

positive_tokenized = []
negative_tokenized = []



for review in positive_reviews:
	tokens = my_tokenizer(review.text)
	positive_tokenized.append(tokens)
	for token in tokens:
		if token not in word_index_map:
			word_index_map[token] = current_index
			current_index +=1

for review in negative_reviews:
	tokens = my_tokenizer(review.text)
	negative_tokenized.append(tokens)
	for token in tokens:
		if token not in word_index_map:
			word_index_map[token] = current_index
			current_index +=1

def tokens_to_vector(tokens, label):
	x = np.zeros(len(word_index_map) +1)
	for t in tokens:
		i = word_index_map[t]
		x[i] += 1
	x = x/x.sum()
	x[-1] = label
	return x

N = len(positive_tokenized) + len(negative_tokenized)

data = np.zeros((N, len(word_index_map) +1))
i = 0

for tokens in positive_tokenized:
	xy = tokens_to_vector(tokens, 1)
	data[i,:] = xy
	i += 1

for tokens in negative_tokenized:
	xy = tokens_to_vector(tokens, 0)
	data[i,:] = xy
	i += 1

np.random.shuffle(data)

X = data[:,:-1]
Y = data[:, -1]

Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

model = LogisticRegression()
model.fit(Xtrain, Ytrain)
print ("rate:", model.score(Xtest,Ytest))


	