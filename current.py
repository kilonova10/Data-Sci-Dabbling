from sklearn.naive_bayes import MultinomialNB
import pandas as pd 
import numpy as numpy

data = pd.read_csv('spambase.data').as_matrix()
np.random.shuffle(data) 

X = data[;, :48]
Y = data[:, -1]

Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print "Rate for  NB:", model.score(Xtest, Ytest)
