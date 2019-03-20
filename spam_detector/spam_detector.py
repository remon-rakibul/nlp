from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np

data = pd.read_csv('spambase/spambase.data').as_matrix()
np.random.shuffle(data)

X = data[:, :48]
Y = data[:, -1]

X_train = X[:-100,]
Y_train = Y[:-100,]
X_test = X[-100:,]
Y_test = Y[-100:,]

model = MultinomialNB()
model.fit(X_train, Y_train)

accurcy = model.score(X_test, Y_test)

print('Classification rate is ' + str(accurcy) + '%')