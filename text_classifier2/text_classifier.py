from sklearn.datasets import load_files
from nltk.corpus import stopwords
import pickle
import nltk
import re
nltk.download('stopwords')

reviews = load_files('train/')
X, y = reviews.data, reviews.target

# Pickiling X, y
with open('X.pickle', 'wb') as f:
    pickle.dump(X, f)
with open('y.pickle', 'wb') as f:
    pickle.dump(y, f)

# Unpickling X, y
with open('X.pickle', 'rb') as f:
    X = pickle.load(f)
with open('y.pickle', 'rb') as f:
    y = pickle.load(f)
    
corpus = []

for i in range(len(X)):
    review = re.sub(r'\W', ' ', str(X[i]))
    review = review.lower()
    review = re.sub(r'\s+[a-z]\s+', ' ', review)
    review = re.sub(r'^[a-z]\s+', ' ', review)
    review = re.sub(r'\s+', ' ', review)
    corpus.append(review)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000, min_df=3, max_df=0.6, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

prediction = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction)

accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0]+cm[1][0]+cm[1][1]+cm[0][1])
print('Positive reviews: Correctly classified {} and failed {}'.format(cm[1][1],cm[0][1]))
print('Negative reviews: Correctly classified {} and failed {}'.format(cm[0][0],cm[1][0]))
print('Accuracy is {}%'.format(accuracy))
