from sklearn.linear_model import LogisticRegression
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.utils import shuffle
import bs4 as bs
import re

lemmatizer = WordNetLemmatizer()

positive_soup = bs.BeautifulSoup(open('sorted_data_acl/electronics/positive.review').read(), 'lxml')
negative_soup = bs.BeautifulSoup(open('sorted_data_acl/electronics/negative.review').read(), 'lxml')

X = []
y = []

for review in positive_soup.find_all('review_text'):
    X.append(review.text)
    y.append(1)

for review in negative_soup.find_all('review_text'):
    X.append(review.text)
    y.append(0)
  
X, y = shuffle(X, y, random_state=3)

corpus = []
for i in range(len(X)):
    review = re.sub(r'\W', ' ', str(X[i]))
    review = review.lower()
    review = re.sub(r'\s+[a-z]\s+', ' ', review)
    review = re.sub(r'^[a-z]\s+', ' ', review)
    review = re.sub(r'\s+', ' ', review)
    corpus.append(review)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=4000, min_df=3, max_df=0.6, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

prediction = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction)

print('Positive reviews: Correctly classified {} and failed {}'.format(cm[1][1],cm[0][1]))
print('Negative reviews: Correctly classified {} and failed {}'.format(cm[0][0],cm[1][0]))
print("Train accuracy:", classifier.score(X_train, y_train))
print("Test accuracy:", classifier.score(X_test, y_test))