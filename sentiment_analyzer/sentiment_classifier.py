import nltk
import numpy as np
from sklearn.utils import shuffle
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
import bs4 as bs
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()

positive_soup = bs.BeautifulSoup(open('sorted_data_acl/electronics/positive.review').read(), 'lxml')
negative_soup = bs.BeautifulSoup(open('sorted_data_acl/electronics/negative.review').read(), 'lxml')

positive_reviews = []
for review in positive_soup.find_all('review_text'):
    positive_reviews.append(review.text)

negative_reviews = []
for review in negative_soup.find_all('review_text'):
    negative_reviews.append(review.text)

def pre_processor(s):
    s = s.lower()
    words = nltk.tokenize.word_tokenize(s)
    words = [word for word in words if len(word) > 2]
    words = [lemmatizer.lemmatize(word) for word in words]
    words = [word for word in words if word not in stopwords.words('english')]
    return words

word_index_map = {}
current_index = 0
positive_tokenized = []
negative_tokenized = []
orig_reviews = []

for review in positive_reviews:
    orig_reviews.append(review)
    words = pre_processor(review)
    positive_tokenized.append(words)
    for word in words:
        if word not in word_index_map:
            word_index_map[word] = current_index
            current_index += 1

for review in negative_reviews:
    orig_reviews.append(review)
    words = pre_processor(review)
    negative_tokenized.append(words)
    for word in words:
        if word not in word_index_map:
            word_index_map[word] = current_index
            current_index += 1

print("len(word_index_map):", len(word_index_map))

# now let's create our input matrices
def words_to_vector(words, label):
    x = np.zeros(len(word_index_map) + 1) # last element is for the label
    for word in words:
        i = word_index_map[word]
        x[i] += 1
    x = x / x.sum() # normalize it before setting label
    x[-1] = label
    return x

N = len(positive_tokenized) + len(negative_tokenized)
# (N x D+1 matrix - keeping them together for now so we can shuffle more easily later
data = np.zeros((N, len(word_index_map) + 1))
i = 0
for words in positive_tokenized:
    xy = words_to_vector(words, 1)
    data[i,:] = xy
    i += 1

for words in negative_tokenized:
    xy = words_to_vector(words, 0)
    data[i,:] = xy
    i += 1

# shuffle the data and create train/test splits
# try it multiple times!
orig_reviews, data = shuffle(orig_reviews, data)

X = data[:,:-1]
Y = data[:,-1]

# last 100 rows will be test
Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

model = LogisticRegression()
model.fit(Xtrain, Ytrain)
print("Train accuracy:", model.score(Xtrain, Ytrain))
print("Test accuracy:", model.score(Xtest, Ytest))