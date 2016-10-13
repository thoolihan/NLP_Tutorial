import project
import pandas as pd
import csv
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# load
train = pd.read_csv(project.labeled, header=0, delimiter="\t", quoting=csv.QUOTE_NONE)

# basic examination
print(train.shape)
print(train.columns.values)
print('\n')

# called review_to_words in kaggle tutorial
def tokenize_review(raw_review):
    # remove markup
    review_text = BeautifulSoup(raw_review, "html.parser").get_text()

    # remove punctuation
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)

    # lowercase and split
    words = letters_only.lower().split()

    # create set of stopwords (faster than list)
    stops = set(stopwords.words("english"))

    # filter stops
    words = [w for w in words if not w in stops]

    # back to string
    return(" ".join(words))

# test with first review
print('First review example clean:')
print(tokenize_review(train.review[0]))
print('\n')

# clean all reviews
print("Cleaning all reviews...\n")
train.review = train.review.map(tokenize_review)

# create bag of words
print("Creating bag of words...\n")
vectorizer = CountVectorizer(analyzer="word", \
                             tokenizer=None, \
                             preprocessor=None, \
                             stop_words=None, \
                             max_features=5000)

train_data_features = vectorizer.fit_transform(train.review).toarray()

vocab = vectorizer.get_feature_names()
freq = np.sum(train_data_features, axis=0)

print("Examine bag of words...")
for token, frq in zip(vocab, freq):
    print("%s: %d" % (token, frq))
