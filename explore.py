import project
import pandas as pd
import csv
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier

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
for token, frq in list(zip(vocab, freq))[:15]:
    print("%s: %d" % (token, frq))
print('\n')

print("Training random forest...\n")
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_data_features, train.sentiment)

print("Create submission...\n")
# load test data
test_df = pd.read_csv(project.test_data, header=0, delimiter="\t", quoting=csv.QUOTE_NONE)

# clean reviews
clean_test_reviews = test_df.review.map(tokenize_review)

# get bag of words
test_data_features = vectorizer.transform(clean_test_reviews).toarray()

# predict with model
test_df['sentiment'] = forest.predict(test_data_features)

# write csv
output_file = project.get_output_name()

test_df.to_csv(output_file, \
               columns=['id', 'sentiment'], \
               index=False, \
               quoting=csv.QUOTE_NONE)

print("Wrote %s\n" % output_file)
