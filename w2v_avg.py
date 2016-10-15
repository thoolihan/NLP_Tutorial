import project
import pandas as pd
import csv
from bs4 import BeautifulSoup
import re
import nltk
import nltk.data
from nltk.corpus import stopwords
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from gensim.models import Word2Vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# load
train = pd.read_csv(project.labeled, header=0, delimiter="\t", quoting=csv.QUOTE_NONE)
test = pd.read_csv(project.test_data, header=0, delimiter="\t", quoting=csv.QUOTE_NONE)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# called review_to_words in kaggle tutorial
def tokenize_review(raw_review, remove_stopwords = False):
    # remove markup
    review_text = BeautifulSoup(raw_review, "html.parser").get_text()

    # remove punctuation
    letters_only = re.sub("[^a-zA-Z0-9]", " ", review_text)

    # lowercase and split
    words = letters_only.lower()

    # split
    words = words.split()

    if remove_stopwords:
        # create set of stopwords (faster than list)
        stops = set(stopwords.words("english"))

        # filter stops
        words = [w for w in words if not w in stops]

    # back to string
    return(words)

def review_to_sentences(review, tokenizer=tokenizer, remove_stopwords = False):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw in raw_sentences:
        if len(raw) > 0:
            sentences.append(tokenize_review(raw, remove_stopwords))
    return(sentences)

def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0.
    index2word_set = set(model.index2word)

    for word in words:
        if word in index2word_set:
            nwords += 1
            featureVec = np.add(featureVec, model[word])

    return(np.divide(featureVec, nwords))

def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0.

    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")

    for review in reviews:
        if counter % 1000. == 0.:
            logging.info("Review %d of %d" % (counter, len(reviews)))
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter += 1

    return reviewFeatureVecs

model = Word2Vec.load(project.w2v_model)

logging.info("Create average feature vecs for training set")
clean_train_reviews = []
for review in train.review:
    clean_train_reviews.append(tokenize_review(review, remove_stopwords=True))

trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, project.num_features)

logging.info("Create average feature vecs for testing set")
clean_test_reviews = []
for review in test.review:
    clean_test_reviews.append(tokenize_review(review, remove_stopwords=True))

testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, project.num_features)

# create model
logging.info("Training random forest...")
param_grid = {'n_estimators': [95, 100, 105]}

forest = RandomForestClassifier()
model = GridSearchCV(forest, param_grid = param_grid)
model.fit(trainDataVecs, train.sentiment)

logging.info("Tuned hyperparameters:")
logging.info(model.best_params_)

logging.info("Create submission...\n")

# predict with model
test['sentiment'] = model.predict(testDataVecs)

# write csv
output_file = project.get_output_name()

test.to_csv(output_file, \
               columns=['id', 'sentiment'], \
               index=False, \
               quoting=csv.QUOTE_NONE)

logging.info("Wrote %s\n" % output_file)
