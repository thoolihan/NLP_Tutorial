import project
from review_processor import ReviewProcessor
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

model = Word2Vec.load(project.w2v_model)
rp = ReviewProcessor()

logging.info("Create average feature vecs for training set")
clean_train_reviews = []
for review in train.review:
    clean_train_reviews.append(rp.tokenize_review(review, remove_stopwords=True))

trainDataVecs = rp.getAvgFeatureVecs(clean_train_reviews, model, project.num_features)

logging.info("Create average feature vecs for testing set")
clean_test_reviews = []
for review in test.review:
    clean_test_reviews.append(rp.tokenize_review(review, remove_stopwords=True))

testDataVecs = rp.getAvgFeatureVecs(clean_test_reviews, model, project.num_features)

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
output_file = project.get_output_name('forest-w2v-avg')

test.to_csv(output_file, \
               columns=['id', 'sentiment'], \
               index=False, \
               quoting=csv.QUOTE_NONE)

logging.info("Wrote %s\n" % output_file)
