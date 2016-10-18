import project
from review_processor import ReviewProcessor
import pandas as pd
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from gensim.models import Word2Vec
import pickle
import logging
import time

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

start = time.time()

# load
train = pd.read_csv(project.labeled, header=0, delimiter="\t", quoting=csv.QUOTE_NONE)
test = pd.read_csv(project.test_data, header=0, delimiter="\t", quoting=csv.QUOTE_NONE)

embeddings = Word2Vec.load(project.w2v_model)
rp = ReviewProcessor()

logging.info("Create vecs for train reviews")
clean_train_reviews = []
for review in train.review:
    clean_train_reviews.append(rp.tokenize_review(review, remove_stopwords=True))

logging.info("Create vecs for test reviews")
clean_test_reviews = []
for review in test.review:
    clean_test_reviews.append(rp.tokenize_review(review, remove_stopwords=True))

word_vectors = embeddings.syn0
num_clusters = int(word_vectors.shape[0] / 5)

logging.info("Loading clusters...")

word_centroid_map = pickle.load(open(project.word_centroid_map_pickle_file, "rb"))

def create_bag_of_centroids(wordlist, word_centroid_map = word_centroid_map):
    num_centroids = max(word_centroid_map.values()) + 1
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1.
    return bag_of_centroids

train_centroids = np.zeros((train.review.size, num_clusters), dtype = "float32")
test_centroids = np.zeros((test.review.size, num_clusters), dtype = "float32")

logging.info("Creating bag of centroids for train set")
counter = 0
for review in clean_train_reviews:
    train_centroids[counter] = create_bag_of_centroids(review)
    counter += 1

logging.info("Creating bag of centroids for test set")
counter = 0
for review in clean_test_reviews:
    test_centroids[counter] = create_bag_of_centroids(review)
    counter += 1

logging.info("Training model")
model = RandomForestClassifier(n_estimators = 100)
model = model.fit(train_centroids, train.sentiment)

logging.info("Predicting test set")
test['sentiment'] = model.predict(test_centroids)

output_file = project.get_output_name('forest-w2v-kmeans')

test.to_csv(output_file, \
               columns=['id', 'sentiment'], \
               index=False, \
               quoting=csv.QUOTE_NONE)

logging.info("Wrote %s\n" % output_file)


end = time.time()
elapsed = end - start
logging.info("Time taken to build model on cluster map: %f seconds" % elapsed)
