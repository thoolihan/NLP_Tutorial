import project
from review_processor import ReviewProcessor
import pandas as pd
import csv
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# load
train = pd.read_csv(project.labeled, header=0, delimiter="\t", quoting=csv.QUOTE_NONE)
unlabeled = pd.read_csv(project.unlabeled, header=0, delimiter="\t", quoting=csv.QUOTE_NONE)

rp = ReviewProcessor()

# clean all reviews
logging.info("Preparing labeled and unlabeled reviews...")
sentences = []

for review in train.review:
    sentences += rp.review_to_sentences(review)

for review in unlabeled.review:
    sentences += rp.review_to_sentences(review)

logging.info("Prepared %d sentences" % len(sentences))

# Set values for various parameters
num_features = project.num_features    # Word vector dimensionality
min_word_count = 40   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
logging.info("Training Word2Vec Model...")
model = word2vec.Word2Vec(sentences, workers = num_workers, \
            size = num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

model.init_sims(replace = True)

model.save(project.w2v_model)
logging.info("Wrote %s" % project.w2v_model)
