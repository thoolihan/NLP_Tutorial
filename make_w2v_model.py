import project
import pandas as pd
import csv
from bs4 import BeautifulSoup
import re
import nltk
import nltk.data
from nltk.corpus import stopwords
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# load
train = pd.read_csv(project.labeled, header=0, delimiter="\t", quoting=csv.QUOTE_NONE)
unlabeled = pd.read_csv(project.unlabeled, header=0, delimiter="\t", quoting=csv.QUOTE_NONE)

# called review_to_words in kaggle tutorial
def tokenize_review(raw_review, stop_words = False):
    # remove markup
    review_text = BeautifulSoup(raw_review, "html.parser").get_text()

    # remove punctuation
    letters_only = re.sub("[^a-zA-Z0-9]", " ", review_text)

    # lowercase and split
    words = letters_only.lower()

    # split
    words = words.split()

    if stop_words:
        # create set of stopwords (faster than list)
        stops = set(stopwords.words("english"))

        # filter stops
        words = [w for w in words if not w in stops]

    # back to string
    return(words)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def review_to_sentences(review, tokenizer=tokenizer, stop_words = False):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw in raw_sentences:
        if len(raw) > 0:
            sentences.append(tokenize_review(raw, stop_words))
    return(sentences)

# clean all reviews
logging.info("Preparing labeled and unlabeled reviews...")
sentences = []

for review in train.review:
    sentences += review_to_sentences(review)

for review in unlabeled.review:
    sentences += review_to_sentences(review)

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
model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

model.init_sims(replace=True)

model.save(project.w2v_name)
logging.info("Wrote %s" % project.w2v_model)
