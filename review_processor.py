from bs4 import BeautifulSoup
import numpy as np
import re
import nltk
import nltk.data
from nltk.corpus import stopwords
import logging

class ReviewProcessor:
    def __init__(self, \
                 tokenizer=nltk.data.load('tokenizers/punkt/english.pickle'), \
                 stopwords = stopwords.words('english')):
        self.tokenizer = tokenizer
        self.stopwords = set(stopwords)

    # called review_to_words in kaggle tutorial
    def tokenize_review(self, raw_review, remove_stopwords = False, remove_numbers = False):
        # remove markup
        review_text = BeautifulSoup(raw_review, "html.parser").get_text()

        # remove punctuation
        if remove_numbers:
            letters_only = re.sub("[^a-zA-Z]", " ", review_text)
        else:
            letters_only = re.sub("[^a-zA-Z0-9]", " ", review_text)

        # lowercase and split
        words = letters_only.lower()

        # split
        words = words.split()

        if remove_stopwords:
            # filter stops
            words = [w for w in words if not w in self.stopwords]

        return(words)

    def review_to_sentences(self, review, remove_stopwords = False):
        raw_sentences = self.tokenizer.tokenize(review.strip())
        sentences = []
        for raw in raw_sentences:
            if len(raw) > 0:
                sentences.append(self.tokenize_review(raw, remove_stopwords))
        return(sentences)

    def makeFeatureVec(self, words, model, num_features):
        featureVec = np.zeros((num_features,), dtype="float32")
        nwords = 0.
        index2word_set = set(model.index2word)

        for word in words:
            if word in index2word_set:
                nwords += 1
                featureVec = np.add(featureVec, model[word])

        return(np.divide(featureVec, nwords))

    def getAvgFeatureVecs(self, reviews, model, num_features):
        counter = 0.

        reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")

        for review in reviews:
            if counter % 1000. == 0.:
                logging.info("Review %d of %d" % (counter, len(reviews)))
            reviewFeatureVecs[counter] = self.makeFeatureVec(review, model, num_features)
            counter += 1

        return reviewFeatureVecs
