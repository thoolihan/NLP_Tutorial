import project
import pandas as pd
import csv
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords

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
print(tokenize_review(train.review[0]))
