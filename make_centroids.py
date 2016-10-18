import project
from review_processor import ReviewProcessor
import pandas as pd
import csv
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import logging
import time

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

start = time.time()

# load
#train = pd.read_csv(project.labeled, header=0, delimiter="\t", quoting=csv.QUOTE_NONE)
#test = pd.read_csv(project.test_data, header=0, delimiter="\t", quoting=csv.QUOTE_NONE)

embeddings = Word2Vec.load(project.w2v_model)

word_vectors = embeddings.syn0
num_clusters = int(word_vectors.shape[0] / 5)

logging.info("Creating clusters...")
kmeans_clustering = KMeans(n_clusters = num_clusters)
idx = kmeans_clustering.fit_predict(word_vectors)

word_centroid_map = dict(zip( embeddings.index2word, idx ))

for cluster in range(0,10):
    print("\nCluster %d" % cluster)
    words = []
    for k,v in word_centroid_map.items():
        if v == cluster:
            words.append(k)
    print(words)

pickle.dump(word_centroid_map, open(project.word_centroid_map_pickle_file, "wb"))
logging.info("Write %s" % project.word_centroid_map_pickle_file)

end = time.time()
elapsed = end - start
logging.info("Time taken for K Means clustering: %f seconds" % elapsed)
