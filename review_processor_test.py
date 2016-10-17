import project
import unittest
from review_processor import ReviewProcessor
from gensim.models import Word2Vec

class ReviewProcessorTestCase(unittest.TestCase):
    def setUp(self):
        self.example = "The Brown Fox"
        self.review_example = "It was a great movie. You should see it! It does not star Tom Cruise."
        self.rp = ReviewProcessor()
        self.model = Word2Vec.load(project.w2v_model)


    def test_tokenize_review(self):
        self.assertEqual(self.rp.tokenize_review(self.example), \
                          ["the","brown", "fox"], \
                          "did not tokenize correctly")

    def test_review_to_sentences(self):
        self.assertEqual(self.rp.review_to_sentences(self.review_example), \
                         [['it', 'was', 'a', 'great', 'movie'], \
                          ['you', 'should', 'see', 'it'], \
                          ['it', 'does', 'not', 'star', 'tom', 'cruise']])

    def test_makeFeatureVec(self):
        reviewVec = self.rp.makeFeatureVec(self.review_example, self.model, project.num_features)
        self.assertEqual(reviewVec.shape, (300,))

    def test_getAvgFeatureVec(self):
        reviewVec = self.rp.getAvgFeatureVecs([self.review_example, self.review_example], \
                                              self.model, \
                                              project.num_features)
        # review collection is num of reviews x num features
        self.assertEqual(reviewVec.shape, (2, project.num_features))
        # each doc is exactly the same (since we passed in the same review twice)
        for i in range(0, project.num_features):
            self.assertEqual(reviewVec[0, i], reviewVec[1, i])
