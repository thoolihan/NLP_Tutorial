import unittest
from review_processor import ReviewProcessor

class ReviewProcessorTestCase(unittest.TestCase):
    def setUp(self):
        self.example = "The Brown Fox"
        self.rp = ReviewProcessor()

    def test_tokenize_review(self):
        self.assertEqual(self.rp.tokenize_review(self.example), \
                          ["the","brown", "fox"], \
                          "did not tokenize correctly")
