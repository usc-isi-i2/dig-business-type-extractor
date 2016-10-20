import sys
import time
import os
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

from digExtractor.extractor import Extractor
from digExtractor.extractor_processor import ExtractorProcessor
from digBusinessTypeExtractor.business_type_extractor import BusinessTypeExtractor
from digBusinessTypeExtractor.documentation_classification import WEDC, DC_DEFAULT_DATASET_PATH

class TestWEDCMethods(unittest.TestCase):

    def setUp(self):
        dc = WEDC(data_path=DC_DEFAULT_DATASET_PATH, vectorizer_model_path=None, vectorizer_type='tfidf', classifier_model_path=None, classifier_type='knn', classifier_algorithm='brute', metrix='cosine')
        dc.train(model_saved=True)

    def tearDown(self):
        pass

    def test_evaluate(self):
        test_size = .4
        n_iter = 1
        random_state = None

        print 'count/jaccard'
        dc = WEDC(data_path=DC_DEFAULT_DATASET_PATH, vectorizer_model_path=None, vectorizer_type='count', classifier_model_path=None, classifier_type='knn', classifier_algorithm='auto', metrix='jaccard')
        dc.evaluate(n_iter=n_iter, test_size=test_size, random_state=random_state)

        print 'tfidf/cosine'
        dc = WEDC(data_path=DC_DEFAULT_DATASET_PATH, vectorizer_model_path=None, vectorizer_type='tfidf', classifier_model_path=None, classifier_type='knn', classifier_algorithm='brute', metrix='cosine')
        dc.evaluate(n_iter=n_iter, test_size=test_size, random_state=random_state)
    

if __name__ == '__main__':
    unittest.main()



