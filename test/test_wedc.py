import unittest

from digBusinessTypeExtractor.documentation_classification\
    import WEDC, DC_DEFAULT_DATASET_PATH


class TestWEDCMethods(unittest.TestCase):

    def setUp(self):
        self.tfidfdc = WEDC(data_path=DC_DEFAULT_DATASET_PATH,
                            vectorizer_type='tfidf', classifier_type='knn',
                            classifier_algorithm='brute', metrix='cosine')
        self.tfidfdc.train()
        self.countdc = WEDC(data_path=DC_DEFAULT_DATASET_PATH,
                            vectorizer_type='count', classifier_type='knn',
                            classifier_algorithm='auto', metrix='jaccard')
        self.countdc.train()

    def tearDown(self):
        pass

    def test_evaluate(self):
        test_size = .4
        n_iter = 1
        random_state = None

        print 'count/jaccard'
        self.countdc.evaluate(n_iter=n_iter, test_size=test_size,
                              random_state=random_state)

        print 'tfidf/cosine'
        self.tfidfdc.evaluate(n_iter=n_iter, test_size=test_size,
                              random_state=random_state)


if __name__ == '__main__':
    unittest.main()
