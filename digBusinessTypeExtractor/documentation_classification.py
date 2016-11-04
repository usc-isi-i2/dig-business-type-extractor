# -*- coding: utf-8 -*-
# @Author: ZwEin
# @Date:   2016-10-06 23:36:45
# @Last Modified by:   ZwEin
# @Last Modified time: 2016-10-19 17:54:04

import os
import csv
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


##################################################################
# Constant
##################################################################

STOP_WORDS = [u'all', u'just', u'being', u'over', u'both', u'through',
              u'yourselves', u'its', u'before', u'o', u'hadn', u'herself',
              u'll', u'had', u'should', u'to', u'only', u'won', u'under',
              u'ours', u'has', u'do', u'them', u'his', u'very', u'they',
              u'not', u'during', u'now', u'him', u'nor', u'd', u'did',
              u'didn', u'this', u'she', u'each', u'further', u'where',
              u'few', u'because', u'doing', u'some', u'hasn', u'are',
              u'our', u'ourselves', u'out', u'what', u'for', u'while',
              u're', u'does', u'above', u'between', u'mustn', u't', u'be',
              u'we', u'who', u'were', u'here', u'shouldn', u'hers', u'by',
              u'on', u'about', u'couldn', u'of', u'against', u's', u'isn',
              u'or', u'own', u'into', u'yourself', u'down', u'mightn',
              u'wasn', u'your', u'from', u'her', u'their', u'aren', u'there',
              u'been', u'whom', u'too', u'wouldn', u'themselves', u'weren',
              u'was', u'until', u'more', u'himself', u'that', u'but', u'don',
              u'with', u'than', u'those', u'he', u'me', u'myself', u'ma',
              u'these', u'up', u'will', u'below', u'ain', u'can', u'theirs',
              u'my', u'and', u've', u'then', u'is', u'am', u'it', u'doesn',
              u'an', u'as', u'itself', u'at', u'have', u'in', u'any', u'if',
              u'again', u'no', u'when', u'same', u'how', u'other', u'which',
              u'you', u'shan', u'needn', u'haven', u'after', u'most', u'such',
              u'why', u'a', u'off', u'i', u'm', u'yours', u'so', u'y', u'the',
              u'having', u'once']

DC_CATEGORY_NAMES = [
    'unknown',
    'others',
    'massage',
    'escort',
    'job_ads'
]
DC_CATEGORY_DICT = {
    'unknown': -1,
    'others': 1,
    'massage': 2,
    'escort': 3,
    'job_ads': 4
}

DC_DEFAULT_DATASET_PATH = os.path.join(os.path.dirname(__file__),
                                       'data/dataset.csv')
DEFAULT_WEDC_CLASSIFIER_FILENAME = 'wedc_classifier.pkl'
DEFAULT_WEDC_VECTORIZER_FILENAME = 'wedc_vectorizer.pkl'


class WEDC(object):

    ##################################################################
    # Basic Methods
    ##################################################################

    def __init__(self, data_path=DC_DEFAULT_DATASET_PATH,
                 vectorizer_model_path=None, classifier_model_path=None,
                 vectorizer_type='tfidf', classifier_type='knn',
                 classifier_algorithm='brute', metrix='cosine'):
        self.corpus = []
        self.labels = []
        self.size = 0

        if data_path:
            new_corpus, new_labels = self.load_data(filepath=data_path)
            self.corpus += new_corpus
            self.labels += new_labels
            self.size += len(new_labels)

        self.vectorizer_type = 'tfidf'
        self.classifier_type = 'knn'
        self.classifier_algorithm = 'brute'
        self.metrix = 'cosine'

        self.vectorizer = None
        self.classifier = None

        self.vectorizer_model_path = vectorizer_model_path
        self.classifier_model_path = classifier_model_path

    def set_vectorizer_model_path(self, vectorizer_model_path):
        self.vectorizer_model_path = vectorizer_model_path
        return vectorizer_model_path

    def set_classifier_model_path(self, classifier_model_path):
        self.classifier_model_path = classifier_model_path
        return classifier_model_path

    def load_data(self, filepath=None):

        def clean_content(content):
            content = content.lower()

            # remove digits and punctuations
            content = ''.join([_ for _ in content if (
                _ >= 'a' and _ <= 'z') or (_ in ' \t')])

            # remove stopwords
            # content = ' '.join([_ for _ in content.split() if _ not in STOP_WORDS])
            # print content
            return content

        dataset = []
        labels = []
        with open(filepath, 'rb') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            for row in reader:
                label = row[0]
                content = row[1].decode(
                    'utf-8', 'ignore').encode('ascii', 'ignore')
                dataset.append(clean_content(content))
                labels.append(label)
        return dataset, labels

    def load_vectorizer(self, handler_type='count', **kwargs):
        vectorizers = {
            'count': CountVectorizer(binary=kwargs.get('binary', True)),
            'tfidf': TfidfVectorizer(min_df=kwargs.get('min_df', 1),
                                     stop_words=kwargs.get('stop_words',
                                                           STOP_WORDS))
            #'tfidf': TfidfVectorizer(min_df=kwargs.get('min_df', 1))
        }
        return vectorizers[handler_type]

    def load_classifier(self, handler_type='knn', **kwargs):
        classifiers = {
            'knn': KNeighborsClassifier(
                algorithm=kwargs.get('algorithm', 'auto'),
                weights=kwargs.get('weights', 'distance'),
                n_neighbors=kwargs.get('n_neighbors', 5),
                metric=kwargs.get('metric', 'jaccard'))
        }
        return classifiers[handler_type]

    def init_classifier_and_vectorizer(self):
        if not self.vectorizer:
            self.vectorizer = self.load_vectorizer(handler_type=self.vectorizer_type,
                                                   binary=True, stop_words=STOP_WORDS)

        if not self.classifier:
            self.classifier = self.load_classifier(
                handler_type=self.classifier_type, algorithm=self.classifier_algorithm, weights='distance', n_neighbors=5, metric=self.metrix)

    def save_model(self, model, path):
        joblib.dump(model, path)

    def save(self):
        if not self.classifier:
            raise Exception("Cannot save because classifier does not exist")
        if not self.vectorizer:
            raise Exception("Cannot save because vectorizer does not exist")
        if not self.classifier_model_path:
            raise Exception("Cannot save because classifier model output path is not defined")
        if not self.vectorizer_model_path:
            raise Exception("Cannot save because vectorizer model output path is not defined")
        self.save_model(self.classifier, self.classifier_model_path)
        self.save_model(self.vectorizer, self.vectorizer_model_path)

    def train(self, data_path=None):
        if data_path:
            new_corpus, new_labels = self.load_data(filepath=data_path)
            self.corpus += new_corpus
            self.labels += new_labels
            self.size += len(new_labels)

        self.init_classifier_and_vectorizer()

        vectors = self.vectorizer.fit_transform(self.corpus)
        self.classifier.fit(vectors, self.labels)
        return self.classifier, self.vectorizer

    def predict(self, data, classifier_model_path=None,
                vectorizer_model_path=None):
        if not data:
            return None
        if not classifier_model_path:
            classifier_model_path = self.classifier_model_path
        if not vectorizer_model_path:
            vectorizer_model_path = self.vectorizer_model_path

        data_corpus = [data] if isinstance(data, basestring) else data

        if not self.classifier or not self.vectorizer:
            # load trained classifier and vectorizer models
            try:
                self.classifier = joblib.load(classifier_model_path)
                self.vectorizer = joblib.load(vectorizer_model_path)
            except:
                self.classifier, self.vectorizer = self.train()

        data_corpus = self.vectorizer.transform(data_corpus).toarray()

        return [DC_CATEGORY_NAMES[int(i)] for i in self.classifier.predict(data_corpus)]

    def evaluate(self, n_iter=1, test_size=.25, random_state=12):
        from sklearn import cross_validation
        from sklearn.metrics import classification_report

        self.init_classifier_and_vectorizer()

        rs = cross_validation.ShuffleSplit(self.size, n_iter=n_iter,
                                           test_size=test_size,
                                           random_state=random_state)

        vectors = self.vectorizer.fit_transform(self.corpus).toarray()

        for train_index, test_index in rs:
            train_X = [vectors[i]
                       for i in range(self.size) if i in train_index]
            train_y = [self.labels[i]
                       for i in range(self.size) if i in train_index]

            test_X = [vectors[i] for i in range(self.size) if i in test_index]
            text_y = [self.labels[i]
                      for i in range(self.size) if i in test_index]

            self.classifier.fit(train_X, train_y)

            pred_y = self.classifier.predict(test_X)

            target_names = ['massage', 'escort', 'job_ads']
            print classification_report(text_y, pred_y,
                                        target_names=target_names)
