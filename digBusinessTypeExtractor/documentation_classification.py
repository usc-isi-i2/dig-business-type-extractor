# -*- coding: utf-8 -*-
# @Author: ZwEin
# @Date:   2016-10-06 23:36:45
# @Last Modified by:   ZwEin
# @Last Modified time: 2016-10-19 16:53:35
# -*- coding: utf-8 -*-
# @Author: ZwEin
# @Date:   2016-09-23 12:58:37
# @Last Modified by:   ZwEin
# @Last Modified time: 2016-09-24 20:29:20

import os
import sys
import csv
import numpy as np
from sklearn.externals import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from os.path import expanduser
import shutil


##################################################################
# Constant
##################################################################

STOP_WORDS = [u'all', u'just', u'being', u'over', u'both', u'through', u'yourselves', u'its', u'before', u'o', u'hadn', u'herself', u'll', u'had', u'should', u'to', u'only', u'won', u'under', u'ours', u'has', u'do', u'them', u'his', u'very', u'they', u'not', u'during', u'now', u'him', u'nor', u'd', u'did', u'didn', u'this', u'she', u'each', u'further', u'where', u'few', u'because', u'doing', u'some', u'hasn', u'are', u'our', u'ourselves', u'out', u'what', u'for', u'while', u're', u'does', u'above', u'between', u'mustn', u't', u'be', u'we', u'who', u'were', u'here', u'shouldn', u'hers', u'by', u'on', u'about', u'couldn', u'of', u'against', u's', u'isn', u'or', u'own', u'into', u'yourself', u'down', u'mightn', u'wasn', u'your', u'from', u'her', u'their', u'aren', u'there', u'been', u'whom', u'too', u'wouldn', u'themselves', u'weren', u'was', u'until', u'more', u'himself', u'that', u'but', u'don', u'with', u'than', u'those', u'he', u'me', u'myself', u'ma', u'these', u'up', u'will', u'below', u'ain', u'can', u'theirs', u'my', u'and', u've', u'then', u'is', u'am', u'it', u'doesn', u'an', u'as', u'itself', u'at', u'have', u'in', u'any', u'if', u'again', u'no', u'when', u'same', u'how', u'other', u'which', u'you', u'shan', u'needn', u'haven', u'after', u'most', u'such', u'why', u'a', u'off', u'i', u'm', u'yours', u'so', u'y', u'the', u'having', u'once']

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

USER_HOME_PATH = expanduser("~")

APP_DIR = os.path.join(USER_HOME_PATH, 'WEDC_DATA')
DATA_DIR = os.path.join(APP_DIR, 'data')
MODEL_DIR = os.path.join(APP_DIR, 'model')
MODEL_CLASSIFIER_DIR = os.path.join(MODEL_DIR, 'classifier')
MODEL_VECTORIZER_DIR = os.path.join(MODEL_DIR, 'vectorizer')

DC_DEFAULT_DATASET_PATH = os.path.join(DATA_DIR, 'dataset.csv')
DC_DEFAULT_CLASSIFIER_MODEL_PATH = os.path.join(MODEL_CLASSIFIER_DIR, 'wedc_classifier.pkl')
DC_DEFAULT_VECTORIZER_MODEL_PATH = os.path.join(MODEL_VECTORIZER_DIR, 'wedc_vectorizer.pkl')

def init_env():
    for path in [DATA_DIR, MODEL_CLASSIFIER_DIR, MODEL_VECTORIZER_DIR]:
        if not os.path.exists(path):
            os.makedirs(path)

    default_data_path = os.path.join(os.path.dirname(__file__), 'data', 'dataset.csv')
    if not os.path.exists(DC_DEFAULT_DATASET_PATH):
        shutil.copy(default_data_path, DC_DEFAULT_DATASET_PATH)
        
init_env()


class WEDC(object):

    ##################################################################
    # Basic Methods
    ##################################################################

    def __init__(self, data_path=DC_DEFAULT_DATASET_PATH, vectorizer_model_path=None, vectorizer_type='tfidf', classifier_model_path=None, classifier_type='knn', classifier_algorithm='brute', metrix='cosine'):
        self.corpus = []
        self.labels = []
        self.size = 0

        if data_path:
            new_corpus, new_labels = self.load_data(filepath=data_path)
            self.corpus += new_corpus
            self.labels += new_labels
            self.size += len(new_labels)

        self.vectorizer = self.load_vectorizer(handler_type=vectorizer_type, binary=True, stop_words=STOP_WORDS)
        self.classifier = self.load_classifier(handler_type=classifier_type, algorithm=classifier_algorithm, weights='distance', n_neighbors=5, metric=metrix)

        if not classifier_model_path:
            self.classifier_model_path = DC_DEFAULT_CLASSIFIER_MODEL_PATH

        if not vectorizer_model_path:
            self.vectorizer_model_path = DC_DEFAULT_VECTORIZER_MODEL_PATH

    def load_data(self, filepath=None):

        def clean_content(content):
            content = content.lower()

            # remove digits and punctuations
            content = ''.join([_ for _ in content if (_ >= 'a' and _ <= 'z') or (_ in ' \t')])
            
            # remove stopwords
            content = ' '.join([_ for _ in content.split() if _ not in STOP_WORDS])
            # print content
            return content

        dataset = []
        labels = []
        with open(filepath, 'rb') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            for row in reader:
                label =row[0]
                content = row[1].decode('utf-8', 'ignore').encode('ascii', 'ignore')
                dataset.append(clean_content(content))
                labels.append(label)
        return dataset, labels

    def load_vectorizer(self, handler_type='count', **kwargs):
        vectorizers = {
            'count': CountVectorizer(binary=kwargs.get('binary', True)),
            'tfidf': TfidfVectorizer(min_df=kwargs.get('min_df', 1), stop_words=kwargs.get('stop_words', STOP_WORDS))
        } 
        return vectorizers[handler_type]

    def load_classifier(self, handler_type='knn', **kwargs):
        classifiers = {
            'knn': KNeighborsClassifier( \
                        algorithm=kwargs.get('algorithm', 'auto'), \
                        weights=kwargs.get('weights', 'distance'), \
                        n_neighbors=kwargs.get('n_neighbors', 5), \
                        metric=kwargs.get('metric', 'jaccard'))
        }
        return classifiers[handler_type]


    def train(self, data_path=None):
        if data_path:
            new_corpus, new_labels = self.load_data(filepath=data_path)
            self.corpus += new_corpus
            self.labels += new_labels
            self.size += len(new_labels)

        # full matrix
        # vectors = self.vectorizer.fit_transform(self.corpus).toarray()
        # self.classifier.fit(vectors, self.labels)
        
        # sparse matrix
        vectors = self.vectorizer.fit_transform(self.corpus)
        self.classifier.fit(vectors, self.labels)

        joblib.dump(self.classifier, self.classifier_model_path) 
        joblib.dump(self.vectorizer, self.vectorizer_model_path) 
        
        return self.classifier, self.vectorizer

    def predict(self, data, classifier_model_path=None, vectorizer_model_path=None):
        if not data :
            return None
        if not classifier_model_path:
            classifier_model_path = self.classifier_model_path
        if not vectorizer_model_path:
            vectorizer_model_path = self.vectorizer_model_path

        data_corpus = [data] if isinstance(data, basestring) else data

        # load trained classifier and vectorizer models
        classifier = vectorizer = None
        try:
            classifier = joblib.load(classifier_model_path) 
            vectorizer = joblib.load(vectorizer_model_path)
        except:
            classifier, vectorizer = self.train()
            
        data_corpus = vectorizer.transform(data_corpus).toarray()
        
        return [DC_CATEGORY_NAMES[int(i)] for i in classifier.predict(data_corpus)]

    def evaluate(self, n_iter=1, test_size=.25, random_state=12):
        from sklearn import cross_validation
        from sklearn.metrics import classification_report

        rs = cross_validation.ShuffleSplit(self.size, n_iter=n_iter, test_size=test_size, random_state=random_state)

        vectors = self.vectorizer.fit_transform(self.corpus).toarray()
        
        for train_index, test_index in rs:
            train_X = [vectors[i] for i in range(self.size) if i in train_index]
            train_y = [self.labels[i] for i in range(self.size) if i in train_index]

            test_origin = [self.corpus[i] for i in range(self.size) if i in test_index]
            test_X = [vectors[i] for i in range(self.size) if i in test_index]
            text_y = [self.labels[i] for i in range(self.size) if i in test_index]

            self.classifier.fit(train_X, train_y)

            pred_y = self.classifier.predict(test_X)

            target_names = ['massage', 'escort', 'job_ads']
            print classification_report(text_y, pred_y, target_names=target_names)    
