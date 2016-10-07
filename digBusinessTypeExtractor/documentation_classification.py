# -*- coding: utf-8 -*-
# @Author: ZwEin
# @Date:   2016-10-06 23:36:45
# @Last Modified by:   ZwEin
# @Last Modified time: 2016-10-07 13:31:44
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

class WEDC(object):

    ##################################################################
    # Basic Methods
    ##################################################################

    def __init__(self, data_path=DC_DEFAULT_DATASET_PATH, classifier_model_path=None, vectorizer_model_path=None, vectorizer_type='count', classifier_type='knn'):
        self.corpus = []
        self.labels = []
        self.size = 0

        if data_path:
            new_corpus, new_labels = self.load_data(filepath=data_path)
            self.corpus += new_corpus
            self.labels += new_labels
            self.size += len(new_labels)

        self.vectorizer = self.load_vectorizer(handler_type=vectorizer_type, binary=True)
        self.classifier = self.load_classifier(handler_type=classifier_type, weights='distance', n_neighbors=5, metric='jaccard')

        if not classifier_model_path:
            self.classifier_model_path = DC_DEFAULT_CLASSIFIER_MODEL_PATH

        if not vectorizer_model_path:
            self.vectorizer_model_path = DC_DEFAULT_VECTORIZER_MODEL_PATH

    def load_data(self, filepath=None):
        dataset = []
        labels = []
        with open(filepath, 'rb') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            for row in reader:
                label =row[0]
                content = row[1].decode('utf-8', 'ignore').encode('ascii', 'ignore')
                dataset.append(content)
                labels.append(label)
        return dataset, labels

    def load_vectorizer(self, handler_type='count', **kwargs):
        vectorizers = {
            'count': CountVectorizer(binary=kwargs.get('binary', True)),
            'tfidf': TfidfVectorizer(min_df=kwargs.get('min_df', .25))
        } 
        return vectorizers[handler_type]

    def load_classifier(self, handler_type='knn', **kwargs):
        classifiers = {
            'knn': KNeighborsClassifier( \
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

        vectors = self.vectorizer.fit_transform(self.corpus).toarray()

        self.classifier.fit(vectors, self.labels)
        joblib.dump(self.classifier, self.classifier_model_path) 
        joblib.dump(self.vectorizer, self.vectorizer_model_path) 
        # DC_DEFAULT_VECTORIZER_MODEL_PATH
        
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

if __name__ == '__main__':

    test_data = ["   DOWNTOWN NEW HOT ASIAN ANGEL 100% REAL PICS Sexy Face Young  - 19      - Click to save or unsave                                                    Posted:  4 months ago                                        Age:  19                                   Category:   Phoenix Escorts                    Hi, Guys,   My Name is Lydia, and I new to the area,   I am a Korean and Spanish mix college student.only paritime, I am 20yrs, 5'4, 34D-23-35.   You will enjoy our time together guaranteed. 100% me.. *I'm ""the REAL deal""* I'm a Sweet, FUN playmate that knows how to have a good time!   Never Say No!! Never Rush!!  Call or Txt: 929-272-7898, TXT: 647-687-7096, Wechat: aa5854660383                        (929) 272-7898                      |  929.272.7898                      |  929-272-7898                      |  (929)272-7898                      |  9292727898     Flag this ad     Hi, Guys, My Name is Lydia, and I new to the area, I am a Korean and Spanish mix college student.only paritime, I am 20yrs, 5'4, 34D-23-35. You will enjoy our time together guaranteed. 100% me.. *I'm ""the REAL deal""* I'm a Sweet, FUN playmate that knows how to have a good time! Never Say No!! Never Rush!! Call or Txt: 929-272-7898, TXT: 647-687-7096, Wechat: aa5854660383 DOWNTOWN NEW HOT ASIAN ANGEL 100% REAL PICS Sexy Face Young  - 19 DOWNTOWN NEW HOT ASIAN ANGEL 100% REAL PICS Sexy Face Young  - 19 - A Sexy Service.com"]


    # data_path = os.path.join(DATA_DIR, 'dataset.csv') # include all training and testing dataset
    dc = WEDC(vectorizer_type='count', classifier_type='knn')
    print dc.predict(test_data)
    # print dc.train()
    # init_env()

