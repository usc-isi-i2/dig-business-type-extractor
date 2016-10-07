# -*- coding: utf-8 -*-
# @Author: ZwEin
# @Date:   2016-10-06 23:36:45
# @Last Modified by:   ZwEin
# @Last Modified time: 2016-10-07 00:44:15
# -*- coding: utf-8 -*-
# @Author: ZwEin
# @Date:   2016-09-23 12:58:37
# @Last Modified by:   ZwEin
# @Last Modified time: 2016-09-24 20:29:20

import os
import sys
import csv
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


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

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

class WEDC(object):

    ##################################################################
    # Basic Methods
    ##################################################################

    def __init__(self, data_path=os.path.join(DATA_DIR, 'dataset.csv'), vectorizer_type='count', classifier_type='knn'):
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

    def predict(self, test_data):
        if not test_data :
            return None

        train_data_corpus, train_data_labels = self.corpus, self.labels
        test_data_corpus = [test_data] if isinstance(test_data, basestring) else test_data

        train_data_size = len(train_data_corpus)
        test_data_size = len(test_data_corpus)

        corpus = train_data_corpus + test_data_corpus
        labels = train_data_labels
        size = train_data_size + test_data_size

        vectors = self.vectorizer.fit_transform(corpus).toarray()

        train_X = vectors[:train_data_size]
        train_y = labels
        test_X = vectors[train_data_size:]

        self.classifier.fit(train_X, train_y)
        pred_y = self.classifier.predict(test_X)
        pred_y = [DC_CATEGORY_NAMES[int(i)] for i in pred_y]
        return pred_y

        


if __name__ == '__main__':

    ## Train and test path provided separately
    # dc = WEDC(vectorizer_type='count', classifier_type='knn')
    # train_path = '/Users/ZwEin/job_works/StudentWork_USC-ISI/projects/dig-groundtruth-data/classification/training-data/dig_memex_eval_datasets.csv'
    # test_path = '/Users/ZwEin/job_works/StudentWork_USC-ISI/projects/dig-groundtruth-data/classification/testing-data/testing_data.csv'
    # dc.run(train_data_path=train_path, test_data_path=test_path)

    ## Only one data path provided
    
    # data_path = '/Users/ZwEin/job_works/StudentWork_USC-ISI/projects/dig-groundtruth-data/classification/testing-data/testing_data.csv'
    

    test_data = ["   DOWNTOWN NEW HOT ASIAN ANGEL 100% REAL PICS Sexy Face Young  - 19      - Click to save or unsave                                                    Posted:  4 months ago                                        Age:  19                                   Category:   Phoenix Escorts                    Hi, Guys,   My Name is Lydia, and I new to the area,   I am a Korean and Spanish mix college student.only paritime, I am 20yrs, 5'4, 34D-23-35.   You will enjoy our time together guaranteed. 100% me.. *I'm ""the REAL deal""* I'm a Sweet, FUN playmate that knows how to have a good time!   Never Say No!! Never Rush!!  Call or Txt: 929-272-7898, TXT: 647-687-7096, Wechat: aa5854660383                        (929) 272-7898                      |  929.272.7898                      |  929-272-7898                      |  (929)272-7898                      |  9292727898     Flag this ad     Hi, Guys, My Name is Lydia, and I new to the area, I am a Korean and Spanish mix college student.only paritime, I am 20yrs, 5'4, 34D-23-35. You will enjoy our time together guaranteed. 100% me.. *I'm ""the REAL deal""* I'm a Sweet, FUN playmate that knows how to have a good time! Never Say No!! Never Rush!! Call or Txt: 929-272-7898, TXT: 647-687-7096, Wechat: aa5854660383 DOWNTOWN NEW HOT ASIAN ANGEL 100% REAL PICS Sexy Face Young  - 19 DOWNTOWN NEW HOT ASIAN ANGEL 100% REAL PICS Sexy Face Young  - 19 - A Sexy Service.com"]


    # data_path = os.path.join(DATA_DIR, 'dataset.csv') # include all training and testing dataset
    dc = WEDC(vectorizer_type='count', classifier_type='knn')
    print dc.predict(test_data)


