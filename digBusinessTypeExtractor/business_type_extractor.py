# -*- coding: utf-8 -*-
# @Author: ZwEin
# @Date:   2016-10-03 23:46:09
# @Last Modified by:   ZwEin
# @Last Modified time: 2016-10-07 00:38:19

import copy 
import types
from digExtractor.extractor import Extractor
from documentation_classification import WEDC

class BusinessTypeExtractor(Extractor):

    def __init__(self):
        self.renamed_input_fields = ['text']
        
    def extract(self, doc):
        if 'text' in doc:
            dc = WEDC(vectorizer_type='count', classifier_type='knn')
            return dc.predict(doc['text'])
        return None

    def get_metadata(self):
        return copy.copy(self.metadata)

    def set_metadata(self, metadata):
        self.metadata = metadata
        return self

    def get_renamed_input_fields(self):
        return self.renamed_input_fields