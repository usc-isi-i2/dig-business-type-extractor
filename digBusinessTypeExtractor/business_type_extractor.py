# -*- coding: utf-8 -*-
# @Author: ZwEin
# @Date:   2016-10-03 23:46:09
# @Last Modified by:   ZwEin
# @Last Modified time: 2016-10-19 17:20:34

import copy
from digExtractor.extractor import Extractor
from documentation_classification import WEDC


class BusinessTypeExtractor(Extractor):

    def __init__(self):
        self.renamed_input_fields = ['text']

    def extract(self, doc):
        if 'text' in doc:
            wedc = self.get_document_classifier()
            return wedc.predict(doc['text'])
        return None

    def get_metadata(self):
        return copy.copy(self.metadata)

    def set_metadata(self, metadata):
        self.metadata = metadata
        return self

    def get_renamed_input_fields(self):
        return self.renamed_input_fields

    def get_document_classifier(self):
        if not hasattr(self, 'document_classifier') or\
           not self.document_classifier:
            self.document_classifier = WEDC()
        return self.document_classifier

    def set_document_classifier(self, document_classifier):
        self.document_classifier = document_classifier
        return self
