import sys
import time
import os
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

from digExtractor.extractor import Extractor
from digExtractor.extractor_processor import ExtractorProcessor
from digBusinessTypeExtractor.business_type_extractor import BusinessTypeExtractor

class TestBusinessTypeExtractorMethods(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_business_type_extractor(self):
        doc = {'content': "  DOWNTOWN NEW HOT ASIAN ANGEL 100% REAL PICS Sexy Face Young  - 19      - Click to save or unsave                                                    Posted:  4 months ago                                        Age:  19                                   Category:   Phoenix Escorts                    Hi, Guys,   My Name is Lydia, and I new to the area,   I am a Korean and Spanish mix college student.only paritime, I am 20yrs, 5'4, 34D-23-35.   You will enjoy our time together guaranteed. 100% me.. *I'm ""the REAL deal""* I'm a Sweet, FUN playmate that knows how to have a good time!   Never Say No!! Never Rush!!  Call or Txt: 929-272-7898, TXT: 647-687-7096, Wechat: aa5854660383                        (929) 272-7898                      |  929.272.7898                      |  929-272-7898                      |  (929)272-7898                      |  9292727898     Flag this ad     Hi, Guys, My Name is Lydia, and I new to the area, I am a Korean and Spanish mix college student.only paritime, I am 20yrs, 5'4, 34D-23-35. You will enjoy our time together guaranteed. 100% me.. *I'm ""the REAL deal""* I'm a Sweet, FUN playmate that knows how to have a good time! Never Say No!! Never Rush!! Call or Txt: 929-272-7898, TXT: 647-687-7096, Wechat: aa5854660383 DOWNTOWN NEW HOT ASIAN ANGEL 100% REAL PICS Sexy Face Young  - 19 DOWNTOWN NEW HOT ASIAN ANGEL 100% REAL PICS Sexy Face Young  - 19 - A Sexy Service.com", 'b': 'world'}

        extractor = BusinessTypeExtractor().set_metadata({'extractor': 'business_type'})
        extractor_processor = ExtractorProcessor().set_input_fields(['content']).set_output_field('extracted').set_extractor(extractor)
        updated_doc = extractor_processor.extract(doc)
        self.assertEqual(updated_doc['extracted'][0]['value'], ['escort'])

if __name__ == '__main__':
    unittest.main()



