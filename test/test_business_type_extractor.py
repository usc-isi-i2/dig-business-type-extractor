import unittest

from digExtractor.extractor_processor import ExtractorProcessor
from digBusinessTypeExtractor.business_type_extractor\
    import BusinessTypeExtractor


class TestBusinessTypeExtractorMethods(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_business_type_extractor(self):
        doc = {'content': "  DOWNTOWN NEW HOT ASIAN ANGEL 100% REAL PICS Sexy Face Young  - 19      - Click to save or unsave                                                    Posted:  4 months ago                                        Age:  19                                   Category:   Phoenix Escorts                    Hi, Guys,   My Name is Lydia, and I new to the area,   I am a Korean and Spanish mix college student.only paritime, I am 20yrs, 5'4, 34D-23-35.   You will enjoy our time together guaranteed. 100% me.. *I'm ""the REAL deal""* I'm a Sweet, FUN playmate that knows how to have a good time!   Never Say No!! Never Rush!!  Call or Txt: 929-272-7898, TXT: 647-687-7096, Wechat: aa5854660383                        (929) 272-7898                      |  929.272.7898                      |  929-272-7898                      |  (929)272-7898                      |  9292727898     Flag this ad     Hi, Guys, My Name is Lydia, and I new to the area, I am a Korean and Spanish mix college student.only paritime, I am 20yrs, 5'4, 34D-23-35. You will enjoy our time together guaranteed. 100% me.. *I'm ""the REAL deal""* I'm a Sweet, FUN playmate that knows how to have a good time! Never Say No!! Never Rush!! Call or Txt: 929-272-7898, TXT: 647-687-7096, Wechat: aa5854660383 DOWNTOWN NEW HOT ASIAN ANGEL 100% REAL PICS Sexy Face Young  - 19 DOWNTOWN NEW HOT ASIAN ANGEL 100% REAL PICS Sexy Face Young  - 19 - A Sexy Service.com", 'b': 'world'}

        metadata = {'extractor': 'business_type'}
        extractor = BusinessTypeExtractor().set_metadata(metadata)
        ep = ExtractorProcessor().set_input_fields(['content'])\
                                 .set_output_field('extracted')\
                                 .set_extractor(extractor)
        updated_doc = ep.extract(doc)
        result = updated_doc['extracted'][0]['result']
        self.assertEqual(result[0]['value'], 'escort')


if __name__ == '__main__':
    unittest.main()
