# -*- coding: utf-8 -*-
# @Author: ZwEin
# @Date:   2016-09-30 14:01:47
# @Last Modified by:   ZwEin
# @Last Modified time: 2016-10-07 13:36:14


from distutils.core import setup
from setuptools import Extension,find_packages
from os import path
from digBusinessTypeExtractor.documentation_classification import init_env

init_env()

setup(
    name = 'digBusinessTypeExtractor',
    version = '0.1.0',
    description = 'digBusinessTypeExtractor',
    author = 'Lingzhe Teng',
    author_email = 'zwein27@gmail.com',
    url = 'https://github.com/usc-isi-i2/dig-business-type-extractor',
    download_url = 'https://github.com/usc-isi-i2/dig-business-type-extractor',
    packages = find_packages(),
    package_data={'digBusinessTypeExtractor': ['data/*.csv']},
    keywords = ['business', 'type', 'extractor'],
    install_requires=['digExtractor', 'numpy', 'scipy', 'scikit-learn']
    )