"""Setup module."""
from setuptools import setup, find_packages

setup(
    name='ar-nlp-toolkit',
    version='0.0.1',
    url='https://github.com/abdelrahman-t/nlp-toolkit',
    author='Abdelrahman Talaat',
    author_email='abdurrahman.talaat@gmail.com',
    description=('NLP toolkit for Arabic'),
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.6.0',
    install_requires=['py4j', 'fuzzywuzzy', 'python-levenshtein>=0.12', 'pyfunctional', 'wrapt'],
)
