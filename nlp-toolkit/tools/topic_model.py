"""Topic modeling."""
import logging
from typing import Sequence

import lightgbm as lgb
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import _preprocess_arabic_text, setup_logger

LOGGER = setup_logger('topic-models', logging.info)


class TopicModel:
    """
    Create a topic model.

    Filtering parts of speech is currently done using tools.Farasa.
    """

    def __init__(self, pos_to_use: Sequence[str], num_workers: int):
        """
        Initialize model.

        :param pos_to_use: Parts of speech to use, possible values are (Farasa-specific)
        ['S', 'E',
         'V', 'NOUN', 'PRON', 'ADJ', 'NUM',
         'CONJ', 'PART', 'NSUFF', 'CASE', 'FOREIGN',
         'DET', 'PREP', 'ABBREV', 'PUNC']

        :param num_workers: Number of worker to use for preprocessing and training.
        """
        self.pos_to_use = pos_to_use
        self.num_workers = num_workers

    def __preprocess(self):
        """Preprocess data."""
        pass

    def fit(self, X: Sequence[str], y: Sequence[str]):
        """Fit model."""
        pass


class CategoryModel:
    """
    Create a topic model.

    Documents are represented using their TF-IDF scores.
    Classifier is trained using LigthGBM.
    """

    def __init__(self, num_workers: int):
        """
        Initialize model.

        :param num_workers: Number of worker to use for preprocessing and training.
        """
        self.num_workers = num_workers
        self.vectorizer = TfidfVectorizer()

    def __preprocess_document(self, document: str) -> str:
        """Preprocess document."""
        return _preprocess_arabic_text(document)

    def fit(self, X: Sequence[str], y: Sequence[str]):
        """Fit model."""
        if not check_is_fitted(self.vectorizer, '_tfidf'):
            pass
