"""Similarity."""
import logging
from typing import Generator, Iterable, List, Sequence, Set

import nltk
from nltk import word_tokenize
from tqdm import tqdm

from .utils import _preprocess_arabic_text, setup_logger
from .word_embedding import WordEmbedding

LOGGER = setup_logger('similarity', logging.DEBUG)

nltk.download('punkt')


class WMDSimilarityClustering:
    """
    Hierarchical clustering based on Word Mover's Distance.

    Based on implementation at https://github.com/src-d/wmd-relax

    Workflow:

    1. pair-wise distances are calculated to form a distance matrix.
    2. agglomerative clustering is then applied on the matrix to obtain cluster.
    """

    def __init__(self, stop_words: Set[str], word_embeddings: WordEmbedding) -> None:
        """
        Initialize model.

        :param stop_words: Set of stop words.
        :param word_embeddings: Pretrained word embeddings.
        """
        self.stop_words = stop_words
        self.word_embeddings = word_embeddings

    def preprocess_document(self, document: str) -> str:
        """
        Preprocess document.

        :param document: document to preprocess.
        """
        return _preprocess_arabic_text(document,
                                       remove_non_arabic=True,
                                       remove_punctuation=True,
                                       remove_numbers=True,
                                       remove_emails_urls_html=True,
                                       remove_hashtags_mentions=True)

    def preprocess_documents(self, documents: Sequence[str]) -> Generator[str, None, None]:
        """
        Preprocess documents.

        :param documents: documents to preprocess.
        """
        progress = tqdm(total=len(documents))

        LOGGER.info('Preprocessing documents..')
        for document in documents:

            if document != '':
                yield document

            progress.update()

        LOGGER.info('Preprocessing is done.')

    def tokenize(self, document: str) -> List[str]:
        """
        Tokenize a document.

        Uses NLTK word tokenizer.
        """
        tokens = word_tokenize(document)
        return [
            token for token in tokens if token not in self.stop_words
        ]

    def fit(self, documents: Sequence[str], preprocess: bool):
        """
        Fit model.

        :param documents: documents to fit the model on.
        :param preprocess: whether to preprocess documens before training the model.
        """
        documents_iter: Iterable = documents if not preprocess else self.preprocess_documents(documents)

        documents_tokens: List[List[str]] = []

        for document in documents_iter:
            documents_tokens.append(
                self.tokenize(document)
            )
