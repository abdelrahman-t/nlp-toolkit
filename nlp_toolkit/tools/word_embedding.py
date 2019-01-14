"""Operations on word embeddings."""
from typing import Generator, List, Optional, Set, Tuple

import numpy
from gensim.models import Word2Vec
from nltk import ngrams as _create_ngrams
from nltk import word_tokenize


class WordEmbedding:
    """Wrapper around gensim Word2vec."""

    def __init__(self, path: str) -> None:
        """
        Initialize word embeddings.

        :param path: path to pretrained word embeddings.
        """
        self._w2v: Word2Vec = Word2Vec.load(path)

    def get_word_vector(self, word: str) -> Optional[numpy.ndarray]:
        """
        Lookup a word in the trained embeddings.

        :param word: Word to lookup.
        """
        return self._w2v.wv.get(word, None)

    def create_ngrams(self, tokens: List[str], nrange: Tuple[int, int] = (1, 3)) -> Generator[str, None, None]:
        """
        Get ngrams.

        :param tokens:
        :param n:
        """
        for n in range(nrange[1], nrange[0] - 1, -1):
            for ngram in _create_ngrams(tokens, n):
                yield '_'.join(ngram)

    def create_valid_trigrams(self, text: str) -> Generator[str, None, None]:
        """
        Create trigrams.

        :param text: Text to extract trigrams from.
        """
        tokens = word_tokenize(text)
        coverage: Set = set()

        for ngram in filter(lambda token: token in self._w2v.wv,
                            self.create_ngrams(tokens, nrange=(1, 3))
                            ):

            parts = set(ngram.split('_'))

            if len(parts & coverage) != len(parts):
                coverage |= parts
                yield ngram

            if len(coverage) == len(tokens):
                break

    def encode_document(self, text: str) -> Generator[numpy.ndarray, None, None]:
        """
        Transform document into its ngram word vectors.

        :param text: text to encode.
        """
        tokens = self.create_valid_trigrams(text)

        yield from map(self.get_word_vector, tokens)

    def get_distance(self, document1: List[str], document2: List[str]) -> float:
        """Get distance between two documents using Word Mover's distance."""
        return self._w2v.wmdistance(document1, document2)
