"""Topic modeling."""
import logging
from collections import Counter
from multiprocessing import Pool
from typing import List, Sequence, Union, Generator, Tuple, Any

from functional import seq
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from .utils import _preprocess_arabic_text, setup_logger
from .farasa import Farasa

LOGGER = setup_logger('topic-models', logging.DEBUG)


class TopicModel:
    """
    Create a topic model.

    Filtering parts of speech is currently done using tools.Farasa.
    """

    def __init__(self, pos_to_use: Sequence[str], min_df: Union[int, float] = 5,
                 max_df: Union[int, float] = 0.85, num_workers: int = 1):
        """
        Initialize model.

        :param pos_to_use: Parts of speech to use, possible values are (Farasa-specific)
        ['S', 'E',
         'V', 'NOUN', 'PRON', 'ADJ', 'NUM',
         'CONJ', 'PART', 'NSUFF', 'CASE', 'FOREIGN',
         'DET', 'PREP', 'ABBREV', 'PUNC']

        :param max_df: When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).

        :param min_df: When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.

        :param num_workers: Number of worker to use for preprocessing and training.
        """
        self.pos_to_use = pos_to_use
        self.num_workers = num_workers

        self.min_df = min_df
        self.max_df = max_df

    @staticmethod
    def _init_pool():
        """
        Intialize pool.

        Ran only one.
        """
        global farasa
        farasa = Farasa()

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

    def _unit_of_work(self, pos_to_use: List[str], document: str) -> str:
        """Apply unit of work."""
        global farasa

        return farasa.filter_pos(  # type: ignore
            self.preprocess_document(document),
            pos_to_use
        )

    def preprocess_documents(self, documents: Sequence[str]) -> Generator[str, None, None]:
        """
        Preprocess documents.

        :param documents: documents to preprocess.
        """
        progress = tqdm(total=len(documents))

        results = []
        pool = Pool(self.num_workers, initializer=self._init_pool)

        for document in documents:
            result = pool.apply_async(
                self._unit_of_work,
                (self.pos_to_use, document),
                callback=lambda *args: progress.update()
            )
            results.append(result)

        for result in results:
            document = result.get()

            if document != '':
                yield document

    def fit(self, documents: Sequence[str]):
        """Fit model."""
        raise NotImplementedError


class CategoryModel:
    """
    Create a category model.

    Documents are represented using their TF-IDF scores.
    Classifier is trained using LigthGBM.
    """

    def __init__(self, discard_categories: float, min_df: Union[int, float] = 5, max_df: Union[int, float] = 0.85):
        """
        Initialize model.

        :param discard_categories: keep only categories that are above this ratio.

        :param max_df: When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).

        :param min_df: When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.
        """
        self.discard_categories = discard_categories

        self.min_df = min_df
        self.max_df = max_df

        self.vectorizer = TfidfVectorizer(min_df=self.min_df, max_df=self.max_df)

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

    def preprocess_corpus(self, X: Sequence[str], y: Sequence[str]) -> Generator[Tuple[str, Any], None, None]:
        """
        Preprocess corpus.

        :param X: training data.
        :param y: labels.
        """
        y_counts = {key: value / len(y) for key, value in Counter(y).items()}

        for document, label in seq(zip(X, y)).filter(lambda item: y_counts[item[1]] > self.discard_categories):
            yield self.preprocess_document(document), label

    def fit(self, X: Sequence[str], y: Sequence[str]):
        """
        Fit model.

        :param X: training data.
        :param y: labels.
        """
        _X, _y = [], []
        progress = tqdm(total=len(X))

        for document, label in filter(lambda item: item[0] != '', self.preprocess_corpus(X, y)):
            _X.append(document)
            _y.append(label)

            progress.update()

        self.vectorizer = self.vectorizer.fit(_X, _y)
