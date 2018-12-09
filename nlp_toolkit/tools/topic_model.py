"""Topic modeling."""
import logging
from collections import Counter
from operator import itemgetter as get
from multiprocessing import Pool
from typing import Any, Dict, Generator, List, Sequence, Set, Tuple, Union, Optional

import nltk
from functional import seq
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore, Phrases
from gensim.models.phrases import Phraser
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from .farasa import Farasa
from .utils import _preprocess_arabic_text, setup_logger

LOGGER = setup_logger('topic-models', logging.DEBUG)


class TopicModel:
    """
    Create a topic model.

    Filtering parts of speech is currently done using tools.Farasa.
    """

    def __init__(self, pos_to_use: Sequence[str], stop_words: Set[str], min_df: Union[int, float] = 5,
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
        nltk.download('punkt')

        self.pos_to_use = pos_to_use
        self.num_workers = num_workers

        self.min_df = min_df
        self.max_df = max_df

        self.stop_words = stop_words

        self.vectorizer = TfidfVectorizer(stop_words=stop_words, max_df=self.max_df)

        self.bigram_model: Optional[Phraser] = None
        self.trigram_model: Optional[Phraser] = None
        self.id2word: Optional[Dict] = None

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

    def tokenize(self, document: str) -> List[str]:
        """
        Tokenize a document.

        Uses NLTK word tokenizer.
        """
        tokens = word_tokenize(document)
        return [
            token for token in tokens if token not in self.stop_words
        ]

    def create_trigrams(self, tokens: List[str]):
        """
        Create trigrams.

        :param tokens: list of tokens.
        :returns: n-gram where n is between 1-3.
        """
        if self.trigram_model and self.bigram_model:
            return self.trigram_model[self.bigram_model[tokens]]

        raise ValueError('trigram model is not fitted yet!')

    def build_vocab(self, documents_tokens: List[List[str]]) -> Tuple[List[List[str]], Dict]:
        """
        Build vocabualry.

        :param documents_tokens: documents as list of tokens, e.g. [
            ['the', 'brown', 'fox'],
            ['another', 'word', ..],
            ...
        ]

        :returns: a tuple consisting of list of documents as word counts (Bag-of-words), 
        and Id2Word dictionary.
        """
        bigram = Phrases(documents_tokens,
                         min_count=self.min_df,
                         threshold=100,
                         progress_per=100,
                         common_terms=self.stop_words)

        self.bigram_model = Phraser(bigram)
        self.trigram_model = Phraser(
            Phrases(bigram[documents_tokens], threshold=100)
        )

        documents_trigrams = []

        for index in range(len(documents_tokens) - 1, -1, -1):
            documents_trigrams.append(
                self.create_trigrams(documents_tokens[index])
            )
            documents_tokens.pop()

        id2word = Dictionary(documents_trigrams)
        return [id2word.doc2bow(text) for text in documents_trigrams], id2word

    def fit(self, documents: Sequence[str], passes: int, random_state: int, num_topics: int, chunksize: int = 1000):
        """
        Fit model.

        :param documents: documents to fit the model on.
        :param passes: number of passes over the training dataset, 1 is enough if dataset is large.
        :param random_state: random state seed for reproducibility.
        :param num_topics: number of topics.
        :param chunksize: number of document to use per update.
        """
        self.vectorizer = self.vectorizer.fit(documents)
        self.stop_words |= self.vectorizer.stop_words_

        corpus, self.id2word = self.build_vocab(
            [self.tokenize(x) for x in documents]
        )

        self._lda_model = LdaMulticore(corpus=corpus,
                                       id2word=self.id2word,
                                       num_topics=num_topics,
                                       random_state=random_state,
                                       chunksize=chunksize,
                                       passes=passes,
                                       per_word_topics=True, workers=self.num_workers, )

        self.topics = self._lda_model.print_topics(num_topics=num_topics, num_words=100)

    def predict(self, document, topics_map: Dict[int, str], num_topics: int) -> List[str]:
        """
        Predict topics distribution for a document.

        :params document: document to predict topics for.
        :params topics_map: a mapping of topic number to topic name.
        :params num_topics: return the top num_topics.
        :returns: a list of topic numbers sorted by their probabilities.
        """
        tokens = (seq([document])
                  .map(self.preprocess_document)
                  .map(self.tokenize)
                  .map(self.create_trigrams)
                  .flat_map(self.id2word.doc2bow)
                  .to_list()
                  )

        topics = (
            seq(self._lda_model[tokens][0])
            .sorted(key=lambda x: -x[1])
            .map(get(0))
            .take(num_topics)
        )

        if topics_map:
            topics = topics.map(lambda topic: topics_map[topic])

        return topics.to_list()


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
