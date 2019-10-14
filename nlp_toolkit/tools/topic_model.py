"""Topic modeling."""
import logging
from functools import lru_cache, partial
from multiprocessing import Pool
from operator import itemgetter as get
from typing import (Dict, Generator, Iterable, List, Optional, Sequence, Set,
                    Tuple, Union)

import dill
from functional import seq
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore, Phrases
from gensim.models.phrases import Phraser
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

import nlp_toolkit.dependencies as dependencies

from .farasa import Farasa
from .pos import filter_pos
from .lemmatization import lemmatize
from .utils import _preprocess_arabic_text, setup_logger

LOGGER = setup_logger('topic-models', logging.DEBUG)


@lru_cache(maxsize=2)
def load_topic_model(model_id: int) -> 'TopicModel':
    """
    Load topic model.

    :param model_id: model id.
    :raises: ValueError if no model with specified model_id can not be found.
    """
    model_config = dependencies.get_topic_model_config(model_id)
    model_config['topic_names'] = {
        int(key): value for key, value in model_config['topic_names'].items()
    }

    model = TopicModel.load(model_config['path'])
    model.predict = partial(model.predict,  # type: ignore
                            topics_map=model_config['topic_names'],
                            num_topics=model_config['predict_top'])

    return model


def infer_topic(model_id: int, document: str):
    """Infer topic."""
    return load_topic_model(model_id).predict(document)  # type: ignore


class TopicModel:
    """
    Create a topic model.

    Filtering parts of speech is currently done using tools.Farasa.
    """

    def __init__(self, pos_to_use: List[str], stop_words: Union[Set[str], List[str], str],
                 min_df: Union[int, float] = 5, max_df: Union[int, float] = 0.85, num_workers: int = 1):
        """
        Initialize model.

        :param pos_to_use: Parts of speech to use, possible values are (Farasa-specific)
        ['S', 'E',
         'V', 'NOUN', 'PRON', 'ADJ', 'NUM',
         'CONJ', 'PART', 'NSUFF', 'CASE', 'FOREIGN',
         'DET', 'PREP', 'ABBREV', 'PUNC']

        :param stop_words: list/set of stop words or filepath to the file containing the stop words.

        :param max_df: When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).

        :param min_df: When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.

        :param num_workers: Number of worker to use for preprocessing and training.
        """
        if isinstance(stop_words, str):
            stop_words = open(stop_words).read().split('\n')

        if isinstance(stop_words, list):
            stop_words = set(stop_words)

        self.pos_to_use = pos_to_use
        self.num_workers = num_workers

        self.min_df = min_df
        self.max_df = max_df

        self.stop_words = stop_words

        self.vectorizer = TfidfVectorizer(stop_words=stop_words, max_df=self.max_df)

        self.bigram_model: Optional[Phraser] = None
        self.trigram_model: Optional[Phraser] = None
        self.id2word: Optional[Dict] = None

        self._farasa: Optional[Farasa] = None

    @staticmethod
    def _init_pool():
        """
        Intialize pool.

        Ran only one.
        """
        global farasa
        farasa = Farasa(singelton=False)

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
        """
        Apply unit of work.

        :param pos_to_use: Parts of speech to keep.
        :param document: Document to process.
        """
        global farasa

        return farasa.filter_pos(  # type: ignore
            self.preprocess_document(document),
            parts_of_speech_to_keep=pos_to_use
        )

    def preprocess_documents(self, documents: Sequence[str]) -> Generator[str, None, None]:
        """
        Preprocess documents.

        :param documents: documents to preprocess.
        """
        progress = tqdm(total=len(documents))

        LOGGER.info('Launching %d workers..', self.num_workers)
        pool = Pool(self.num_workers, initializer=self._init_pool)

        LOGGER.info('Preprocessing documents using %d workers..', self.num_workers)

        results = []
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

    def create_trigrams(self, tokens: List[str]) -> List[str]:
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
        LOGGER.info('Fitting bigram model..')
        bigram = Phrases(documents_tokens,
                         min_count=self.min_df,
                         threshold=100,
                         progress_per=100,
                         common_terms=self.stop_words)

        self.bigram_model = Phraser(bigram)

        LOGGER.info('Fitting trigram model..')
        self.trigram_model = Phraser(
            Phrases(bigram[documents_tokens], threshold=100)
        )

        documents_trigrams = []

        LOGGER.info('Creating trigrams..')
        for index in range(len(documents_tokens) - 1, -1, -1):
            documents_trigrams.append(
                self.create_trigrams(documents_tokens[index])
            )
            documents_tokens.pop()

        id2word = Dictionary(documents_trigrams)
        return [id2word.doc2bow(text) for text in documents_trigrams], id2word

    def fit(self, documents: Sequence[str], preprocess: bool, passes: int, random_state: int, num_topics: int,
            chunksize: int = 1000):
        """
        Fit model.

        :param documents: documents to fit the model on.
        :param preprocess: whether to preprocess documens before training the model.
        :param passes: number of passes over the training dataset, 1 is enough if dataset is large.
        :param random_state: random state seed for reproducibility.
        :param num_topics: number of topics.
        :param chunksize: number of document to use per update.
        """
        self.vectorizer = self.vectorizer.fit(documents)
        self.stop_words |= self.vectorizer.stop_words_

        documents_iter: Iterable = documents if not preprocess else self.preprocess_documents(documents)

        LOGGER.info('Building vocab..')
        corpus, self.id2word = self.build_vocab(
            [self.tokenize(x) for x in documents_iter]
        )

        LOGGER.info('Fitting lda..')
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
                  .map(lemmatize)  # type: ignore
                  .map(self.tokenize)
                  .map(self.create_trigrams)
                  .flat_map(self.id2word.doc2bow)  # type: ignore
                  .to_list()
                  )

        topics = (
            seq(self._lda_model[tokens][0])
            .sorted(key=lambda x: -x[1])
            .map(get(0))
            .filter(None)
            .distinct()
            .take(num_topics)
        )

        if topics_map:
            topics = topics.map(lambda topic: topics_map[topic])

        return topics.to_list()

    @staticmethod
    def load(path: str) -> 'TopicModel':
        """
        Load model.

        :param path: path to the model.
        """
        return dill.load(open(path, 'rb'))

    def save(self, path: str):
        """
        Save model.

        :param path: path to save the model to.
        """
        farasa: Farasa = self.__dict__.pop('_farasa')
        dill.dump(self, open(path, 'wb'))

        self._farasa = farasa
