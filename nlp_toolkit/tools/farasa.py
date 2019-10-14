"""
Pythonic and thread-safe wrapper around Farasa.

Farasa is developed at QCRI and can be found at http://qatsdemo.cloudapp.net/farasa/
Paper can be found at http://www.aclweb.org/anthology/N16-3003
"""
import logging
from collections import defaultdict
from operator import concat, itemgetter
from threading import RLock
from typing import Dict, List, Optional, Tuple

from functional import seq
from py4j.java_gateway import GatewayParameters, JavaGateway, launch_gateway

import nlp_toolkit.dependencies as dependencies

from .utils import break_input_into_chuncks, setup_logger

LOGGER = setup_logger('farasa', logging.INFO)

FARASA_JARS = [
    dependencies.get_language_model_path('ner'),
    dependencies.get_language_model_path('pos'),
    dependencies.get_language_model_path('diacritizer')
]

CLASS_PATH = ':'.join(FARASA_JARS)


class Farasa:
    """
    Pythonic wrapper around Farasa.

    Supports Farasa Segmenter, POS and NER taggers.
    """

    SEGMENT_TYPES = ['S', 'E',
                     'V', 'NOUN', 'ADJ', 'NUM',
                     'CONJ', 'PART', 'NSUFF', 'CASE', 'FOREIGN',
                     'DET', 'PREP', 'ABBREV', 'PUNC']

    NER_TOKEN_TYPES = ['B-LOC', 'B-ORG', 'B-PERS',
                       'I-LOC', 'I-ORG', 'I-PERS']

    __instance: Optional['Farasa'] = None
    __global_lock: RLock = RLock()

    def __new__(cls, singelton: bool) -> 'Farasa':
        """
        Create a Farasa instance.

        :param singelton: whether to create a single shared instance of Farasa.
        """
        if singelton:
            with cls.__global_lock:
                return cls.__instance or super(Farasa, cls).__new__(cls)  # type: ignore

        return super(Farasa, cls).__new__(cls)  # type: ignore

    def __init__(self, singelton: bool = True) -> None:
        """
        Initialize Farasa.

        :param singelton: whether to create a single shared instance of Farasa.
        """
        if not self.__class__.__instance or not singelton:
            self.gateway = self.__launch_java_gateway()

            base = self.gateway.jvm.com.qcri.farasa

            self.segmenter = base.segmenter.Farasa()
            self.pos_tagger = base.pos.FarasaPOSTagger(self.segmenter)
            self.ner = base.ner.ArabicNER(self.segmenter, self.pos_tagger)
            self.diacritizer = base.diacritize.DiacritizeText(self.segmenter, self.pos_tagger)

            if singelton:
                self.__class__.__instance = self
                self.__lock = self.__global_lock

            else:
                self.__lock = RLock()

            self.is_singelton = singelton

    @break_input_into_chuncks(concat=concat)
    def tag_pos(self, text: str) -> List[Tuple[str, str]]:
        """
        Tag part of speech.

        :param text: text to process.

        :returns: List of (token, token_type) pairs.
        """
        text = text.replace(';', ' ')  # to handle a bug in FARASA.

        result = []

        segments = self.segment(text)
        for segment in self.pos_tagger.tagLine(segments).clitics:
            result.append(
                (segment.surface, segment.guessPOS)
            )

        return result

    def merge_iffix(self, tags):
        """Merge iffix."""
        length = len(tags)

        for i in range(length):
            word, pos = tags[i]

            if word.startswith('+'):
                tags[i-1] = (tags[i-1][0] + word.replace('+', ''),
                             tags[i-1][1])

            elif word.endswith('+'):
                tags[i+1] = (word.replace('+', '') + tags[i+1][0],
                             tags[i+1][1])

        return tags

    @break_input_into_chuncks(concat=lambda x, y: x + ' ' + y)
    def filter_pos(self, text: str, parts_of_speech_to_keep: List[str]) -> str:
        """
        Break text into chuncks and then calls _filter_pos.

        :param text: text to process.
        :param parts_of_speech_to_keep: list of parts of speech to keep

        SEGMENT_TYPES = ['S', 'E',
                         'V', 'NOUN', 'PRON', 'ADJ', 'NUM',
                         'CONJ', 'PART', 'NSUFF', 'CASE', 'FOREIGN',
                         'DET', 'PREP', 'ABBREV', 'PUNC'].

        :returns: filtered text.
        """
        if 'VERB' in parts_of_speech_to_keep:
            parts_of_speech_to_keep = parts_of_speech_to_keep + ['V']

        pos = self.merge_iffix(self.tag_pos(text))
        return ' '.join(seq(pos)
                        .filter(lambda x: x[1] in parts_of_speech_to_keep and '+' not in x[0])
                        .map(itemgetter(0))
                        )

    @break_input_into_chuncks(concat=concat)
    def lemmetize(self, text: str) -> str:
        """
        Lemmetize text.

        :param text: text to process.
        """
        text = text.replace(';', ' ')  # to handle a bug in FARASA.

        return ' '.join(self.segmenter.lemmatizeLine(text))

    @break_input_into_chuncks(concat=concat)
    def segment(self, text: str) -> List[str]:
        """
        Segment piece of text.

        :param text: text to process.

        :returns: Unaltered Farasa segmenter output.
        """
        text = text.replace(';', ' ')  # to handle a bug in FARASA.

        return self.segmenter.segmentLine(text)

    @break_input_into_chuncks(concat=concat)
    def _get_named_entities(self, text: str, lemmatize: bool) -> List[Tuple[str, str]]:
        """
        Get named entities.

        :param text: text to process.
        :param lemmatize: whether to lemmatize results.

        :returns: List of (token, token_type) pairs.
        """
        text = text.replace(';', ' ')  # to handle a bug in FARASA.

        tokens = (seq(self.ner.tagLine(text))
                  .map(lambda token: token.split('/'))
                  .filter(lambda token: token[1] in self.NER_TOKEN_TYPES)
                  )

        result: Dict[Tuple[int, str], List[str]] = defaultdict(list)
        entities: List[Tuple[str, str]] = []

        index = -1
        # Farasa returns named entities in IOB Style (Inside, Outside and Begninning).
        # Related Entities are grouped together.
        for token, info in tokens:
            position, token_type = info.split('-')

            if position == 'B':
                index += 1

            result[(index, token_type)].append(token)

        # Return NE as a name and type pairs, i.e. ('Egypt', 'LOC').
        for key in sorted(result.keys(), key=lambda value: value[0]):
            entity = ' '.join(result[key])

            if lemmatize:
                entity = self.lemmetize(entity)

            entities.append(
                (entity, key[1])
            )

        return seq(entities).to_list()

    def get_named_entities(self, text: str, lemmatize: bool = False) -> List[Tuple[str, str]]:
        """
        Wrap _get_named_entities.

        :param text: text to process.
        :param lemmatize: whether to lemmatize results.

        :returns: List of (token, token_type) pairs.
        """
        return seq(self._get_named_entities(text, lemmatize=lemmatize)).to_list()

    @break_input_into_chuncks(concat=lambda x, y: x + ' ' + y)
    def diacritize(self, text: str, keep_original_diacritics: bool = False) -> str:
        """
        Diacritize.

        :param text: text to process.
        :param keep_original_diacritics: whether to keep original diacritics.
        """
        raise NotImplementedError('This feature is currently disabled')
        return self.diacritizer.diacritize(text, keep_original_diacritics)

    @classmethod
    def __launch_java_gateway(cls) -> JavaGateway:
        """Launch java gateway."""
        LOGGER.info('Initializing Farasa..')

        port = launch_gateway(classpath=CLASS_PATH, die_on_exit=True)
        params = GatewayParameters(
            port=port, auto_convert=True, auto_field=True, eager_load=True
        )

        return JavaGateway(gateway_parameters=params)
