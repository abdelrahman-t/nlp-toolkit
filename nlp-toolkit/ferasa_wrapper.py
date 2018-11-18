"""
Pythonic and thread-safe wrapper around Farasa.

Farasa is developed at QCRI and can be found at http://qatsdemo.cloudapp.net/farasa/
Paper can be found at http://www.aclweb.org/anthology/N16-3003
"""
import logging
import os.path
from functools import partial
from collections import defaultdict
from operator import itemgetter
from typing import List, Tuple, Dict

from functional import seq
from fuzzywuzzy import process
from py4j.java_gateway import GatewayParameters, JavaGateway, launch_gateway

import utils
from utils import preprorcess_text

LOGGER = utils.setup_logger('farasa', logging.INFO)

FILE_PATH = os.path.dirname(__file__)
FARASA_JARS = [
    os.path.join(FILE_PATH, 'Farasa/NER/NER.jar'),
    os.path.join(FILE_PATH, 'Farasa/POS/POS.jar'),
]

CACHE_SIZE = 100

if not seq(FARASA_JARS).map(os.path.isfile).all():
    raise FileNotFoundError(
        'could not locate Farasa .jar files, %s are required' % FARASA_JARS
    )

CLASS_PATH = ':'.join(FARASA_JARS)


class Farasa:
    """
    Pythonic wrapper around Farasa.

    Supports Farasa Segmenter, POS and NER taggers.
    """
    SEGMENT_TYPES = ['S', 'E',
                     'V', 'NOUN', 'PRON', 'ADJ', 'NUM',
                     'CONJ', 'PART', 'NSUFF', 'CASE', 'FOREIGN',
                     'DET', 'PREP', 'ABBREV', 'PUNC']

    NER_TOKEN_TYPES = ['B-LOC', 'B-ORG', 'B-PERS',
                       'I-LOC', 'I-ORG', 'I-PERS']

    def __init__(self) -> None:
        """Initialize Farasa."""
        self.gateway = self.__launch_java_gateway()

        base = self.gateway.jvm.com.qcri.farasa

        self.segmenter = base.segmenter.Farasa()
        self.pos_tagger = base.pos.FarasaPOSTagger(self.segmenter)
        self.ner = base.ner.ArabicNER(self.segmenter, self.pos_tagger)

    @preprorcess_text(remove_punct=False)
    def tag_pos(self, text: str) -> List[Tuple[str, str]]:
        """
        Tag part of speech.

        :param text: text to process.

        :returns: List of (token, token_type) pairs.
        """
        result = []

        segments = self.segment(text)
        for segment in self.pos_tagger.tagLine(segments).clitics:
            result.append(
                (segment.surface, segment.guessPOS)
            )

        return result

    def filter_pos(self, text: str, keep: List[str]) -> str:
        """
        Filter parts of speech

        :param text: text to process.
        :param keep: list of parts of speech to keep.

        :returns: filtered text.
        """
        pos = self.tag_pos(text)
        get_match = partial(process.extractOne, choices=text.split())

        return ' '.join(seq(pos)
                        .filter(lambda x: x[1] in keep and '+' not in x[1])
                        .map(itemgetter(0))
                        .map(lambda word: get_match(word)[0])
                        .to_list()
                        )

    @preprorcess_text(remove_punct=False)
    def segment(self, text: str) -> List[str]:
        """
        Segment piece of text.

        :param text: text to process.

        :returns: Unaltered Farasa segmenter output.
        """
        return self.segmenter.segmentLine(text)

    @preprorcess_text(remove_punct=False)
    def get_named_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Get named entities.

        :param text: text to process.

        :returns: List of (token, token_type) pairs.
        """
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
            entities.append(
                (' '.join(result[key]), key[1])
            )

        # Keep distinct NE ONLY.
        return seq(entities).distinct().to_list()

    @staticmethod
    def __launch_java_gateway() -> JavaGateway:
        """Launch java gateway."""
        port = launch_gateway(classpath=CLASS_PATH, die_on_exit=True)
        params = GatewayParameters(
            port=port, auto_convert=True, auto_field=True, eager_load=True
        )

        return JavaGateway(gateway_parameters=params)
