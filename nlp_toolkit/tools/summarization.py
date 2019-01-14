"""Summarization."""
import logging
from typing import List, Tuple

from gensim.summarization import keywords as _extract_keywords
from gensim.summarization.summarizer import summarize as _summarize

from .farasa import Farasa
from .utils import _preprocess_arabic_text, setup_logger

LOGGER = setup_logger('summarization', logging.DEBUG)

farasa = Farasa()


def summarize(text, ratio) -> str:
    """
    Summarize a piece of text.

    :param text: text to summarize.
    :param ratio: summarization ratio, 1.0 to return the original text without summarization.
    """
    try:
        summary = _summarize(
            _preprocess_arabic_text(text, remove_emails_urls_html=True),
            ratio
        )

    except ValueError:
        summary = text

    return summary or text


def extract_keywords(text: str, pos_filter: List[str], top_n: int = None) -> List[str]:
    """Extract list of keywords from text."""
    text = _preprocess_arabic_text(text, remove_emails_urls_html=True)

    try:
        keywords = _extract_keywords(
            farasa.filter_pos(text, keep=pos_filter),
            split=True
        )

        return keywords[:top_n]

    except Exception as e:
        LOGGER.exception(e)
        return []


def extract_entities(text: str) -> List[Tuple[str, str]]:
    """
    Extract named entities in text.

    :param text: text to extract entities from.
    :returns: list of (entity, entity_type) pairs, e.g. [('egypt', 'LOC')]
    """
    try:
        text = _preprocess_arabic_text(text, remove_emails_urls_html=True)
        return farasa.get_named_entities(text)

    except Exception as e:
        LOGGER.exception(e)
        return []
