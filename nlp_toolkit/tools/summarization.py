"""Summarization."""
import logging
from typing import List

from gensim.summarization import keywords
from gensim.summarization.summarizer import summarize as _summarize

from .farasa import Farasa
from .utils import _preprocess_arabic_text, setup_logger

LOGGER = setup_logger('summarization', logging.DEBUG)

farasa = Farasa()


def summarize(text, ratio) -> str:
    """Summarize a piece of text."""
    try:
        summary = _summarize(
            _preprocess_arabic_text(text, remove_emails_urls_html=True),
            ratio
        )

    except ValueError:
        summary = text

    return summary or text


def extract_keywords(text: str, pos_filter: List[str]) -> List[str]:
    """Extract list of keywords from text."""
    text = _preprocess_arabic_text(text, remove_emails_urls_html=True)

    return keywords(
        farasa.filter_pos(text, keep=pos_filter),
        split=True, words=True
    )
