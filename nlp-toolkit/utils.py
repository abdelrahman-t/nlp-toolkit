"""Utils"""
import logging
import re
import string

import wrapt

RE_WHITESPACE = r'\s+'

RE_NUM = r'([0-9]+|[\u0660-\u0669]+)'
RE_HASHTAG = r'#([a-zA-Z0-9_]+|[\u0621-\u064A\u0660-\u0669\uFE70-\uFEFF0-9_]+)'
RE_MENTION = r'@([a-zA-Z0-9_]+|[\u0621-\u064A\u0660-\u0669\uFE70-\uFEFF0-9_]+)'

RE_URL = r'(?:https?://|www\.)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
RE_EMAIL = r'\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b'
RE_HTML = r'<.*?>'
RE_SPECIAL_SYMBOL = r'[^\u0600-\u06FF]+'

TOKENS = [
    RE_URL,
    RE_EMAIL,
    RE_HASHTAG,
    RE_MENTION,
    RE_NUM,
    RE_HTML,
    RE_SPECIAL_SYMBOL
]

IGNORED = [
    RE_WHITESPACE
]

RE_PATTERN = re.compile(r'|'.join(IGNORED) +
                        r'|(' + r'|'.join(TOKENS) + r')', re.UNICODE)


LONGATION = re.compile(r'(.)\1+')
TASHKEEL = re.compile(r'[\u0617-\u061A\u064B-\u0652]')

TRANSLATE_TABLE = dict((ord(char), None) for char in string.punctuation)


def remove_extra_spaces(text: str) -> str:
    """Remove extranious spaces."""
    template = [('\r', ' '), ('\n', ' '), ('\t', ' ')]

    for token, replacement in template:
        text.replace(token, replacement)

    return ' '.join(
        text.split()
    )


def _preprocess_text(text, remove_punct: bool = False) -> str:
    """
    Provide a Modified version of https://github.com/bakrianoo/aravec .

    :param text: text to preprocess.
    :returns: preprocessed text.
    """
    template = [
        ('أ', 'ا'),
        ('إ', 'ا'),
        ('آ', 'ا'),
        ('ة', 'ه'),
        ('_', ' '),
        ('-', ' '),
        ('/', ''),
        ('.', '.'),
        ('،', ','),
        (' و ', ' و'),
        (' يا ', ' يا'),
        ('"', ''),
        ('ـ', ''),
        ("'", ''),
        ('ى', 'ي'),
        ('\\', ''),
        ('\r', ''),
        ('\n', ' '),
        ('\t', ' '),
        ('&quot;', ' '),
        ('?', ' ? '),
        ('؟', ' ؟ '),
        ('!', ' ! '),
    ]

    text = re.sub(RE_PATTERN, ' ', text)
    text = re.sub(TASHKEEL, '', text)
    text = re.sub(LONGATION, r'\1\1', text)
    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')

    for token, replacement in template:
        text = text.replace(token, replacement)

    if remove_punct:
        text = text.translate(TRANSLATE_TABLE)

    return remove_extra_spaces(text)


def preprorcess_text(remove_punct: bool):
    """Preprocess text."""
    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):  # pylint: disable=unused-argument
        try:
            if not isinstance(args[0], str):
                raise TypeError

            text = _preprocess_text(args[0], remove_punct=remove_punct)

        except (IndexError, TypeError):
            raise TypeError('text must be a string!')

        return wrapped(text)
    return wrapper


def setup_logger(name: str, level) -> logging.Logger:
    """Create a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    file_handler = logging.FileHandler(f'{name}.log')
    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        f'%(asctime)s [{name}] [%(levelname)-5.5s]  %(message)s')

    # set logging level.
    file_handler.setLevel(level)
    stream_handler.setLevel(level)

    # set formatter.
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # add file and stream handlers.
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
