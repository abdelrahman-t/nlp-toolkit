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
RE_NON_ARABIC = r'[^\u0600-\u06FF]'

TWITTER = [
    RE_HASHTAG,
    RE_MENTION
]

WEB = [
    RE_URL,
    RE_EMAIL,
    RE_HTML,
]

IGNORED = [
    RE_WHITESPACE
]

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


def _preprocess_arabic_text(text,  # pylint: disable=too-many-arguments
                            remove_non_arabic: bool = False,
                            remove_punctuation: bool = False,
                            remove_numbers: bool = False,
                            remove_emails_urls_html: bool = False,
                            remove_hashtags_mentions: bool = False) -> str:
    """
    Provide a Modified version of https://github.com/bakrianoo/aravec .

    This function removes non-arabic characters, urls, emails, twitter hashtags and mentions among other things.

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
        ('\r', ' '),
        ('\n', ' '),
        ('\t', ' '),
        ('&quot;', ' '),
        ('?', ' ? '),
        ('؟', ' ؟ '),
        ('!', ' ! '),
    ]

    expressions = []

    if remove_non_arabic:
        expressions.append(RE_NON_ARABIC)

    if remove_numbers:
        expressions.append(RE_NUM)

    if remove_emails_urls_html:
        expressions.extend(WEB)

    if remove_hashtags_mentions:
        expressions.extend(TWITTER)

    if expressions:
        re_pattern = r'|'.join(IGNORED) + r'|(' + r'|'.join(expressions) + r')'  # type: ignore
        text = re.sub(re_pattern, ' ', text)

    text = re.sub(TASHKEEL, '', text)
    text = re.sub(LONGATION, r'\1\1', text)
    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')

    for token, replacement in template:
        text = text.replace(token, replacement)

    if remove_punctuation:
        text = text.translate(TRANSLATE_TABLE)

    return remove_extra_spaces(text)


def preprorcess_arabic_text(**kwargs):
    """Preprocess text."""
    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):  # pylint: disable=unused-argument
        try:
            if not isinstance(args[0], str):
                raise TypeError

            text = _preprocess_arabic_text(args[0], **kwargs)

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
