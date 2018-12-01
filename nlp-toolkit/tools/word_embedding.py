"""Operations on word embeddings."""

import gensim.models


def load_embeddings(path: str) -> gensim.models.Word2Vec:
    """
    Load pre-trained word embeddings (Gensim).

    :param path: path to the pre-trained embeddings.
    """
    return gensim.models.Word2Vec.load(path)
