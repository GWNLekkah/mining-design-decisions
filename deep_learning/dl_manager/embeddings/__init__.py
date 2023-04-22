from .embedding_generator import AbstractEmbeddingGenerator
from .word2vec import Word2VecGenerator
from .doc2vec import Doc2VecGenerator

_gens = [
    Word2VecGenerator,
    Doc2VecGenerator
]

generators = {g.__name__: g for g in _gens}
