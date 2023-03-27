from .generator import AbstractFeatureGenerator
from .generator import ParameterSpec
from ..model_io import OutputMode
from .word2vec_1D import Word2Vec1D
from .doc2vec import Doc2Vec
from .bow_frequency import BOWFrequency
from .bow_normalized import BOWNormalized
from .bert import Bert
from .tfidf import TfidfGenerator
from .metadata import Metadata
from .ontology_features import OntologyFeatures
from .auto_encoder import AutoEncoder
from .kate_auto_encoder import KateAutoEncoder
from .generator import FeatureEncoding

_generators = (
    Word2Vec1D,
    Doc2Vec,
    BOWFrequency,
    BOWNormalized,
    Bert,
    TfidfGenerator,
    Metadata,
    OntologyFeatures,
    AutoEncoder,
    KateAutoEncoder
)
generators = {
    cls.__name__: cls for cls in _generators
}
del _generators
