from .base import AbstractUpSampler
from .smote import SmoteUpSampler
from .synonyms import SynonymUpSampler
from .random import RandomUpSampler

_upsamplers = [
    SynonymUpSampler,
    SmoteUpSampler,
    RandomUpSampler
]

upsamplers = {cls.__name__: cls for cls in _upsamplers}
