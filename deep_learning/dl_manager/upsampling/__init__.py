import typing

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


def upsample(conf, name, labels, keys, *features):
    upsampler_cls: typing.Type[AbstractUpSampler] = upsamplers[name]
    upsampler = upsampler_cls(conf, **conf.get('run.upsampler-params')[f'{name}.0'])
    return upsampler.upsample_to_majority(labels, keys, *features)
