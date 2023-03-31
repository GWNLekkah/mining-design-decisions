import numpy

from . import feature_generators
from .model_io import InputEncoding


class Vector2Vector:

    def __init__(self):
        pass

    def forward_transform(self, m):
        return m

    def backward_transform(self, m):
        return m


class Matrix2Vector:

    def __init__(self, shape: tuple[int, int, int]):
        self.__shape = shape

    def forward_transform(self, m: numpy.array) -> numpy.array:
        x, y, z = self.__shape
        return m.reshape(x, y*z)

    def backward_transform(self, m: numpy.array) -> numpy.array:
        x, y, z = self.__shape
        return m.reshape(x, y, z)


class List2Vector:

    def __init__(self, sizes: tuple[int, ...]):
        self.__cum_sizes = []
        total = 0
        for s in sizes:
            total += s
            self.__cum_sizes.append(s)

    def forward_transform(self, m):
        return numpy.column_stack(m)

    def backward_transform(self, m):
        return numpy.split(m, self.__cum_sizes, axis=1)


class Features2Vector:

    def __init__(self, generators: list[str], features):
        self.__generators = generators
        self.__transform = None
        self.__converters = [
            Vector2Vector()
            if self._check_encoding(feature_generators.generators[g].input_encoding_type()) else
            Matrix2Vector(f[0].shape)
            for f, g in zip(features, self.__generators)
        ]

    @staticmethod
    def _check_encoding(e: InputEncoding) -> bool:
        match e:
            case InputEncoding.Vector:
                return True
            case InputEncoding.Matrix:
                return False
            case _ as x:
                raise ValueError(f'Cannot convert {x} encoded feature to vector')

    def forward_transform(self, features):
        data = [t.forward_transform(f) for f, t in zip(features, self.__converters)]
        self.__transform = List2Vector(tuple(len(d) for d in data))
        return self.__transform.forward_transform(data)

    def backward_transform(self, f):
        data = self.__transform.backward_transform(f)
        return [t.backward_transform(d) for t, d in zip(self.__converters, data)]
