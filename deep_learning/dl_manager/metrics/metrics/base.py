from __future__ import annotations

import abc
import math
import typing

from .confusion import MetricSet


metrics: list[typing.Type[AbstractMetric]] = []


class MetricMeta(abc.ABCMeta):

    def __new__(mcs, *args, **kwargs):
        cls = super().__new__(mcs, *args, **kwargs)
        metrics.append(cls)
        return cls


class AbstractMetric(metaclass=MetricMeta):

    @staticmethod
    @abc.abstractmethod
    def name() -> str:
        pass

    @abc.abstractmethod
    def set_params(self, **kwargs):
        if kwargs:
            names = ', '.join(kwargs)
            raise ValueError(
                f'Invalid parameters for metrics {self.__class__.__name__!r}: {names}'
            )

    @abc.abstractmethod
    def calculate_class(self, confusion: MetricSet) -> float:
        pass

    def macro(self, classes: dict[str, MetricSet]) -> float:
        return self.weighted(
            classes,
            {key: 1 / len(classes) for key in classes}
        )

    def weighted(self,
                 classes: dict[str, MetricSet],
                 weights: dict[str, float]) -> float:
        assert math.isclose(math.fsum(weights.values()), 1.0)
        return math.fsum(
            weights[cls] * self.calculate_class(m)
            for cls, m in classes.items()
        )

    @abc.abstractmethod
    def micro(self, classes: dict[str, MetricSet]) -> float:
        pass
