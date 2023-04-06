from .base import MetricSet, AbstractMetric


class TruePositiveCount(AbstractMetric):

    @staticmethod
    def name() -> str:
        return 'true_positives'

    def set_params(self, **kwargs):
        super().set_params(**kwargs)

    def calculate_class(self, confusion: MetricSet) -> float:
        return confusion.true_positives

    def micro(self, classes: dict[str, MetricSet]) -> float:
        return sum(m.true_positives for m in classes.values())


class TrueNegativeCount(AbstractMetric):

    @staticmethod
    def name() -> str:
        return 'true_negatives'

    def set_params(self, **kwargs):
        super().set_params(**kwargs)

    def calculate_class(self, confusion: MetricSet) -> float:
        return confusion.true_negatives

    def micro(self, classes: dict[str, MetricSet]) -> float:
        return sum(m.true_negatives for m in classes.values())


class FalsePositiveCount(AbstractMetric):

    @staticmethod
    def name() -> str:
        return 'false_positives'

    def set_params(self, **kwargs):
        super().set_params(**kwargs)

    def calculate_class(self, confusion: MetricSet) -> float:
        return confusion.false_positives

    def micro(self, classes: dict[str, MetricSet]) -> float:
        return sum(m.false_positives for m in classes.values())


class FalseNegativeCount(AbstractMetric):

    @staticmethod
    def name() -> str:
        return 'false_negatives'

    def set_params(self, **kwargs):
        super().set_params(**kwargs)

    def calculate_class(self, confusion: MetricSet) -> float:
        return confusion.false_negatives

    def micro(self, classes: dict[str, MetricSet]) -> float:
        return sum(m.false_negatives for m in classes.values())
