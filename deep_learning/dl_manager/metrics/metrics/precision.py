from .base import MetricSet, AbstractMetric


class Precision(AbstractMetric):

    @staticmethod
    def name() -> str:
        return 'precision'

    def set_params(self, **kwargs):
        super().set_params(**kwargs)

    def calculate_class(self, confusion: MetricSet) -> float:
        denominator = confusion.true_positives + confusion.false_positives
        if denominator == 0:
            return 0.0
        return confusion.true_positives / denominator

    def micro(self, classes: dict[str, MetricSet]) -> float:
        denominator = sum(
            m.true_positives + m.false_positives
            for m in classes.values()
        )
        if denominator == 0:
            return 0.0
        numerator = sum(m.true_positives for m in classes.values())
        return numerator / denominator
