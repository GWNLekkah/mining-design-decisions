from .base import MetricSet, AbstractMetric

class Specificity(AbstractMetric):

    @staticmethod
    def name() -> str:
        return 'specificity'

    def set_params(self, **kwargs):
        super().set_params(**kwargs)

    def calculate_class(self, confusion: MetricSet) -> float:
        denominator = confusion.true_negatives + confusion.false_positives
        if denominator == 0:
            return 0.0
        return confusion.true_negatives / denominator

    def micro(self, classes: dict[str, MetricSet]) -> float:
        denominator = sum(
            m.true_negatives + m.false_positives
            for m in classes.values()
        )
        if denominator == 0:
            return 0.0
        numerator = sum(m.true_negatives for m in classes.values())
        return numerator / denominator
