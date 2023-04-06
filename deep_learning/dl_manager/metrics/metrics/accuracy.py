from .base import MetricSet, AbstractMetric


class Accuracy(AbstractMetric):

    @staticmethod
    def name() -> str:
        return 'accuracy'

    def set_params(self, **kwargs):
        super().set_params(**kwargs)

    def calculate_class(self, confusion: MetricSet) -> float:
        numerator = confusion.true_positives + confusion.true_negatives
        denominator = numerator + confusion.false_positives + confusion.false_negatives
        return numerator / denominator

    def micro(self, classes: dict[str, MetricSet]) -> float:
        tp = sum(m.true_positives for m in classes.values())
        tn = sum(m.true_negatives for m in classes.values())
        fp = sum(m.false_positives for m in classes.values())
        fn = sum(m.false_negatives for m in classes.values())
        return (tn + tp) / (tn + tp + fp + fn)
