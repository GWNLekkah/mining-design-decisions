from .base import MetricSet, AbstractMetric


class FBetaScore(AbstractMetric):

    def __init__(self):
        self._beta = 1.0

    @staticmethod
    def name() -> str:
        return 'f_beta_score'

    def set_params(self, **kwargs):
        self._beta = float(kwargs.pop('beta', '1.0'))
        super().set_params(**kwargs)

    def calculate_class(self, confusion: MetricSet) -> float:
        return self._compute(
            confusion.true_positives,
            confusion.false_negatives,
            confusion.false_positives,
            self._beta
        )

    def micro(self, classes: dict[str, MetricSet]) -> float:
        return self._compute(
            tp=sum(m.true_positives for m in classes.values()),
            fn=sum(m.false_negatives for m in classes.values()),
            fp=sum(m.false_positives for m in classes.values()),
            beta=self._beta
        )

    @staticmethod
    def _compute(tp, fn, fp, beta) -> float:
        x = (1 + beta**2) * tp
        y = beta**2 * fn
        z = x + y + fp
        if z == 0:
            return 0.0
        return x / z


class F1Score(FBetaScore):

    def set_params(self, **kwargs):
        super().set_params(**kwargs)
        super().set_params(beta='1.0')

