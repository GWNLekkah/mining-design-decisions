import typing

import numpy

from ...model_io import OutputMode, OutputEncoding
from . import confusion
from . import base


class MetricCalculationManager:

    def __init__(self,
                 y_pred,
                 y_true,
                 output_mode: OutputMode, *,
                 include_non_arch=False,
                 classification_as_detection=False):
        self._cache: dict[str, typing.Type[base.AbstractMetric]] = {}
        self._classification_as_detection = classification_as_detection
        self._include_non_arch = include_non_arch
        self._output_mode = output_mode
        y_pred_classes = self._convert_confidences(y_pred)
        y_true_converted = self._convert_ground_truths(y_true)
        if self._classification_as_detection:
            if output_mode == OutputMode.Detection:
                raise ValueError(
                    'Can only evaluate classification as '
                    'detection using classification predictions.'
                )
            self._y_pred = self._convert_labels_to_binary(y_pred_classes)
            self._y_true = self._convert_labels_to_binary(y_true_converted)
            self._global_acc = self._confusion = confusion.compute_confusion_binary(
                self._y_true, self._y_pred
            )
        else:
            self._y_pred = y_pred_classes
            self._y_true = y_true_converted
            self._global_acc, self._confusion = self._compute_confusion()
        self._weights = self._compute_weights()

    def _convert_confidences(self, y_pred: numpy.ndarray):
        match self._output_mode:
            case OutputMode.Detection:
                flattened = y_pred.flatten()
                flattened[flattened < 0.5] = 0
                flattened[flattened >= 0.5] = 1
                return flattened
            case OutputMode.Classification3:
                rounded = y_pred.copy()
                rounded[rounded < 0.5] = 0
                rounded[rounded >= 0.5] = 1
                return rounded
            case OutputMode.Classification3Simplified:
                return y_pred.argmax(axis=1)
            case OutputMode.Classification8:
                return y_pred.argmax(axis=1)

    def _convert_ground_truths(self, y_true: numpy.ndarray):
        match self._output_mode:
            case OutputMode.Detection:
                return y_true
            case OutputMode.Classification3:
                return y_true
            case OutputMode.Classification3Simplified:
                return y_true.argmax(axis=1)
            case OutputMode.Classification8:
                return y_true.argmax(axis=1)

    def _convert_labels_to_binary(self, y: numpy.ndarray):
        match self._output_mode:
            case OutputMode.Detection:
                return y
            case OutputMode.Classification3:
                return ~(y == self._output_mode.non_architectural_pattern).all(axis=0)
            case OutputMode.Classification3Simplified:
                return y != self._output_mode.non_architectural_pattern.index(1)
            case OutputMode.Classification8:
                return y != self._output_mode.non_architectural_pattern.index(1)

    def _compute_confusion(self):
        if self._classification_as_detection:
            raise ValueError(
                '_compute_confusion should not be called when '
                'evaluating classification as detection'
            )
        labels = self._output_mode.output_vector_field_names
        match self._output_mode:
            case OutputMode.Detection:
                return confusion.compute_confusion_binary(self._y_true,
                                                          self._y_pred)
            case OutputMode.Classification3:
                return confusion.compute_confusion_multi_class(self._y_true,
                                                               self._y_pred,
                                                               labels)
            case OutputMode.Classification3Simplified:
                return confusion.compute_confusion_multi_label(self._y_true,
                                                               self._y_pred,
                                                               labels)
            case OutputMode.Classification8:
                return confusion.compute_confusion_multi_class(self._y_true,
                                                               self._y_pred,
                                                               labels)

    def _compute_weights(self) -> dict[str, float] | None:
        if self._classification_as_detection or self._output_mode == OutputMode.Detection:
            return {
                cls: (self._y_true == key).sum().item()
                for key, cls in OutputMode.Detection.label_encoding.items()
            }
        if self._output_mode.output_encoding == OutputEncoding.OneHot:
            return {
                cls: (self._y_true == key.index(1)).sum().item()
                for key, cls in self._output_mode.output_encoding.items()
            }
        if self._output_mode == OutputMode.Classification3:
            return None
        raise NotImplementedError(self._output_mode)

    def calc_metric(self, name: str, category: str) -> float:
        if name == 'accuracy' and category == 'global':
            return self._global_acc
        if (metric_cls := self._cache.get(name)) is None:
            self._cache = {m.name: m for m in base.metrics}
            metric_cls = self._cache[name]
        metric = metric_cls()
        match category:
            case 'macro':
                return metric.macro(self._confusion)
            case 'minor':
                return metric.micro(self._confusion)
            case 'weighted':
                if self._weights is None:
                    raise ValueError(
                        f'Weighted {name} not supported for output mode {self._output_mode}'
                    )
                return metric.weighted(self._confusion, self._weights)
            case 'class':
                return {
                    cls: metric.calculate_class(m)
                    for cls, m in self._confusion.items()
                }
