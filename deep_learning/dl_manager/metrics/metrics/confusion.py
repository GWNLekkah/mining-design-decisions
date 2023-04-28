import dataclasses

import numpy
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix

from ...model_io import OutputMode

##############################################################################
##############################################################################
# Utility classes
##############################################################################


@dataclasses.dataclass(slots=True, frozen=True)
class MetricSet:
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int


##############################################################################
##############################################################################
# Utility functions
##############################################################################


def minor(matrix, i, j):
    return numpy.delete(numpy.delete(matrix, i, axis=0), j, axis=1)


##############################################################################
##############################################################################
# Confusion matrix calculations
##############################################################################


def compute_confusion_binary(y_true, y_pred) -> tuple[float, MetricSet]:
    # labels = OutputMode.Detection.output_vector_field_names
    # accuracy, classes = compute_confusion_multi_class(y_true,
    #                                                   y_pred,
    #                                                   labels)
    matrix = confusion_matrix(y_true, y_pred)
    metrics = extract_confusion(matrix, 1, len(y_true))
    accuracy = (metrics.true_positives + metrics.true_negatives) / len(y_true)
    return accuracy, {'Architectural': metrics}


def compute_confusion_multi_class(y_true,
                                  y_pred,
                                  labels) -> tuple[float, dict[str, MetricSet]]:
    # Matrix format: truth in row, prediction in column
    matrix = confusion_matrix(y_true, y_pred)
    # Compute class specific metrics
    class_metrics = {}
    for index, cls_name in enumerate(labels):
        class_metrics[cls_name] = extract_confusion(matrix, index, len(y_true))
    accuracy = sum(m.true_positives for m in class_metrics.values()) / len(y_pred)
    return accuracy, class_metrics


def compute_confusion_multi_label(y_true,
                                  y_pred,
                                  labels, *,
                                  all_negative_is_class=False,
                                  all_negative_class_name='negative',
                                  all_negative_class_pattern=None) -> tuple[float, dict[str, MetricSet]]:
    matrices = multilabel_confusion_matrix(y_true, y_pred)
    class_metrics = {}
    for matrix, label in zip(matrices, labels):
        class_metrics[label] = extract_confusion(matrix, 1, len(y_true))
    if all_negative_is_class:
        binary_y_true = ~(y_true == all_negative_class_pattern).all(axis=1)
        binary_y_pred = ~(y_pred == all_negative_class_pattern).all(axis=1)
        assert len(binary_y_true) == len(y_true)
        # _, result = compute_confusion_binary(binary_y_true, binary_y_pred)
        # class_metrics[all_negative_class_name] = result
        matrix = confusion_matrix(binary_y_true, binary_y_pred)
        class_metrics[all_negative_class_name] = extract_confusion(matrix, 0, len(binary_y_true))
    x, y = y_pred.shape
    total = x * y
    correct = sum(
        m.true_positives + m.true_negatives
        for m in class_metrics.values()
    )
    if all_negative_is_class:
        accuracy = correct / (total ** 2)
    else:
        accuracy = correct / total
    assert 0 <= accuracy <= 1
    return accuracy, class_metrics


def extract_confusion(matrix, index, expected_sum):
    # The true positive count for a given class is the value
    # on the diagonal.
    true_positives = matrix[index, index]
    # The true negative count for a given class is the sum of
    # the minor of the matrix
    true_negatives = minor(matrix, index, index).sum()
    # The false positive count for a given class is the sum of the
    # column with the diagonal entry removed.
    false_positives = matrix[:, index].sum() - matrix[index, index]
    # The false negative count for a given class is the sum of the
    # row with the diagonal entry removed
    false_negatives = matrix[index, :].sum() - matrix[index, index]
    assert true_positives + true_negatives + false_positives + false_negatives == expected_sum
    return MetricSet(true_positives=true_positives.item(),
                     true_negatives=true_negatives.item(),
                     false_positives=false_positives.item(),
                     false_negatives=false_negatives.item())
