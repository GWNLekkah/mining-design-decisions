import collections
import numpy

from .model_io import OutputMode, OutputEncoding


def get_voting_predictions(output_mode: OutputMode, predictions, voting_mode: str):
    if voting_mode == 'hard':
        return _get_hard_voting_predictions(output_mode, predictions.copy())
    else:
        return _get_soft_voting_predictions(output_mode, predictions.copy())


def _get_hard_voting_predictions(output_mode: OutputMode, predictions):
    if output_mode.output_encoding == OutputEncoding.Binary:
        return _get_hard_voting_predictions_binary(output_mode, predictions)
    else:
        return _get_hard_voting_predictions_one_hot(output_mode, predictions)


def _get_hard_voting_predictions_binary(output_mode: OutputMode, predictions):
    n = len(predictions)
    rounded = predictions.copy()
    rounded[rounded < 0.5] = False
    rounded[rounded >= 0.5] = True
    totals = numpy.sum(rounded, axis=0)
    if n % 2 == 1:
        totals[totals < n / 2] = False
        totals[totals > n / 2] = True
    else:
        totals[totals < n // 2] = False
        totals[totals > n // 2] = True
        # Resolve conflicts
        # Get confidences of conflicts
        loc = numpy.where(totals == n // 2)
        assert len(loc) == 2
        conflict_confidences = predictions[:, loc[0], loc[1]]
        # Get matrix of rounded predictions
        conflict_rounded = rounded[:, loc[0], loc[1]]
        # Get sum of confidences for both classes, by prediction
        conf_arch = conflict_confidences * conflict_rounded
        conf_non_arch = (1 - conflict_confidences) * (1 - conflict_rounded)
        # Use difference (highest sum) to break ties
        conflict = conf_arch.sum(axis=0) - conf_non_arch.sum(axis=0)
        for c, x, y in zip(conflict > 0, loc[0], loc[1]):
            totals[x, y] = c
    if output_mode.output_size == 1:
        totals = totals.flatten()
    return totals


def _get_hard_voting_predictions_one_hot(output_mode: OutputMode, predictions):
    indices = numpy.argmax(predictions, axis=2)
    classes = []
    for index, axis in enumerate(indices.transpose()):
        assert len(axis) == len(predictions)
        c = collections.Counter(axis)
        if len(c) == 1:
            classes.append(c.most_common()[0][0])
        else:
            (x, cx), (y, cy) = c.most_common(2)
            if cx > cy:
                classes.append(x)
            else:
                candidates = [y for y, cy in c.most_common() if cy == cx]
                scores = {}
                for candidate in candidates:
                    local_confidences = predictions[:, index]
                    local_predictions = numpy.argmax(local_confidences, axis=1)
                    scores[candidate] = local_confidences[local_predictions == candidate][candidate].sum()
                cls = max(scores.items(), key=lambda pair: pair[x])[0]
                classes.append(cls)
    return numpy.array(classes)


def _get_soft_voting_predictions(output_mode: OutputMode, predictions):
    confidences = numpy.sum(predictions, axis=0) / len(predictions)
    if output_mode.output_encoding == OutputEncoding.Binary:
        confidences[confidences < 0.5] = 0
        confidences[confidences >= 0.5] = 1
        if output_mode.output_size == 1:
            confidences = confidences.flatten()
        return confidences
    else:
        return numpy.argmax(confidences, axis=1)


def get_voting_confidences(output_mode: OutputMode, predictions, voting_mode):
    if voting_mode == 'hard':
        return _get_hard_voting_confidences(output_mode, predictions.copy())
    else:
        return _get_soft_voting_confidences(output_mode, predictions.copy())


def _get_hard_voting_confidences(output_mode: OutputMode, predictions):
    if output_mode.output_encoding == OutputEncoding.Binary:
        n = len(predictions)
        rounded = predictions.copy()
        rounded[rounded < 0.5] = False
        rounded[rounded >= 0.5] = True
        confidences = numpy.sum(rounded, axis=0) / n
        if output_mode.output_size == 1:
            confidences = confidences.flatten()
        return confidences
    else:
        confidences = []
        for index, cls in enumerate(_get_hard_voting_predictions_one_hot(output_mode, predictions)):
            confidences.append(predictions[:, index, cls].mean())


def _get_soft_voting_confidences(output_mode: OutputMode, predictions):
    confidences = numpy.sum(predictions, axis=0) / len(predictions)
    if output_mode.output_size == 1:
        confidences = confidences.flatten()
    return confidences
