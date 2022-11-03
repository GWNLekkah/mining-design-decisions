import collections
import numpy
from .classifiers import OutputEncoding


def get_voting_predictions(output_mode, predictions):
    if output_mode.output_encoding == OutputEncoding.Binary:
        hard_predictions = []
        for pred in predictions:
            hard_pred = pred.flatten()
            hard_pred[hard_pred < 0.5] = 0
            hard_pred[hard_pred >= 0.5] = 1
            hard_predictions.append(hard_pred)
    else:
        hard_predictions = []
        for pred in predictions:
            hard_pred = numpy.argmax(pred, axis=1)
            hard_predictions.append(hard_pred)
    # Step 1: Determine whether there is a majority
    prediction_matrix = numpy.asarray(hard_predictions).transpose().tolist()
    # noinspection PyTypeChecker
    final_predictions = [mode(x) for x in prediction_matrix]
    # Step 2: Break ties using probabilities
    probability_matrix = numpy.asarray(predictions).sum(axis=0)
    # noinspection PyTypeChecker
    probability_classes: list = numpy.argmax(probability_matrix, axis=1).tolist()
    final_predictions = numpy.asarray([
        final_pred if final_pred is not None else probability_classes[index]
        for index, final_pred in enumerate(final_predictions)
    ])
    return final_predictions


def mode(x):
    counter = collections.Counter(x)
    if len(counter) == 1:
        return x[0]
    best_two = counter.most_common(2)
    (value_1, count_1), (value_2, count_2) = best_two
    if count_1 > count_2:
        return value_1
    if count_2 > count_1:
        return value_2
    return None
