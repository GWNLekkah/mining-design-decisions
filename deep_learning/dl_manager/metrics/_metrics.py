##############################################################################
##############################################################################
# Imports
##############################################################################

from copy import copy
import collections

import numpy
import matplotlib.pyplot as pyplot
import texttable

##############################################################################
##############################################################################
# Helper Functions for Predictions
##############################################################################


def round_binary_predictions(predictions: numpy.ndarray) -> numpy.ndarray:
    rounded_predictions = copy(predictions)
    rounded_predictions[predictions <= 0.5] = 0
    rounded_predictions[predictions > 0.5] = 1
    return rounded_predictions.flatten().astype(bool)


def round_binary_predictions_no_flatten(predictions: numpy.ndarray) -> numpy.ndarray:
    rounded_predictions = copy(predictions)
    rounded_predictions[predictions <= 0.5] = 0
    rounded_predictions[predictions > 0.5] = 1
    return rounded_predictions


def round_onehot_predictions(predictions: numpy.ndarray) -> numpy.ndarray:
    return (predictions == predictions.max(axis=1)).astype(numpy.int64)


def onehot_indices(predictions: numpy.ndarray) -> numpy.ndarray:
    return predictions.argmax(axis=1)


##############################################################################
##############################################################################
# Functionality for model comparison
##############################################################################


class ComparisonManager:

    def __init__(self):
        self.__results = []
        self.__current = []
        self.__truths = []

    def mark_end_of_fold(self):
        self.__check_finalized(False)
        self.__results.append(self.__current)
        self.__current = []

    def finalize(self):
        self.__check_finalized(False)
        if self.__current:
            self.__results.append(self.__current)
        self.__current = None

    def add_result(self, results):
        self.__check_finalized(False)
        self.__current.append(results['predictions'])

    def add_truth(self, truth):
        self.__truths.append(truth)

    def compare(self):
        self.__check_finalized(True)
        print(len(self.__results))
        print(len(self.__truths))
        assert len(self.__results) == len(self.__truths)
        prompt = f'How to order {len(self.__results)} plots? [nrows ncols]: '
        rows, cols = map(int, input(prompt).split())
        fig, axes = pyplot.subplots(nrows=rows, ncols=cols, squeeze=False)
        axes = axes.flatten()
        for ax, results, truth in zip(axes, self.__results, self.__truths):
            self.__make_comparison_plot(ax, results, truth)
        pyplot.show()

    def __check_finalized(self, expected_state: bool):
        is_finalized = self.__current is None
        if is_finalized and not expected_state:
            raise ValueError('Already finalized')
        if expected_state and not is_finalized:
            raise ValueError('Not yet finalized')

    def __make_comparison_plot(self, ax, results, truth):
        matrix = [result[-1] for result in results]
        table = texttable.Texttable()
        table.header(
            ['Ground Truth'] + [f'Model {i}' for i in range(1, len(results) + 1)] + ['Amount']
        )
        counter = collections.defaultdict(int)
        for truth, *predictions in zip(truth, *matrix):
            key = (truth,) + tuple(predictions)
            counter[key] += 1
        for key, value in counter.items():
            table.add_row([str(x) for x in key] + [str(value)])
        print(table.draw())



