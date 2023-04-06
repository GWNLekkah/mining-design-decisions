##############################################################################
##############################################################################
# Imports
##############################################################################

from copy import copy
import collections
import dataclasses

import numpy
import keras.callbacks
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as pyplot
import texttable
from transformers.modeling_tf_outputs import TFSequenceClassifierOutput

from ..model_io import OutputMode, OutputEncoding
from ..config import conf


##############################################################################
##############################################################################
# General Utility
##############################################################################


def _check_output_mode():
    # output_mode = OutputMode.from_string(conf.get('run.output-mode'))
    # if output_mode == OutputMode.Classification3:
    #     raise ValueError(
    #         'Metric computation not supported for 3-Classification')
    pass


def get_binary_metrics():
    return {
        'true_positives', 'false_positives',
        'true_negatives', 'false_negatives',
        'accuracy', 'precision',
        'recall', 'f_score_tf_macro',
        'loss'
    }


def get_multi_class_metrics():
    return {
        'accuracy', 'loss', 'f_score_tf_macro'
    }


def get_multi_label_metrics():
    return {
        'accuracy', 'loss', 'f_score_tf_macro'
    }


def get_metrics():
    output_mode = OutputMode.from_string(conf.get('run.output-mode'))
    if output_mode.output_encoding == OutputEncoding.OneHot:
        return get_multi_class_metrics()
    _check_output_mode()
    if output_mode.output_size == 1:
        return get_binary_metrics()
    return get_multi_label_metrics()


def get_metric_translation_table():
    return {
        'true_positives': 'tp',
        'false_positives': 'fp',
        'true_negatives': 'tn',
        'false_negatives': 'fn',
        'loss': 'loss',
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f_score_tf_macro': 'f-score'
    }


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



