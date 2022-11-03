##############################################################################
##############################################################################
# Imports
##############################################################################

import collections
import dataclasses
import statistics

import numpy
import keras.callbacks
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as pyplot
import texttable

from .feature_generators import OutputMode
from .classifiers import OutputEncoding
from .config import conf


##############################################################################
##############################################################################
# General Utility
##############################################################################


def _check_output_mode():
    output_mode = OutputMode.from_string(conf.get('run.output-mode'))
    if output_mode == OutputMode.Classification3:
        raise ValueError(
            'Metric computation not supported for 3-Classification')


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


def get_metrics():
    output_mode = OutputMode.from_string(conf.get('run.output-mode'))
    if output_mode.output_encoding == OutputEncoding.OneHot:
        return get_multi_class_metrics()
    _check_output_mode()
    return get_binary_metrics()


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
    predictions[predictions <= 0.5] = 0
    predictions[predictions > 0.5] = 1
    return predictions.flatten().astype(numpy.bool)


def round_onehot_predictions(predictions: numpy.ndarray) -> numpy.ndarray:
    return (predictions == predictions.max(axis=1)).astype(numpy.int64)


def onehot_indices(predictions: numpy.ndarray) -> numpy.ndarray:
    return predictions.argmax(axis=1)


def map_labels_to_names(predictions: numpy.ndarray,
                        label_mapping) -> numpy.ndarray:
    array: list = predictions.tolist()
    if isinstance(array[0], list):
        array = [tuple(x) for x in array]
    return numpy.asarray([label_mapping[x] for x in array])


##############################################################################
##############################################################################
# Manual Metric Computation
##############################################################################


@dataclasses.dataclass(slots=True, frozen=True)
class MetricSet:
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        if self.true_positives + self.false_positives == 0:
            # We want to be consistent with tensorflow
            if self.true_positives:
                return float('inf')
            return 0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        if self.true_positives + self.false_negatives == 0:
            if self.true_positives:
                return float('inf')
            return 0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f_score(self) -> float:
        if self.precision + self.recall == 0:
            return 0    # Correction by continuity
        return 2*self.precision*self.recall / (self.precision + self.recall)

    def as_dictionary(self, prefix=None):
        prefix = '' if not prefix else f'{prefix}_'
        return {
            f'{prefix}tp': self.true_positives,
            f'{prefix}tn': self.true_negatives,
            f'{prefix}fp': self.false_positives,
            f'{prefix}fn': self.false_negatives,
            f'{prefix}precision': self.precision,
            f'{prefix}recall': self.recall,
            f'{prefix}f-score': self.f_score
        }


def compute_confusion_binary(y_true,
                             y_pred,
                             label_mapping) -> (float, MetricSet):
    _check_output_mode()
    output_mode = OutputMode.from_string(conf.get('run.output-mode'))
    accuracy, classes = compute_confusion_multi_class(y_true,
                                                      y_pred,
                                                      label_mapping)
    return accuracy, classes[output_mode.true_category]


def compute_confusion_multi_class(y_true,
                                  y_pred,
                                  label_mapping) -> (float, dict[str, MetricSet]):
    _check_output_mode()
    y_true = map_labels_to_names(y_true, label_mapping)
    y_pred = map_labels_to_names(y_pred, label_mapping)
    labels = list(label_mapping.values())
    # Matrix format: truth in row, prediction in column
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    # Compute accuracy: the sum of the diagonal of the matrix.
    accuracy = numpy.diagonal(matrix).sum().item() / matrix.sum().item()
    # Compute class specific metrics
    class_metrics = {}
    for index, cls_name in enumerate(labels):
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
        assert true_positives + true_negatives + false_positives + false_negatives == len(y_true)
        class_metrics[cls_name] = MetricSet(true_positives=true_positives.item(),
                                            true_negatives=true_negatives.item(),
                                            false_positives=false_positives.item(),
                                            false_negatives=false_negatives.item())
    return accuracy, class_metrics


def minor(matrix, i, j):
    return numpy.delete(numpy.delete(matrix, i, axis=0), j, axis=1)


##############################################################################
##############################################################################
# Helper for metric collection
##############################################################################


class MetricCollection:

    def __init__(self):
        self.__metrics = {
            'tp': [], 'fp': [], 'tn': [], 'fn': [],
            'loss': [],
            'accuracy': [], 'precision': [], 'recall': [], 'f-score': [],
            'class-precision': collections.defaultdict(list),
            'class-recall': collections.defaultdict(list),
            'class-f-score': collections.defaultdict(list)
        }
        output_mode = OutputMode.from_string(conf.get('run.output-mode'))
        self.__is_multi_class = output_mode.output_size > 1
        _check_output_mode()

    def update_metrics(self, **kwargs):
        trans = get_metric_translation_table()
        for raw_key in get_metrics():
            key = trans[raw_key]
            if key not in kwargs:
                pass
            self.__metrics[key].append(kwargs.pop(key))
        if kwargs:
            raise ValueError('Not all **kwargs were consumed:', kwargs)

    def update_from_metric_set(self, accuracy, metrics: MetricSet):
        metrics_as_dict = {'accuracy': accuracy}
        metrics_as_dict |= metrics.as_dictionary()
        self.update_metrics(**metrics_as_dict)

    def update_class_metrics(self, class_name, metrics: MetricSet):
        self.__metrics['class-precision'][class_name].append(metrics.precision)
        self.__metrics['class-recall'][class_name].append(metrics.recall)
        self.__metrics['class-f-score'][class_name].append(metrics.f_score)

    def as_dictionary(self, prefix=None):
        prefix = f'{prefix}-' if prefix is not None else ''
        return {
            f'{prefix}{key}': value for key, value in self.__metrics.items()
        }

    # def fix_fscore(self):
    #     print(self.__metrics['class-f-score'])
    #     assert self.__metrics['class-f-score']
    #     assert all(x for x in self.__metrics['class-f-score'].values())
    #     fscore = [statistics.mean(v)
    #               for v in zip(self.__metrics['class-f-score'].items())]
    #     self.__metrics['f-score'] = fscore

##############################################################################
##############################################################################
# Metric Logger
##############################################################################


class MetricLogger(keras.callbacks.Callback):

    def __init__(self,
                 model,
                 test_x,
                 test_y,
                 output_mode: OutputMode,
                 label_mapping,
                 issue_keys):
        _check_output_mode()
        super().__init__()
        self.__model = model
        self.__test_x = test_x
        self.__test_y = test_y
        # Metrics
        self.__train_metrics = MetricCollection()
        self.__val_metrics = MetricCollection()
        self.__test_metrics = MetricCollection()
        self.__predictions = []
        # End metrics
        self.__output_mode = output_mode
        self.__task_is_binary = output_mode == OutputMode.Detection
        if self.__output_mode.output_encoding == OutputEncoding.OneHot:
            self.__label_mapping = {key.index(1): value
                                    for key, value in self.__output_mode.label_encoding.items()}
        else:
            self.__label_mapping = label_mapping
        self.__issue_keys = issue_keys
        self.__stopped_on_last_epoch = False

    # =================================================================
    # Early Stopping Support
    # =================================================================

    def __check_for_early_stopping(self, epoch):
        if conf.get('run.use-early-stopping'):
            self.__stopped_on_last_epoch = epoch + 1 == conf.get('run.epochs')
        else:
            self.__stopped_on_last_epoch = True
            conf.set('run.early-stopping-patience', 0)
            conf.set('run.early-stopping-min-delta', [0])

    # =================================================================
    # Metric Logging
    # =================================================================

    def __log_binary_metrics(self, logs, epoch):
        # Get the performance on the test set;
        # The logs contain the performance on the
        # training and validation sets
        results = self.__model.evaluate(x=self.__test_x,
                                        y=self.__test_y,
                                        return_dict=True)
        self.__update_metrics_helper(results,
                                     logs,
                                     epoch)

    def __log_multi_class_metrics(self, logs, epoch):
        results = self.__model.evaluate(x=self.__test_x,
                                        y=self.__test_y,
                                        return_dict=True)
        self.__update_metrics_helper(results,
                                     logs,
                                     epoch)

    def __update_metrics_helper(self, results, logs, epoch):
        trans = get_metric_translation_table()
        metrics = get_metrics()
        # Log test metrics
        self.__test_metrics.update_metrics(
            **{trans[metric]: results[metric]
               for metric in metrics})
        # Log training metrics
        self.__train_metrics.update_metrics(
            **{trans[metric]: logs[metric]
               for metric in metrics})
        # Log validation metrics
        self.__val_metrics.update_metrics(
            **{trans[metric]: logs[f'val_{metric}']
               for metric in metrics})

        # Also print the output to the console
        print(f'Test accuracy ({epoch}):', results['accuracy'])
        print(f'Test F-score ({epoch}):', results['f_score_tf_macro'])

    # =================================================================
    # Main Function -- Called on epoch end
    # =================================================================

    def on_epoch_end(self, epoch, logs=None):
        self.__check_for_early_stopping(epoch)

        # Evaluate the model.
        if self.__task_is_binary:
            self.__log_binary_metrics(logs, epoch)
        else:
            self.__log_multi_class_metrics(logs, epoch)

        y_pred = numpy.asarray(self.model.predict(self.__test_x))
        if self.__output_mode.output_encoding == OutputEncoding.OneHot:
            y_true = onehot_indices(self.__test_y)
            y_pred_class = onehot_indices(y_pred)
        else:
            _check_output_mode()
            y_true = numpy.asarray(self.__test_y)
            y_pred_class = round_binary_predictions(y_pred)

        self.__predictions.append(y_pred_class.tolist())

        if self.__output_mode.output_encoding == OutputEncoding.OneHot:
            accuracy, class_metrics = compute_confusion_multi_class(y_true,
                                                                    y_pred_class,
                                                                    self.__label_mapping)
            for cls, metrics_for_class in class_metrics.items():
                self.__test_metrics.update_class_metrics(cls,
                                                         metrics_for_class)
        else:
            _check_output_mode()
            # We do not actually have to compute the metrics here,
            # because they have already been computed by Keras.
            pass

    def get_model_results_for_all_epochs(self):
        basis = {
            'classes': self.__label_mapping,
            'truth': self.__test_y.tolist(),
            'early-stopping-settings': {
                'stopped-early': not self.__stopped_on_last_epoch,
                'patience': conf.get('run.early-stopping-patience'),
                'min-delta': conf.get('run.early-stopping-min-delta')
            },
            'predictions': self.__predictions
        }
        # if self.__output_mode.output_encoding == OutputEncoding.OneHot:
        #     self.__train_metrics.fix_fscore()
        #     self.__val_metrics.fix_fscore()
        #     self.__test_metrics.fix_fscore()
        train = self.__train_metrics.as_dictionary('train')
        val = self.__val_metrics.as_dictionary('val')
        test = self.__test_metrics.as_dictionary()
        return basis | train | val | test

    def get_main_model_metrics_at_stopping_epoch(self):
        data = self.get_model_results_for_all_epochs()
        if self.__stopped_on_last_epoch:
            return {'accuracy': data['accuracy'][-1],
                    'f-score': data['f-score'][-1]}
        else:
            offset = conf.get('run.early-stopping-patience') + 1
            return {'accuracy': data['accuracy'][-offset],
                    'f-score': data['f-score'][-offset]}


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



