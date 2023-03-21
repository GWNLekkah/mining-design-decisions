import abc
import collections
import statistics
import typing
import warnings

import alive_progress
import numpy as np
from keras.activations import softmax
import json
from scipy.special import softmax, expit

from .classifiers import models
from .model_io import OutputMode, OutputEncoding
from .config import conf
from . import data_manager_bootstrap

import tensorflow as tf

class KeywordEntry(typing.NamedTuple):
    keyword: str
    probability: float

    def as_dict(self):
        return {
            'keyword': self.keyword,
            'probability': self.probability
        }


def model_is_convolution() -> bool:
    classifiers = conf.get('run.classifier')
    if len(classifiers) > 1:
        return False
    return models[classifiers[0]].input_must_support_convolution()


def doing_one_run() -> bool:
    k = conf.get('run.k-cross')
    if k > 0:
        return False
    if conf.get('run.cross-project'):
        return False
    return True


def enabled() -> bool:
    return conf.get('run.analyze-keywords')


def analyze_keywords(model, test_x, test_y, issue_keys, suffix):
    def _to_str(y):
        return OutputMode.Classification3Simplified.label_encoding[y]

    def _pop(x):
        y = x.copy()
        del y['key']
        return y

    def _trim(z):
        if z.count('-') == 1:
            return z
        p = z.split('-')
        return f'{p[0]}-{p[1]}'

    output_mode = OutputMode.from_string(conf.get('run.output-mode'))
    print('Analyzing Keywords...')
    if output_mode.output_encoding == OutputEncoding.Binary:
        analyzer = BinaryConvolutionKeywordAnalyzer(model)
        with alive_progress.alive_bar(len(issue_keys)) as bar:
            keywords_per_class = analyzer.get_keywords(test_x, issue_keys, test_y, bar)
    else:
        analyzer = OneHotConvolutionKeywordAnalyzer(model)
        with alive_progress.alive_bar(len(issue_keys)) as bar:
            keywords_per_class = analyzer.get_keywords(test_x, issue_keys, test_y, bar)


    with open('../datasets/labels/bottom-up.csv') as file:
        bottom_up = [line.strip() for line in file]
    with open('../datasets/labels/maven.csv') as file:
        maven = [line.strip() for line in file]
    with open('../datasets/labels/top-down.csv') as file:
        top_down = [line.strip() for line in file]
    with open('../datasets/labels/BHAT_labels.json') as file:
        bhat = [item['key'] for item in json.load(file)]

    with open(f'./{conf.get("system.storage.file_prefix")}_maven-keywords-{suffix}.json', 'w') as file:
        print(keywords_per_class)
        maven_keywords = {
            cls: [
                _pop(entry) for entry in entries if _trim(entry['key']) in maven
            ]
            for cls, entries in keywords_per_class.items()
        }
        json.dump(maven_keywords, file)

    with open(f'./{conf.get("system.storage.file_prefix")}_bottom-up-keywords-{suffix}.json', 'w') as file:
        bottom_up_keywords = {
            cls: [
                _pop(entry) for entry in entries if _trim(entry['key']) in bottom_up
            ]
            for cls, entries in keywords_per_class.items()
        }
        json.dump(bottom_up_keywords, file)

    with open(f'./{conf.get("system.storage.file_prefix")}_top-down-keywords-{suffix}.json', 'w') as file:
        top_down_keywords = {
            cls: [
                _pop(entry) for entry in entries if _trim(entry['key']) in top_down
            ]
            for cls, entries in keywords_per_class.items()
        }
        json.dump(top_down_keywords, file)

    with open(f'./{conf.get("system.storage.file_prefix")}_bhat-keywords-{suffix}.json', 'w') as file:
        bhat_keywords = {
            cls: [
                _pop(entry) for entry in entries if _trim(entry['key']) in bhat
            ]
            for cls, entries in keywords_per_class.items()
        }
        json.dump(bhat_keywords, file)



def sigmoid(x):
    return expit(x)


class _ConvolutionKeywordAnalyzer(abc.ABC):

    def __init__(self, model):
        output_mode = OutputMode.from_string(conf.get('run.output-mode'))
        self.__binary = output_mode.output_encoding == OutputEncoding.Binary

        self.__number_of_classes = output_mode.number_of_classes

        # Store model
        self.__model = model

        # Get original text
        with open(data_manager_bootstrap.get_raw_text_file_name()) as file:
            self.__original_text_lookup = json.load(file)

        # Store weights of last dense layer
        self.__dense_layer_weights = self.__model.layers[-1].get_weights()[0]

        # Build model to get outputs in second to last layer
        self.__pre_output_model = tf.keras.Model(inputs=model.inputs,
                                                 outputs=model.layers[-2].output)
        self.__pre_output_model.compile()

        # Build models to get outputs of convolutions.
        self.__convolutions = {}
        convolution_number = 0
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv1D):
                self.__convolutions[convolution_number] = tf.keras.Model(inputs=model.inputs,
                                                                         outputs=layer.output)
                self.__convolutions[convolution_number].compile()
                convolution_number += 1
        print(f'Found {len(self.__convolutions)} convolutions')

        # Get number of filters
        params = conf.get('run.params')
        conv_params = params.get('default', {}) | params.get('Word2Vec1D', {})
        self.__input_size = int(conv_params['max-len'])

        hy_params = conf.get('run.hyper-params')
        conv_params = hy_params.get('default', {}) | hy_params.get('LinearConv1Model', {})
        self.__num_filters = int(conv_params.get('filters', 32))
        self.__convolution_sizes = {}
        for i in range(len(self.__convolutions)):
            self.__convolution_sizes[i] = int(conv_params[f'kernel-{i+1}-size'])

    @property
    def number_of_classes(self) -> int:
        return self.__number_of_classes

    @abc.abstractmethod
    def get_candidates(self, pre_predictions, truth, dense_weights):
        return []

    @abc.abstractmethod
    def get_minimum_strength(self) -> float:
        return 0.0

    def get_keywords(self, vectors, keys, truths, bar):
        output_mode = OutputMode.from_string(conf.get('run.output-mode'))

        # Compute all predictions and features.
        # Even though we might make more predictions than strictly
        # necessary, doing everything at once is significantly
        # faster than per-sample computation.
        pre_predictions = self.__pre_output_model.predict(np.array(vectors))
        feature_map = {
            i: self.__convolutions[i].predict(np.array(vectors))
            for i in self.__convolutions
        }

        # Compute indices of the ground truth
        #truth_indices = np.argmax(np.array(truths), axis=1)

        min_strength = self.get_minimum_strength()

        # Map for the outputs
        output = {}

        for j, (truth, issue_key) in enumerate(zip(truths, keys)):
            pre_predictions_for_sample = pre_predictions[j, :]
            list_tuple_prob = self.get_candidates(pre_predictions_for_sample, truth, dense_weights=self.__dense_layer_weights)

            # Get text of the original issue
            word_text = self.__original_text_lookup[issue_key]

            votes_per_convolution = collections.defaultdict(
                lambda: collections.defaultdict(
                    lambda: collections.defaultdict(list)
                )
            )
            for (ind, label, prob, w) in list_tuple_prob:
                # localize the convolutional layer
                conv_num = int(ind / self.__num_filters)
                # localize the index in the convolutional layer
                conv_ind = ind % self.__num_filters
                # localize keywords index
                features = feature_map[conv_num][j, :, :]
                keywords_index = np.where(features[:, conv_ind] == pre_predictions_for_sample[ind])[0][0]
                # Record the keywords
                votes_per_convolution[conv_num][label][keywords_index].append(prob)

            keywords_per_convolution = collections.defaultdict(
                lambda: collections.defaultdict(list)
            )
            for convolution, votes_per_label in votes_per_convolution.items():
                for label, votes in votes_per_label.items():
                    for keyword_index, probabilities in votes.items():
                        mean_strength = float(statistics.mean(probabilities))
                        if mean_strength >= min_strength:
                            keyword_stop = min(
                                keyword_index + self.__convolution_sizes[convolution],
                                len(word_text)
                            )
                            keywords_per_convolution[convolution][label].append(
                                (
                                    ' '.join([word_text[index] for index in range(keyword_index, keyword_stop)]),
                                    mean_strength
                                )
                            )

            kw = (
                [(label, KeywordEntry(keyword, prob))
                 for keywords_by_label in keywords_per_convolution.values()
                 for label, keywords in keywords_by_label.items()
                 for (keyword, prob) in keywords])
            for label, entry in kw:
                output.setdefault(output_mode.label_encoding[label], []).append(
                    entry.as_dict() | {'ground_truth': output_mode.label_encoding[label], 'key': issue_key}
                )

            bar()
        return output


class OneHotConvolutionKeywordAnalyzer(_ConvolutionKeywordAnalyzer):

    def get_candidates(self, pre_predictions, truth, dense_weights):
        list_tuple_prob = []
        truth_index = np.argmax(truth)
        for i, f in enumerate(pre_predictions):
            w = f * dense_weights[i]
            prob = softmax(w)
            if np.argmax(prob) == truth_index:
                list_tuple_prob.append((i, tuple(truth), prob[truth_index], w[truth_index]))
        return list_tuple_prob

    def get_minimum_strength(self) -> float:
        return 1 / self.number_of_classes


class BinaryConvolutionKeywordAnalyzer(_ConvolutionKeywordAnalyzer):

    def get_candidates(self, pre_predictions, truth, dense_weights):
        warnings.warn(f'{self.__class__.__name__} does not collect keywords for the negative class')
        output_mode = OutputMode.from_string(conf.get('run.output-mode'))
        list_tuple_prob = []
        if not isinstance(truth, np.ndarray):
            truth = np.array([truth])
        for i, f in enumerate(pre_predictions):
            w = f * dense_weights[i]
            prob = 1 / (1 + np.exp(w))  # element-wise sigmoid
            assert prob.shape == truth.shape
            for j, (x, y) in enumerate(zip(prob, truth)):
                if x >= 0.5 and y >= 0.5:
                    match (output_mode, j):
                        case (OutputMode.Detection, 0): # index is always 0
                            label = True
                        case (OutputMode.Classification3, 0):
                            label = (1, 0, 0)
                        case (OutputMode.Classification3, 1):
                            label = (0, 1, 0)
                        case (OutputMode.Classification3, 2):
                            label = (0, 0, 1)
                        case _ as x:
                            raise ValueError(f'Cannot determine label for key {x}')
                    list_tuple_prob.append((i, label, x, w[j]))
        return list_tuple_prob

    def get_minimum_strength(self) -> float:
        return 0.5