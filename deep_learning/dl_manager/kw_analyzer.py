import collections
import csv
import statistics
import typing

import alive_progress
import numpy as np
from keras.activations import softmax
import json
from scipy.special import softmax, expit

from .classifiers import models
from .feature_generators import OutputMode
from .config import conf
from . import data_manager_bootstrap

import tensorflow as tf
import numpy


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


def analyze_keywords(model, test_x, test_y, issue_keys):
    output_mode = OutputMode.from_string(conf.get('run.output-mode'))
    analyzer = ConvolutionKeywordAnalyzer(model)
    classes = list(output_mode.label_encoding.keys())
    keywords_per_class = collections.defaultdict(list)
    print('Analyzing Keywords...')
    with alive_progress.alive_bar(len(issue_keys) * len(classes)) as bar:
        for input_x, truth, issue_key in zip(test_x, test_y, issue_keys):
            for cls in classes:
                words: list[KeywordEntry] = analyzer.get_keywords_for_input(input_x, issue_key, cls)
                for entry in words:
                    if output_mode == output_mode.Detection:
                        keywords_per_class[truth].append(entry.as_dict() | {'ground_truth': truth})
                    else:
                        keywords_per_class[tuple(truth)].append(entry.as_dict() | {'ground_truth': tuple(truth)})
                bar()

    with open('keywords.json', 'w') as file:
        json.dump(dict(keywords_per_class), file)
        
    # for label, keywords in keywords_per_class.items():
    #     label_as_text = output_mode.label_encoding[label]
    #     with open(f'{label_as_text}_keywords.csv', 'w', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(['Keyword', 'Frequency'])
    #         counts = collections.Counter()
    #         for keyword_dict in keywords:
    #             counts.update(keyword_dict.keys())
    #         for kw, freq in counts.items():
    #             writer.writerow([kw, freq])


def sigmoid(x):
    return expit(x)


class ConvolutionKeywordAnalyzer:

    def __init__(self, model):
        # Pre-flight check: output mode must be detection
        output_mode = OutputMode.from_string(conf.get('run.output-mode'))
        self.__binary = output_mode == OutputMode.Detection

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
        conv_params = params.get('default', {}) | params.get('Word2Vec2D', {})
        self.__input_size = int(conv_params['max-len'])

        hy_params = conf.get('run.hyper-params')
        conv_params = hy_params.get('default', {}) | hy_params.get('LinearConv1Model', {})
        self.__num_filters = int(conv_params.get('filters', 32))
        self.__convolution_sizes = {}
        for i in range(len(self.__convolutions)):
            self.__convolution_sizes[i] = int(conv_params[f'kernel-{i+1}-size'])

    def get_keywords_for_input(self, vector, issue_key, ground_truth):
        if self.__binary:
            return self.get_keywords_for_input_detection(vector, issue_key, ground_truth)
        return self.get_keywords_for_input_classification(vector, issue_key, ground_truth)

    def get_keywords_for_input_classification(self, vector, issue_key, ground_truth):
        pre_predictions = self.__pre_output_model.predict(np.array([vector]))[0]

        truth_index = numpy.argmax(ground_truth)

        list_tuple_prob = []
        for i, f in enumerate(pre_predictions):  # Loop over individual feature items
            w = f * self.__dense_layer_weights[i]
            prob = softmax(w)

            if prob[truth_index] > float(1.0 / self.__number_of_classes):
                list_tuple_prob.append((i, prob, w[truth_index]))

        word_text = self.__original_text_lookup[issue_key]

        votes_per_convolution = collections.defaultdict(lambda: collections.defaultdict(list))
        # keywords_length = self.__convolution_sizes[conv_num]

        for (ind, prob, w) in list_tuple_prob:
            # localize the convolutional layer
            conv_num = int(ind / self.__num_filters)
            # localize the index in the convolutional layer
            conv_ind = ind % self.__num_filters

            # localize keywords index
            features = self.__convolutions[conv_num].predict(np.array([vector]))[0]

            keywords_index = np.where(features[:, conv_ind] == pre_predictions[ind])[0][0]

            votes_per_convolution[conv_num][keywords_index].append(prob)

        keywords_per_convolution = collections.defaultdict(list)
        for convolution, votes in votes_per_convolution.items():
            for keyword_index, probabilities in votes.items():
                mean_strength = statistics.mean(votes)
                if mean_strength >= 0.5:
                    keyword_stop = min(
                        keyword_index + self.__convolution_sizes[convolution],
                        len(word_text)
                    )
                    keywords_per_convolution[convolution].append(
                        (
                            ' '.join([word_text[index] for index in range(keyword_index, keyword_stop)]),
                            mean_strength
                        )
                    )

        return [KeywordEntry(keyword, prob)
                for keywords in keywords_per_convolution.values()
                for keyword, prob in keywords]

    def get_keywords_for_input_detection(self, vector, issue_key, ground_truth):
        pre_predictions = self.__pre_output_model.predict(np.array([vector]))[0]

        list_tuple_prob = []
        for i, f in enumerate(pre_predictions):  # Loop over individual feature items
            w = f * self.__dense_layer_weights
            prob = sigmoid(w[i])
            # prob = softmax(w)

            # if prob[pred] > float(1.0 / len(labels)):
            if abs(prob - ground_truth) < 0.5:     # Strong vote towards a label
                list_tuple_prob.append((i, prob, w[i]))

        # dict_keywords = {}
        # list_keywords = []
        word_text = self.__original_text_lookup[issue_key]

        votes_per_convolution = collections.defaultdict(lambda: collections.defaultdict(list))
        # keywords_length = self.__convolution_sizes[conv_num]

        for (ind, prob, w) in list_tuple_prob:
            # localize the convolutional layer
            conv_num = int(ind / self.__num_filters)
            # localize the index in the convolutional layer
            conv_ind = ind % self.__num_filters

            # localize keywords index
            features = self.__convolutions[conv_num].predict(np.array([vector]))[0]

            keywords_index = np.where(features[:, conv_ind] == pre_predictions[ind])[0][0]

            votes_per_convolution[conv_num][keywords_index].append(prob)

        keywords_per_convolution = collections.defaultdict(list)
        for convolution, votes in votes_per_convolution.items():
            for keyword_index, probabilities in votes.items():
                mean_strength = statistics.mean(votes)
                if mean_strength >= 0.5:
                    keyword_stop = min(
                        keyword_index + self.__convolution_sizes[convolution],
                        len(word_text)
                    )
                    keywords_per_convolution[convolution].append(
                        (
                            ' '.join([word_text[index] for index in range(keyword_index, keyword_stop)]),
                            mean_strength
                        )
                    )

        return [KeywordEntry(keyword, prob)
                for keywords in keywords_per_convolution.values()
                for keyword, prob in keywords]

