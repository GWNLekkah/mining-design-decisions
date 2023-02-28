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
    analyzer = ConvolutionKeywordAnalyzer(model)
    classes = list(output_mode.label_encoding.keys())
    keywords_per_class = collections.defaultdict(list)
    print('Analyzing Keywords...')
    if output_mode == OutputMode.Detection:
        with alive_progress.alive_bar(len(issue_keys) * len(classes)) as bar:
            for input_x, truth, issue_key in zip(test_x, test_y, issue_keys):
                for cls in classes:
                    words: list[KeywordEntry] = analyzer.get_keywords_for_input(input_x, issue_key, cls)
                    for entry in words:
                        if output_mode == output_mode.Detection:
                            keywords_per_class[truth].append(
                                entry.as_dict() | {'ground_truth': _to_str(truth), 'key': issue_key}
                            )
                        else:
                            keywords_per_class[tuple(truth)].append(
                                entry.as_dict() | {'ground_truth': _to_str(tuple(truth)), 'key': issue_key}
                            )
                    bar()
    else:
        #issue_keys = [_trim(key) for key in issue_keys]
        with alive_progress.alive_bar(len(issue_keys)) as bar:
            keywords_per_class = analyzer.get_bulk_keywords_for_input_classification(test_x, issue_keys, test_y, bar)

    #print(keywords_per_class)

    with open('../datasets/labels/bottom-up.csv') as file:
        bottom_up = [line.strip() for line in file]
    with open('../datasets/labels/maven.csv') as file:
        maven = [line.strip() for line in file]
    with open('../datasets/labels/top-down.csv') as file:
        top_down = [line.strip() for line in file]
    with open('../datasets/labels/BHAT_labels.json') as file:
        bhat = [item['key'] for item in json.load(file)]

    with open(f'./maven-keywords-{suffix}.json', 'w') as file:
        maven_keywords = {
            _to_str(cls): [
                _pop(entry) for entry in entries if _trim(entry['key']) in maven
            ]
            for cls, entries in keywords_per_class.items()
        }
        json.dump(maven_keywords, file)

    with open(f'./bottom-up-keywords-{suffix}.json', 'w') as file:
        bottom_up_keywords = {
            _to_str(cls): [
                _pop(entry) for entry in entries if _trim(entry['key']) in bottom_up
            ]
            for cls, entries in keywords_per_class.items()
        }
        json.dump(bottom_up_keywords, file)

    with open(f'./top-down-keywords-{suffix}.json', 'w') as file:
        top_down_keywords = {
            _to_str(cls): [
                _pop(entry) for entry in entries if _trim(entry['key']) in top_down
            ]
            for cls, entries in keywords_per_class.items()
        }
        json.dump(top_down_keywords, file)

    with open(f'./bhat-keywords-{suffix}.json', 'w') as file:
        bhat_keywords = {
            _to_str(cls): [
                _pop(entry) for entry in entries if _trim(entry['key']) in bhat
            ]
            for cls, entries in keywords_per_class.items()
        }
        json.dump(bhat_keywords, file)

    #with open('keywords.json', 'w') as file:
    #    json.dump(dict(keywords_per_class), file)
        
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
        conv_params = params.get('default', {}) | params.get('Word2Vec1D', {})
        self.__input_size = int(conv_params['max-len'])

        hy_params = conf.get('run.hyper-params')
        conv_params = hy_params.get('default', {}) | hy_params.get('LinearConv1Model', {})
        self.__num_filters = int(conv_params.get('filters', 32))
        self.__convolution_sizes = {}
        for i in range(len(self.__convolutions)):
            self.__convolution_sizes[i] = int(conv_params[f'kernel-{i+1}-size'])

    def get_keywords_for_one_hot_output(self, vectors, keys, truths, bar):
        # Get output mode
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
        truth_indices = np.argmax(np.array(truths), axis=1)

        # Mapping of indices to keys suitable for the output
        # mode encoding map
        l_map = {vec.index(1): vec for vec in output_mode.label_encoding}

        # Map for the outputs
        output = {}

        # l_map = {
        #     0: (1, 0, 0, 0),
        #     1: (0, 1, 0, 0),
        #     2: (0, 0, 1, 0),
        #     3: (0, 0, 0, 1)
        # }

        for j, (truth_index, issue_key) in enumerate(zip(truth_indices, keys)):
            list_tuple_prob = []

            pre_predictions_for_sample = pre_predictions[j, :]

            # Loop over individual feature items,
            # and record all items which would result in
            # a correct classification.
            for i, f in enumerate(pre_predictions_for_sample):
                w = f * self.__dense_layer_weights[i]
                prob = softmax(w)
                if np.argmax(prob) == truth_index:
                    list_tuple_prob.append((i, prob[truth_index], w[truth_index]))

            # Get text of the original issue
            word_text = self.__original_text_lookup[issue_key]

            votes_per_convolution = collections.defaultdict(lambda: collections.defaultdict(list))
            for (ind, prob, w) in list_tuple_prob:
                # localize the convolutional layer
                conv_num = int(ind / self.__num_filters)
                # localize the index in the convolutional layer
                conv_ind = ind % self.__num_filters
                # localize keywords index
                features = feature_map[conv_num][j, :, :]
                keywords_index = np.where(features[:, conv_ind] == pre_predictions_for_sample[ind])[0][0]
                # Record the keywords
                votes_per_convolution[conv_num][keywords_index].append(prob)

            keywords_per_convolution = collections.defaultdict(list)
            for convolution, votes in votes_per_convolution.items():
                for keyword_index, probabilities in votes.items():
                    mean_strength = float(statistics.mean(probabilities))
                    if mean_strength >= float(1.0 / self.__number_of_classes):
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

            kw = (
                [KeywordEntry(keyword, prob)
                 for keywords in keywords_per_convolution.values()
                 for keyword, prob in keywords])
            output.setdefault(l_map[truth_index], []).extend(
                entry.as_dict() | {
                    'ground_truth': output_mode.label_encoding[l_map[truth_index]],
                    'key': issue_key
                }
                for entry in kw
            )

            bar()
        return output

    def get_keywords_for_binary_output(self, vector, issue_key, ground_truth):
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

