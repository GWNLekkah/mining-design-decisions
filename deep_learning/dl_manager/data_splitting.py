##############################################################################
##############################################################################
# Imports
##############################################################################

import abc
import collections
import functools
import random

import numpy
import sklearn.model_selection
from sklearn.model_selection import train_test_split

from . import kfold
from . import custom_kfold
from .config import Config

##############################################################################
##############################################################################
# Utility Classes
##############################################################################


class DeepLearningData:

    _NAIVE_SPLITTABLE = {
        'BOWFrequency',
        'BOWNormalized',
        'TfidfGenerator',
        'Word2Vec',
        'Doc2Vec',
        'AutoEncoder',
        'KateAutoEncoder',
        'Metadata',
        'OntologyFeatures'
    }

    def __init__(self, labels, issue_keys, issue_ids, generators, *features):
        self.labels = numpy.array(labels)
        self.issue_keys = numpy.array(issue_keys)
        self.issue_ids = numpy.array(issue_ids)
        self.features = [
            self.prepare_feature(f, g)
            for f, g in zip(features, generators)
        ]
        self.generators = generators

    def prepare_feature(self, f, g):
        if g in self._NAIVE_SPLITTABLE:
            return numpy.array(f)
        elif g == 'Bert':
            return f
        raise ValueError(f'Cannot prepare features from generator {g}')

    def to_dataset_and_keys(self):
        return self.to_dataset(), self.issue_keys, self.issue_ids

    def to_dataset(self):
        """Get a dataset which can be readily passed to a TensorFlow model.
        """
        return make_dataset(self.labels, *self.features)

    @property
    def size(self):
        return len(self.labels)

    @property
    def extended_labels(self):
        return [
            (label, key.split('-')[0])
            for label, key in zip(self.labels, self.issue_keys)
        ]

    @property
    def custom_kfold_extended_labels(self):
        if not isinstance(self.labels[0], (numpy.ndarray, list)):
            return self.extended_labels
        return [
            (tuple(label), key.split('-')[0])
            for label, key in zip(self.labels, self.issue_keys)
        ]

    def split_fraction(self, size: float):
        size_left = int(size * len(self.labels))
        left = self.sample_indices(range(0, size_left))
        right = self.sample_indices(range(size_left, self.size))
        return left, right

    def split_fraction_stratified(self, size: float):
        indices = list(range(len(self.labels)))
        left, right, _, _ = sklearn.model_selection.train_test_split(
            indices,
            self.extended_labels,
            train_size=size,
            shuffle=True,
            stratify=True
        )
        return self.sample_indices(left), self.sample_indices(right)

    def split_k_cross(self, k: int):
        splitter = kfold.StratifiedKFold(k)
        stream = splitter.split(self.extended_labels, self.extended_labels)
        for indices_left, indices_right in stream:
            left = self.sample_indices(indices_left)
            right = self.sample_indices(indices_right)
            yield left, right

    def split_k_cross_three(self, k: int):
        for ix, iy, iz in custom_kfold.stratified_kfold(k, self.custom_kfold_extended_labels):
            x = self.sample_indices(ix)
            y = self.sample_indices(iy)
            z = self.sample_indices(iz)
            yield x, y, z

    def split_cross_project(self, val_split: float):
        # Collect bins of indices
        bins = collections.defaultdict(set)
        for index, issue_key in enumerate(self.issue_keys):
            project = issue_key.split('-')[0]
            bins[project].add(index)
        # Loop over bins
        for test_project, test_indices in bins.items():
            remaining_indices = functools.reduce(
                set.union,
                [indices
                 for project, indices in bins.items()
                 if project != test_project],
                set()
            )
            remaining = self.sample_indices(list(remaining_indices))
            validation,  training = remaining.split_fraction(val_split)
            testing = self.sample_indices(list(test_indices))
            yield training, validation, testing

    def limit_size(self, max_size: int):
        if len(self.labels) < max_size:
            return self
        return self.sample_indices(list(range(0, max_size)))

    def shuffle(self):
        indices = list(range(self.size))
        random.shuffle(indices)
        return self.sample_indices(indices)

    def sample_indices(self, indices):
        return DeepLearningData(
            self.labels[indices],
            self.issue_keys[indices],
            self.issue_ids[indices],
            self.generators,
            *(
                self.sample_from(f, indices, g)
                for f, g in zip(self.features, self.generators, strict=True)
            )
        )

    def sample_from(self, f, indices, g):
        if g in self._NAIVE_SPLITTABLE:
            return f[indices]
        elif g == 'Bert':
            return {
                'input_ids': f['input_ids'][indices],
                'token_type_ids': f['token_type_ids'][indices],
                'attention_mask': f['attention_mask'][indices]
            }
        raise ValueError(f'Cannot split data from generator {g}')



##############################################################################
##############################################################################
# Utility Functions
##############################################################################


def shuffle_raw_data(*x):
    c = list(zip(*x))
    random.shuffle(c)
    return map(numpy.asarray, zip(*c))


def make_dataset(labels, *features):
    if len(features) == 1:
        return features[0], labels
    return list(features), labels


##############################################################################
##############################################################################
# Abstract Splitter
##############################################################################


class DataSplitter(abc.ABC):

    def __init__(self, conf: Config, /, **kwargs):
        self.conf = conf
        self.max_train = t if (t := kwargs.pop('max_train', -1)) != -1 else None
        if kwargs:
            keys = ', '.join(kwargs)
            raise ValueError(
                f'Illegal options for splitter {self.__class__.__name__}: {keys}'
            )

    @abc.abstractmethod
    def split(self, training_data_raw, testing_data=None, *, generators=None):
        pass


##############################################################################
##############################################################################
# Single Data Splitter
##############################################################################


class SimpleSplitter(DataSplitter):

    def __init__(self, conf: Config, /, **kwargs):
        self.val_split = kwargs.pop('val_split_size')
        self.test_split = kwargs.pop('test_split_size')
        super().__init__(conf, **kwargs)

    def split(self, training_data_raw, testing_data=None, *, generators=None):
        if generators is None:
            generators = self.conf.get('run.input-mode')
        features, labels, issue_keys, issue_ids  = training_data_raw
        # labels, issue_keys, *features = shuffle_raw_data(labels,
        #                                                  issue_keys,
        #                                                  *features)
        data = DeepLearningData(labels, issue_keys, issue_ids, generators, *features).shuffle()
        if testing_data is None:
            size = self.val_split + self.test_split
            #training_data, remainder = data.split_fraction(1 - size)
            training_data, remainder = data.split_fraction_stratified(1 - size)
            size = self.val_split / (self.val_split + self.test_split)
            #val_data, test_data = remainder.split_fraction(size)
            val_data, test_data = remainder.split_fraction_stratified(size)
        else:
            #training_data, val_data = data.split_fraction(1 - self.val_split)
            training_data, val_data = data.split_fraction_stratified(1 - self.val_split)
            assert (
                (self.val_split < 0.5 and training_data.labels.size > val_data.labels.size) or
                (self.val_split > 0.5 and training_data.labels.size < val_data.labels.size)
            )
            features_test, labels_test, keys_test, ids_test  = testing_data
            #labels_test, keys_test, *features_test = shuffle_raw_data(labels, issue_keys, *features_test)
            test_data = DeepLearningData(labels_test, keys_test, ids_test, *features_test).shuffle()
        if self.max_train is not None:
            training_data = training_data.limit_size(self.max_train)
        yield (
            training_data.to_dataset(),
            test_data.to_dataset(),
            val_data.to_dataset(),
            training_data.issue_keys,
            val_data.issue_keys,
            test_data.issue_keys,
            training_data.issue_ids,
            val_data.issue_ids,
            test_data.issue_ids
        )


class CrossFoldSplitter(DataSplitter):

    def __init__(self, conf: Config, /, **kwargs):
        self.k = kwargs.pop('k')
        super().__init__(conf, **kwargs)

    def split(self, training_data_raw, testing_data=None, *, generators=None):
        if generators is None:
            generators = self.conf.get('run.input-mode')
        if testing_data is not None:
            raise ValueError(
                f'{self.__class__.__name__} does not support splitting with explicit testing data'
            )
        features, labels, issue_keys, issue_ids = training_data_raw
        # labels, issue_keys, *features = shuffle_raw_data(labels,
        #                                                  issue_keys,
        #                                                  *features)
        data = DeepLearningData(labels, issue_keys, issue_ids, generators, *features).shuffle()
        for inner, test_data in data.split_k_cross(self.k):
            for training_data, validation_data in inner.split_k_cross(self.k - 1):
                if self.max_train is not None:
                    training_data = training_data.limit_size(self.max_train)
                yield (
                    training_data.to_dataset(),
                    test_data.to_dataset(),
                    validation_data.to_dataset(),
                    training_data.issue_keys,
                    validation_data.issue_keys,
                    test_data.issue_keys,
                    training_data.issue_ids,
                    validation_data.issue_ids,
                    test_data.issue_ids
                )


class QuickCrossFoldSplitter(DataSplitter):

    def __init__(self, conf: Config, /, **kwargs):
        self.k = kwargs.pop('k')
        super().__init__(conf, **kwargs)

    def split(self, training_data_raw, testing_data=None, *, generators=None):
        if generators is None:
            generators = self.conf.get('run.input-mode')
        if testing_data is not None:
            raise ValueError(
                f'{self.__class__.__name__} does not support splitting with explicit testing data'
            )
        features, labels, issue_keys, issue_ids = training_data_raw
        # labels, issue_keys, *features = shuffle_raw_data(labels,
        #                                                  issue_keys,
        #                                                  *features)
        data = DeepLearningData(labels, issue_keys, issue_ids, generators, *features).shuffle()
        # if self.test_project is not None or self.test_study is not None:
        #     if self.test_project is not None:
        #         testing_data, remainder = data.split_on_project(self.test_project)
        #     else:
        #         testing_data, remainder = data.split_on_study(self.test_study)
        #     for training, validation in data.split_k_cross(self.k):
        #         if self.max_train is not None:
        #             training = training.limit_size(self.max_train)
        #         yield (
        #             training.to_dataset(),
        #             testing_data.to_dataset(),
        #             validation.to_dataset(),
        #             training.issue_keys,
        #             validation.issue_keys,
        #             testing_data.issue_keys
        #         )
        if True:
            for training, validation, testing in data.split_k_cross_three(self.k):
                if self.max_train is not None:
                    training = training.limit_size(self.max_train)
                yield (
                    training.to_dataset(),
                    testing.to_dataset(),
                    validation.to_dataset(),
                    training.issue_keys,
                    validation.issue_keys,
                    testing.issue_keys,
                    training.issue_ids,
                    validation.issue_ids,
                    testing.issue_ids
                )


class CrossProjectSplitter(DataSplitter):

    def __init__(self, conf: Config, /, **kwargs):
        self.val_split = kwargs.pop('val_split_size')
        super().__init__(conf, **kwargs)
        if self.max_train is not None:
            raise ValueError(f'{self.__class__.__name__} does not support max_train')

    def split(self, training_data_raw, testing_data=None, *, generators=None):
        if generators is None:
            generators = self.conf.get('run.input-mode')
        if testing_data is not None:
            raise ValueError(
                f'{self.__class__.__name__} does not support splitting with explicit testing data'
            )
        features, labels, issue_keys, issue_ids = training_data_raw
        # labels, issue_keys, *features = shuffle_raw_data(labels,
        #                                                  issue_keys,
        #                                                  *features)
        data = DeepLearningData(labels, issue_keys, issue_ids, generators, *features).shuffle()
        for training, validation, testing in data.split_cross_project(self.val_split):
            if self.max_train is not None:
                training = training.limit_size(self.max_train)
            yield (
                training.to_dataset(),
                testing.to_dataset(),
                validation.to_dataset(),
                training.issue_keys,
                validation.issue_keys,
                testing.issue_keys,
                training.issue_ids,
                validation.issue_ids,
                testing.issue_ids
            )
