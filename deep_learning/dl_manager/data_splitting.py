##############################################################################
##############################################################################
# Imports
##############################################################################

import abc
import collections
import functools
import random

import numpy

from . import kfold
from . import custom_kfold

##############################################################################
##############################################################################
# Utility Classes
##############################################################################


class DeepLearningData:

    def __init__(self, labels, issue_keys, *features):
        self.labels = numpy.array(labels)
        self.issue_keys = numpy.array(issue_keys)
        self.features = [
            numpy.array(f) for f in features
        ]

    def to_dataset_and_keys(self):
        return self.to_dataset(), self.issue_keys

    def to_dataset(self):
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

    def split_on_project(self, project: str):
        return self._split_on_predicate(
            # Split: project, number, study
            lambda issue_key: issue_key.split('-')[0] == project
        )

    def split_on_study(self, study: str):
        return self._split_on_predicate(
            lambda issue_key: issue_key.split('-')[2] == study
        )

    def _split_on_predicate(self, predicate):
        project_indices = [
            index
            for index, issue_key in enumerate(self.issue_keys)
            if predicate(issue_key)
        ]
        project_complement = [
            index
            for index, issue_key in enumerate(self.issue_keys)
            if not predicate(issue_key)
        ]
        project_data = DeepLearningData(
            self.labels[project_indices],
            self.issue_keys[project_indices],
            *(
                f[project_indices] for f in self.features
            )
        )
        complement_data = DeepLearningData(
            self.labels[project_complement],
            self.issue_keys[project_complement],
            *(
                f[project_complement] for f in self.features
            )
        )
        return project_data, complement_data

    def split_fraction(self, size: float):
        size_left = int(size * len(self.labels))
        left = DeepLearningData(
            self.labels[:size_left],
            self.issue_keys[:size_left],
            *(
                f[:size_left] for f in self.features
            )
        )
        right = DeepLearningData(
            self.labels[size_left:],
            self.issue_keys[size_left:],
            *(
                f[size_left:] for f in self.features
            )
        )
        return left, right

    def split_k_cross(self, k: int):
        splitter = kfold.StratifiedKFold(k)
        stream = splitter.split(self.features[0], self.extended_labels)
        for indices_left, indices_right in stream:
            left = self.sample_indices(indices_left)
            right = self.sample_indices(indices_right)
            yield left, right

    def split_k_cross_three(self, k: int):
        for ix, iy, iz in custom_kfold.stratified_kfold(k, self.extended_labels):
            x = self.sample_indices(ix)
            y = self.sample_indices(iy)
            z = self.sample_indices(iz)
            yield x, y, z

    def split_cross_project(self, val_split: float):
        # Collect bins of indices
        bins = collections.defaultdict(set)
        for index, issue_key in enumerate(self.issue_keys):
            project = (project
                       if (project := issue_key.split('-')[0]) != 'HBASE'
                       else 'HADOOP')
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
            testing = self.sample_indices(test_indices)
            yield training, validation, testing

    def limit_size(self, max_size: int):
        if len(self.labels) < max_size:
            return self
        return DeepLearningData(
            self.labels[:max_size],
            self.issue_keys[:max_size],
            *(
                f[:max_size] for f in self.features
            )
        )

    def sample_indices(self, indices):
        return DeepLearningData(
            self.labels[indices],
            self.issue_keys[indices],
            *(
                f[indices] for f in self.features
            )
        )


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

    def __init__(self, **kwargs):
        self.max_train = kwargs.pop('max_training_samples', None)
        self.test_project = kwargs.pop('test_project', None)
        self.test_study = kwargs.pop('test_study', None)
        if kwargs:
            keys = ', '.join(kwargs)
            raise ValueError(
                f'Illegal options for splitter {self.__class__.__name__}: {keys}'
            )
        if self.test_project is not None and self.test_study is not None:
            raise ValueError('Cannot combine test_project and test_study')

    @abc.abstractmethod
    def split(self, labels, issue_keys, *features):
        pass


##############################################################################
##############################################################################
# Single Data Splitter
##############################################################################


class SimpleSplitter(DataSplitter):

    def __init__(self, **kwargs):
        self.val_split = kwargs.pop('val_split_size')
        self.test_split = kwargs.pop('test_split_size')
        super().__init__(**kwargs)

    def split(self, labels, issue_keys, *features):
        labels, issue_keys, *features = shuffle_raw_data(labels,
                                                         issue_keys,
                                                         *features)
        data = DeepLearningData(labels, issue_keys, *features)
        if self.test_project is not None:
            test_data, remainder = data.split_on_project(self.test_project)
            val_data, training_data = remainder.split_fraction(self.val_split)
        elif self.test_study is not None:
            test_data, remainder = data.split_on_study(self.test_study)
            val_data, training_data = remainder.split_fraction(self.val_split)
        else:
            size = self.val_split + self.test_split
            training_data, remainder = data.split_fraction(size)
            size = self.val_split / (self.val_split + self.test_split)
            val_data, test_data = remainder.split_fraction(size)
        if self.max_train is not None:
            training_data = training_data.limit_size(self.max_train)
        yield (
            training_data.to_dataset(),
            test_data.to_dataset(),
            val_data.to_dataset(),
            test_data.issue_keys
        )


class CrossFoldSplitter(DataSplitter):

    def __init__(self, **kwargs):
        self.k = kwargs.pop('k')
        super().__init__(**kwargs)
        if self.test_study is not None or self.test_project is not None:
            raise ValueError(
                f'{self.__class__.__name__} does not support test_study or test_project'
            )

    def split(self, labels, issue_keys, *features):
        labels, issue_keys, *features = shuffle_raw_data(labels,
                                                         issue_keys,
                                                         *features)
        data = DeepLearningData(labels, issue_keys, *features)
        for inner, test_data in data.split_k_cross(self.k):
            for training_data, validation_data in inner.split_k_cross(self.k - 1):
                if self.max_train is not None:
                    training_data = training_data.limit_size(self.max_train)
                yield (
                    training_data.to_dataset(),
                    test_data.to_dataset(),
                    validation_data.to_dataset(),
                    test_data.issue_keys
                )


class QuickCrossFoldSplitter(DataSplitter):

    def __init__(self, **kwargs):
        self.k = kwargs.pop('k')
        super().__init__(**kwargs)

    def split(self, labels, issue_keys, *features):
        labels, issue_keys, *features = shuffle_raw_data(labels,
                                                         issue_keys,
                                                         *features)
        data = DeepLearningData(labels, issue_keys, *features)
        if self.test_project is not None or self.test_study is not None:
            if self.test_project is not None:
                testing_data, remainder = data.split_on_project(self.test_project)
            else:
                testing_data, remainder = data.split_on_study(self.test_study)
            for training, validation in data.split_k_cross(self.k):
                if self.max_train is not None:
                    training = training.limit_size(self.max_train)
                yield (
                    training.to_dataset(),
                    testing_data.to_dataset(),
                    validation.to_dataset(),
                    testing_data.issue_keys
                )
        else:
            for training, validation, testing in data.split_k_cross_three(self.k):
                if self.max_train is not None:
                    training = training.limit_size(self.max_train)
                yield (
                    training.to_dataset(),
                    testing.to_dataset(),
                    validation.to_dataset(),
                    testing.issue_keys
                )


class CrossProjectSplitter(DataSplitter):

    def __init__(self, **kwargs):
        self.val_split = kwargs.pop('val_split_size')
        super().__init__(**kwargs)
        if self.test_study is not None or self.test_project is not None:
            raise ValueError(
                f'{self.__class__.__name__} does not support test_study or test_project'
            )
        if self.max_train is not None:
            raise ValueError(f'{self.__class__.__name__} does not support max_train')

    def split(self, labels, issue_keys, *features):
        labels, issue_keys, *features = shuffle_raw_data(labels,
                                                         issue_keys,
                                                         *features)
        data = DeepLearningData(labels, issue_keys, *features)
        for training, validation, testing in data.split_cross_project(self.val_split):
            if self.max_train is not None:
                training = training.limit_size(self.max_train)
            yield (
                training.to_dataset(),
                testing.to_dataset(),
                validation.to_dataset(),
                testing.issue_keys
            )
