import collections

from sklearn.model_selection import StratifiedKFold as _StratifiedKFold
import numpy


class StratifiedKFold:

    def __init__(self, k):
        self.__k = k
        self.__kfold = _StratifiedKFold(n_splits=self.__k)

    def split(self, x, y):
        labels = self.__simplify_labels(y)
        return self.__kfold.split(x, labels)

    def __simplify_labels(self, labels):
        return list(self.__simplify_labels_iter(labels))

    def __simplify_labels_iter(self, labels):
        label_mapping = {}
        next_number = 0
        for label in labels:
            key = _to_tuple(label)
            if key not in label_mapping:
                label_mapping[key] = next_number
                next_number += 1
            yield label_mapping[key]


def _to_tuple(x):
    if not isinstance(x, (list, tuple, numpy.ndarray)):
        return x
    return tuple(_to_tuple(y) for y in x)


# sklean.model_selection.train_test_split complains if a
# label only occurs once; this class is a workaround.
class StratifiedSplit:

    def __init__(self, size: float):
        self._size = size

    def split(self, x, y):
        # Input validation
        if isinstance(x, list):
            x = numpy.array(x)
        if isinstance(y, list):
            y = numpy.array(y)
        assert x.shape[0] == y.shape[0]

        # Target size
        n = int(self._size * x.shape[0])

        # Compute auxiliary data
        bins = collections.defaultdict(int)
        indices_by_label = collections.defaultdict(list)
        for i, z in enumerate(y):
            z = _to_tuple(z)
            bins[z] += 1
            indices_by_label[z].append(i)

        # Idea: compute the amount of samples from
        # each class that should be in the "left" set,
        # and compare with the exact (fractional)
        # amount. Use the sum of fractional amounts
        # to determine how many items should be added
        # from the "leftovers". Select those based on
        # largest fractional remainders.
        deltas = {cls: amount*self._size - int(amount * self._size)
                  for cls, amount in bins.items()}
        remainder = int(sum(deltas.values()))
        order = sorted(deltas, key=lambda cls: deltas[cls])

        left = []
        right = []
        for i, cls in enumerate(order):
            amount = int(self._size * bins[cls]) + (i < remainder)
            left.extend(indices_by_label[cls][:amount])
            right.extend(indices_by_label[cls][amount:])

        return numpy.array(left), numpy.array(right)
