import abc

import numpy

from ..model_io import OutputMode
from ..config import Config, Argument, ArgumentConsumer


class AbstractUpSampler(abc.ABC, ArgumentConsumer):

    def __init__(self, conf: Config, /, **hyper_params):
        self.hyper_params = hyper_params
        self.conf = conf
        self.__synthetic_key = 1

    def synthetic_keys(self, n: int):
        keys = [
            f'$SYNTHETIC-{i}'
            for i in range(self.__synthetic_key, self.__synthetic_key + n)
        ]
        self.__synthetic_key += n
        return numpy.asarray(keys)

    def upsample_to_majority(self, labels: numpy.ndarray, keys, *features):
        output_mode = OutputMode.from_string(self.conf.get('run.output-mode'))
        counts = [
            (labels == label).sum() for label in output_mode.label_encoding
        ]
        target = max(counts)
        return self.upsample_to_size(target, labels, keys, *features)

    def upsample_to_size(self, size: int, labels: numpy.ndarray, keys, *features):
        output_mode = OutputMode.from_string(self.conf.get('run.output-mode'))
        targets = {label: size for label in output_mode.label_encoding}
        return self.upsample_to(targets, labels, keys, *features)

    def upsample_to(self, targets: object, labels: object, keys: object, *features: object) -> object:
        indices = {
            target: numpy.where(labels == target)
            for target in targets
        }
        return self.upsample(indices, targets, labels, keys, *features)

    @abc.abstractmethod
    def upsample(self, indices, targets, labels, keys, *features):
        all_labels = [labels]
        all_keys = [keys]
        all_features = [[f] for f in features]
        for label, target in targets.items():
            new_labels, new_keys, *new_features = self.upsample_class(indices[label],
                                                                      target,
                                                                      labels,
                                                                      keys,
                                                                      *features)
            all_labels.append(new_labels)
            all_keys.append(new_keys)
            for f, n_f in zip(all_features, new_features):
                f.append(n_f)
        return (
            numpy.vstack(all_labels),
            numpy.vstack(all_keys),
            [numpy.vstack(f) for f in features]
        )

    @abc.abstractmethod
    def upsample_class(self, indices, target, labels, keys, *features):
        pass

    @staticmethod
    @abc.abstractmethod
    def get_arguments() -> dict[str, Argument]:
        return {}
