import random

from . import base


class RandomUpSampler(base.AbstractUpSampler):

    def upsample(self, indices, targets, labels, keys, *features):
        # Delegate to super and use upsample_class instead
        return super().upsample(indices, targets, labels, keys, *features)

    def upsample_class(self, indices, target, labels, keys, *features):
        new = tuple(random.choices(indices, k=target - len(indices)))
        return (
            labels[new],
            keys[new]
            *(
                f[new] for f in features
            )
        )

    @staticmethod
    def get_arguments():
        return {}
