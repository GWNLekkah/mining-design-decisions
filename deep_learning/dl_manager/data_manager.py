##############################################################################
##############################################################################
# Imports
##############################################################################

import typing

##############################################################################
##############################################################################
# Result Object
##############################################################################


class Dataset(typing.NamedTuple):
    features: typing.Any
    labels: list
    binary_labels: list
    shape: int | tuple[int]
    embedding_weights: None | list[float]
    vocab_size: None | int
    weight_vector_length: None | int
    issue_keys: list
    ids: list

    def is_embedding(self):
        return (
            self.embedding_weights is not None and
            self.vocab_size is not None and
            self.weight_vector_length is not None
        )
