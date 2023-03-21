from __future__ import annotations

import enum
import warnings

classification8_lookup = {
    (False, False, False): (1, 0, 0, 0, 0, 0, 0, 0),
    (False, False, True): (0, 0, 0, 1, 0, 0, 0, 0),
    (False, True, False): (0, 0, 1, 0, 0, 0, 0, 0),
    (False, True, True): (0, 0, 0, 0, 0, 0, 1, 0),
    (True, False, False): (0, 1, 0, 0, 0, 0, 0, 0),
    (True, False, True): (0, 0, 0, 0, 0, 1, 0, 0),
    (True, True, False): (0, 0, 0, 0, 1, 0, 0, 0),
    (True, True, True): (0, 0, 0, 0, 0, 0, 0, 1)
}

class InputEncoding(enum.Enum):
    Vector = enum.auto()
    Matrix = enum.auto()
    Embedding = enum.auto()
    Text = enum.auto()


class OutputEncoding(enum.Enum):
    OneHot = enum.auto()
    Binary = enum.auto()



class OutputMode(enum.Enum):
    Detection = enum.auto()
    Classification3 = enum.auto()
    Classification3Simplified = enum.auto()
    Classification8 = enum.auto()

    @classmethod
    def from_string(cls, string: str) -> OutputMode:
        match string:
            case 'Detection':
                return cls.Detection
            case 'Classification3':
                warnings.warn('Classification3 (Binary) is outdated. '
                              'Using this option requires carefully re-implementing '
                              'support for this mode in all places where we deal with '
                              'labels. This can be done by following the code paths '
                              'for all normal and ensemble models, both in training/evaluation mode, '
                              'and in prediction mode. The involved code will raise exceptions '
                              'when Classification3 is used as input mode.')
                return cls.Classification3
            case 'Classification3Simplified':
                return cls.Classification3Simplified
            case 'Classification8':
                return cls.Classification8
        raise ValueError(f'Invalid input: {string}')

    @property
    def output_encoding(self):
        match self:
            case self.Detection:
                return OutputEncoding.Binary
            case self.Classification3:
                return OutputEncoding.Binary
            case self.Classification3Simplified:
                return OutputEncoding.OneHot
            case self.Classification8:
                return OutputEncoding.OneHot

    @property
    def output_size(self):
        match self:
            case self.Detection:
                return 1
            case self.Classification3:
                return 3
            case self.Classification3Simplified:
                return 4
            case self.Classification8:
                return 8

    @property
    def true_category(self) -> str:
        if self != self.Detection:
            raise ValueError(f'No true category exists in mode {self}')
        return 'Architectural'

    @property
    def index_label_encoding(self):
        if self not in (self.Classification3Simplified, self.Classification8):
            raise NotImplementedError
        mapping: dict[tuple[int, ...], str] = self.label_encoding
        return {key.index(1): value for key, value in mapping.items()}

    @property
    def label_encoding(self):
        match self:
            case self.Detection:
                return {
                    0: 'Non-Architectural',
                    1: 'Architectural'
                }
            case self.Classification3:
                # Existence, Executive, Property
                return {
                    (0, 0, 0): 'Non-Architectural',
                    (0, 0, 1): 'Property',
                    (0, 1, 0): 'Executive',
                    (0, 1, 1): 'Executive/Property',
                    (1, 0, 0): 'Existence',
                    (1, 0, 1): 'Existence/Property',
                    (1, 1, 0): 'Existence/Executive',
                    (1, 1, 1): 'Existence/Executive/Property',
                }
            case self.Classification3Simplified:
                return {
                    (1, 0, 0, 0): 'Existence',
                    (0, 1, 0, 0): 'Executive',
                    (0, 0, 1, 0): 'Property',
                    (0, 0, 0, 1): 'Non-Architectural'
                }
            case self.Classification8:
                return {
                    classification8_lookup[(False, False, False)]: 'Non-Architectural',
                    classification8_lookup[(False, False, True)]: 'Property',
                    classification8_lookup[(False, True, False)]: 'Executive',
                    classification8_lookup[(False, True, True)]: 'Executive/Property',
                    classification8_lookup[(True, False, False)]: 'Existence',
                    classification8_lookup[(True, False, True)]: 'Existence/Property',
                    classification8_lookup[(True, True, False)]: 'Existence/Executive',
                    classification8_lookup[(True, True, True)]: 'Existence/Executive/Property',
                }

    @property
    def number_of_classes(self) -> int:
        match self:
            case self.Detection:
                return 2
            case self.Classification3:
                return 8
            case self.Classification3Simplified:
                return 4
            case self.Classification8:
                return 8


