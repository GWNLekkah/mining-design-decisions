##############################################################################
##############################################################################
# Imports
##############################################################################

import enum

import contractions
import gensim
import nltk

__all__ = [
    'clean_issue_text',
    'FormattingHandling'
]

##############################################################################
##############################################################################
# Public Interface
##############################################################################


def clean_issue_text(text: str) -> list[str]:
    text = fix_punctuation(text)
    sentences = nltk.tokenize.sent_tokenize(text)
    return [f"{' '.join(gensim.utils.tokenize(sent))}" for sent in sentences]


##############################################################################
##############################################################################
# Utility
##############################################################################

class FormattingHandling(enum.Enum):
    Remove = enum.auto()
    Markers = enum.auto()
    Keep = enum.auto()

    @classmethod
    def from_string(cls, x: str) -> 'FormattingHandling':
        match x:
            case 'keep':
                return cls.Keep
            case 'markers':
                return cls.Markers
            case 'remove':
                return cls.Remove
            case _:
                raise ValueError(f'Invalid formatting handling: {x}')

    def as_string(self):
        match self:
            case self.Keep:
                return 'keep'
            case self.Markers:
                return 'markers'
            case self.Remove:
                return 'remove'


def fix_punctuation(text: str) -> str:
    text = contractions.fix(text)
    return text
