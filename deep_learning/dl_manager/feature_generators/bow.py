import abc
import collections

from ..logger import timer

from .generator import AbstractFeatureGenerator, ParameterSpec, FeatureEncoding
from ..model_io import InputEncoding


class AbstractBOW(AbstractFeatureGenerator, abc.ABC):
    @staticmethod
    def input_encoding_type() -> InputEncoding:
        return InputEncoding.Vector

    def generate_vectors(self,
                         tokenized_issues: list[list[str]],
                         metadata,
                         args: dict[str, str]):
        with timer('BOW Feature Generation'):
            if self.pretrained is None:
                doc_count = collections.defaultdict(int)
                for tokenized_issue in tokenized_issues:
                    for token in set(tokenized_issue):
                        # if token not in word_to_idx:
                        #     word_to_idx[token] = idx
                        #     idx += 1
                        doc_count[token] += 1
                min_count = int(self.params.get('min-doc-count', '0'))
                print(min_count)
                included = [w for w, c in doc_count.items() if c >= min_count]
                idx = len(included)
                word_to_idx = {w: i for i, w in enumerate(included)}
                self.save_pretrained(
                    {
                        'word-to-index-mapping': word_to_idx,
                        'max-index': len(word_to_idx)
                    }
                )
            else:
                word_to_idx = self.pretrained['word-to-index-mapping']
                idx = self.pretrained['max-index']

            bags = []
            for tokenized_issue in tokenized_issues:
                bag = [0] * idx
                for token in tokenized_issue:
                    if token in word_to_idx:    # In pretrained mode, ignore unknown words.
                        token_idx = word_to_idx[token]
                        bag[token_idx] += self.get_word_value(len(tokenized_issue))
                bags.append(bag)

        return {
            'features': bags,
            'feature_shape': idx,
            'feature_encoding': {
                'encoding': self.feature_encoding(),
                'metadata': []
            }
        }

    @staticmethod
    @abc.abstractmethod
    def get_word_value(divider):
        pass

    @staticmethod
    def feature_encoding() -> FeatureEncoding:
        return FeatureEncoding.Numerical

    @staticmethod
    def get_parameters() -> dict[str, ParameterSpec]:
        return {
            'min-doc-count': ParameterSpec(
                description='Minimum number of document occurrences for a word to be included',
                type='int'
            )
        } | super(AbstractBOW, AbstractBOW).get_parameters()
