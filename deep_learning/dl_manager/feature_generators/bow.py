import abc

from .generator import AbstractFeatureGenerator, InputEncoding, ParameterSpec


class AbstractBOW(AbstractFeatureGenerator, abc.ABC):
    @staticmethod
    def input_encoding_type() -> InputEncoding:
        return InputEncoding.Vector

    def generate_vectors(self,
                         tokenized_issues: list[list[str]],
                         metadata,
                         args: dict[str, str]):
        if self.pretrained is None:
            word_to_idx = dict()
            idx = 0
            for tokenized_issue in tokenized_issues:
                for token in tokenized_issue:
                    if token not in word_to_idx:
                        word_to_idx[token] = idx
                        idx += 1
            self.save_pretrained(
                {
                    'word-to-index-mapping': word_to_idx,
                    'max-index': idx
                }
            )
        else:
            word_to_idx = self.pretrained['word-to-index-mapping']
            idx = self.pretrained['max-index']

        bags = []
        for tokenized_issue in tokenized_issues:
            bag = [0] * idx
            for token in tokenized_issue:
                token_idx = word_to_idx[token]
                bag[token_idx] += self.get_word_value(len(tokenized_issue))
            bags.append(bag)

        return {
            'features': bags,
            'feature_shape': idx
        }

    @staticmethod
    @abc.abstractmethod
    def get_word_value(divider):
        pass

    @staticmethod
    def get_parameters() -> dict[str, ParameterSpec]:
        return {} | super(AbstractBOW, AbstractBOW).get_parameters()
