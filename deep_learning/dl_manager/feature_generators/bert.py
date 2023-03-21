from ..model_io import InputEncoding
from .word2vec import AbstractFeatureGenerator, ParameterSpec
from transformers import AutoTokenizer


class Bert(AbstractFeatureGenerator):
    @staticmethod
    def input_encoding_type() -> InputEncoding:
        return InputEncoding.Text

    def generate_vectors(self,
                         tokenized_issues: list[list[str]],
                         metadata,
                         args: dict[str, str]):
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        tokens = tokenizer(
            [' '.join(issue) for issue in tokenized_issues],
            padding=True,
            max_length=512,
            truncation=True,
            return_tensors='np'
        ).data

        self.save_pretrained({})

        return {
            'features': tokens,
            'feature_shape': None
        }

    @staticmethod
    def get_parameters() -> dict[str, ParameterSpec]:
        return super(Bert, Bert).get_parameters()
