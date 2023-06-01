from ..config import Argument
from .generator import AbstractFeatureGenerator, FeatureEncoding
from ..model_io import InputEncoding


CATEGORICAL_ATTRIBUTES = {
    'parent',
    'labels',
    'priority',
    'resolution',
    'status '
}


def concat(stream):
    result = []
    for item in stream:
        result.extend(item)
    return result


class Metadata(AbstractFeatureGenerator):

    @staticmethod
    def input_encoding_type() -> InputEncoding:
        return InputEncoding.Vector

    def generate_vectors(self,
                         tokenized_issues: list[list[str]],
                         metadata,
                         args: dict):
        # When adding any preprocessing here, be sure to
        # implement handling pretrained generators.

        if self.pretrained is None:
            self.save_pretrained({})

        attrs = self.params.get('metadata-attributes', '').split(',')

        return {
            'features': [
                concat(m[a] for a in attrs) for m in metadata
            ],
            'feature_shape': len(metadata[0]),
            'feature_encoding': {
                'encoding': self.feature_encoding(),
                'metadata': [
                    attrs.index(a) for a in CATEGORICAL_ATTRIBUTES if a in attrs
                ]
            }
        }

    @staticmethod
    def feature_encoding() -> FeatureEncoding:
        return FeatureEncoding.Numerical

    @staticmethod
    def get_arguments() -> dict[str, Argument]:
        return {} | super(Metadata, Metadata).get_arguments()
