import collections
import math

from . import ParameterSpec
from .generator import AbstractFeatureGenerator
from ..classifiers import InputEncoding


class TfidfGenerator(AbstractFeatureGenerator):

    @staticmethod
    def input_encoding_type() -> InputEncoding:
        return InputEncoding.Vector

    def generate_vectors(self,
                         tokenized_issues: list[list[str]],
                         metadata,
                         args: dict[str, str]):
        if self.pretrained is None:
            document_frequency = collections.defaultdict(int)
            for document in tokenized_issues:
                for word in set(document):
                    document_frequency[word] += 1
            inverse_document_frequency = collections.defaultdict(float)
            inverse_document_frequency.update({
                term: math.log10(len(tokenized_issues) / count)
                for term, count in document_frequency.items()
            })
            # Layout (in words) of the resulting feature vectors
            layout = sorted(document_frequency)
            self.save_pretrained(
                {
                    'idf': inverse_document_frequency,
                    'word-order': layout
                }
            )
        else:
            inverse_document_frequency = self.pretrained['idf']
            layout = self.pretrained['word-order']
        feature_vectors = []
        for document in tokenized_issues:
            term_counts = collections.defaultdict(int)
            for word in document:
                term_counts[word] += 1
            term_frequency = collections.defaultdict(float)
            for word, count in term_counts.items():
                term_frequency[word] = count / len(document)
            vector = [
                term_frequency[term] * inverse_document_frequency[term]
                for term in layout
            ]
            feature_vectors.append(vector)
        assert len(set(len(x) for x in feature_vectors)) == 1
        print(len(feature_vectors))
        print(set(len(x) for x in feature_vectors))
        return {
            'features': feature_vectors,
            'feature_shape': len(layout)
        }

    @staticmethod
    def get_parameters() -> dict[str, ParameterSpec]:
        return super(TfidfGenerator, TfidfGenerator).get_parameters()
