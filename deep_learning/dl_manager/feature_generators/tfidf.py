import collections
import json
import os

import issue_db_api

from ..config import Argument, StringArgument
from .generator import AbstractFeatureGenerator, FeatureEncoding
from ..model_io import InputEncoding


class TfidfGenerator(AbstractFeatureGenerator):

    @staticmethod
    def input_encoding_type() -> InputEncoding:
        return InputEncoding.Vector

    def generate_vectors(self,
                         tokenized_issues: list[list[str]],
                         metadata,
                         args: dict[str, str]):
        if self.pretrained is None:
            db: issue_db_api.IssueRepository = self.conf.get('system.storage.database-api')
            embedding = db.get_embedding_by_id(self.params['dictionary-id'])
            filename = self.params['dictionary-id'] + '.bin'
            if os.path.exists(filename):
                os.remove(filename)
            embedding.download_binary(filename)

            with open(filename) as file:
                tfidf_data = json.load(file)

            layout = tfidf_data['layout']
            inverse_document_frequency = tfidf_data['idf']

            self.save_pretrained(
                {
                    'idf-file': filename
                },
                [
                    filename
                ]
            )
        else:
            aux_map = self.conf.get('system.storage.auxiliary_map')
            filename = aux_map[self.pretrained['idf-file']]
            with open(filename) as file:
                tfidf_data = json.load(file)

            layout = tfidf_data['layout']
            inverse_document_frequency = tfidf_data['idf']


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
        return {
            'features': feature_vectors,
            'feature_shape': len(layout),
            'feature_encoding': {
                'encoding': self.feature_encoding(),
                'metadata': []
            }
        }

    @staticmethod
    def feature_encoding() -> FeatureEncoding:
        return FeatureEncoding.Numerical

    @staticmethod
    def get_arguments() -> dict[str, Argument]:
        return super(TfidfGenerator, TfidfGenerator).get_arguments() | {
            'dictionary-id': StringArgument(
                name='dictionary-id',
                description='ID of the (pretrained) (idf) dictionary to use for TF/IDF feature generation.'
            )
        }
