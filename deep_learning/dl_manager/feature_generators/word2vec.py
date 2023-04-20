import abc
import datetime
import os

import issue_db_api
from gensim.models import Word2Vec as GensimWord2Vec
from gensim import models

from .generator import FeatureEncoding
from ..config import conf

from ..feature_generators import AbstractFeatureGenerator, ParameterSpec


class AbstractWord2Vec(AbstractFeatureGenerator, abc.ABC):
    def generate_vectors(self,
                         tokenized_issues: list[list[str]],
                         metadata,
                         args: dict[str, str]):
        # Train or load a model
        if self.pretrained is None:
            db: issue_db_api.IssueRepository = conf.get('system.storage.database.api')
            embedding = db.get_embedding_by_id(self.params['embedding-id'])
            filename = self.params['embedding-id'] + '.bin'
            if os.path.exists(filename):
                os.remove(filename)
            embedding.download_binary(filename)

            # Load the model
            wv = models.KeyedVectors.load_word2vec_format(filename, binary=True)
        else:
            aux_map = conf.get('system.storage.auxiliary_map')
            filename = aux_map[self.pretrained['model']]
            wv = models.KeyedVectors.load_word2vec_format(filename, binary=True)

        # Build the final feature vectors.
        # This function should also save the pretrained model
        return self.finalize_vectors(tokenized_issues, wv, args)

    @staticmethod
    @abc.abstractmethod
    def finalize_vectors(tokenized_issues, wv, args):
        pass

    @staticmethod
    def feature_encoding() -> FeatureEncoding:
        return FeatureEncoding.Numerical

    @staticmethod
    def get_parameters() -> dict[str, ParameterSpec]:
        return {
            'vector-length': ParameterSpec(
                description='specify the length of the output vector',
                type='int'
            ),
            'min-count': ParameterSpec(
                description='minimum occurrence for a word to be in the word2vec',
                type='int'
            ),
           'embedding-id': ParameterSpec(
               description='ID of the word embedding to use',
               type='str',
           )
        } | super(AbstractWord2Vec, AbstractWord2Vec).get_parameters()
