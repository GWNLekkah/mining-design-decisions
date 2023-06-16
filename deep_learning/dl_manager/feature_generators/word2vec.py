import abc
import os

from gensim import models
import issue_db_api

from ..config import Argument, IntArgument, StringArgument

from .generator import FeatureEncoding
from ..embeddings.util import load_embedding
from ..feature_generators import AbstractFeatureGenerator

class AbstractWord2Vec(AbstractFeatureGenerator, abc.ABC):
    def generate_vectors(self,
                         tokenized_issues: list[list[str]],
                         metadata,
                         args: dict[str, str]):
        # Train or load a model
        if self.pretrained is None:
            db: issue_db_api.IssueRepository = self.conf.get('system.storage.database-api')
            filename = load_embedding(self.params['embedding-id'], db, self.conf)

            # Load the model
            wv = models.KeyedVectors.load_word2vec_format(filename, binary=True)
        else:
            aux_map = self.conf.get('system.storage.auxiliary_map')
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
    def get_arguments() -> dict[str, Argument]:
        args = {
            'vector-length': IntArgument(
                name='vector-length',
                minimum=1,
                description='specify the length of the output vector',
            ),
           'embedding-id': StringArgument(
               name='embedding-id',
               description='ID of the word embedding to use',
           )
        } | super(AbstractWord2Vec, AbstractWord2Vec).get_arguments()
        args['max-len'] = IntArgument(
            name='max-len',
            description='Maximum length of the input issues. Must be positive for embeddings',
            minimum=1
        )
        return args
