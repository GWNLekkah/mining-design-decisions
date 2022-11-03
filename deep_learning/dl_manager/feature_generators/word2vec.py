import abc
import datetime

from gensim.models import Word2Vec as GensimWord2Vec
from gensim import models
from ..feature_generators import AbstractFeatureGenerator, ParameterSpec


class AbstractWord2Vec(AbstractFeatureGenerator, abc.ABC):
    def generate_vectors(self,
                         tokenized_issues: list[list[str]],
                         metadata,
                         args: dict[str, str]):
        # Train or load a model
        if self.pretrained is None:
            if 'pretrained-file' not in args:
                model = GensimWord2Vec(tokenized_issues, min_count=int(args['min-count']),
                                       vector_size=int(args['vector-length']))
                filename = 'word2vec_' + datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S') + '.bin'
                model.wv.save_word2vec_format(filename, binary=True)
                args['pretrained-file'] = filename
                args['pretrained-binary'] = 'True'

            # Load the model
            wv = models.KeyedVectors.load_word2vec_format(
                args['pretrained-file'], binary=bool(args['pretrained-binary'])
            )
        else:
            wv = models.KeyedVectors.load_word2vec_format(
                self.pretrained['model'],
                binary=self.pretrained['model-binary']
            )

        # Build the final feature vectors.
        # This function should also save the pretrained model
        return self.finalize_vectors(tokenized_issues, wv, args)

    @staticmethod
    @abc.abstractmethod
    def finalize_vectors(tokenized_issues, wv, args):
        pass

    @staticmethod
    def get_parameters() -> dict[str, ParameterSpec]:
        return {
            'vector-length': ParameterSpec(
                description='specify the length of the output vector'
            ),
            'min-count': ParameterSpec(
                description='minimum occurrence for a word to be in the word2vec'
            ),
            'pretrained-file': ParameterSpec(
                description='specify path to the pretrained word2vec model'
            ),
            'pretrained-binary': ParameterSpec(
                description='specify is pretrained word2vec is binary'
            ),
        } | super(AbstractWord2Vec, AbstractWord2Vec).get_parameters()
