import os

import issue_db_api
from gensim.models.doc2vec import Doc2Vec as GensimDoc2Vec


from ..config import Argument, IntArgument, StringArgument
from .generator import AbstractFeatureGenerator, FeatureEncoding
from ..embeddings.util import load_embedding
from ..model_io import InputEncoding


class Doc2Vec(AbstractFeatureGenerator):

    @staticmethod
    def input_encoding_type() -> InputEncoding:
        return InputEncoding.Vector

    def generate_vectors(self,
                         tokenized_issues: list[list[str]],
                         metadata,
                         args: dict[str, str]):
        if self.pretrained is None:
            # documents = []
            # for idx in range(len(tokenized_issues)):
            #     documents.append(TaggedDocument(tokenized_issues[idx], [idx]))

            # if 'pretrained-file' not in args:
            #     model = GensimDoc2Vec(documents, vector_size=int(args['vector-length']))
            #     filename = 'doc2vec_' + datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S') + '.bin'
            #     model.save(filename)
            #     args['pretrained-file'] = filename

            db: issue_db_api.IssueRepository = self.conf.get('system.storage.database-api')
            filename = load_embedding(self.params['embedding-id'], db, self.conf)

            model = GensimDoc2Vec.load(filename)

            shape = int(args['vector-length'])

            self.save_pretrained(
                {
                    'pretrained-file': filename,
                    'vector-length': shape
                },
                [
                    filename
                ]
            )
        else:
            aux_map = self.conf.get('system.storage.auxiliary_map')
            filename = aux_map[self.pretrained['pretrained-file']]
            model = GensimDoc2Vec.load(filename)
            shape = self.pretrained['vector-length']

        return {
            'features': [
                    model.infer_vector(
                        issue
                    ).tolist()
                    for issue in tokenized_issues],
            'feature_shape': shape,
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
        return {
            'vector-length': IntArgument(
                description='specify the length of the output vector',
                name='vector-length',
                minimum=1
            ),
            'embedding-id': StringArgument(
                name='embedding-id',
                description='ID of the embedding to use',
            )
        } | super(Doc2Vec, Doc2Vec).get_arguments()
