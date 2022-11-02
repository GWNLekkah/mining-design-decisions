import datetime

from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec as GensimDoc2Vec

from .generator import AbstractFeatureGenerator, InputEncoding, ParameterSpec


class Doc2Vec(AbstractFeatureGenerator):

    @staticmethod
    def input_encoding_type() -> InputEncoding:
        return InputEncoding.Vector

    def generate_vectors(self,
                         tokenized_issues: list[list[str]],
                         metadata,
                         args: dict[str, str]):
        if self.pretrained is None:
            documents = []
            for idx in range(len(tokenized_issues)):
                documents.append(TaggedDocument(tokenized_issues[idx], [idx]))

            if 'pretrained-file' not in args:
                model = GensimDoc2Vec(documents, vector_size=int(args['vector-length']))
                filename = 'doc2vec_' + datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S') + '.bin'
                model.save(filename)
                args['pretrained-file'] = filename

            model = GensimDoc2Vec.load(args['pretrained-file'])
        else:
            raise NotImplementedError(
                f'{self.__class__.__name__} does not implement loading a pre-trained generator'
            )

        return {'features': [
                    model.infer_vector(
                        issue
                    ).tolist()
                    for issue in tokenized_issues],
                'feature_shape': int(args['vector-length'])}

    @staticmethod
    def get_parameters() -> dict[str, ParameterSpec]:
        return {
            'vector-length': ParameterSpec(
                description='specify the length of the output vector'
            ),
        } | super(Doc2Vec, Doc2Vec).get_parameters()
