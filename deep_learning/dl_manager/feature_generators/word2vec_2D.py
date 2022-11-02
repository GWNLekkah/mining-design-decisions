import warnings

import numpy

from ..classifiers import InputEncoding
from .word2vec import AbstractWord2Vec


class Word2Vec2D(AbstractWord2Vec):

    @staticmethod
    def input_encoding_type() -> InputEncoding:
        return InputEncoding.Matrix

    def finalize_vectors(self, tokenized_issues, wv, args):
        if self.pretrained is None:
            idx = 0
            word_to_idx = dict()
            embedding_weights = []
            word_vector_length = int(args['vector-length'])
        else:
            word_to_idx = self.pretrained['word-to-index-mapping']
            idx = self.pretrained['max-index']
            embedding_weights = self.pretrained['embedding-weights']
            word_vector_length = self.pretrained['word-vector-length']
            model = ...
            raise NotImplementedError('Word2Vec Model Loading not implemented')
        features = []
        original_text = []
        for tokenized_issue in tokenized_issues:
            feature = []
            current_issue_original_text = []
            for token in tokenized_issue:
                if token in word_to_idx:
                    current_issue_original_text.append(token)
                    feature.append(wv[token].tolist())
                else:
                    if token in wv:
                        current_issue_original_text.append(token)
                        word_to_idx[token] = idx
                        embedding_weights.append(wv[token].tolist())
                        feature.append(wv[token].tolist())
                        idx += 1
            original_text.append(current_issue_original_text)
            feature.extend([[0] * int(args['vector-length'])] * (int(args['max-len']) - len(feature)))
            features.append(feature)

        if self.pretrained is None:
            warnings.warn('Word2Vec Model saving not implemented')
            # self.save_pretrained(
            #     {
            #         'word-to-index-mapping': word_to_idx,
            #         'max-index': idx,
            #         'embedding-weights': embedding_weights,
            #         'feature-shape': feature_shape,
            #         'word-vector-length': word_vector_length,
            #         'model': ...
            #     }
            # )

        return {'features': features,
                'weights': embedding_weights,
                'feature_shape': numpy.shape(features[0]),
                'vocab_size': idx,
                'word_vector_length': word_vector_length,
                'original': original_text
                }
