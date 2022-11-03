import warnings

from ..classifiers import InputEncoding
from .word2vec import AbstractWord2Vec


class Word2Vec1D(AbstractWord2Vec):

    @staticmethod
    def input_encoding_type() -> InputEncoding:
        return InputEncoding.Embedding

    def finalize_vectors(self, tokenized_issues, wv, args):
        if self.pretrained is None:
            idx = 0
            word_to_idx = dict()
            embedding_weights = []
            feature_shape = int(args['max-len'])
            word_vector_length = int(args['vector-length'])
        else:
            word_to_idx = self.pretrained['word-to-index-mapping']
            idx = self.pretrained['max-index']
            embedding_weights = self.pretrained['embedding-weights']
            feature_shape = self.pretrained['feature-shape']
            word_vector_length = self.pretrained['word-vector-length']
        features = []
        original_text = []
        for tokenized_issue in tokenized_issues:
            feature = []
            current_issue_original_text = []
            for token in tokenized_issue:
                if token in word_to_idx:
                    current_issue_original_text.append(token)
                    feature.append([word_to_idx[token]])
                else:
                    # Be sure to only add to mapping when not
                    # using a pretrained generator.
                    if token in wv and self.pretrained is None:
                        current_issue_original_text.append(token)
                        word_to_idx[token] = idx
                        embedding_weights.append(wv[token].tolist())
                        feature.append([idx])
                        idx += 1
            original_text.append(current_issue_original_text)
            feature.extend([[0]] * (int(args['max-len']) - len(feature)))
            features.append(feature)

        if self.pretrained is None:
            self.save_pretrained(
                {
                    'word-to-index-mapping': word_to_idx,
                    'max-index': idx,
                    'embedding-weights': embedding_weights,
                    'feature-shape': feature_shape,
                    'word-vector-length': word_vector_length,
                    'model': self.__params['pretrained-file'],
                    'model-binary': self.__params['pretrained-binary'].lower() == 'true'
                }
            )

        return {'features': features,
                'weights': embedding_weights,
                'feature_shape': feature_shape,
                'vocab_size': idx,
                'word_vector_length': word_vector_length,
                'original': original_text
                }
