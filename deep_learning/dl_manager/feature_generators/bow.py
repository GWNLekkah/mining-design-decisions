import abc
import json
import os

import issue_db_api

from ..logger import timer

from ..config import Argument, StringArgument
from .generator import AbstractFeatureGenerator, FeatureEncoding
from ..embeddings.util import load_embedding
from ..model_io import InputEncoding


class AbstractBOW(AbstractFeatureGenerator, abc.ABC):
    @staticmethod
    def input_encoding_type() -> InputEncoding:
        return InputEncoding.Vector

    def generate_vectors(self,
                         tokenized_issues: list[list[str]],
                         metadata,
                         args: dict[str, str]):
        with timer('BOW Feature Generation'):
            if self.pretrained is None:
                db: issue_db_api.IssueRepository = self.conf.get('system.storage.database-api')
                filename = load_embedding(self.params['dictionary-id'], db, self.conf)

                with open(filename) as file:
                    word_to_idx = json.load(file)

                directory = os.path.split(filename)[0]
                self.save_pretrained(
                    {
                        'dict-file': filename
                    },
                    [
                        os.path.join(directory, path)
                        for path in os.listdir(directory)
                    ]
                )
            else:
                aux_map = self.conf.get('system.storage.auxiliary_map')
                filename = aux_map[self.pretrained['dict-file']]
                with open(filename) as file:
                    word_to_idx = json.load(file)

            bags = []
            for tokenized_issue in tokenized_issues:
                bag = [0] * len(word_to_idx)
                for token in tokenized_issue:
                    if token in word_to_idx:    # In pretrained mode, ignore unknown words.
                        token_idx = word_to_idx[token]
                        bag[token_idx] += self.get_word_value(len(tokenized_issue))
                bags.append(bag)

        return {
            'features': bags,
            'feature_shape': len(word_to_idx),
            'feature_encoding': {
                'encoding': self.feature_encoding(),
                'metadata': []
            }
        }

    @staticmethod
    @abc.abstractmethod
    def get_word_value(divider):
        pass

    @staticmethod
    def feature_encoding() -> FeatureEncoding:
        return FeatureEncoding.Numerical

    @staticmethod
    def get_arguments() -> dict[str, Argument]:
        return {
            'dictionary-id': StringArgument(
                name='dictionary-id',
                description='ID of the (pretrained) dictionary to use for BOW feature generation.'
            )
        } | super(AbstractBOW, AbstractBOW).get_arguments()
