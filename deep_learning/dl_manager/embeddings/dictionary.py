import collections
import json
import pathlib

from .embedding_generator import AbstractEmbeddingGenerator
from ..config import Argument, IntArgument


class DictionaryGenerator(AbstractEmbeddingGenerator):

    def generate_embedding(self, issues: list[list[str]], path: pathlib.Path):
        # Generate a frequency table
        frequencies = collections.defaultdict(int)
        for issue in issues:
            for word in set(issue):
                frequencies[word] += 1
        # Select words to be contained in dictionary
        threshold = self.params['min-doc-count']
        dictionary = [
            word for word, frequency in frequencies.items()
            if frequency >= threshold
        ]
        # Let's make the dictionary sorted
        dictionary.sort()
        # Convert to a word -> index mapping
        #dictionary = {v: k for k, v in dict(enumerate(dictionary)).items()}
        dictionary = {w: i for i, w in enumerate(dictionary)}
        # Save dictionary
        with open(path, 'w') as file:
            json.dump(dictionary, file)

    @staticmethod
    def get_arguments() -> dict[str, Argument]:
        return super(DictionaryGenerator, DictionaryGenerator).get_arguments() | {
            'min-doc-count': IntArgument(
                name='min-doc-count',
                description='Minimum document count for a word to be included in the dictionary',
                minimum=0
            )
        }
