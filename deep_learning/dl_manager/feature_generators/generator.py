##############################################################################
##############################################################################
# Imports
##############################################################################

from __future__ import annotations

import abc
import ast
import csv
import enum
import json
import pathlib
import random
import typing
import warnings

import gensim
import nltk

from ..classifiers import InputEncoding, OutputEncoding
from ..custom_kfold import stratified_trim
from .util import ontology
from ..config import conf

from ..data_manager_bootstrap import get_raw_text_file_name

csv.field_size_limit(100000000)

classification8_lookup = {
    (False, False, False): (1, 0, 0, 0, 0, 0, 0, 0),
    (False, False, True): (0, 0, 0, 1, 0, 0, 0, 0),
    (False, True, False): (0, 0, 1, 0, 0, 0, 0, 0),
    (False, True, True): (0, 0, 0, 0, 0, 0, 1, 0),
    (True, False, False): (0, 1, 0, 0, 0, 0, 0, 0),
    (True, False, True): (0, 0, 0, 0, 0, 1, 0, 0),
    (True, True, False): (0, 0, 0, 0, 1, 0, 0, 0),
    (True, True, True): (0, 0, 0, 0, 0, 0, 0, 1)
}

POS_CONVERSION = {
    "JJ": "a",
    "JJR": "a",
    "JJS": "a",
    "NN": "n",
    "NNS": "n",
    "NNP": "n",
    "NNPS": "n",
    "RB": "r",
    "RBR": "r",
    "RBS": "r",
    "VB": "v",
    "VBD": "v",
    "VBG": "v",
    "VBN": "v",
    "VBP": "v",
    "VBZ": "v",
    "WRB": "r",
}

ATTRIBUTE_CONSTANTS = {
    'n_attachments': 'n_attachments',
    'n_comments': 'n_comments',
    'len_comments': 'len_comments',
    'n_components': 'n_components',
    'len_description': 'len_description',
    'n_issuelinks': 'n_issuelinks',
    'n_labels': 'n_labels',
    'parent': 'parent',
    'n_subtasks': 'n_subtasks',
    'len_summary': 'len_summary',
    'n_votes': 'n_votes',
    'n_watches': 'n_watches',
    'components': 'components',
    'issuetype': 'issuetype',
    'labels': 'labels',
    'priority': 'priority',
    'resolution': 'resolution',
    'status': 'status'
}

##############################################################################
##############################################################################
# Auxiliary Classes
##############################################################################


class _NullDict(dict):
    def __missing__(self, key):
        return key


class ParameterSpec(typing.NamedTuple):
    description: str
    type: str


class OutputMode(enum.Enum):
    Detection = enum.auto()
    Classification3 = enum.auto()
    Classification3Simplified = enum.auto()
    Classification8 = enum.auto()

    @classmethod
    def from_string(cls, string: str) -> OutputMode:
        match string:
            case 'Detection':
                return cls.Detection
            case 'Classification3':
                warnings.warn('Classification3 (Binary) is outdated. '
                              'Using this option requires carefully re-implementing '
                              'support for this mode in all places where we deal with '
                              'labels. This can be done by following the code paths '
                              'for all normal and ensemble models, both in training/evaluation mode, '
                              'and in prediction mode. The involved code will raise exceptions '
                              'when Classification3 is used as input mode.')
                return cls.Classification3
            case 'Classification3Simplified':
                return cls.Classification3Simplified
            case 'Classification8':
                return cls.Classification8
        raise ValueError(f'Invalid input: {string}')

    @property
    def output_encoding(self):
        match self:
            case self.Detection:
                return OutputEncoding.Binary
            case self.Classification3:
                return OutputEncoding.Binary
            case self.Classification3Simplified:
                return OutputEncoding.OneHot
            case self.Classification8:
                return OutputEncoding.OneHot

    @property
    def output_size(self):
        match self:
            case self.Detection:
                return 1
            case self.Classification3:
                return 3
            case self.Classification3Simplified:
                return 4
            case self.Classification8:
                return 8

    @property
    def true_category(self) -> str:
        if self != self.Detection:
            raise ValueError(f'No true category exists in mode {self}')
        return 'Architectural'

    @property
    def index_label_encoding(self):
        if self not in (self.Classification3Simplified, self.Classification8):
            raise NotImplementedError
        mapping: dict[tuple[int, ...], str] = self.label_encoding
        return {key.index(1): value for key, value in mapping.items()}

    @property
    def label_encoding(self):
        match self:
            case self.Detection:
                return {
                    0: 'Non-Architectural',
                    1: 'Architectural'
                }
            case self.Classification3:
                # Existence, Executive, Property
                return {
                    (0, 0, 0): 'Non-Architectural',
                    (0, 0, 1): 'Property',
                    (0, 1, 0): 'Executive',
                    (0, 1, 1): 'Executive/Property',
                    (1, 0, 0): 'Existence',
                    (1, 0, 1): 'Existence/Property',
                    (1, 1, 0): 'Existence/Executive',
                    (1, 1, 1): 'Existence/Executive/Property',
                }
            case self.Classification3Simplified:
                return {
                    (1, 0, 0, 0): 'Existence',
                    (0, 1, 0, 0): 'Executive',
                    (0, 0, 1, 0): 'Property',
                    (0, 0, 0, 1): 'Non-Architectural'
                }
            case self.Classification8:
                return {
                    classification8_lookup[(0, 0, 0)]: 'Non-Architectural',
                    classification8_lookup[(0, 0, 1)]: 'Property',
                    classification8_lookup[(0, 1, 0)]: 'Executive',
                    classification8_lookup[(0, 1, 1)]: 'Executive/Property',
                    classification8_lookup[(1, 0, 0)]: 'Existence',
                    classification8_lookup[(1, 0, 1)]: 'Existence/Property',
                    classification8_lookup[(1, 1, 0)]: 'Existence/Executive',
                    classification8_lookup[(1, 1, 1)]: 'Existence/Executive/Property',
                }

    @property
    def number_of_classes(self) -> int:
        match self:
            case self.Detection:
                return 2
            case self.Classification3:
                return 8
            case self.Classification3Simplified:
                return 4
            case self.Classification8:
                return 8


##############################################################################
##############################################################################
# Main Class
##############################################################################


class AbstractFeatureGenerator(abc.ABC):

    def __init__(self, *,
                 pretrained_generator_settings: dict | None = None,
                 **params):
        self.__params = params
        self.__pretrained = pretrained_generator_settings
        if self.__pretrained is not None:
            if self.__params:
                raise ValueError(
                    'Feature generator does not take params when pretrained settings are given'
                )
            # Populate params for default pre-processing,
            # which does not require any trained settings.
            for name in AbstractFeatureGenerator.get_parameters():
                if name in self.__pretrained:
                    self.__params[name] = self.__pretrained[name]

    @property
    def params(self) -> dict[str, str]:
        return self.__params

    @property
    def pretrained(self) -> dict | None:
        return self.__pretrained

    def save_pretrained(self, pretrained_settings: dict):
        settings = '_'.join(
            f'{key}-{value}' for key, value in self.__params.items()
        )
        filename = f'{self.__class__.__name__}__{settings}.json'
        for name in AbstractFeatureGenerator.get_parameters():
            if name in self.__params:
                pretrained_settings[name] = self.__params[name]
        while filename in conf.get('system.storage.generators'):
            filename = f'x_{filename}'
        conf.get('system.storage.generators').append(filename)
        with open(filename, 'w') as file:
            json.dump(
                {
                    'settings': pretrained_settings,
                    'generator': self.__class__.__name__,
                },
                file
            )

    @staticmethod
    @abc.abstractmethod
    def input_encoding_type() -> InputEncoding:
        """Type of input encoding generated by this generator.
        """

    @abc.abstractmethod
    def generate_vectors(self,
                         tokenized_issues: list[list[str]],
                         metadata,
                         args: dict[str, str]):
        # TODO: implement this method
        # TODO: this method should take in data, and generate
        # TODO: the corresponding feature vectors
        pass

    @staticmethod
    @abc.abstractmethod
    def get_parameters() -> dict[str, ParameterSpec]:
        return {
            'max-len': ParameterSpec(
                description='words limit of the issue text',
                type='int'
            ),
            'disable-lowercase': ParameterSpec(
                description='transform words to lowercase',
                type='bool'
            ),
            'disable-stopwords': ParameterSpec(
                description='remove stopwords from text',
                type='bool'
            ),
            'use-stemming': ParameterSpec(
                description='stem the words in the text',
                type='bool'
            ),
            'use-lemmatization': ParameterSpec(
                description='Use lemmatization on words in the text',
                type='bool'
            ),
            'use-pos': ParameterSpec(
                'Enhance words in the text with part of speech information',
                type='bool'
            ),
            'class-limit': ParameterSpec(
                description='limit the amount of items per class',
                type='int'
            ),
        }

    def generate_features(self,
                          source_filename: pathlib.Path,
                          target_filename: pathlib.Path):
        """Generate features from the data in the given source file,
        and store the results in the given target file.
        """
        with open(source_filename) as file:
            issues = json.load(file)
            # DONT USE EVAL!
            #metadata_attributes = eval(self.__params.get('metadata-attributes', '[]'), ATTRIBUTE_CONSTANTS)
            metadata_attributes = [
                attr for attr in self.__params.get('metadata-attributes', '').split(',') if attr
            ]
            for attr in metadata_attributes:
                if attr not in ATTRIBUTE_CONSTANTS:
                    raise ValueError(f'Unknown metadata attribute: {attr}')

            texts = []
            metadata = []
            labels = {
                'detection': [],
                'classification3': [],
                'classification3simplified': [],
                'classification8': [],
                'issue_keys': []
            }
            classification_indices = {
                'Existence': [],
                'Property': [],
                'Executive': [],
                'Non-Architectural': []
            }
            current_index = 0
            for issue in issues:
                if self.pretrained is None:
                    # Add labels if not using a pre-trained dataset.
                    is_cat1 = issue['is-cat1']['value'] == 'True'
                    is_cat2 = issue['is-cat2']['value'] == 'True'
                    is_cat3 = issue['is-cat3']['value'] == 'True'
                    key = (is_cat1, is_cat2, is_cat3)

                    if is_cat2:  # Executive
                        labels['classification3simplified'].append((0, 1, 0, 0))
                        classification_indices['Executive'].append(current_index)
                    elif is_cat3:  # Property
                        labels['classification3simplified'].append((0, 0, 1, 0))
                        classification_indices['Property'].append(current_index)
                    elif is_cat1:  # Existence
                        labels['classification3simplified'].append((1, 0, 0, 0))
                        classification_indices['Existence'].append(current_index)
                    else:  # Non-architectural
                        labels['classification3simplified'].append((0, 0, 0, 1))
                        classification_indices['Non-Architectural'].append(current_index)

                    if issue['is-design'] == 'True':
                        labels['detection'].append(True)
                    else:
                        labels['detection'].append(False)

                    labels['classification8'].append(classification8_lookup[key])
                    labels['classification3'].append(key)
                    labels['issue_keys'].append(issue['key'] + '-' + issue['study'])

                texts.append(issue['summary'] + issue['description'])

                new_metadata = []
                for attribute in metadata_attributes:
                    new_metadata.extend(issue['metadata'][attribute])
                metadata.append(new_metadata)

                current_index += 1

        try:
            texts
        except NameError:   # This is necessary to make Pycharm shut up
            raise ValueError('BUG')

        limit = int(self.params.get('class-limit', -1))
        if limit != -1 and self.pretrained is None:     # Only execute if not pretrained
            random.seed(42)
            stratified_indices = []
            for issue_type in classification_indices.keys():
                project_labels = [label for index, label in enumerate([label.split('-')[0]
                                                                       for label in labels['issue_keys']])
                                  if index in classification_indices[issue_type]]
                trimmed_indices = stratified_trim(limit, project_labels)
                stratified_indices.extend([classification_indices[issue_type][idx] for idx in trimmed_indices])
            texts = [text for idx, text in enumerate(texts) if idx in stratified_indices]
            for key in labels.keys():
                labels[key] = [label for idx, label in enumerate(labels[key]) if idx in stratified_indices]

        if self.input_encoding_type() == InputEncoding.Text:
            tokenized_issues = [['. '.join(text)] for text in texts]
        else:
            tokenized_issues = self.preprocess(texts)

        output = self.generate_vectors(tokenized_issues, metadata, self.__params)
        output['labels'] = labels   # labels is empty when pretrained

        if 'original' in output and not self.pretrained:    # Only dump original text when not pre-trained.
            with open(get_raw_text_file_name(), 'w') as file:
                mapping = {key: text
                           for key, text in zip(labels['issue_keys'], output['original'])}
                json.dump(mapping, file)
            del output['original']
        elif 'original' in output:
            del output['original']

        with open(target_filename, 'w') as file:
            json.dump(output, file)

    def preprocess(self, issues):
        ontology_path = conf.get('make-features.ontology-classes')
        if ontology_path != '':
            ontology_table = ontology.load_ontology(ontology_path)
        else:
            ontology_table = None

        tokenized_issues = []
        for issue in issues:
            all_words = []

            # Tokenize
            for sentence in issue:
                words = nltk.word_tokenize(sentence)

                # Transform lowercase
                if self.__params.get('disable-lowercase', 'False') != 'True':
                    words = [word.lower() for word in words]

                # Apply ontology simplification. Must be done before stemming/lemmatization
                if conf.get('make-features.apply-ontology-classes'):
                    assert ontology_table is not None, 'Missing --ontology-classes'
                    words = ontology.apply_ontologies_to_sentence(words, ontology_table)

                words = nltk.pos_tag(words)

                # Remove stopwords
                if self.__params.get('disable-stopwords', 'False') != 'True':
                    stopwords = nltk.corpus.stopwords.words('english')
                    #words = [word for word in words if word not in stopwords]
                    words = [(word, tag) for word, tag in words if word not in stopwords]

                use_stemming = self.__params.get('use-stemming', 'False') == 'True'
                use_lemmatization = self.__params.get('use-lemmatization', 'False') == 'True'
                use_pos = self.__params.get('use-pos', 'False') == 'True'
                if use_stemming and use_lemmatization:
                    raise ValueError('Cannot use both stemming and lemmatization')

                if use_stemming:
                    stemmer = nltk.stem.PorterStemmer()
                    words = [(stemmer.stem(word), tag) for word, tag in words]
                if use_lemmatization:
                    lemmatizer = nltk.stem.WordNetLemmatizer()
                    words = [(lemmatizer.lemmatize(word, pos=POS_CONVERSION.get(tag, 'n')), tag)
                             for word, tag in words]
                if use_pos:
                    words = [f'{word}_{POS_CONVERSION.get(tag, tag)}' for word, tag in words]
                else:
                    words = [word for word, _ in words]

                # At this point, we forget about sentence order
                all_words.extend(words)

            # Limit issue length
            if 'max-len' in self.__params:
                if len(all_words) > int(self.__params['max-len']):
                    all_words = all_words[0:int(self.__params['max-len'])]

            tokenized_issues.append(all_words)

        return tokenized_issues
