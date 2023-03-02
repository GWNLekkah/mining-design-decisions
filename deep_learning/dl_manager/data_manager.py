##############################################################################
##############################################################################
# Imports
##############################################################################

import json
import pathlib
import string
import typing

from .feature_generators import generators, OutputMode

from .config import conf

FEATURE_FILE_DIRECTORY = pathlib.Path('./features')

##############################################################################
##############################################################################
# Result Object
##############################################################################


class Dataset(typing.NamedTuple):
    features: typing.Any
    labels: list
    binary_labels: list
    shape: int | tuple[int]
    embedding_weights: None | list[float]
    vocab_size: None | int
    weight_vector_length: None | int
    issue_keys: list

    def is_embedding(self):
        return (
            self.embedding_weights is not None and
            self.vocab_size is not None and
            self.weight_vector_length is not None
        )


##############################################################################
##############################################################################
# Functionality
##############################################################################


def get_feature_file(source_file: pathlib.Path,
                     input_mode: str,
                     output_mode: str,
                     **params):
    base_name = f'{source_file.stem}_features_{input_mode}'
    suffix = '_'.join(f'{key}-{_escape(value)}'
                      for key, value in params.items()
                      if key != 'metadata-attributes')
    if len(suffix) >= 200:  # Temporary fix for long file names
        suffix = str(hash(suffix))
    if conf.get('system.peregrine'):
        data = pathlib.Path(conf.get('system.peregrine.data'))
        directory = data / 'features'
        if not directory.exists():
            directory.mkdir(exist_ok=True)
        return directory / f'{base_name}_{suffix}.json'
    if not FEATURE_FILE_DIRECTORY.exists():
        FEATURE_FILE_DIRECTORY.mkdir(exist_ok=True)
    return FEATURE_FILE_DIRECTORY / f'{base_name}_{suffix}.json'


def _escape(x):
    for ws in string.whitespace:
        x = x.replace(ws, '_')
    x = x.replace('.', 'dot')
    for illegal in '/<>:"/\\|?*\'':
        x = x.replace(illegal, '')
    return x


def escape_filename(x):
    return _escape(x)


def get_features(source_file: pathlib.Path,
                 input_mode: str,
                 output_mode: str,
                 **params) -> Dataset:
    feature_file = get_feature_file(source_file,
                                    input_mode,
                                    output_mode,
                                    **params)
    if not feature_file.exists():
        make_features(source_file,
                      feature_file,
                      input_mode,
                      **params)
    return load_features(feature_file, output_mode)


def make_features(source_file: pathlib.Path,
                  feature_file: pathlib.Path,
                  input_mode: str,
                  **params):
    try:
        generator = generators[input_mode](**params)
    except KeyError:
        raise ValueError(f'Invalid input mode {input_mode}')
    generator.generate_features(source_file, feature_file)


def load_features(filename: pathlib.Path, output_mode: str) -> Dataset:
    with open(filename) as file:
        data = json.load(file)
    dataset = Dataset(
        features=data['features'],
        labels=data['labels'][output_mode.lower()],
        shape=data['feature_shape'],
        embedding_weights=data.get('weights', None),
        vocab_size=data.get('vocab_size', None),
        weight_vector_length=data.get('word_vector_length', None),
        binary_labels=data['labels']['detection'],
        issue_keys=data['labels']['issue_keys']
    )
    return dataset




