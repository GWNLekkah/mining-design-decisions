##############################################################################
##############################################################################
# Imports
##############################################################################

import json
import os
import pathlib
import shutil

from .config import conf

##############################################################################
##############################################################################
# Utility Functions
##############################################################################


def _prepare_directory(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    if len(os.listdir(path)) != 0:
        raise RuntimeError(
            f'Cannot store model in {path}: directory is not empty.'
        )


def _get_and_copy_feature_generators(directory: str):
    filename = conf.get('system.unstable.generator')
    full_path = os.path.join(directory, filename)
    shutil.copy(filename, full_path)
    return filename


##############################################################################
##############################################################################
# Model Saving
##############################################################################


def save_single_model(directory: str, model):
    _prepare_directory(directory)
    _store_model(directory, 0, model)
    metadata = {
        'model_type': 'single',
        'model_path': '0',
        'feature_generator': _get_and_copy_feature_generators(directory),
    } | _get_cli_settings()
    with open(os.path.join(directory, 'model.json'), 'w') as file:
        json.dump(metadata, file, indent=4)


def save_stacking_model(directory: str,
                        meta_model,
                        conversion_strategy: str,
                        *child_models):
    _prepare_directory(directory)
    _store_model(directory, 0, meta_model)
    for nr, model in enumerate(child_models, start=1):
        _store_model(directory, nr, model)
    metadata = {
        'model_type': 'stacking',
        'meta_model': '0',
        'feature_generator': _get_and_copy_feature_generators(directory),
        'input_conversion_strategy': conversion_strategy,
        'child_models': [
            str(i) for i in range(1, len(child_models) + 1)
        ],
    } | _get_cli_settings()
    with open(os.path.join(directory, 'model.json'), 'w') as file:
        json.dump(metadata, file, indent=4)


def save_voting_model(directory: str, *models):
    _prepare_directory(directory)
    for nr, model in enumerate(models):
        _store_model(directory, nr, model)
    metadata = {
        'model_type': 'voting',
        'child_models': [str(x) for x in range(len(models))],
        'feature_generator': _get_and_copy_feature_generators(directory),
    } | _get_cli_settings()
    with open(os.path.join(directory, 'model.json'), 'w') as file:
        json.dump(metadata, file, indent=4)


def _store_model(directory, number, model):
    path = os.path.join(directory, str(number))
    model.save(path)


def _get_cli_settings():
    return {
        'feature_settings': {
            key: _convert_value(value)
            for key, value in conf.get_all('make-features').items()
        },
        'model_settings': {
            key: _convert_value(value)
            for key, value in conf.get_all('run').items()
        },
    }


def _convert_value(x):
    if isinstance(x, pathlib.Path):
        return str(x)
    return x

##############################################################################
##############################################################################
# Model Loading
##############################################################################


def load_single_model(path: str):
    pass


def load_stacking_model(path: str):
    pass


def load_voting_model(path: str):
    pass

