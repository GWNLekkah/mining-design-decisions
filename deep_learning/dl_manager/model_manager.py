##############################################################################
##############################################################################
# Imports
##############################################################################

import json
import os
import pathlib
import shutil
import zipfile

import issue_db_api

from .config import Config

MODEL_DIR = 'model'
MODEL_FILE = 'pretrained_model.zip'

##############################################################################
##############################################################################
# Utility Functions
##############################################################################


def _prepare_directory(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _get_and_copy_feature_generators(directory: str, conf: Config):
    filenames = conf.get('system.storage.generators')
    for filename in filenames:
        full_path = os.path.join(directory, filename)
        shutil.copy(filename, full_path)
    return filenames


def _get_and_copy_auxiliary_files(directory: str, conf: Config):
    filenames = conf.get('system.storage.auxiliary')
    os.makedirs(os.path.join(directory, 'auxiliary'), exist_ok=True)
    result = {}
    for filename in filenames:
        full_path = os.path.join(directory, 'auxiliary', filename)
        result[filename] = os.path.join('auxiliary', filename)
        stem = os.path.split(full_path)[0]
        os.makedirs(stem, exist_ok=True)
        shutil.copy(filename, full_path)
    return result



##############################################################################
##############################################################################
# Model Saving
##############################################################################


def save_single_model(model, conf: Config):
    directory = MODEL_DIR
    _prepare_directory(directory)
    _store_model(directory, 0, model)
    metadata = {
        'model_type': 'single',
        'model_path': '0',
        'feature_generators': _get_and_copy_feature_generators(directory, conf),
        'auxiliary_files': _get_and_copy_auxiliary_files(directory, conf)
    } | _get_cli_settings(conf)
    with open(os.path.join(directory, 'model.json'), 'w') as file:
        json.dump(metadata, file, indent=4)
    _upload_zip_data(directory, conf)


def save_stacking_model(meta_model,
                        conversion_strategy: str,
                        *child_models,
                        conf: Config):
    directory = MODEL_DIR
    _prepare_directory(directory)
    _store_model(directory, 0, meta_model)
    for nr, model in enumerate(child_models, start=1):
        _store_model(directory, nr, model)
    metadata = {
        'model_type': 'stacking',
        'meta_model': '0',
        'feature_generators': _get_and_copy_feature_generators(directory, conf),
        'input_conversion_strategy': conversion_strategy,
        'child_models': [
            str(i) for i in range(1, len(child_models) + 1)
        ],
        'auxiliary_files': _get_and_copy_auxiliary_files(directory, conf)
    } | _get_cli_settings(conf)
    with open(os.path.join(directory, 'model.json'), 'w') as file:
        json.dump(metadata, file, indent=4)
    _upload_zip_data(directory, conf)


def save_voting_model(*models, conf: Config):
    directory = MODEL_DIR
    _prepare_directory(directory)
    for nr, model in enumerate(models):
        _store_model(directory, nr, model)
    metadata = {
        'model_type': 'voting',
        'child_models': [str(x) for x in range(len(models))],
        'feature_generators': _get_and_copy_feature_generators(directory, conf),
        'auxiliary_files': _get_and_copy_auxiliary_files(directory, conf)
    } | _get_cli_settings(conf)
    with open(os.path.join(directory, 'model.json'), 'w') as file:
        json.dump(metadata, file, indent=4)
    _upload_zip_data(directory, conf)


def _store_model(directory, number, model):
    path = os.path.join(directory, str(number))
    if hasattr(model, 'save_pretrained') and callable(model.save_pretrained):
        model.save_pretrained(path)
    else:
        model.save(path)
    os.makedirs(os.path.join(directory, 'arch'), exist_ok=True)
    with open(os.path.join(directory, 'arch', f'{number}.json'), 'w') as file:
        file.write(model.to_json(indent=4))


def _get_cli_settings(conf: Config):
    return {
        'model_settings': {
            key: _convert_value(value)
            for key, value in conf.get_all('run').items()
        },
    }


def _convert_value(x):
    if isinstance(x, pathlib.Path):
        return str(x)
    return x


def _upload_zip_data(path, conf: Config):
    filename = shutil.make_archive('model', 'zip', path)
    db: issue_db_api.IssueRepository = conf.get('system.storage.database-api')
    model = db.get_model_by_id(conf.get('run.model-id'))
    model.add_version(filename)


##############################################################################
##############################################################################
# Model Loading
##############################################################################


def load_model_from_zip(filename: str):
    zip_file = zipfile.ZipFile(filename, 'r')
    zip_file.extractall(MODEL_DIR)
    zip_file.close()

