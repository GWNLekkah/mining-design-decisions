##############################################################################
##############################################################################
# Imports
##############################################################################

import json
import os
import pathlib
import shutil
import typing

import zipfile

import issue_db_api

from .config import Config
from .logger import get_logger
log = get_logger('Model Saver')

MODEL_DIR = 'model'
MODEL_FILE = 'pretrained_model.zip'

##############################################################################
##############################################################################
# JSON support for queries
##############################################################################


class QueryJSONEncoder(json.JSONEncoder):

    def default(self, o: typing.Any) -> typing.Any:
        if isinstance(o, issue_db_api.Query):
            return o.to_json()
        return super().default(o)


##############################################################################
##############################################################################
# Utility Functions
##############################################################################


def _prepare_directory(path: str):
    log.debug(f'Preparing directory: {path}')
    if os.path.exists(path):
        log.info(f'Removing existing directory: {path}')
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _transform(file: str, directory: str, *, aux=False):
    base_path, filename = os.path.split(file)
    infix = os.path.relpath(directory, base_path)
    if not aux:
        return os.path.join(base_path, infix, filename)
    return os.path.join(base_path, infix, 'auxiliary', filename)


def _strip_path(filename: str, conf: Config):
    return os.path.relpath(filename,
                           conf.get('system.os.scratch-directory'))


def _get_and_copy_feature_generators(directory: str, conf: Config):
    filenames = conf.get('system.storage.generators')
    for filename in filenames:
        # suffix = os.path.relpath(filename, directory)
        # full_path = os.path.join(directory, suffix)
        full_path = _transform(filename, directory)
        log.info(f'Copying feature generator: {filename} -> {full_path}')
        shutil.copy(filename, full_path)
    return [_strip_path(filename, conf) for filename in filenames]


def _get_and_copy_auxiliary_files(directory: str, conf: Config):
    filenames = conf.get('system.storage.auxiliary')
    os.makedirs(os.path.join(directory, 'auxiliary'), exist_ok=True)
    result = {}
    for filename in filenames:
        # suffix = os.path.relpath(filename, directory)
        # full_path = os.path.join(directory, 'auxiliary', suffix)
        full_path = _transform(filename, directory, aux=True)
        #result[filename] = _strip_path(full_path, conf)
        result[filename] = os.path.relpath(
            full_path,
            os.path.join(
                conf.get('system.os.scratch-directory'),
                'model'
            )
        )
        stem = os.path.split(full_path)[0]
        os.makedirs(stem, exist_ok=True)
        log.info(f'Copying auxiliary file: {filename} -> {full_path}')
        shutil.copy(filename, full_path)
    return result



##############################################################################
##############################################################################
# Model Saving
##############################################################################


def save_single_model(model, conf: Config):
    log.info('Storing single model')
    directory = os.path.join(conf.get('system.os.scratch-directory'), MODEL_DIR)
    _prepare_directory(directory)
    _store_model(directory, 0, model)
    metadata = {
        'model-type': 'single',
        'model-path': '0',
        'feature-generators': _get_and_copy_feature_generators(directory, conf),
        'auxiliary-files': _get_and_copy_auxiliary_files(directory, conf)
    } | _get_cli_settings(conf)
    with open(os.path.join(directory, 'model.json'), 'w') as file:
        json.dump(metadata, file, indent=4, cls=QueryJSONEncoder)
    return _upload_zip_data(directory, conf)


def save_stacking_model(meta_model,
                        conversion_strategy: str,
                        *child_models,
                        conf: Config):
    log.info('Storing stacking ensemble')
    directory = os.path.join(conf.get('system.os.scratch-directory'), MODEL_DIR)
    _prepare_directory(directory)
    _store_model(directory, 0, meta_model)
    for nr, model in enumerate(child_models, start=1):
        _store_model(directory, nr, model)
    metadata = {
        'model-type': 'stacking',
        'meta-model': '0',
        'feature-generators': _get_and_copy_feature_generators(directory, conf),
        'input-conversion_strategy': conversion_strategy,
        'child-models': [
            str(i) for i in range(1, len(child_models) + 1)
        ],
        'auxiliary-files': _get_and_copy_auxiliary_files(directory, conf)
    } | _get_cli_settings(conf)
    with open(os.path.join(directory, 'model.json'), 'w') as file:
        json.dump(metadata, file, indent=4, cls=QueryJSONEncoder)
    return _upload_zip_data(directory, conf)


def save_voting_model(*models, conf: Config):
    log.info('Storing voting ensemble')
    directory = os.path.join(conf.get('system.os.scratch-directory'), MODEL_DIR)
    _prepare_directory(directory)
    for nr, model in enumerate(models):
        _store_model(directory, nr, model)
    metadata = {
        'model-type': 'voting',
        'child-models': [str(x) for x in range(len(models))],
        'feature-generators': _get_and_copy_feature_generators(directory, conf),
        'auxiliary-files': _get_and_copy_auxiliary_files(directory, conf)
    } | _get_cli_settings(conf)
    with open(os.path.join(directory, 'model.json'), 'w') as file:
        json.dump(metadata, file, indent=4, cls=QueryJSONEncoder)
    return _upload_zip_data(directory, conf)


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
        'model-settings': {
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
    return model.add_version(filename).version_id


##############################################################################
##############################################################################
# Model Loading
##############################################################################


def load_model_from_zip(filename: str, conf: Config):
    zip_file = zipfile.ZipFile(filename, 'r')
    zip_file.extractall(os.path.join(conf.get('system.os.scratch-directory'), MODEL_DIR))
    zip_file.close()

