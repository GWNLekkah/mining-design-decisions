##############################################################################
##############################################################################
# Imports
##############################################################################

import collections
import datetime
import json
import warnings

import requests
import typing

from .config import conf
from .logger import get_logger

log = get_logger('Database Manager')

##############################################################################
##############################################################################
# Query Parsing
##############################################################################

def parse_query(q):
    try:
        return json.loads(q)
    except json.JSONDecodeError as e:
        raise ValueError(f'Failed to parse {q}') from e

##############################################################################
##############################################################################
# Query Validation
##############################################################################

def validate_query(query, *, __force_eq=False):
    return True
    # if not isinstance(query, dict):
    #     raise _invalid_query(query)
    # if len(query) != 1:
    #     raise _invalid_query(query, 'expected exactly 1 element')
    # match query:
    #     case {'$and': operands}:
    #         if __force_eq:
    #             raise _invalid_query(query, '$and was not expected here')
    #         if not isinstance(operands, list):
    #             raise _invalid_query(query, '$and operand must be a list')
    #         for o in operands:
    #             validate_query(o)
    #     case {'$or': operands}:
    #         if __force_eq:
    #             raise _invalid_query(query, '$or was not expected here')
    #         if not isinstance(operands, list):
    #             raise _invalid_query(query, '$or operand must be a list')
    #         for o in operands:
    #             validate_query(o)
    #     case {'tags': operand}:
    #         if not isinstance(operand, dict):
    #             raise _invalid_query('tag operand must be an object')
    #         validate_query(operand, __force_eq=True)
    #     case {'project': operand}:
    #         if not isinstance(operand, dict):
    #             raise _invalid_query('project operand must be an object')
    #         validate_query(operand, __force_eq=True)
    #     case {'$eq': operand}:
    #         if not __force_eq:
    #             raise _invalid_query(query, '$eq not expected here')
    #         if not isinstance(operand, str):
    #             raise _invalid_query(query, '$eq operand must be a string')
    #     case {'$neq': operand}:
    #         if not __force_eq:
    #             raise _invalid_query(query, '$neq not expected here')
    #         if not isinstance(operand, str):
    #             raise _invalid_query(query, '$neq operand must be a string')
    #     case _ as x:
    #         raise _invalid_query(x, 'Invalid operation')


def _invalid_query(q, msg=None):
    if msg is not None:
        return ValueError(f'Invalid (sub-)query ({msg}): {q}')
    return ValueError(f'Invalid (sub-)query: {q}')


##############################################################################
##############################################################################
# Requests
##############################################################################


def _call_endpoint(endpoint, payload, verb):
    url = f'{conf.get("system.storage.database-url")}/{endpoint}'
    log.info(f'Calling endpoint {endpoint}')
    log.debug(f'Request payload: {payload}')
    match verb:
        case 'GET':
            response = requests.get(url,
                                    json=payload,
                                    verify=conf.get('system.security.certificate-authority'))
            response_payload = response.json()
            log.debug(f'Response payload: {response_payload}')
            return response_payload
        case 'POST':
            response = requests.post(url,
                                     json=payload,
                                     verify=conf.get('system.security.certificate-authority'),
                                     headers={
                                         'Authorization': 'Bearer ' + conf.get('system.security.db-token')
                                     })
            response.raise_for_status()
        case _ as x:
            raise ValueError(f'Invalid verb: {x}')


def get_token(username, password):
    endpoint = f'token'
    url = f'{conf.get("system.storage.database-url")}/{endpoint}'
    log.info(f'Calling endpoint {endpoint}')
    response = requests.post(
        url,
        files={
            'username': (None, username),
            'password': (None, password)
        },
        verify=conf.get('system.security.certificate-authority')
    )
    return response.json()['token']


def get_model_config(config_id: str):
    return _call_endpoint(f'models/{config_id}', {}, 'GET')['config']


def select_issue_ids(query) -> list[str]:
    parsed = parse_query(query)
    validate_query(parsed)
    return _call_endpoint('issue-ids', {'filter': parsed}, 'GET')['ids']


def get_issue_labels_by_key(ids: list[str]):
    return _call_endpoint('manual-labels', {'ids': ids}, 'GET')['labels']


def get_issue_data_by_keys(ids: list[str], attributes: list[str]):
    return _call_endpoint(
        'issue-data',
        {
            'ids': ids,
            'attributes': attributes
        },
        'GET'
    )['data']


def add_tag_to_issues(ids: list[str], tag: str):
    return _call_endpoint(
        'add-tags',
        {
            'ids': ids,
            'tags': [tag]
        },
        'POST'
    )


def save_predictions(model_id: str,
                     model_version: str,
                     predictions_by_id: dict[str, dict[str, typing.Any]]):
    _call_endpoint(
        f'models/{model_id}/versions/{model_version}/predictions',
        {
            'predictions': predictions_by_id
        },
        'POST'
    )


def store_model(model_id: str, time: str, filename: str) -> str:
    # Special endpoint, takes form data
    endpoint = f'models/{model_id}/versions'
    url = f'{conf.get("system.storage.database-url")}/{endpoint}'
    log.info(f'Calling endpoint {endpoint}')
    response = requests.post(
        url,
        files={
            'time': (None, time),
            'file': (filename, open(filename, 'rb'))
        },
        verify=conf.get('system.security.certificate-authority'),
        headers={
            'Authorization': 'Bearer ' + conf.get('system.security.db-token')
        }
    )
    return response.json()['version-id']


def get_most_recent_model(model_id: str) -> str:
    versions = _call_endpoint(f'models/{model_id}/versions', {}, 'GET')['versions']
    return max(versions, key=lambda x: datetime.datetime.fromisoformat(x['time']).timestamp())['id']


def retrieve_model(model_id: str, version_id: str) -> bytes:
    endpoint = f'models/{model_id}/versions/{version_id}'
    url = f'{conf.get("system.storage.database-url")}/{endpoint}'
    log.info(f'Calling endpoint {endpoint}')
    response = requests.get(url, verify=conf.get('system.security.certificate-authority'))
    return response.content


def save_model_results(model_id: str, version_id: str, results):
    endpoint = f'models/{model_id}/performances'
    payload = {'time': version_id, 'performance': results}
    _call_endpoint(endpoint, payload, 'POST')


##############################################################################
##############################################################################
# High level DB interface
##############################################################################


class DatabaseAPI:

    def __init__(self):
        self.__cache = collections.defaultdict(dict)

    def get_token(self, username, password):
        return get_token(username, password)

    def get_model_config(self, config_id: str):
        return get_model_config(config_id)

    def store_model(self, model_id: str, filename: str) -> str:
        return store_model(model_id, conf.get('system.training-start-time'), filename)

    def get_most_recent_model(self, model_id: str) -> str:
        return get_most_recent_model(model_id)

    def retrieve_model(self, model_id: str, version_id: str):
        return retrieve_model(model_id, version_id)

    def save_training_results(self, results):
        save_model_results(
            conf.get('run.model-id'),
            conf.get('system.training-start-time'),
            results
        )

    def select_issues(self, query):
        key = json.dumps(query)
        if key not in self.__cache['select-issue-ids']:
            self.__cache['select-issue-ids'][key] = select_issue_ids(query)
        return self.__cache['select-issue-ids'][key]

    def get_labels(self, ids: list[str]):
        local_cache = self.__cache['issue-labels']
        cached_keys = {key
                       for key in ids
                       if key in local_cache}
        required_keys = [key for key in ids if key not in cached_keys]
        if required_keys:
            labels = get_issue_labels_by_key(required_keys)
            local_cache.update(labels)
        return [local_cache[key] for key in ids]

    def get_issue_data(self,
                       issue_ids: list[str],
                       attributes: list[str],
                       *,
                       raise_on_partial_result=False):
        local_cache = self.__cache['issue-data']
        attrs = set(attributes)
        required_keys = []
        for key in issue_ids:
            if key not in local_cache:
                required_keys.append(key)
            elif not (attrs.issubset(local_cache['key'])):
                if raise_on_partial_result:
                    raise ValueError(f'Partially loaded set of attributes for key {key}')
                required_keys.append(key)
        if required_keys:
            data = get_issue_data_by_keys(required_keys, attributes)
            local_cache.update(data)
        wrong = [key for key in issue_ids if key not in local_cache]
        warnings.warn('Remember to remove `wrong` array once database has been updated!')
        return [local_cache[key] for key in issue_ids if key not in wrong]

    def add_tag(self, ids: list[str], tag: str):
        add_tag_to_issues(ids, tag)

    def save_predictions(self,
                         model_name: str,
                         model_version: str,
                         predictions_by_id: dict[str, dict[str, typing.Any]]):
        save_predictions(model_name, model_version, predictions_by_id)
