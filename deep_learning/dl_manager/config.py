"""
This module provides special functionality for
handling configuration options in a centralized
manner.

It provides an easy to configure way to set up
an elaborate, automatically handled command
line interface
"""

from __future__ import annotations

##############################################################################
##############################################################################
# Imports
##############################################################################

import importlib
import json

import fastapi
import uvicorn

##############################################################################
##############################################################################
# Custom Exceptions
##############################################################################


class NoSuchSetting(LookupError):

    def __init__(self, attribute, action):
        message = (f'Cannot perform action {action!r} on setting '
                   f'{attribute} since it does not exist')
        super().__init__(message)


class NotSet(Exception):

    def __init__(self, attribute):
        message = f'Attribute {attribute!r} has not been initialized'
        super().__init__(message)


class IllegalNamespace(Exception):

    def __init__(self, attribute):
        message = f'The namespace containing {attribute!r} is currently not accessible'
        super().__init__(message)


##############################################################################
##############################################################################
# Configuration Class (not thread safe)
##############################################################################


class ConfigFactory:

    def __init__(self):
        self._namespace = {}

    @staticmethod
    def _normalize_name(x: str) -> str:
        return x.lower().replace('-', '_')

    def register_namespace(self, name: str):
        if not name:
            raise ValueError('Name must be non-empty')
        parts = self._normalize_name(name).split('.')
        current = self._namespace
        for part in parts:
            if part not in current:
                current[part] = {}
            current = current[part]
            if current is None:
                raise ValueError(f'{name} is a property, and cannot be (part of) a namespace')

    def register(self, name: str):
        if not name:
            raise ValueError('Name must be non-empty')
        parts = self._normalize_name(name).split('.')
        if len(parts) < 2:
            raise ValueError(f'A property must be contained in the non-global namespace ({name})')
        current = self._namespace
        for part in parts[:-1]:
            if part not in current:
                raise ValueError(f'Cannot register property {name}; namespace does not exist')
            current = current[part]
        current[parts[-1]] = None

    def build_config(self, *namespaces) -> Config:
        legal = [self._normalize_name(n) for n in namespaces]
        for n in legal:
            if '.' in n:
                raise ValueError(f'Can only register top-level namespaces as legal, not {n}')
        Config(
            legal,
            self._namespace,
            self._new_namespace_tree(self._namespace)
        )

    def _new_namespace_tree(self, obj):
        if obj is None:
            return Config.NOT_SET
        return {key: self._new_namespace_tree(value)
                for key, value in obj.items()}


class Config:

    NOT_SET = object()

    def __init__(self, legal_namespaces, namespaces, data):
        self._legal = legal_namespaces
        self._namespaces = namespaces
        self._data = data

    @staticmethod
    def _normalize_name(x):
        return x.lower().replace('-', '_').split('.')

    def _resolve(self, name, action, path):
        if path[0] not in self._legal:
            raise IllegalNamespace(name)
        current_n = self._namespaces
        current_d = self._data
        for part in path:
            if current_n is None:
                raise NoSuchSetting(name, action)
            if path not in current_n:
                raise NoSuchSetting(name, action)
            current_n = current_n[part]
            current_d = current_d[part]
        return current_d

    def get_all(self, name: str):
        return self._resolve(name, 'get_all', self._normalize_name(name))

    def get(self, name: str):
        *path, prop = self._normalize_name(name)
        namespace = self._resolve(name, 'get', path)
        if prop not in namespace:
            raise NoSuchSetting(prop, 'get')
        return namespace[prop]

    def set(self, name: str, value):
        *path, prop = self._normalize_name(name)
        namespace = self._resolve(name, 'set', path)
        if prop not in namespace:
            raise NoSuchSetting(prop, 'set')
        namespace[prop] = value

    def clone(self, from_: str, to: str):
        self.set(from_, self.get(to))

    def transfer(self, target: Config, *properties):
        for prop in properties:
            target.set(prop, self.get(prop))

    def update(self, prefix: str | None = None, /, **items):
        for key, value in items.items():
            if prefix is not None:
                key = f'{prefix}.{key}'
            self.set(key, value)


##############################################################################
##############################################################################
# Web API builder
##############################################################################


class WebApp:

    def __init__(self, filename: str):
        self._app = fastapi.FastAPI()
        self._router = fastapi.APIRouter()
        with open(filename) as file:
            spec = json.load(file)
        self._build_endpoints(spec['commands'])
        self._callbacks = {}
        self._setup_callbacks = []
        self._constraints = []
        self._endpoints = []
        self._config_factory = ConfigFactory()
        self._register_system_properties()

    def _register_system_properties(self):
        self._config_factory.register_namespace('system.storage')
        self._config_factory.register_namespace('system.security')
        self._config_factory.register_namespace('system.os')
        self._config_factory.register_namespace('system.resources')
        self._config_factory.register_namespace('system.management')

        # Variable for tracking the amount of available threads
        self._config_factory.register('system.resources.threads')

        # Database credentials
        self._config_factory.register('system.security.db-username')
        self._config_factory.register('system.security.db-password')

        # Control of self-signed certificates
        self._config_factory.register(
            'system.security.allow-self-signed-certificates'
        )

        # Database connection
        self._config_factory.register('system.storage.database-url')
        self._config_factory.register('system.storage.database-api')

        # Model Saving Support
        # - system.storage.generators
        #       A list of filenames pointing to the files containing
        #       the configurations for the trained feature generators.
        # - system.storage.auxiliary
        #       A list of filenames containing auxiliary files which
        #       must be included in the folder for saving a
        #       pretrained model.
        # - system.storage.auxiliary_map
        #       A mapping which is necessary to resolve filenames
        #       when loading a pretrained model.
        #       It maps "local" filenames to the actual location
        #       in the folder containing the pretrained model.
        # - system.storage.file_prefix
        #       A prefix which _should_ be used for all files
        #       generated by the pipeline, which is often
        #       forgotten about in practice.
        self._config_factory.register('system.storage.generators')
        self._config_factory.register('system.storage.auxiliary')
        self._config_factory.register('system.storage.auxiliary_map')
        self._config_factory.register('system.storage.file_prefix')

        # Current system state
        self._config_factory.register('system.management.active-command')

        # Target home and data directories.
        self._config_factory.register('system.os.peregrine')
        self._config_factory.register('system.os.home-directory')
        self._config_factory.register('system.os.data-directory')

    def _build_endpoints(self, commands):
        for command in commands:
            self._build_endpoint(command)

    def _build_endpoint(self, spec):
        endpoint = _Endpoint(spec, self._config_factory, self.dispatch)
        self._router.add_api_route(endpoint.name, endpoint, description=endpoint.description)

    def register_callback(self, event, func):
        self._callbacks[event] = func

    def register_setup_callback(self, func):
        self._setup_callbacks.append(func)

    def add_constraint(self, predicate, message, *keys):
        self._constraints.append((keys, predicate, message))

    def deploy(self, port, keyfile, certfile):
        self._app.include_router(self._router)
        uvicorn.run(
            self._app,
            host='0.0.0.0',
            port=port,
            ssl_keyfile=keyfile,
            ssl_certfile=certfile
        )

    def dispatch(self, name, conf: Config):
        for keys, predicate, message in self._constraints:
            try:
                values = [(conf.get(key) if key != '#config' else conf) for key in keys]
            except IllegalNamespace:
                continue    # Constraint not relevant
            if not predicate(values):
                error = f'Constraint check on {",".join(keys)} failed: {message}'
                raise ValueError(error)
        conf.set('system.management.active-command', name)
        conf.set('system.management.app', self)
        for callback in self._setup_callbacks:
            callback()
        self._callbacks[name](conf)

    def new_config(self, *namespaces) -> Config:
        self._config_factory.build_config(*namespaces)


class _Endpoint:

    def __init__(self,
                 spec,
                 config_factory: ConfigFactory,
                 callback):
        self.name = spec['name']
        self.description = spec['help']
        self._args = spec
        validators = [_ArgumentValidator(arg) for arg in self._args]
        self._validators = {arg.name: arg for arg in validators}
        self._required = {arg.name for arg in validators if arg.required}
        self._defaults = {arg.name: arg.default
                          for arg in validators
                          if arg.default is not _ArgumentValidator.NOT_SET}
        self._config_factory = config_factory
        self._dispatcher = callback
        self._config_factory.register_namespace(self.name)
        for v in self._validators:
            self._config_factory.register(f'{self.name}.{v.name}')


    def __call__(self, req: fastapi.Request):
        payload = req.json()
        args = self._validate(payload)
        conf = self._config_factory.build_config(self.name, 'system')
        for name, value in args.items():
            conf.set(f'{self.name}.{name}', value)
        self._dispatcher(self.name, conf)

    def _validate(self, obj):
        if not isinstance(obj, dict):
            raise fastapi.HTTPException(detail='Expected a JSON object',
                                        status_code=400)
        parsed = {}
        for name, value in obj.items():
            if name not in self._validators:
                raise fastapi.HTTPException(
                    status_code=400,
                    detail=f'Invalid argument for endpoint {self.name!r}: {name}'
                )
            parsed[name] = self._validators[name].validate(value)
        missing = self._required - set(parsed.keys())
        if missing:
            raise fastapi.HTTPException(
                status_code=400,
                detail=f'Endpoint {self.name!r} is missing the following required arguments: {", ".join(missing)}'
            )
        for name in set(self._defaults) - set(parsed):
            parsed[name] = self._validators[name](self._defaults[name])
        return parsed


class _ArgumentValidator:

    NOT_SET = object()

    def __init__(self, spec):
        self.name = spec['name']
        self.description = spec['help']
        self.required = spec.get('required', False)
        self.default = spec.get('default', self.NOT_SET)
        self._nargs = '1' if 'nargs' not in spec else spec['nargs']
        self._type = spec['type']
        self._options = spec['options']
        if self._nargs not in ('1', '*', '+'):
            raise ValueError(f'[{self.name}] Invalid nargs: {self._nargs}')
        if self._type not in ('str', 'int', 'bool', 'enum', 'class', 'args'):
            raise ValueError(f'[{self.name}] Invalid type: {self._type}')
        if self._type == 'class':
            if len(self._options) != 1:
                raise ValueError(f'[{self.name}] Argument of type "class" requires exactly one option.')
            dotted_name = self._options[0]
            module, item = dotted_name.split('.')
            mod = importlib.import_module(module)
            cls = getattr(mod, item)
            self._options = [cls]


    def validate(self, value):
        if self._nargs == '1':
            return self._validate(value)
        else:
            if not isinstance(value, list):
                raise fastapi.HTTPException(
                    detail=f'{self.name!r} is a multi-valued argument. Expected a list.',
                    status_code=400
                )
            if self._nargs == '+' and not value:
                raise fastapi.HTTPException(
                    detail=f'{self.name!r} requires at least 1 value.',
                    status_code=400
                )
            return [self._validate(x) for x in value]

    def _validate(self, x):
        match self._type:
            case 'str':
                if not isinstance(x, str):
                    self._raise_invalid_type('string', x)
                return x
            case 'int':
                if not isinstance(x, int):
                    self._raise_invalid_type('int', x)
                return x
            case 'bool':
                if not isinstance(x, bool):
                    self._raise_invalid_type('bool', x)
                return x
            case 'enum':
                if not isinstance(x, str):
                    raise fastapi.HTTPException(
                        detail=f'{self.name!r} enum argument must be of type string, got {x.__class__.__name__}',
                        status_code=400
                    )
                if x not in self._options:
                    raise fastapi.HTTPException(
                        detail=f'Invalid option for {self.name!r}: {x} (valid options: {self._options})',
                        status_code=400
                    )
                return x
            case 'class':
                try:
                    return self._options[0](x)
                except Exception as e:
                    raise fastapi.HTTPException(
                        detail=f'Error while converting {self.name!r} to {self._options[0].__class__.__name__}: {e}',
                        status_code=400
                    )
            case 'args':
                raise NotImplementedError

    def _raise_invalid_type(self, expected, got):
        raise fastapi.HTTPException(
            detail=f'{self.name!r} must be of type {expected}, got {got.__class__.__name__}',
            status_code=400
        )
