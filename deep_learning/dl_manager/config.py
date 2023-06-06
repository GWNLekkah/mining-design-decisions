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

import abc
import collections
import copy
import graphlib
import importlib
import json
import math
import typing

import fastapi
import issue_db_api
import requests
import uvicorn

from . import db_util
from . import logger

log = logger.get_logger("App Builder")

##############################################################################
##############################################################################
# Custom Exceptions
##############################################################################


class NoSuchSetting(LookupError):
    def __init__(self, attribute, action):
        message = (
            f"Cannot perform action {action!r} on setting "
            f"{attribute!r} since it does not exist"
        )
        super().__init__(message)


class NotSet(Exception):
    def __init__(self, attribute):
        message = f"Attribute {attribute!r} has not been initialized"
        super().__init__(message)


class IllegalNamespace(Exception):
    def __init__(self, attribute):
        message = f"The namespace containing {attribute!r} is currently not accessible"
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
        return x.lower().replace("-", "_")

    def register_namespace(self, name: str):
        if not name:
            raise ValueError("Name must be non-empty")
        parts = self._normalize_name(name).split(".")
        current = self._namespace
        for part in parts:
            if part not in current:
                current[part] = {}
            current = current[part]
            if current is None:
                raise ValueError(
                    f"{name} is a property, and cannot be (part of) a namespace"
                )

    def register(self, name: str):
        if not name:
            raise ValueError("Name must be non-empty")
        parts = self._normalize_name(name).split(".")
        if len(parts) < 2:
            raise ValueError(
                f"A property must be contained in the non-global namespace ({name})"
            )
        current = self._namespace
        for part in parts[:-1]:
            if part not in current:
                raise ValueError(
                    f"Cannot register property {name}; namespace does not exist"
                )
            current = current[part]
        current[parts[-1]] = None

    def build_config(self, *namespaces) -> Config:
        legal = [self._normalize_name(n) for n in namespaces]
        for n in legal:
            if "." in n:
                raise ValueError(
                    f"Can only register top-level namespaces as legal, not {n}"
                )
        return Config(legal, self._namespace, self._new_namespace_tree(self._namespace))

    def _new_namespace_tree(self, obj):
        if obj is None:
            return Config.NOT_SET
        return {key: self._new_namespace_tree(value) for key, value in obj.items()}


class Config:
    NOT_SET = object()

    def __init__(self, legal_namespaces, namespaces, data):
        self._legal = legal_namespaces
        self._namespaces = namespaces
        self._data = data

    @staticmethod
    def _normalize_name(x):
        return x.lower().replace("-", "_").split(".")

    def _resolve(self, name, action, path):
        if path[0] not in self._legal:
            raise IllegalNamespace(name)
        current_n = self._namespaces
        current_d = self._data
        for part in path:
            if current_n is None:
                raise NoSuchSetting(name, action)
            if part not in current_n:
                raise NoSuchSetting(name, action)
            current_n = current_n[part]
            current_d = current_d[part]
        return current_d

    def get_all(self, name: str):
        return {
            key: (value if value is not self.NOT_SET else None)
            for key, value in self._resolve(
                name, "get_all", self._normalize_name(name)
            ).items()
        }

    def get(self, name: str):
        *path, prop = self._normalize_name(name)
        namespace = self._resolve(name, "get", path)
        if prop not in namespace:
            raise NoSuchSetting(name, "get")
        value = namespace[prop]
        if value is self.NOT_SET:
            raise NotSet(name)
        return value

    def set(self, name: str, value):
        *path, prop = self._normalize_name(name)
        namespace = self._resolve(name, "set", path)
        if prop not in namespace:
            raise NoSuchSetting(name, "set")
        namespace[prop] = value

    def clone(self, from_: str, to: str):
        self.set(to, self.get(from_))

    def transfer(self, target: Config, *properties):
        for prop in properties:
            target.set(prop, self.get(prop))

    def update(self, prefix: str | None = None, /, **items):
        for key, value in items.items():
            if prefix is not None:
                key = f"{prefix}.{key}"
            self.set(key, value)


##############################################################################
##############################################################################
# Web API builder
##############################################################################


class WebApp:
    def __init__(self, filename: str):
        self._app = fastapi.FastAPI()
        self._router = fastapi.APIRouter()
        self._callbacks = {}
        self._setup_callbacks = []
        self._constraints = []
        self._endpoints = {}
        self._config_factory = ConfigFactory()
        self._register_system_properties()
        with open(filename) as file:
            self._spec = json.load(file)
        self._build_endpoints(copy.deepcopy(self._spec)["commands"])
        self._add_static_endpoints()

    def _register_system_properties(self):
        self._config_factory.register_namespace("system.storage")
        self._config_factory.register_namespace("system.security")
        self._config_factory.register_namespace("system.os")
        self._config_factory.register_namespace("system.resources")
        self._config_factory.register_namespace("system.management")

        # Variable for tracking the amount of available threads
        self._config_factory.register("system.resources.threads")

        # Database credentials
        # self._config_factory.register('system.security.db-username')
        # self._config_factory.register('system.security.db-password')
        self._config_factory.register("system.security.db-token")

        # Control of self-signed certificates
        self._config_factory.register("system.security.allow-self-signed-certificates")

        # Database connection
        self._config_factory.register("system.storage.database-url")
        self._config_factory.register("system.storage.database-api")

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
        self._config_factory.register("system.storage.generators")
        self._config_factory.register("system.storage.auxiliary")
        self._config_factory.register("system.storage.auxiliary_map")
        self._config_factory.register("system.storage.file_prefix")

        # Current system state
        self._config_factory.register("system.management.active-command")
        self._config_factory.register("system.management.app")

        # Target home and data directories.
        self._config_factory.register("system.os.peregrine")
        self._config_factory.register("system.os.home-directory")
        self._config_factory.register("system.os.data-directory")
        self._config_factory.register("system.os.scratch-directory")

    def _build_endpoints(self, commands):
        for command in commands:
            self._build_endpoint(command)

    def _build_endpoint(self, spec):
        endpoint = _Endpoint(spec, self._config_factory, self.dispatch)
        if not endpoint.private:
            self._router.post("/" + endpoint.name, description=endpoint.description)(
                endpoint.invoke
            )
        self._endpoints[endpoint.name] = endpoint

    def _add_static_endpoints(self):
        @self._router.get("/endpoints")
        async def get_endpoints():
            return self._spec

        for cmd in self._spec["commands"]:
            for arg in cmd["args"]:
                if arg["type"] != "arglist":
                    continue
                self._add_arglist_endpoint(cmd["name"], arg)

    def _add_arglist_endpoint(self, cmd_name, spec):
        @self._router.get(f'/arglists/{cmd_name}/{spec["name"]}')
        async def get_arglist():
            module, item = spec["options"][0]["map-path"].rsplit(".", maxsplit=1)
            mapping = getattr(importlib.import_module(module), item)
            return {
                name: [arg.get_json_spec() for arg in cls.get_arguments().values()]
                for name, cls in mapping.items()
            }

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
            host="0.0.0.0",
            port=port,
            ssl_keyfile=keyfile,
            ssl_certfile=certfile,
        )

    def execute_script(self, filename):
        with open(filename) as file:
            script = json.load(file)
        token = script["auth"]["token"]
        for command in script["script"]:
            # Refresh the token
            response = requests.post(
                url=script["auth"]["token-endpoint"],
                headers={"Authorization": "Bearer " + token},
            )
            response.raise_for_status()
            token = response.json()["access_token"]
            # Call the endpoint internally
            endpoint = command["cmd"]
            payload = {"auth": {"token": token}, "config": command["args"]}
            self._endpoints[endpoint].invoke_with_json(payload)

    def invoke_endpoint(self, name: str, conf: Config, payload):
        return self._endpoints[name].run(conf, payload)

    def dispatch(self, name, conf: Config):
        for keys, predicate, message in self._constraints:
            try:
                values = [(conf.get(key) if key != "#config" else conf) for key in keys]
            except IllegalNamespace:
                continue  # Constraint not relevant
            if not predicate(*values):
                error = f'Constraint check on {",".join(keys)} failed: {message}'
                raise fastapi.HTTPException(detail=error, status_code=400)
        conf.set("system.management.active-command", name)
        conf.set("system.management.app", self)
        for callback in self._setup_callbacks:
            callback(conf)
        return self._callbacks[name](conf)

    def new_config(self, *namespaces) -> Config:
        return self._config_factory.build_config(*namespaces)


class _Endpoint:
    def __init__(self, spec, config_factory: ConfigFactory, callback):
        self.name = spec["name"]
        log.info(f"Registering endpoint {self.name!r}")
        self.description = spec["help"]
        self._args = spec["args"]
        self.private = spec["private"]
        validators = [_ArgumentValidator(arg) for arg in self._args]
        self._validators = {arg.name: arg for arg in validators}
        for v in self._validators.values():
            if v.depends is None:
                continue
            if v.depends not in self._validators:
                raise ValueError(
                    f"[{self.name}] Argument {v.name!r} depends on unknown argument {v.depends!r}"
                )
            # if self._validators[v.depends].dtype != 'bool':
            #    raise ValueError(f'[{self.name}] Argument {v.name!r} depends on non-Boolean argument {v.depends!r}')
        self._order = []
        self._compute_validation_order()
        self._required = {arg.name for arg in validators if arg.required}
        self._defaults = {
            arg.name: arg.default
            for arg in validators
            if arg.default is not _ArgumentValidator.NOT_SET
        }
        self._config_factory = config_factory
        self._dispatcher = callback
        self._config_factory.register_namespace(self.name)
        for v in self._validators:
            self._config_factory.register(f"{self.name}.{v}")

    def _compute_validation_order(self):
        sorter = graphlib.TopologicalSorter()
        for v in self._validators.values():
            if v.depends is not None:
                sorter.add(v.name, v.depends)
            else:
                sorter.add(v.name)
        try:
            self._order = list(sorter.static_order())
        except graphlib.CycleError:
            raise ValueError(
                f"[{self.name}] Cycle in if_null/unless_null declarations."
            )

    async def invoke(self, req: fastapi.Request):
        if self.private:
            fastapi.HTTPException(
                detail=f"Endpoint {self.name} is private/internal", status_code=406
            )
        payload = await req.json()
        return self.invoke_with_json(payload)

    def invoke_with_json(self, payload):
        conf = self._config_factory.build_config(self.name, "system")
        if "auth" in payload:
            auth = payload["auth"]
            # conf.set('system.security.db-username', auth['username'])
            # conf.set('system.security.db-password', auth['password'])
            conf.set("system.security.db-token", auth["token"])
        return self.run(conf, payload["config"])

    def run(self, conf: Config, payload):
        args = self.validate(payload)
        for name, value in args.items():
            conf.set(f"{self.name}.{name}", value)
        return self._dispatcher(self.name, conf)

    def validate(self, obj):
        return self._validate(obj)

    def _validate(self, obj):
        if not isinstance(obj, dict):
            raise fastapi.HTTPException(
                detail="Expected a JSON object", status_code=400
            )
        parsed = {}
        not_handled = set(obj)
        for name in self._order:
            try:
                value = obj[name]
            except KeyError:
                if name in self._defaults:
                    value = self._defaults[name]
                else:
                    continue
            if name not in self._validators:
                raise fastapi.HTTPException(
                    status_code=400,
                    detail=f"Invalid argument for endpoint {self.name!r}: {name}",
                )
            validator = self._validators[name]
            if validator.depends is not None:
                parsed[name] = validator.validate_conditionally(
                    value, parsed[validator.depends]
                )
            else:
                parsed[name] = validator.validate(value)
            if name in not_handled:
                not_handled.remove(name)
        missing = self._required - set(parsed.keys())
        if missing:
            raise fastapi.HTTPException(
                status_code=400,
                detail=f'Endpoint {self.name!r} is missing the following required arguments: {", ".join(missing)}',
            )
        if not_handled:
            raise fastapi.HTTPException(
                status_code=400,
                detail=f'Endpoint {self.name!r} received unknown arguments: {", ".join(not_handled)}',
            )
        # for name in set(self._defaults) - set(parsed):
        #    val = self._validators[name]
        #    if val.depends is not None:
        #        parsed[name] = val.validate_conditionally(value, parsed[validator.depends])
        #    else:
        #        parsed[name] = self._validators[name].validate(self._defaults[name])
        return parsed


class _ArgumentValidator:
    NOT_SET = object()

    def __init__(self, spec):
        self.name = spec["name"]
        log.info(f"Registering argument {self.name!r}")
        self.description = spec["help"]
        self.required = spec.get("required", False)
        self.default = spec.get("default", self.NOT_SET)
        self._nargs = "1" if "nargs" not in spec else spec["nargs"]
        self._type = spec["type"]
        self._null_if = spec.get("null-if", None)
        if self._null_if is not None:
            if not isinstance(self._null_if, dict):
                raise ValueError(f'[{self.name}] "null-if" value must be a dict')
            if "name" not in self._null_if or "value" not in self._null_if:
                raise ValueError(
                    f'[{self.name}] "null-if" must have keys "name" and "value"'
                )
        self._null_unless = spec.get("null-unless", None)
        if self._null_unless is not None:
            if not isinstance(self._null_unless, dict):
                raise ValueError(f'[{self.name}] "null-unless" value must be a dict')
            if "name" not in self._null_unless or "value" not in self._null_unless:
                raise ValueError(
                    f'[{self.name}] "null-unless" must have keys "name" and "value"'
                )
        if self._null_if is not None and self._null_unless is not None:
            raise ValueError(
                f'[{self.name}] Cannot set both "null-if" and "null-unless"'
            )
        # self._options = spec.get('options', [])
        self._options: typing.Any = spec["options"]
        if self._nargs not in ("1", "*", "+"):
            raise ValueError(f"[{self.name}] Invalid nargs: {self._nargs}")
        if self._type not in (
            "str",
            "int",
            "bool",
            "enum",
            "class",
            "arglist",
            "float",
            "query",
            "dynamic_enum",
            "object",
            "hyper_arglist",
        ):
            raise ValueError(f"[{self.name}] Invalid type: {self._type}")
        if self._type == "class":
            if len(self._options) != 1:
                raise ValueError(
                    f'[{self.name}] Argument of type "class" requires exactly one option.'
                )
            dotted_name = self._options[0]
            module, item = dotted_name.rsplit(".", maxsplit=1)
            mod = importlib.import_module(module)
            cls = getattr(mod, item)
            self._options = [cls]
        if self._type == "arglist":
            if len(self._options) != 1 or not isinstance(self._options[0], dict):
                raise ValueError(
                    f'[{self.name}] Argument of type "arglist" requires exactly one option of type "dict".'
                )
            if (
                "map-path" not in self._options[0]
                or "multi-valued" not in self._options[0]
            ):
                raise ValueError(
                    f'[{self.name}] Option of "arglist" argument must contain "map-path" and "multi-valued".'
                )
            module, item = self._options[0]["map-path"].rsplit(".", maxsplit=1) # type: ignore
            self._options[0] = ArgumentListParser(
                name=self.name,
                lookup_map=getattr(importlib.import_module(module), item),
                multi_valued=self._options[0]["multi-valued"],  # type: ignore
            )
        if self._type == "hyper_arglist":
            if len(self._options) != 1 or not isinstance(self._options[0], dict):
                raise ValueError(
                    f'[{self.name}] Argument of type "hyper_arglist" requires exactly one option of type "dict".'
                )
            if (
                "map-path" not in self._options[0]
                or "multi-valued" not in self._options[0]
            ):
                raise ValueError(
                    f'[{self.name}] Option of "hyper_arglist" argument must contain "map-path" and "multi-valued".'
                )
            module, item = self._options[0]["map-path"].rsplit(".", maxsplit=1) # type: ignore
            self._options[0] = HyperArgumentListParser(
                name=self.name,
                lookup_map=getattr(importlib.import_module(module), item),
                multi_valued=self._options[0]["multi-valued"], # type: ignore
            )
        if self._type == "dynamic_enum":
            if len(self._options) != 1 or not isinstance(self._options[0], str):
                raise ValueError(
                    f'[{self.name}] Argument of type "dynamic_enum" requires exactly one option of type "str".'
                )
            module, item = self._options[0].rsplit(".", maxsplit=1)
            self._options[0] = set(getattr(importlib.import_module(module), item))

    @property
    def depends(self):
        if self._null_if:
            return self._null_if["name"]
        if self._null_unless:
            return self._null_unless["name"]
        return None

    @property
    def dtype(self):
        return self._type

    def validate_conditionally(self, value, flag):
        if self._null_if and self._null_if["value"] == flag:
            if value is not None:
                raise fastapi.HTTPException(
                    detail=f'Argument {self.name!r} must be null because {self.depends!r} == {self._null_if["value"]}',
                    status_code=400,
                )
            return None
        elif self._null_if and self._null_if["value"] != flag:
            return self.validate(value)
        elif self._null_unless and self._null_unless["value"] == flag:
            return self.validate(value)
        else:
            if value is not None:
                raise fastapi.HTTPException(
                    detail=f'Argument {self.name!r} must be null because {self.depends!r} != {self._null_unless["value"]}',
                    status_code=400,
                )
            return None

    def validate(self, value):
        if self._nargs == "1":
            return self._validate(value)
        else:
            if not isinstance(value, list):
                raise fastapi.HTTPException(
                    detail=f"{self.name!r} is a multi-valued argument. Expected a list. (got {value})",
                    status_code=400,
                )
            if self._nargs == "+" and not value:
                raise fastapi.HTTPException(
                    detail=f"{self.name!r} requires at least 1 value.", status_code=400
                )
            return [self._validate(x) for x in value]

    def _validate(self, x):
        match self._type:
            case "str":
                if not isinstance(x, str):
                    self._raise_invalid_type("string", x)
                return x
            case "int":
                if not isinstance(x, int):
                    self._raise_invalid_type("int", x)
                return x
            case "bool":
                if not isinstance(x, bool) and (
                    not isinstance(x, int) or x not in (0, 1)
                ):
                    self._raise_invalid_type("bool", x)
                return x
            case "float":
                if not isinstance(x, float):
                    self._raise_invalid_type("float", x)
                return x
            case "enum":
                if not isinstance(x, str):
                    raise fastapi.HTTPException(
                        detail=f"{self.name!r} enum argument must be of type string, got {x.__class__.__name__}",
                        status_code=400,
                    )
                if x not in self._options:
                    raise fastapi.HTTPException(
                        detail=f"Invalid option for {self.name!r}: {x} (valid options: {self._options})",
                        status_code=400,
                    )
                return x
            case "class":
                try:
                    return self._options[0](x)  # type: ignore
                except Exception as e:
                    raise fastapi.HTTPException(
                        detail=f"Error while converting {self.name!r} to {self._options[0].__class__.__name__}: {e}",
                        status_code=400,
                    )
            case "query":
                try:
                    return db_util.json_to_query(x)
                except Exception as e:
                    raise fastapi.HTTPException(
                        detail=f"Invalid query for param {self.name!r}: {x} ({e})",
                        status_code=400,
                    )
            case "arglist":
                return self._options[0].validate(x)
            case "hyper_arglist":
                return self._options[0].validate(x)
            case "dynamic_enum":
                if not isinstance(x, str):
                    raise fastapi.HTTPException(
                        detail=f"{self.name!r} dynamic_enum argument must be of type string, got {x.__class__.__name__}",
                        status_code=400,
                    )
                if x not in self._options[0]:
                    raise fastapi.HTTPException(
                        detail=f"Invalid option for {self.name!r}: {x} (valid options: {self._options})",
                        status_code=400,
                    )
                return x
            case "object":
                return x

    def _raise_invalid_type(self, expected, got):
        raise fastapi.HTTPException(
            detail=f"{self.name!r} must be of type {expected}, got {got.__class__.__name__}",
            status_code=400,
        )


##############################################################################
##############################################################################
# ArgList Support
##############################################################################


class ArgumentConsumer:
    @staticmethod
    def get_arguments() -> dict[str, Argument]:
        return {}


class Argument(abc.ABC):
    _NOT_SET = object()

    def __init__(self, name: str, description: str, data_type, default=_NOT_SET):
        self._name = name
        self._description = description
        self._data_type = data_type
        self._default = default

    @property
    def argument_name(self):
        return self._name

    @property
    def argument_description(self):
        return self._description

    @property
    def default(self):
        return self._default

    @property
    def has_default(self):
        return self._default is not self._NOT_SET

    @property
    def argument_type(self):
        return self._data_type

    @abc.abstractmethod
    def validate(self, value, *, tuning=False):
        pass

    @property
    @abc.abstractmethod
    def legal_values(self):
        pass

    @abc.abstractmethod
    def get_json_spec(self):
        return {
            "name": self._name,
            "description": self._description,
            "type": self._data_type.__name__,
            "has-default": self.has_default,
            "default": self._default if self.has_default else None,
            "readable-options": self.legal_values,
        }

    @staticmethod
    @abc.abstractmethod
    def supported_hyper_param_specs():
        return []

    def raise_invalid(self, msg):
        raise fastapi.HTTPException(
            detail=f"Argument {self.argument_name!r} is invalid: {msg}", status_code=400
        )


class FloatArgument(Argument):
    def __init__(
        self,
        name: str,
        description: str,
        default=Argument._NOT_SET,
        minimum: float | None = None,
        maximum: float | None = None,
    ):
        super().__init__(name, description, float, default)
        self._min = minimum
        self._max = maximum

    def validate(self, value, *, tuning=False):
        if not isinstance(value, float):
            self.raise_invalid(f"Must be float, got {value.__class__.__name__}")
        if self._min is not None and value < self._min:
            self.raise_invalid(f"Must be >= {self._min}")
        if self._max is not None and value > self._max:
            self.raise_invalid(f"Must be <= {self._max}")
        return value

    def legal_values(self):
        lo = self._min if self._min is not None else -float("inf")
        hi = self._max if self._max is not None else float("inf")
        return f"[{lo}, {hi}]"

    def get_json_spec(self):
        return super().get_json_spec() | {"minimum": self._min, "maximum": self._max}

    @staticmethod
    def supported_hyper_param_specs():
        return ["values", "floats"]


class IntArgument(Argument):
    def __init__(
        self,
        name: str,
        description: str,
        default=Argument._NOT_SET,
        minimum: int | None = None,
        maximum: int | None = None,
    ):
        super().__init__(name, description, int, default)
        self._min = minimum
        self._max = maximum

    def validate(self, value, *, tuning=False):
        if not isinstance(value, int):
            self.raise_invalid(f"Must be int, got {value.__class__.__name__}")
        if self._min is not None and value < self._min:
            self.raise_invalid(f"Must be >= {self._min}")
        if self._max is not None and value > self._max:
            self.raise_invalid(f"Must be <= {self._max}")
        return value

    def legal_values(self):
        lo = self._min if self._min is not None else -float("inf")
        hi = self._max if self._max is not None else float("inf")
        if math.isinf(lo) or math.isinf(hi):
            return f"[{lo}, {hi}]"
        return range(self._min, self._max + 1)

    def get_json_spec(self):
        return super().get_json_spec() | {"minimum": self._min, "maximum": self._max}

    @staticmethod
    def supported_hyper_param_specs():
        return ["values", "range"]


class EnumArgument(Argument):
    def __init__(
        self,
        name: str,
        description: str,
        default=Argument._NOT_SET,
        options: list[str] = None,
    ):
        super().__init__(name, description, str, default)
        if options is None:
            self._options = []
        else:
            self._options = options

    def validate(self, value, *, tuning=False):
        if not isinstance(value, str):
            self.raise_invalid(f"Must be string, got {value.__class__.__name__}")
        if value not in self._options:
            self.raise_invalid(f'Must be one of {", ".join(self._options)}')
        return value

    def legal_values(self):
        return self._options

    def get_json_spec(self):
        return super().get_json_spec() | {"options": self._options}

    @staticmethod
    def supported_hyper_param_specs():
        return ["values"]


class BoolArgument(Argument):
    def __init__(self, name: str, description: str, default=Argument._NOT_SET):
        super().__init__(name, description, bool, default)

    def validate(self, value, *, tuning=False):
        if isinstance(value, bool) or (isinstance(value, int) and value in (0, 1)):
            return value
        self.raise_invalid(f"Must be Boolean, got {value.__class__.__name__}")

    def legal_values(self):
        return [False, True]

    def get_json_spec(self):
        return super().get_json_spec() | {}

    @staticmethod
    def supported_hyper_param_specs():
        return ["values"]


class StringArgument(Argument):
    def __init__(self, name: str, description: str, default=Argument._NOT_SET):
        super().__init__(name, description, str, default)

    def validate(self, value, *, tuning=False):
        if not isinstance(value, str):
            self.raise_invalid(f"Must be string, got {value.__class__.__name__}")
        return value

    def legal_values(self):
        return "Any"

    def get_json_spec(self):
        return super().get_json_spec() | {}

    @staticmethod
    def supported_hyper_param_specs():
        return ["values"]


class QueryArgument(Argument):
    def __init__(self, name: str, description: str, default=Argument._NOT_SET):
        super().__init__(name, description, issue_db_api.Query, default)

    def validate(self, value, *, tuning=False):
        if value is None:
            return value
        try:
            return db_util.object_to_query(value)
        except Exception as e:
            self.raise_invalid(f"Invalid query: {e}")

    def legal_values(self):
        return ""

    def get_json_spec(self):
        return super().get_json_spec() | {}

    @staticmethod
    def supported_hyper_param_specs():
        return ["values"]


class NestedArgument(Argument):

    class _Wrapper(ArgumentConsumer):
        def __init__(self, payload):
            self._payload = payload
        def get_arguments(self) -> dict[str, Argument]:
            return self._payload


    def __init__(self,
                 name: str,
                 description: str, *,
                 spec: dict[str, dict[str, Argument]]):
        super().__init__(name, description, dict)
        self._spec = {key: self._Wrapper(value) for key, value in spec.items()}
        self._parser = ArgumentListParser(name, self._spec)
        self._hyper_parser = HyperArgumentListParser(name, self._spec)

    def validate(self, value, *, tuning=False):
        # if not isinstance(value, dict):
        #     self.raise_invalid(f'Expected a dictionary, got {value.__class__.__name__}')
        # if len(value) != 1:
        #     self.raise_invalid(f'Expected a single key, got {len(value)}')
        # key, nested = next(iter(value.items()))
        # if key not in self._spec:
        #     self.raise_invalid(f'Illegal key {key!r}')
        try:
            if tuning:
                return self._hyper_parser.validate(value)
            return self._parser.validate(value)
        except fastapi.HTTPException as e:
            self.raise_invalid(e.detail)

    @property
    def legal_values(self):
        return {}

    def get_json_spec(self):
        return super().get_json_spec()

    @staticmethod
    def supported_hyper_param_specs():
        return ['nested']


class ArgumentListParser:
    def __init__(self, name: str, lookup_map, *, multi_valued=False):
        self._map = lookup_map
        self._name = name
        self._multi_valued = multi_valued

    def validate(self, values):
        if not isinstance(values, dict):
            raise fastapi.HTTPException(
                detail=f"Parameter {self._name!r} requires a dict.", status_code=400
            )
        if (not self._multi_valued) and len(values) != 1:
            raise fastapi.HTTPException(
                detail=f"[{self._name}] Non-multivalued argument list can only contain a single key",
                status_code=400,
            )
        # if not self._multi_valued and 'default' in values:
        #    raise fastapi.HTTPException(detail='Single-valued arglist cannot contain a default section',
        #                                status_code=400)
        result = {}
        defaults_by_class = collections.defaultdict(dict)
        defaults = None
        for name, args in values.items():
            if "." in name:
                cls, index = name.split(".")
                index = int(index)
                if (not self._multi_valued) and index != 0:
                    raise fastapi.HTTPException(
                        detail="arglist index should be 0 in single-valued arglist",
                        status_code=400,
                    )
                if cls in result and index in result[cls]:
                    raise fastapi.HTTPException(
                        detail=f"Duplicate entry for {cls}.{index} in arglist",
                        status_code=400,
                    )
                result.setdefault(cls, {})[index] = self._validate(cls, args)
            elif name != "default":
                # if name in indices:
                #    raise fastapi.HTTPException(detail=f'Un-numbered occurrence of name {name!r} in arglist',
                #                                status_code=400)
                if not self._multi_valued:
                    raise fastapi.HTTPException(
                        detail=f"Class default {name!r} not allowed in single valued arglist",
                        status_code=400,
                    )
                if name in defaults_by_class:
                    raise fastapi.HTTPException(
                        detail=f"Encountered duplicate class default for class {name!r} in arglist",
                        status_code=400,
                    )
                defaults_by_class[name] = self._validate(name, args)
            else:
                if not self._multi_valued:
                    raise fastapi.HTTPException(
                        detail=f"default section not allowed in single valued arglist",
                        status_code=400,
                    )
                if defaults is not None:
                    raise fastapi.HTTPException(
                        detail="Encountered duplicate default section in arglist",
                        status_code=400,
                    )
                # Find a common base class for all classes in the map
                classes = list(self._map.values())
                proposal_class, *test_classes = classes
                result = None
                for base in reversed(proposal_class.__mro__):
                    if all(issubclass(c, base) for c in test_classes):
                        result = base
                    else:
                        break
                if result is None or not issubclass(result, ArgumentConsumer):
                    raise fastapi.HTTPException(
                        detail=f"Failed to find base class for resolving global defaults for {self._name!r}",
                        status_code=500,
                    )
                defaults = self._validate(result, args)
        for cls_name, value in defaults_by_class.items():
            if cls_name not in result:
                raise fastapi.HTTPException(
                    detail=f"No entries (indices) to apply default {cls_name!r} to",
                    status_code=400,
                )
            for v in result[cls_name].values():
                v.update(value)
        if defaults is not None:
            if not result:
                raise fastapi.HTTPException(
                    detail=f"No entries (indices) to apply defaults to in arglist",
                    status_code=400,
                )
            for indices in result.values():
                for v in indices.values():
                    v.update(defaults)
        return result

    def _validate(self, name, obj, *, klass=None):
        log.info(f"Parsing argument list for {name!r}")
        log.info(f"Raw input: {obj}")
        if klass is None:
            try:
                cls: typing.Type[ArgumentConsumer] = self._map[name]
            except KeyError:
                raise fastapi.HTTPException(
                    detail=f'Cannot find class {name!r} in map (available: {", ".join(self._map)})',
                    status_code=400,
                )
        else:
            cls = klass
        args = cls.get_arguments()
        if not isinstance(obj, dict):
            raise fastapi.HTTPException(
                detail=f"arglist must contain dicts, got {obj.__class__.__name__}",
                status_code=400,
            )
        required = {arg.argument_name for arg in args.values() if not arg.has_default}
        result = {}
        for key, value in obj.items():
            if key not in args:
                raise fastapi.HTTPException(
                    detail=f"Unknown argument {key} for {name}", status_code=400
                )
            # result[key] = args[key].validate(value)
            result[key] = self.validate_value(args[key], value)
            log.info(f"Parsed argument {key!r}: {result[key]}")

        for arg in args.values():
            if arg.argument_name not in result and arg.has_default:
                # result[arg.argument_name] = arg.validate(arg.default)
                result[arg.argument_name] = self.validate_default(arg, arg.default)
                log.info(
                    f"Applied default: {arg.argument_name!r}: {result[arg.argument_name]}"
                )
        missing = required - set(result)
        if missing:
            raise fastapi.HTTPException(
                detail=f'Missing arguments for {name}: {", ".join(missing)}',
                status_code=400,
            )
        return result

    def validate_value(self, validator: Argument, value: typing.Any) -> typing.Any:
        return validator.validate(value)

    def validate_default(self, validator: Argument, value: typing.Any) -> typing.Any:
        return validator.validate(value)


class HyperArgumentListParser(ArgumentListParser):
    def validate_value(self, validator: Argument, value: typing.Any) -> typing.Any:
        if not isinstance(value, dict):
            raise fastapi.HTTPException(
                detail="hyper param arglist entry must be a dict", status_code=400
            )
        if "type" not in value:
            raise fastapi.HTTPException(
                detail='hyper param arglist entry must contain a field "type"',
                status_code=400,
            )
        if "options" not in value:
            raise fastapi.HTTPException(
                detail='hyper param arglist entry must contain a field "options"',
                status_code=400,
            )
        if not isinstance(value["options"], dict):
            raise fastapi.HTTPException(
                detail='hyper param arglist "options" must be a dict', status_code=400
            )
        match value["type"]:
            case "range":
                return {
                    "type": "range",
                    "options": self._validate_range(validator, value["options"]),
                }
            case "values":
                return {
                    "type": "values",
                    "options": self._validate_values(validator, value["options"]),
                }
            case "floats":
                return {
                    "type": "floats",
                    "options": self._validate_floats(validator, value["options"]),
                }
            case _ as x:
                raise fastapi.HTTPException(
                    detail=f"Invalid hyper param arglist type: {x}", status_code=400
                )

    def _validate_range(self, validator: Argument, options: dict):
        if "range" not in validator.supported_hyper_param_specs():
            raise fastapi.HTTPException(
                detail=f"Hyper param arglist item of type {validator.__class__.__name__} does not support range.",
                status_code=400,
            )
        if "step" not in options:
            options["step"] = None
        else:
            self._check_opt_type(options["step"], int, "step")
        if "sampling" not in options:
            options["sampling"] = "linear"
        else:
            self._check_opt_type(options["sampling"], str, "sampling")
        self._check_opt_key(options, {"start", "stop", "step", "sampling"})
        self._check_opt_type(options["start"], int, "start")
        self._check_opt_type(options["stop"], int, "stop")
        return {
            "start": options["start"],
            "stop": options["stop"],
            "step": options["step"],
            "sampling": options["sampling"],
        }

    def _validate_values(self, validator: Argument, options: dict):
        if "values" not in validator.supported_hyper_param_specs():
            raise fastapi.HTTPException(
                detail=f'Hyper param arglist item of type {validator.__class__.__name__} does not support "values".',
                status_code=400,
            )
        self._check_opt_key(options, {"values"})
        self._check_opt_type(options["values"], list, "options")
        return {"values": [validator.validate(value) for value in options["values"]]}

    def _validate_floats(self, validator: Argument, options: dict):
        if "floats" not in validator.supported_hyper_param_specs():
            raise fastapi.HTTPException(
                detail=f'Hyper param arglist item of type {validator.__class__.__name__} does not support "floats".',
                status_code=400,
            )
        if "step" not in options:
            options["step"] = None
        else:
            self._check_opt_type(options["step"], float, "step")
        if "sampling" not in options:
            options["sampling"] = "linear"
        else:
            self._check_opt_type(options["sampling"], str, "sampling")
        self._check_opt_key(options, {"start", "stop", "step", "sampling"})
        self._check_opt_type(options["start"], float, "start")
        self._check_opt_type(options["stop"], float, "stop")
        return {
            "start": options["start"],
            "stop": options["stop"],
            "step": options["step"],
            "sampling": options["sampling"],
        }

    def _require_opt_field(self, options, key):
        if key not in options:
            raise fastapi.HTTPException(
                detail=f"Hyper param arglist must have options field {key}",
                status_code=400,
            )

    def _check_opt_type(self, obj, typ, name):
        if not isinstance(obj, typ):
            raise fastapi.HTTPException(
                detail=f"Hyper param arglist item option {name} be of type {typ}",
                status_code=400,
            )

    def _check_opt_key(self, options, keys):
        for key in keys:
            self._require_opt_field(options, key)
        if rem := set(options) - keys:
            raise fastapi.HTTPException(
                detail=f"Superfluous option keys: {rem}", status_code=400
            )

    def validate_default(self, validator: Argument, value: typing.Any) -> typing.Any:
        return self.validate_value(
            validator,
            {"type": "values", "options": {"values": [validator.validate(value)]}},
        )
