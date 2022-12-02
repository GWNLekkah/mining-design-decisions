"""
This module provides special functionality for
handling configuration options in a centralized
manner.

It provides an easy to configure way to set up
an elaborate, automatically handled command
line interface
"""

##############################################################################
##############################################################################
# Imports
##############################################################################

import argparse
import collections
import importlib
import json

##############################################################################
##############################################################################
# Custom Exceptions
##############################################################################


class NoSuchSetting(LookupError):

    def __init__(self, attribute, action):
        message = (f'Cannot perform action {action!r} on setting '
                   f'{attribute} since it does not exist')
        super().__init__(message)


class NoDefault(Exception):

    def __init__(self, attribute):
        message = f'No default available for non-given attribute {attribute}'
        super().__init__(message)


##############################################################################
##############################################################################
# Configuration Class (not thread safe)
##############################################################################


class Config:

    _self = None
    _NOT_SET = object()

    def __new__(cls):
        if cls._self is None:
            cls._self = self = super().__new__(cls)
        return cls._self

    def __init__(self):
        self._namespace = {}
        self._types = {}

    def reset(self):
        self._namespace = {}
        self._types = {}

    def set(self, name, value):
        name = name.replace('-', '_')
        if name not in self._namespace:
            raise NoSuchSetting(name, 'set')
        converter: type = self._types[name]
        if (not isinstance(converter, type)) or not isinstance(value, converter):
            if value is not None:
                value = converter(value)
        self._namespace[name] = value

    def get(self, name):
        name = name.replace('-', '_')
        if name not in self._namespace:
            raise NoSuchSetting(name, 'get')
        value = self._namespace[name]
        if value is self._NOT_SET:
            raise NoDefault(name)
        return value

    def get_all(self, name):
        name = name.replace('-', '_')
        return {
            key: value
            for key, value in self._namespace.items()
            if key.startswith(name) and value is not self._NOT_SET
        }

    def clone(self, source, target):
        self.register(target, self._types[source])
        self.set(target, self.get(source))

    def register(self, name, type_or_validator, default=_NOT_SET):
        name = name.replace('-', '_')
        self._namespace[name] = default
        self._types[name] = type_or_validator

    def is_registered(self, name: str) -> bool:
        return name in self._namespace


conf = Config()


##############################################################################
##############################################################################
# Command Line Interface Wrapper
##############################################################################


class CLIApp:

    def __init__(self, filename: str):
        with open(filename) as file:
            spec = json.load(file)
        (self.__parser,
         self.__dict_values,
         self.__borrowed,
         self.__qualnames) = _build_cli_app(spec)
        self.__callbacks = {}
        self.__constraints = []
        self.__setup_callbacks = []

    def callback(self, event):
        def wrapper(func):
            self.register_callback(event, func)
            return func
        return wrapper

    def register_callback(self, event, func):
        self.__callbacks[event] = func

    def register_setup_callback(self, func):
        self.__setup_callbacks.append(func)

    def add_constraint(self, predicate, message, *keys):
        self.__constraints.append((keys, predicate, message))

    def parse_and_dispatch(self, cli_args=None):
        if cli_args is None:
            args = self.__parser.parse_args()
        else:
            args = self.__parser.parse_args(cli_args)
        self.__expand_dictionaries(args)
        active_name, active_qualname = self.__get_active_command(args)
        self.__resolve_borrows(args, active_name)
        for name, value in vars(args).items():
            if name.startswith('subcommand_name_'):
                continue
            full_name = f'{active_qualname}.{name}'
            conf.set(full_name, value)
        # Check constraints
        for keys, predicate, message in self.__constraints:
            try:
                values = [conf.get(key) for key in keys]
            except (NoSuchSetting, NoDefault):  # This command might have not been called
                continue
            if not predicate(*values):
                error = f'Constraint check on {",".join(keys)} failed: {message}'
                raise ValueError(error)
        for callback in self.__setup_callbacks:
            callback()
        self.__callbacks[active_qualname]()

    @staticmethod
    def __get_active_command(args):
        i = 0
        while f'subcommand_name_{i+1}' in args:
            i += 1
        pieces = [
            getattr(args, f'subcommand_name_{j}')
            for j in range(i + 1)
        ]
        return pieces[-1], '.'.join(pieces)

    def __expand_dictionaries(self, args):
        for name in self.__dict_values:
            name = name.replace('-', '_')
            if name not in vars(args):
                continue
            value = getattr(args, name)
            if value is None:
                setattr(args, name, {})
            else:
                setattr(args, name, _dict_converter(value))

    def __resolve_borrows(self, args, owner):
        borrowed_from = set()
        for original, name in self.__borrowed[owner]:
            full_name = f'{original}.{name}'
            full_name = self.__qualnames[full_name]
            conf.set(full_name, getattr(args, name.replace('-', '_')))
            borrowed_from.add(original)
        for parent in borrowed_from:
            self.__resolve_borrows(args, parent)


def _dict_converter(items):
    if isinstance(items, dict):
        return items
    result = collections.defaultdict(dict)
    for item in items:
        key, value = item.split('=')
        if '.' in key:
            super_key, sub_key = key.split('.')
        else:
            super_key = 'default'
            sub_key = key
        result[super_key][sub_key] = value
    return result


def _enum_converter_factory(options):
    def enum_converter(x):
        if x not in options:
            raise ValueError(f'Invalid option: {x!r}')
        return x
    return enum_converter


def _bool_converter(x):
    if x == '0' or x == 'False':
        return False
    if x == '1' or x == 'True':
        return True
    return bool(x)


def _list(of):
    def wrapper(x):
        return [of(y) for y in x]
    return wrapper


##############################################################################
##############################################################################
# Command Line Interface Builder
##############################################################################


def _build_cli_app(spec):
    cached_args = {}
    dict_values = set()
    borrowed = {}
    qualnames = {}
    parser = argparse.ArgumentParser(spec.get('name', ''))
    subparsers = parser.add_subparsers(
        dest='subcommand_name_0', title='Sub-commands'
    )
    _add_subparser_section(spec,
                           subparsers,
                           cached_args,
                           dict_values,
                           borrowed,
                           1,
                           '',
                           qualnames)
    return parser, dict_values, borrowed, qualnames


def _add_subparser_section(spec,
                           subparsers,
                           cached_args,
                           dict_values,
                           borrowed,
                           section_count,
                           prefix,
                           qualnames):
    # First, add the regular sub-commands
    for command in spec.get('commands', []):
        _add_command(subparsers,
                     command,
                     cached_args,
                     dict_values,
                     borrowed,
                     prefix,
                     qualnames)
    # Now, add the nested sub-parser sections
    for subparser_section in spec.get('subparsers', []):
        parser = subparsers.add_parser(
            subparser_section['name'],
            help=subparser_section.get('help', '')
        )
        nested_subparsers = parser.add_subparsers(
            dest=f'subcommand_name_{section_count}',
            title=('-'.join(['sub'*(section_count + 1)]) + '-commands').capitalize()
        )
        if not prefix:
            prefix = subparser_section['name']
        else:
            prefix = f'{prefix}.{subparser_section["name"]}'
        _add_subparser_section(
            subparser_section,
            nested_subparsers,
            cached_args,
            dict_values,
            borrowed,
            section_count + 1,
            prefix,
            qualnames
        )


def _add_command(subparsers,
                 command,
                 cached_args,
                 dict_values,
                 borrowed,
                 prefix,
                 qualnames):
    parser = subparsers.add_parser(
        command['name'],
        help=command.get('help', '')
    )
    cmd_name = command['name']
    cached_args[cmd_name] = {}
    for argument in command.get('args', []):
        # First, store the argument in the cache
        cached_args[cmd_name][argument['name']] = argument
        # Add the argument to the parser
        _add_argument(parser,
                      argument,
                      cmd_name,
                      dict_values,
                      prefix,
                      qualnames)
    borrowed[cmd_name] = local_borrowed = []
    for reference in command.get('import_args', []):
        owning_command, borrowed_name = reference.split('/')
        if borrowed_name == '*':
            for argument in cached_args[owning_command].values():
                _add_argument_if_not_present(parser,
                                             argument,
                                             owning_command,
                                             local_borrowed,
                                             cmd_name,
                                             dict_values,
                                             cached_args,
                                             prefix,
                                             qualnames)
        else:
            argument = cached_args[owning_command][borrowed_name]
            _add_argument_if_not_present(parser,
                                         argument,
                                         owning_command,
                                         local_borrowed,
                                         cmd_name,
                                         dict_values,
                                         cached_args,
                                         prefix,
                                         qualnames)


def _add_argument_if_not_present(parser,
                                 argument,
                                 owning_command,
                                 local_borrowed,
                                 cmd_name,
                                 dict_values,
                                 cached_args,
                                 prefix,
                                 qualnames):
    arg_name = argument['name']
    # First, check whether we already have this argument
    # registered because of a diamond structure
    if any(pair[1] == arg_name for pair in local_borrowed):
        # Add the item to the cache in case other
        # commands re-use arguments from this command
        cached_args[cmd_name][arg_name] = argument
        return
    # This is a new command, so we add it to the list of
    # borrowed commands.
    # Note that we need the owning_command for
    # argument resolving later when parsing CLI parameters.
    local_borrowed.append((owning_command, arg_name))
    cached_args[cmd_name][arg_name] = argument
    _add_argument(parser, argument, cmd_name, dict_values, prefix, qualnames)


def _add_argument(parser,
                  argument,
                  parent_cmd_name,
                  dict_values,
                  prefix,
                  qualnames):
    if argument['style'] == 'positional':
        names = [argument['name']]
    else:
        names = [f'--{argument["name"]}']
        if 'alias' in argument:
            names.append(f'-{argument["alias"]}')
    kwargs = {'help': argument.get('help', '')}
    conf_kwargs = {}
    if argument['style'] == 'flag':
        kwargs |= {
            'action': 'store_true',
            'default': False,
        }
        arg_type = _bool_converter
        conf_converter = _bool_converter
        conf_kwargs['default'] = False
    else:
        arg_type, extra_args, conf_converter = _resolve_type(argument,
                                                             dict_values)
        kwargs['type'] = arg_type
        kwargs |= extra_args
    if argument.get('nargs', '1') != '1':
        kwargs['nargs'] = argument['nargs']
    if argument.get('required', False):
        kwargs['required'] = argument['required']
    if 'default' in argument:
        default_value = arg_type(argument['default'])
        if argument.get('nargs', '1') != '1':
            default_value = [default_value]
        kwargs['default'] = default_value
        conf_kwargs['default'] = default_value
    parser.add_argument(*names, **kwargs)
    full_name = f'{parent_cmd_name}.{argument["name"]}'
    if argument.get('nargs', '1') != '1' and conf_converter not in [dict, list, tuple]:
        conf_converter = _list(conf_converter)
    if prefix:
        full_name = f'{prefix}.{full_name}'
    qualnames[f'{parent_cmd_name}.{argument["name"]}'] = full_name
    conf.register(full_name, conf_converter, **conf_kwargs)


def _resolve_type(argument, dict_values):
    argument_type = argument['type']
    match argument_type:
        case 'str':
            return str, {}, str
        case 'int':
            return int, {}, int
        case 'float':
            return float, {}, float
        case 'dict':
            dict_values.add(argument['name'])
            return str, {}, dict
        case 'enum':
            return str, {'choices': argument['options']}, str
        case 'class':
            dotted_name = argument['options'][0]
            module, item = dotted_name.split('.')
            mod = importlib.import_module(module)
            cls = getattr(mod, item)
            return cls, {}, cls
        case _:
            raise ValueError(f'Unknown type {argument_type}')
