import contextlib
import io

from .cli import invoke_pipeline_with_command as _invoke
from .cli import invoke_pipeline_with_config as _invoke_api
from .cli import main as _main
from .classifiers import AbstractModel
from .classifiers import models as _models
from .feature_generators import AbstractFeatureGenerator
from .feature_generators import generators as _generators
from . import OutputMode


def run_cli_app():
    _main()


def run_command(cmd: str, *, capture_output=False):
    if capture_output:
        out = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(out):
            with contextlib.redirect_stderr(err):
                _invoke(cmd)
        return out.getvalue(), err.getvalue()
    _invoke(cmd)


def run_config(config: dict, *, capture_output=False):
    if capture_output:
        out = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(out):
            with contextlib.redirect_stderr(err):
                _invoke_api(config)
        return out.getvalue(), err.getvalue()
    _invoke_api(config)


def get_available_models() -> dict[str, type[AbstractModel]]:
    return _models


def get_model_by_name(name: str) -> type[AbstractModel]:
    return _models[name]


def get_feature_generators() -> dict[str, type[AbstractFeatureGenerator]]:
    return _generators


def get_feature_generator_by_name(name: str) -> type[AbstractFeatureGenerator]:
    return _generators[name]


def get_output_modes() -> dict[str, OutputMode]:
    return {
        'Detection': OutputMode.Detection,
        'Classification3': OutputMode.Classification3,
        'Classification3Simplified': OutputMode.Classification3Simplified,
        'Classification8': OutputMode.Classification8
    }

def get_output_mode_by_name(name: str) -> OutputMode:
    return OutputMode.from_string(name)
