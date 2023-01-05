"""
Command line utility for managing and training
deep learning classifiers.
"""

##############################################################################
##############################################################################
# Imports
##############################################################################

import argparse
import collections
import json
import os.path
import pathlib
import getpass
import warnings

from . import classifiers, kw_analyzer
from .classifiers import HyperParameter

from . import feature_generators
from .feature_generators import ParameterSpec, OutputMode
from . import data_manager

from . import learning
from .config import conf, CLIApp

from . import analysis
from . import prediction


##############################################################################
##############################################################################
# Parser Setup
##############################################################################


def main(args=None):
    conf.reset()
    app = build_app()
    app.parse_and_dispatch(args)


def build_app():
    location = os.path.split(__file__)[0]
    app = CLIApp(os.path.join(location, 'cli.json'))

    def add_eq_len_constraint(p, q):
        app.add_constraint(lambda x, y: len(x) == len(y),
                           'Argument lists must have equal length.',
                           p, q)

    def add_min_delta_constraints(cmd):
        app.add_constraint(lambda deltas, attrs: len(deltas) == len(attrs) or len(deltas) == 1,
                           'Requirement not satisfied: len(min-delta) = len(trimming-attributes) or len(min-delta) = 1',
                           f'{cmd}.min-delta', f'{cmd}.trimming-attribute')

    add_min_delta_constraints('run_analysis.summarize')
    add_min_delta_constraints('run_analysis.plot')
    add_min_delta_constraints('run_analysis.plot-attributes')
    add_min_delta_constraints('run_analysis.confusion')
    add_min_delta_constraints('run_analysis.compare')
    add_min_delta_constraints('run_analysis.compare-stats')

    add_eq_len_constraint('run.classifier', 'run.input_mode')
    add_eq_len_constraint('run.early-stopping-min-delta', 'run.early-stopping-attribute')

    app.add_constraint(lambda ensemble, test_sep: ensemble == 'none' or not test_sep,
                       'Cannot use ensemble when using separate testing mode.',
                       'run.ensemble-strategy', 'run.test-separately')
    app.add_constraint(lambda store, test_separately: not (store and test_separately),
                       'Cannot store model when using separate testing mode.',
                       'run.store-model', 'run.test-separately')
    app.add_constraint(lambda store, k: not (store and k > 0),
                       'Cannot store model when using k-fold cross validation',
                       'run.store-model', 'run.k-cross')
    app.add_constraint(lambda project, study: project == 'None' or study == 'None',
                       'Cannot use test-project and test-study at the same time.',
                       'run.test-project', 'run.test-study')
    app.add_constraint(lambda cross_project, k: k == 0 or not cross_project,
                       'Cannot use --k-cross and --cross-project at the same time.',
                       'run.cross-project', 'run.k-cross')
    app.add_constraint(
        lambda k, quick_cross, test_study, test_project: not (
                (k > 0 and not quick_cross) and (test_study != 'None' or test_project != 'None')
        ),
        'Cannot use --test-study or --test-project without --quick-cross when k > 0',
        'run.k-cross', 'run.quick-cross', 'run.test-study', 'run.test-project'
    )
    app.add_constraint(
        lambda k, quick_cross: not quick_cross or k > 0,
        'Must specify k when running with --quick-cross',
        'run.k-cross', 'run.quick-cross'
    )
    app.add_constraint(
        lambda cross_project, test_study, test_project: not (
                cross_project and (test_study != 'None' or test_project != 'None')
        ),
        'Cannot use --test-study or --test-project in --cross-project mode',
        'run.cross-project', 'run.test-study', 'run.test-project'
    )
    app.add_constraint(
        lambda do_save, path: (not do_save) or (do_save and path),
        '--target-model-path must be given when storing a model.',
        'run.store-model', 'run.target-model-path'
    )
    app.add_constraint(
        lambda do_save, force_regenerate: (not do_save) or (do_save and force_regenerate),
        'Must use --force-regenerate-data when using --store-model.',
        'run.store-model', 'run.force-regenerate-data'
    )
    app.add_constraint(
        lambda do_analyze, _:
            not do_analyze or kw_analyzer.model_is_convolution(),
        'Can only analyze keywords when using a convolutional model',
        'run.analyze-keywords', 'run.classifier'
    )
    app.add_constraint(
        lambda do_analyze: (not do_analyze) or kw_analyzer.doing_one_run(),
        'Can not perform cross validation when extracting keywords',
        'run.analyze-keywords'
    )

    app.register_callback('predict', run_prediction_command)
    app.register_callback('run', run_classification_command)
    app.register_callback('visualize', run_visualize_command)
    app.register_callback('make-features', run_make_features_command)
    app.register_callback('list', run_list_command)
    app.register_callback('hyperparams', run_hyper_params_command)
    app.register_callback('generator-params', run_generator_params_command)
    app.register_callback('gui', run_gui_command)
    app.register_callback('combination-strategies', show_combination_strategies)
    app.register_callback('run_analysis.summarize',
                          analysis.run_summarize_command)
    app.register_callback('run_analysis.plot-attributes',
                          analysis.run_plot_attributes_command)
    app.register_callback('run_analysis.plot',
                          analysis.run_bar_plot_command)
    app.register_callback('run_analysis.compare',
                          analysis.run_comparison_command)
    app.register_callback('run_analysis.confusion',
                          analysis.run_confusion_matrix_command)
    app.register_callback('run_analysis.compare-stats',
                          analysis.run_stat_command)

    app.register_setup_callback(setup_peregrine)
    app.register_setup_callback(setup_storage)

    return app


def setup_peregrine():
    conf.clone('run.peregrine', 'system.peregrine')
    if conf.get('system.peregrine'):
        print('Running on Peregrine!')
        conf.register('system.peregrine.home', str, os.path.expanduser('~'))
        conf.register('system.peregrine.data', str, f'/data/{getpass.getuser()}')
        print(f'system.peregrine.home: {conf.get("system.peregrine.home")}')
        print(f'system.peregrine.data: {conf.get("system.peregrine.data")}')


def setup_storage():
    conf.register('system.storage.generators', list, [])


##############################################################################
##############################################################################
# Command Dispatch - GUI command
##############################################################################


def run_gui_command():
    from .ui import start_ui
    start_ui()


##############################################################################
##############################################################################
# Command Dispatch - Combination Strategies
##############################################################################


STRATEGIES = {
    'add': 'Add the values of layers to combine them.',
    'subtract': 'Subtract values of layers to combine them. Order matters',
    'multiply': 'Multiply the values of layers to combine them.',
    'max': 'Combine two inputs or layers by taking the maximum.',
    'min': 'Combine two inputs or layers by taking the minimum.',
    'dot': 'Combine two inputs or layers by computing their dot product.',
    'concat': 'Combine two inputs or layers by combining them into one single large layer.',
    'boosting': 'Train a strong classifier using boosting. Only a single model must be given.',
    'stacking': 'Train a strong classifier using stacking. Ignores the simple combination strategy.',
    'voting': 'Train a strong classifier using voting. Ignores the simple combination strategy.'
}


def show_combination_strategies():
    margin = max(map(len, STRATEGIES))
    for strategy in sorted(STRATEGIES):
        print(f'{strategy.rjust(margin)}: {STRATEGIES[strategy]}')


##############################################################################
##############################################################################
# Command Dispatch - list command
##############################################################################


def run_list_command():
    match conf.get('list.arg'):
        case 'classifiers':
            _show_classifier_list()
        case 'inputs':
            _show_input_mode_list()
        case 'outputs':
            _show_enum_list('Output Mode', feature_generators.OutputMode)


def _show_classifier_list():
    print(f'Available Classifiers:')
    _print_keys(list(classifiers.models))


def _show_input_mode_list():
    print(f'Available Input Modes:')
    _print_keys(list(feature_generators.generators))


def _show_enum_list(name: str, obj):
    print(f'Possible values for {name} setting:')
    keys = [key for key in vars(obj) if not key.startswith('_')]
    _print_keys(keys)


def _print_keys(keys):
    keys.sort()
    for key in keys:
        print(f'\t* {key}')


##############################################################################
##############################################################################
# Command Dispatch - hyperparams command
##############################################################################


def run_hyper_params_command():
    classifier = conf.get('hyperparams.classifier')
    if classifier not in classifiers.models:
        return print(f'Unknown classifier: {classifier}')
    cls = classifiers.models[classifier]
    keys = []
    name: str
    param: HyperParameter
    for name, param in cls.get_hyper_parameters().items():
        keys.append((f'{name} -- '
                     f'[min, max] = [{param.minimum}, {param.maximum}] -- '
                     f'default = {param.default}'))
    print(f'Hyper-parameters for {classifier}:')
    _print_keys(keys)


##############################################################################
##############################################################################
# Command Dispatch - generator-params command
##############################################################################


def run_generator_params_command():
    generator = conf.get('generator-params.generator')
    if generator not in feature_generators.generators:
        return print(f'Unknown feature generator: {generator}')
    cls = feature_generators.generators[generator]
    keys = []
    name: str
    param: ParameterSpec
    for name, param in cls.get_parameters().items():
        keys.append(f'{name} -- {param.description}')
    print(f'Parameters for {generator}:')
    _print_keys(keys)


##############################################################################
##############################################################################
# Command Dispatch - make-features command
##############################################################################


def run_make_features_command():
    source_file = conf.get('make-features.file')
    input_mode = conf.get('make-features.input_mode')
    output_mode = conf.get('make-features.output_mode')
    params = conf.get('make-features.params')
    imode_counts = collections.defaultdict(int)
    for imode in input_mode:
        number = imode_counts[imode]
        imode_counts[imode] += 1
        # Get the parameters for the feature generator
        mode_params = _normalize_param_names(
            params.get(imode, {}) |
            params.get('default', {}) |
            params.get(f'{imode}[{number}]', {})
        )
        # Validate that the parameters are valid
        valid_params = feature_generators.generators[imode].get_parameters()
        for param_name in mode_params:
            if param_name not in valid_params:
                raise ValueError(f'Invalid parameter for feature generator {imode}: {param_name}')
        # Generate the features
        filename = data_manager.get_feature_file(source_file,
                                                 imode,
                                                 output_mode,
                                                 **mode_params)
        data_manager.make_features(source_file,
                                   filename,
                                   imode,
                                   **mode_params)


##############################################################################
##############################################################################
# Command Dispatch - run command
##############################################################################


def run_visualize_command():
    classifier = conf.get('visualize.classifier')
    input_mode = conf.get('visualize.input_mode')
    output_mode = conf.get('visualize.output_mode')
    source_file = conf.get('visualize.file')
    params = conf.get('visualize.params')
    hyper_parameters = conf.get('visualize.hyperparams')

    _, _, factory = _get_model_factory(
        input_mode, output_mode, params, hyper_parameters,
        source_file, False, False, classifier
    )

    model = factory()

    import visualkeras.visualkeras as visualkeras
    visualkeras.graph_view(model,
                           to_file='output.png', legend=True).show()


def run_classification_command():
    classifier = conf.get('run.classifier')
    input_mode = conf.get('run.input_mode')
    output_mode = conf.get('run.output_mode')
    source_file = conf.get('run.file')
    params = conf.get('run.params')
    epochs = conf.get('run.epochs')
    split_size = conf.get('run.split_size')
    max_train = conf.get('run.max_train')
    k_cross = conf.get('run.k_cross')
    quick_cross = conf.get('run.quick_cross')
    regenerate_data = conf.get('run.force_regenerate_data')
    architectural_only = conf.get('run.architectural_only')
    hyper_parameters = conf.get('run.hyper-params')
    test_project = conf.get('run.test_project')

    datasets, labels, factory = _get_model_factory(
        input_mode, output_mode, params, hyper_parameters,
        source_file, architectural_only, regenerate_data, classifier
    )

    if conf.get('run.ensemble-strategy') != 'none':
        learning.run_ensemble(factory,
                              [dataset.features for dataset in datasets],
                              labels,
                              datasets[0].issue_keys,
                              OutputMode.from_string(output_mode).label_encoding)
        return

    # 5) Invoke actual DL process
    if k_cross == 0 and not conf.get('run.cross-project'):
        learning.run_single(factory(),
                            epochs,
                            split_size,
                            max_train,
                            labels,
                            OutputMode.from_string(output_mode),
                            OutputMode.from_string(output_mode).label_encoding,
                            *[dataset.features for dataset in datasets],
                            issue_keys=datasets[0].issue_keys,
                            test_project=test_project)
    else:
        learning.run_cross(factory,
                           epochs,
                           k_cross,
                           max_train,
                           quick_cross,
                           labels,
                           OutputMode.from_string(output_mode),
                           OutputMode.from_string(output_mode).label_encoding,
                           *[dataset.features for dataset in datasets],
                           issue_keys=datasets[0].issue_keys)


def _get_model_factory(input_mode,
                       output_mode,
                       params,
                       hyper_parameters,
                       source_file,
                       architectural_only,
                       regenerate_data,
                       classifier):
    # 1) Re-generate data
    if regenerate_data:
        # We can directly delegate to the `make-features` command,
        # which will also perform --param validation
        run_make_features_command()

    # 2) Collect all datasets
    datasets = []
    labels = None
    binary_labels = None
    imode_numbers = collections.defaultdict(int)
    for imode in input_mode:
        number = imode_numbers[imode]
        imode_numbers[imode] += 1
        mode_params = params.get(imode, {}) | params.get('default', {}) | params.get(f'{imode}[{number}]', {})
        dataset = data_manager.get_features(source_file,
                                            imode,
                                            output_mode,
                                            **mode_params)
        if labels is not None:
            assert labels == dataset.labels
            assert binary_labels == dataset.binary_labels
        else:
            labels = dataset.labels
            binary_labels = dataset.binary_labels
        datasets.append(dataset)

    if architectural_only:
        new_features = [[] for _ in range(len(datasets))]
        for index, is_architectural in enumerate(binary_labels):
            if is_architectural:
                for j, dataset in enumerate(datasets):
                    new_features[j].append(dataset.features[index])
        new_datasets = []
        for old_dataset, new_feature_list in zip(datasets, new_features):
            new_dataset = data_manager.Dataset(
                features=new_feature_list,
                labels=[label for bin_label, label in zip(binary_labels, labels) if bin_label],
                shape=old_dataset.shape,
                embedding_weights=old_dataset.embedding_weights,
                vocab_size=old_dataset.vocab_size,
                weight_vector_length=old_dataset.weight_vector_length,
                binary_labels=old_dataset.binary_labels,
                issue_keys=old_dataset.issue_keys
            )
            new_datasets.append(new_dataset)
        datasets = new_datasets
        labels = datasets[0].labels

    # 3) Define model factory

    def factory():
        models = []
        keras_models = []
        output_encoding = OutputMode.from_string(output_mode).output_encoding
        output_size = OutputMode.from_string(output_mode).output_size
        stream = zip(classifier, input_mode, datasets)
        model_counts = collections.defaultdict(int)
        for name, mode, data in stream:
            try:
                generator = feature_generators.generators[mode]
            except KeyError:
                raise ValueError(f'Unknown input mode: {mode}')
            input_encoding = generator.input_encoding_type()
            try:
                model_factory = classifiers.models[name]
            except KeyError:
                raise ValueError(f'Unknown classifier: {name}')
            if input_encoding not in model_factory.supported_input_encodings():
                raise ValueError(
                    f'Input encoding {input_encoding} not compatible with model {name}'
                )
            model: classifiers.AbstractModel = model_factory(data.shape,
                                                             input_encoding,
                                                             output_size,
                                                             output_encoding)
            models.append(model)
            number = model_counts[name]
            model_counts[name] += 1
            hyperparams = _normalize_param_names(
                hyper_parameters.get(name, {}) |
                hyper_parameters.get(f'{name}[{number}]', {}) |
                hyper_parameters.get('default', {})
            )
            allowed_hyper_params = model.get_hyper_parameters()
            for param_name in hyperparams:
                if param_name not in allowed_hyper_params:
                    raise ValueError(f'Illegal hyperparameter for model {name}: {param_name}')
            if data.is_embedding():
                keras_model = model.get_compiled_model(embedding=data.embedding_weights,
                                                       embedding_size=data.vocab_size,
                                                       embedding_output_size=data.weight_vector_length,
                                                       **hyperparams)
            else:
                keras_model = model.get_compiled_model(**hyperparams)
            keras_models.append(keras_model)
        # 4) If necessary, combine models
        if len(models) == 1:
            final_model = keras_models[0]
        elif conf.get('run.ensemble_strategy') not in ('stacking', 'voting') and not conf.get('run.test-separately'):
            final_model = classifiers.combine_models(
                models[0], *keras_models, fully_connected_layers=(None, None)
            )
        else:
            return keras_models  # Return all models separately, required for stacking or separate testing
        final_model.summary()
        return final_model

    return datasets, labels, factory


def _normalize_param_names(params):
    return {key.replace('_', '-'): value for key, value in params.items()}


##############################################################################
##############################################################################
# Command Dispatch - Prediction Command
##############################################################################


def run_prediction_command():
    # Step 1: Load model data
    data: pathlib.Path = conf.get('predict.data')
    model: pathlib.Path = conf.get('predict.model')
    with open(model / 'model.json') as file:
        model_metadata = json.load(file)
    output_mode = OutputMode.from_string(
        model_metadata['feature_settings']['make_features.output_mode']
    )

    # Step 2: Load data
    datasets = []
    warnings.warn('The predict command does not cache features!')
    for generator in model_metadata['feature_generators']:
        with open(model / generator) as file:
            generator_data = json.load(file)
        feature_file = pathlib.Path('prediction_features.json')
        generator_class = feature_generators.generators[generator_data['generator']]
        generator = generator_class(
            pretrained_generator_settings=generator_data['settings']
        )
        generator.generate_features(data, feature_file)
        features = data_manager.load_features(feature_file, output_mode.name).features
        datasets.append(features)

    # Step 3: Load the model and get the predictions
    match model_metadata['model_type']:
        case 'single':
            prediction.predict_simple_model(model, model_metadata, datasets, output_mode)
        case 'stacking':
            prediction.predict_stacking_model(model, model_metadata, datasets, output_mode)
        case 'voting':
            prediction.predict_voting_model(model, model_metadata, datasets, output_mode)
        case _ as tp:
            raise ValueError(f'Invalid model type: {tp}')
