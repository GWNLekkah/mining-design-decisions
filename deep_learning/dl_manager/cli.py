"""
Command line utility for managing and training
deep learning classifiers.
"""

##############################################################################
##############################################################################
# Imports
##############################################################################

import collections
import json
import os.path
import pathlib
import getpass
import random
import statistics
import typing
import warnings

import fastapi
import numpy

import issue_db_api

from . import classifiers, kw_analyzer, model_manager

from . import feature_generators
from .model_io import OutputMode
from . import data_manager
from . import embeddings

from . import learning
from .config import WebApp, Config
from .logger import get_logger
from . import metrics
log = get_logger('CLI')

from . import prediction


##############################################################################
##############################################################################
# Parser Setup
##############################################################################


def main(port, keyfile, certfile, script):
    app = build_app()
    if not script:
        app.deploy(port, keyfile,  certfile)
    else:
        app.execute_script(script)


def get_arg_spec():
    location = os.path.split(__file__)[0]
    return os.path.join(location, 'web_api.json')


def build_app():
    location = get_arg_spec()
    log.debug(f'Building app from file {location}')
    app = WebApp(location)
    setup_app_constraints(app)
    return app


def setup_app_constraints(app):
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
    app.add_constraint(lambda cross_project, k: k == 0 or not cross_project,
                       'Cannot use --k-cross and --cross-project at the same time.',
                       'run.cross-project', 'run.k-cross')
    app.add_constraint(
        lambda k, quick_cross: not quick_cross or k > 0,
        'Must specify k when running with --quick-cross',
        'run.k-cross', 'run.quick-cross'
    )
    app.add_constraint(
        lambda k, cross_project: k == 0 or not cross_project,
        'k-cross must be 0 when running with --cross-project',
        'run.k-cross', 'run.cross-project'
    )
    app.add_constraint(
        lambda do_save, model_id: (not do_save) or (do_save and model_id),
        '--model-id must be given when storing a model.',
        'run.store-model', 'run.model-id'
    )
    app.add_constraint(
        lambda do_save, cache_features: (not do_save) or (do_save and not cache_features),
        'May not use --cache-features when using --store-model.',
        'run.store-model', 'run.cache-features'
    )
    app.add_constraint(
        lambda do_save, k, cross_project, quick_cross:
            (not do_save)
            or (k == 0
                and not cross_project
                and not quick_cross),
        'Cannot run cross validation (or cross study) scheme when saving a model.',
        'run.store-model',
        'run.k-cross',
        'run.cross-project',
        'run.quick-cross'
    )
    app.add_constraint(
        lambda do_analyze, _, conf:
            not do_analyze or kw_analyzer.model_is_convolution(conf),
        'Can only analyze keywords when using a convolutional model',
        'run.analyze-keywords', 'run.classifier', '#config'
    )
    app.add_constraint(
        lambda do_analyze, conf: (not do_analyze) or kw_analyzer.doing_one_run(conf),
        'Can not perform cross validation when extracting keywords',
        'run.analyze-keywords', '#config'
    )
    app.add_constraint(
        lambda k_cross, test_with_training_data: k_cross == 0 or test_with_training_data,
        'Must test with training data when performing cross validation!',
        'run.k-cross', 'run.test-with-training-data'
    )
    app.add_constraint(
        lambda ontology_id, apply_ontology, models: True if 'OntologyFeatures' not in models and not apply_ontology else ontology_id != '',
        'Ontology class file must be given when applying ontology classes or using ontology features',
        'run.ontology-classes', 'run.apply-ontology-classes', 'run.classifier'
    )

    # Enforced using null-if
    # app.add_constraint(
    #     lambda test_query, test_with_train: (
    #         (not test_with_train) or test_query
    #     ),
    #     'Must either test with training data, or give a testing data query',
    #     'run.test-data-query', 'run.test-with-training-data'
    # )


    app.register_callback('predict', run_prediction_command)
    app.register_callback('run', run_classification_command)
    app.register_callback('train', run_training_session)
    app.register_callback('generate-embedding', run_embedding_generation_command)
    app.register_callback('generate-embedding-internal', run_embedding_generation_command_internal)
    app.register_callback('metrics', run_metrics_calculation_command)

    app.register_setup_callback(setup_security)
    app.register_setup_callback(setup_os)
    app.register_setup_callback(setup_storage)
    app.register_setup_callback(setup_resources)

    log.debug('Finished building app')
    return app


def setup_security(conf: Config):
    conf.set(
        'system.security.allow-self-signed-certificates',
        os.environ.get('DL_MANAGER_ALLOW_UNSAFE_SSL', 'false').lower() == 'true'
    )


def setup_os(conf: Config):
    conf.set(
        'system.os.peregrine',
        os.environ.get('DL_MANAGER_RUNNING_ON_PEREGRINE', 'false').lower() == 'true'
    )
    if conf.get('system.os.peregrine'):
        conf.set('system.os.home-directory', os.path.expanduser('~'))
        conf.set('system.os.data-directory', f'/projects/{getpass.getuser()}')
        conf.set('system.os.scratch-directory', f'/scratch/{getpass.getuser()}')
        # We assume these exist
    else:
        conf.set('system.os.home-directory', os.path.expanduser('~'))
        conf.set('system.os.data-directory', f'./data')
        conf.set('system.os.scratch-directory', f'./temp')
        os.makedirs(conf.get('system.os.data-directory'), exist_ok=True)
        os.makedirs(conf.get('system.os.scratch-directory'), exist_ok=True)


def setup_storage(conf: Config):
    # Storage space and constants
    conf.set('system.storage.generators', [])
    conf.set('system.storage.auxiliary', [])
    conf.set('system.storage.auxiliary_map', {})
    conf.set('system.storage.file_prefix', 'dl_manager')

    endpoints_with_database = [
        'run',
        'train',
        'predict',
        'generate-embedding',
        'generate-embedding-internal',
        'metrics'
    ]
    if (cmd := conf.get('system.management.active-command')) in endpoints_with_database:
        conf.clone(f'{cmd}.database-url', 'system.storage.database-url')
        log.info(f'Registered database url: {conf.get("system.storage.database-url")}')
        try:
            api = issue_db_api.IssueRepository.from_token(
                url=conf.get('system.storage.database-url'),
                token=conf.get('system.security.db-token'),
                allow_self_signed_certificates=conf.get('system.security.allow-self-signed-certificates'),
                label_caching_policy='use_local_after_load'
            )
        except issue_db_api.InvalidCredentialsException:
            raise fastapi.HTTPException(detail='Invalid credentials for database', status_code=401)
        conf.set('system.storage.database-api', api)

    if conf.get('system.management.active-command') == 'run':
        class IdentityDict(dict):
            def __missing__(self, key):
                return key

        conf.set('system.storage.auxiliary_map', IdentityDict())


def setup_resources(conf: Config):
    conf.set(
        'system.resources.threads',
        int(os.environ.get('DL_MANAGER_NUM_THREADS', '1'))
    )


##############################################################################
##############################################################################
# Command Dispatch - Training Commands
##############################################################################


def run_training_session(conf: Config):
    config_id = conf.get('train.model-id')
    db: issue_db_api.IssueRepository = conf.get('system.storage.database-api')
    settings = db.get_model_by_id(config_id).config
    settings |= {
        'database-url': conf.get('system.storage.database-url'),
        'model-id': conf.get('train.model-id')
    }
    app: WebApp = conf.get('system.management.app')
    new_conf = app.new_config('run', 'system')
    # conf.transfer(new_conf,
    #               'system.security.db-username',
    #               'system.security.db-password')
    conf.transfer(new_conf, 'system.security.db-token')
    return app.invoke_endpoint('run', new_conf, settings)


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

##############################################################################
##############################################################################
# Command Dispatch - Embedding Generation
#############################################################################


def run_embedding_generation_command_internal(conf: Config):
    embedding_config = conf.get('generate-embedding-internal.embedding-config')
    generator_name = conf.get('generate-embedding-internal.embedding-generator')
    generator: typing.Type[embeddings.AbstractEmbeddingGenerator] = embeddings.generators[
        generator_name
    ]
    query = conf.get('generate-embedding-internal.training-data-query')
    g = generator(**embedding_config[generator_name][0])
    g.make_embedding(query, conf=conf)


def run_embedding_generation_command(conf: Config):
    db: issue_db_api.IssueRepository = conf.get('system.storage.database-api')
    embedding_id = conf.get('generate-embedding.embedding-id')
    embedding = db.get_embedding_by_id(embedding_id)
    settings = embedding.config
    app: WebApp = conf.get('system.management.app')
    new_conf = app.new_config('generate-embedding-internal', 'system')
    conf.transfer(new_conf, 'system.security.db-token')
    payload = {
        'embedding-id': conf.get('generate-embedding.embedding-id'),
        'training-data-query': settings['training-data-query'],
        'embedding-config': settings['params'],
        'embedding-generator': settings['generator'],
        'database-url': conf.get('generate-embedding.database-url')
    }
    return app.invoke_endpoint('generate-embedding-internal', new_conf, payload)


##############################################################################
##############################################################################
# Feature Generation
##############################################################################


def generate_features_and_get_data(architectural_only: bool = False,
                                   force_regenerate: bool = False, *,
                                   conf: Config):
    input_mode = conf.get('run.input_mode')
    output_mode = conf.get('run.output_mode')
    params = conf.get('run.params')
    imode_counts = collections.defaultdict(int)
    datasets_train = []
    labels_train = None
    binary_labels_train = None
    datasets_test = []
    labels_test = None
    binary_labels_test = None
    for imode in input_mode:
        number = imode_counts[imode]
        imode_counts[imode] += 1
        # Get the parameters for the feature generator
        mode_params = params[imode][number]
        # Validate that the parameters are valid
        valid_params = feature_generators.generators[imode].get_arguments()
        for param_name in mode_params:
            if param_name not in valid_params:
                raise ValueError(f'Invalid parameter for feature generator {imode}: {param_name}')
        training_query = issue_db_api.Query().land(
            issue_db_api.Query().tag('has-label'),
            issue_db_api.Query().not_tag('needs-review'),
            conf.get('run.training-data-query')
        )
        generator = feature_generators.generators[imode](conf, **mode_params)
        dataset = generator.generate_features(training_query, output_mode)
        if labels_train is not None:
            assert labels_train == dataset.labels
            assert binary_labels_train == dataset.binary_labels
        else:
            labels_train = dataset.labels
            binary_labels_train = dataset.binary_labels
        datasets_train.append(dataset)
        if not conf.get('run.test-with-training-data'):
            testing_query = issue_db_api.Query().land(
                issue_db_api.Query().tag('has-label'),
                issue_db_api.Query().not_tag('needs-review'),
                conf.get('run.test-data-query')
            )
            # Load the most recently saved generator.
            # The auxiliary map is set to the IdentityMap
            # class, so everything should work.
            with open(conf.get('system.storage.generators')[-1]) as file:
                data = json.load(file)
            generator_class = feature_generators.generators[data['generator']]
            generator = generator_class(
                conf, pretrained_generator_settings=data['settings']
            )
            dataset = generator.generate_features(testing_query, output_mode)
            if labels_test is not None:
                assert labels_test == dataset.labels
                assert binary_labels_test == dataset.binary_labels
            else:
                labels_test = dataset.labels
                binary_labels_test = dataset.binary_labels
            datasets_test.append(dataset)

    if architectural_only:
        datasets_train, labels_train = select_architectural_only(datasets_train,
                                                                 labels_train,
                                                                 binary_labels_train)
        if not conf.get('run.test-with-training-data'):
            datasets_test, labels_test = select_architectural_only(datasets_test,
                                                                   labels_test,
                                                                   binary_labels_test)

    return (
        (datasets_train, labels_train),
        (datasets_test, labels_test)
    )


def select_architectural_only(datasets, labels, binary_labels):
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
            issue_keys=old_dataset.issue_keys,
            ids=old_dataset.ids
        )
        new_datasets.append(new_dataset)
    #datasets = new_datasets
    #labels = datasets[0].labels
    return new_datasets, new_datasets[0].labels


##############################################################################
##############################################################################
# Command Dispatch - run command
##############################################################################


def run_classification_command(conf: Config):
    if (seed := conf.get('run.seed')) != -1:
        random.seed(seed)
        numpy.random.seed(seed)
        import tensorflow
        tensorflow.random.set_seed(seed)

    datasets_train, labels_train, datasets_test, labels_test, factory = _get_model_factory(conf)

    training_data = (
        [ds.features for ds in datasets_train],
        labels_train,
        datasets_train[0].issue_keys,
        datasets_train[0].ids
    )
    if datasets_test:
        testing_data = (
            [ds.features for ds in datasets_test],
            labels_test,
            datasets_test[0].issue_keys,
            datasets_test[0].ids
        )
    else:
        testing_data = None

    if conf.get('run.ensemble-strategy') != 'none':
        if conf.get('run.k-cross') != 0 or conf.get('run.cross-project'):
            assert testing_data is None, 'testing_data should be None'
        version, performances, kw_files = learning.run_ensemble(factory,
                                                                training_data,
                                                                testing_data,
                                                                OutputMode.from_string(
                                                                    conf.get('run.output-mode')).label_encoding,
                                                                conf=conf)
        return {'version-id': version, 'run-ids': performances, 'keyword-ids': kw_files}

    # 5) Invoke actual DL process
    if conf.get('run.k-cross') == 0 and not conf.get('run.cross-project'):
        version, performances, kw_files = learning.run_single(factory(),
                                                              conf.get('run.epochs'),
                                                              OutputMode.from_string(conf.get('run.output-mode')),
                                                              OutputMode.from_string(
                                                                  conf.get('run.output-mode')).label_encoding,
                                                              training_data,
                                                              testing_data,
                                                              conf=conf)
    else:
        assert testing_data is None, 'testing_data should be None'
        version, performances, kw_files = learning.run_cross(factory,
                                                             conf.get('run.epochs'),
                                                             OutputMode.from_string(conf.get('run.output-mode')),
                                                             OutputMode.from_string(
                                                                 conf.get('run.output-mode')).label_encoding,
                                                             training_data,
                                                             testing_data,
                                                             conf=conf)
    return {'version-id': version, 'run-ids': performances, 'keyword-ids': kw_files}

def _get_model_factory(conf: Config):
    ((datasets, labels), (datasets_test, labels_test)) = generate_features_and_get_data(
        conf.get('run.architectural-only'),
        not conf.get('run.cache-features'),
        conf=conf
    )

    # 3) Define model factory

    def factory():
        models = []
        keras_models = []
        output_encoding = OutputMode.from_string(conf.get('run.output-mode')).output_encoding
        output_size = OutputMode.from_string(conf.get('run.output-mode')).output_size
        stream = zip(conf.get('run.classifier'), conf.get('run.input-mode'), datasets)
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
            model_number = model_counts[name]
            model_counts[name] += 1
            hyper_parameters = conf.get('run.hyper-params')
            hyperparams = hyper_parameters[name][model_number]
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
        elif conf.get('run.ensemble-strategy') not in ('stacking', 'voting') and not conf.get('run.test-separately'):
            final_model = classifiers.combine_models(
                models[0], *keras_models, fully_connected_layers=(None, None), conf=conf
            )
        else:
            return keras_models  # Return all models separately, required for stacking or separate testing
        final_model.summary()
        return final_model

    return datasets, labels, datasets_test, labels_test, factory


##############################################################################
##############################################################################
# Command Dispatch - Prediction Command
##############################################################################


def run_prediction_command(conf: Config):
    # Step 1: Load model data
    data_query = conf.get('predict.data-query')
    model_id: str = conf.get('predict.model')
    model_version = conf.get('predict.version')
    # Load model from DB
    db: issue_db_api.IssueRepository = conf.get('system.storage.database-api')
    model = db.get_model_by_id(model_id)
    if model_version == 'most-recent':
        trained_model = max(model.versions, key=lambda v: v.version_id)
        model_version = trained_model.version_id
    else:
        trained_model = model.get_version_by_id(model_version)
    trained_model.download(
        os.path.join(conf.get('system.os.scratch-directory'), model_manager.MODEL_FILE)
    )
    model_manager.load_model_from_zip(model_manager.MODEL_FILE, conf)
    # Load model from file
    model = pathlib.Path(os.path.join(conf.get('system.os.scratch-directory'))) / model_manager.MODEL_DIR
    with open(model / 'model.json') as file:
        model_metadata = json.load(file)
    output_mode = OutputMode.from_string(
        model_metadata['model-settings']['output_mode']
    )

    # Step 2: Load data
    datasets = []
    warnings.warn('The predict command does not cache features!')
    auxiliary_files = {
        file: os.path.join(model, path)
        for file, path in model_metadata['auxiliary-files'].items()
    }
    conf.get('system.storage.auxiliary-map').update(auxiliary_files)
    ids = None
    for generator in model_metadata['feature-generators']:
        with open(model / generator) as file:
            generator_data = json.load(file)
        generator_class = feature_generators.generators[generator_data['generator']]
        generator = generator_class(
            conf, pretrained_generator_settings=generator_data['settings']
        )
        data_stuff = generator.generate_features(data_query, output_mode.name)
        if ids is None:
            ids = data_stuff.ids
        if type(data_stuff.features) is dict:
            datasets.append(data_stuff.features)
        else:
            datasets.append(numpy.asarray(data_stuff.features))

    # Step 3: Load the model and get the predictions
    match model_metadata['model-type']:
        case 'single':
            prediction.predict_simple_model(model,
                                            model_metadata,
                                            datasets,
                                            output_mode,
                                            ids,
                                            model_id,
                                            model_version,
                                            conf=conf)
        case 'stacking':
            prediction.predict_stacking_model(model,
                                              model_metadata,
                                              datasets,
                                              output_mode,
                                              ids,
                                              model_id,
                                              model_version,
                                              conf=conf)
        case 'voting':
            prediction.predict_voting_model(model,
                                            model_metadata,
                                            datasets,
                                            output_mode,
                                            ids,
                                            model_id,
                                            model_version,
                                            conf=conf)
        case _ as tp:
            raise ValueError(f'Invalid model type: {tp}')


##############################################################################
##############################################################################
# Command Dispatch - Metric Calculation
##############################################################################


def run_metrics_calculation_command(conf: Config):
    db: issue_db_api.IssueRepository = conf.get('system.storage.database-api')
    model_id = conf.get('metrics.model-id')
    model = db.get_model_by_id(model_id)
    model_config = model.config
    version_id = conf.get('metrics.version-id')
    metric_settings = conf.get('metrics.metrics')
    if isinstance(metric_settings, str):
        metric_settings = json.loads(metric_settings.replace("'", '"'))
    results = model.get_run_by_id(version_id).data
    results_per_fold = []
    for fold in results:
        match conf.get('metrics.epoch'):
            case 'last':
                epoch = -1
                results_per_fold.append([
                    _calculate_metrics(metric_settings, fold, epoch, model_config, conf=conf)
                ])
            case 'stopping-point':
                es_settings = fold['early_stopping_settings']
                if es_settings['use_early_stopping']:
                    if es_settings['stopped_early']:
                        epoch = -1 - es_settings['patience']
                    else:
                        epoch = -1
                else:
                    epoch = -1
                results_per_fold.append([
                    _calculate_metrics(metric_settings, fold, epoch, model_config, conf=conf)
                ])
            case 'all':
                results_per_fold.append(
                    [
                        _calculate_metrics(metric_settings, fold, e, model_config, conf=conf)
                        for e in range(len(fold['predictions']['training']))
                    ]
                )
            case _ as x:
                epoch = int(x) - 1
                results_per_fold.append([
                    _calculate_metrics(metric_settings, fold, epoch, model_config, conf=conf)
                ])
    result = {
        'folds': results_per_fold,
    }
    if conf.get('metrics.epoch') != 'all':
        result['aggregated'] = _compute_aggregate_metrics(metric_settings,
                                                          [fold[0] for fold in results_per_fold])
    return result


def _compute_aggregate_metrics(metric_settings, results_per_fold):
    result = {'training': {}, 'validation': {}, 'testing': {}}
    for metric in metric_settings:
        mode = metric['dataset']
        metric_name = metric['metric']
        variant = metric['variant']
        key = f'{metric_name}[{variant}]'
        if variant != 'class':
            result[mode][key] = {
                'mean': statistics.mean(fold[mode][key] for fold in results_per_fold),
                'std': statistics.stdev(fold[mode][key] for fold in results_per_fold) if len(results_per_fold) > 1 else None
            }
        else:
            result[mode][key] = {
                cls: {
                    'mean': statistics.mean(fold[mode][key][cls] for fold in results_per_fold),
                    'std': statistics.stdev(fold[mode][key][cls] for fold in results_per_fold) if len(results_per_fold) > 1 else None
                }
                for cls in results_per_fold[0][mode][key].keys()
            }
    return result


def _calculate_metrics(metric_settings, results, epoch, model_config, *, conf: Config, get_confusion=False, key='metrics'):
    training_manager = metrics.MetricCalculationManager(
        y_true=numpy.asarray(results['truth']['training']),
        y_pred=numpy.asarray(results['predictions']['training'][epoch]),
        output_mode=OutputMode.from_string(
            model_config['output_mode'] if 'output_mode' in model_config else model_config['output-mode']
        ),
        classification_as_detection=conf.get(f'{key}.classification-as-detection'),
        include_non_arch=conf.get(f'{key}.include-non-arch')
    )
    validation_manager = metrics.MetricCalculationManager(
        y_true=numpy.asarray(results['truth']['validation']),
        y_pred=numpy.asarray(results['predictions']['validation'][epoch]),
        output_mode=OutputMode.from_string(
            model_config['output_mode'] if 'output_mode' in model_config else model_config['output-mode']
        ),
        classification_as_detection=conf.get(f'{key}.classification-as-detection'),
        include_non_arch=conf.get(f'{key}.include-non-arch')
    )
    testing_manager = metrics.MetricCalculationManager(
        y_true=numpy.asarray(results['truth']['testing']),
        y_pred=numpy.asarray(results['predictions']['testing'][epoch]),
        output_mode=OutputMode.from_string(
            model_config['output_mode'] if 'output_mode' in model_config else model_config['output-mode']
        ),
        classification_as_detection=conf.get(f'{key}.classification-as-detection'),
        include_non_arch=conf.get(f'{key}.include-non-arch')
    )
    if get_confusion:
        return {
            'training': training_manager.get_raw_confusion_matrix(),
            'validation': validation_manager.get_raw_confusion_matrix(),
            'testing': testing_manager.get_raw_confusion_matrix(),
        }
    managers = {
        'training': training_manager,
        'validation': validation_manager,
        'testing': testing_manager
    }
    result = {'training': {}, 'validation': {}, 'testing': {}}
    for metric in metric_settings:
        mode = metric['dataset']
        metric_name = metric['metric']
        variant = metric['variant']
        if mode not in result:
            raise ValueError(f'Invalid mode: {mode}')
        if metric_name != 'loss':
            result[mode][f'{metric_name}[{variant}]'] = managers[mode].calc_metric(metric_name, variant)
        else:
            result[mode][f'loss[{variant}]'] = results['loss'][mode][epoch]
    return result


##############################################################################
##############################################################################
# Command Dispatch - Confusion Matrix Calculation
##############################################################################


def compute_confusion_matrix(conf: Config):
    db: issue_db_api.IssueRepository = conf.get('system.storage.database-api')
    model_id = conf.get('confusion-matrix.model-id')
    model = db.get_model_by_id(model_id)
    model_config = model.config
    version_id = conf.get('confusion-matrix.version-id')
    results = model.get_run_by_id(version_id).data
    total_results = []
    for fold in results:
        fold_result = []
        total_results.append(fold_result)
        match conf.get('confusion-matrix.epoch'):
            case 'last':
                epoch = -1
                fold_result.append([
                    _calculate_metrics({}, fold, epoch, model_config, conf=conf, get_confusion =True, key='confusion-matrix')
                ])
            case 'stopping-point':
                es_settings = fold['early_stopping_settings']
                if es_settings['use_early_stopping']:
                    if es_settings['stopped_early']:
                        epoch = -1 - es_settings['patience']
                    else:
                        epoch = -1
                else:
                    epoch = -1
                fold_result.append([
                    _calculate_metrics({}, fold, epoch, model_config, conf=conf, get_confusion=True, key='confusion-matrix')
                ])
            case 'all':
                fold_result.append([
                    _calculate_metrics({}, fold, e, model_config, conf=conf, get_confusion=True, key='confusion-matrix')
                    for e in range(len(fold['predictions']['training']))
                ])
            case _ as x:
                epoch = int(x) - 1
                fold_result.append([
                    _calculate_metrics({}, fold, epoch, model_config, conf=conf, get_confusion=True, key='confusion-matrix')
                ])
    return total_results