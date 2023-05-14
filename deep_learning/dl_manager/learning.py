"""
This code contains the core of the training algorithms.

Yes, it is a mess... but a relatively easy-to-oversee mess,
and it works.
"""
import collections
##############################################################################
##############################################################################
# Imports
##############################################################################

import gc
import random
import statistics
import warnings

import numpy
import keras.callbacks

import tensorflow as tf
import numpy as np

from collections import Counter

import issue_db_api

from .model_io import OutputMode

from . config import Config
from . import stacking
from .metrics.metric_logger import PredictionLogger
from . import metrics
from . import data_splitting as splitting
from . import model_manager
from . import voting_util
from . import run_identifiers
from . import upsampling


EARLY_STOPPING_GOALS = {
    'loss': 'min',
    'accuracy': 'max',
    'precision': 'max',
    'recall': 'max',
    'f_score_tf': 'max',
    'true_positives': 'max',
    'true_negatives': 'max',
    'false_positives': 'min',
    'false_negatives': 'min'
}


##############################################################################
##############################################################################
# Model Training/Testing
##############################################################################


def _coerce_none(x: str) -> str | None:
    match x:
        case 'None':
            return None
        case 'none':
            return None
        case _:
            return x


def run_single(model_or_models,
               epochs: int,
               output_mode: OutputMode,
               label_mapping: dict,
               training_data,
               testing_data, *,
               conf: Config):
    id_generator = run_identifiers.IdentifierFactory()
    max_train = conf.get('run.max-train')
    if max_train > 0:
        warnings.warn('The --max-train parameter is ignored in single runs.')
    spitter = splitting.SimpleSplitter(conf,
                                       val_split_size=conf.get('run.split-size'),
                                       test_split_size=conf.get('run.split-size'),
                                       max_train=conf.get('run.max-train'))
    # Split returns an iterator; call next() to get data splits
    train, test, validation, train_keys, val_keys, test_issue_keys = next(spitter.split(training_data, testing_data))
    comparator = metrics.ComparisonManager()
    if not conf.get('run.test-separately'):
        models = [model_or_models]
        inputs = [(train, test, validation)]
    else:
        models = model_or_models
        inputs = _separate_datasets(train, test, validation)
    ident = None
    version_id = None
    for model, (m_train, m_test, m_val) in zip(models, inputs):
        trained_model, metrics_, best = train_and_test_model(model,
                                                             m_train,
                                                             m_val,
                                                             m_test,
                                                             epochs,
                                                             output_mode,
                                                             label_mapping,
                                                             test_issue_keys,
                                                             training_keys=train_keys,
                                                             validation_keys=val_keys,
                                                             conf=conf)
        # Save model can only be true if not testing separately,
        # which means the loop only runs once.
        if conf.get('run.store-model'):
            version_id = model_manager.save_single_model(trained_model, conf)
        ident = dump_metrics([metrics_],
                             conf=conf,
                             description=id_generator.generate_id('single-training-run'))
        comparator.add_result(metrics_)
    assert ident is not None
    comparator.add_truth(test[1])
    comparator.finalize()
    if conf.get('run.test-separately'):
        comparator.compare()
    return version_id, [ident]


def run_cross(model_factory,
              epochs: int,
              output_mode: OutputMode,
              label_mapping: dict,
              training_data,
              testing_data, *,
              conf: Config):
    results = []
    best_results = []
    id_generator = run_identifiers.IdentifierFactory()
    if conf.get('run.quick-cross'):
        splitter = splitting.QuickCrossFoldSplitter(
            conf,
            k=conf.get('run.k-cross'),
            max_train=conf.get('run.max-train'),
        )
    elif conf.get('run.cross-project'):
        splitter = splitting.CrossProjectSplitter(
            conf,
            val_split_size=conf.get('run.split-size'),
            max_train=conf.get('run.max-train'),
        )
    else:
        splitter = splitting.CrossFoldSplitter(
            conf,
            k=conf.get('run.k-cross'),
            max_train=conf.get('run.max-train'),
        )
    comparator = metrics.ComparisonManager()
    stream = splitter.split(training_data, testing_data)
    for train, test, validation, training_keys, validation_keys, test_issue_keys in stream:
        model_or_models = model_factory()
        if conf.get('run.test-separately'):
            models = model_or_models
            inputs = _separate_datasets(train, test, validation)
        else:
            models = [model_or_models]
            inputs = [(train, test, validation)]
        for model, (m_train, m_test, m_val) in zip(models, inputs):
            _, metrics_, best_metrics = train_and_test_model(model,
                                                             m_train,
                                                             m_val,
                                                             m_test,
                                                             epochs,
                                                             output_mode,
                                                             label_mapping,
                                                             test_issue_keys,
                                                             training_keys=training_keys,
                                                             validation_keys=validation_keys,
                                                             conf=conf)
            results.append(metrics_)
            best_results.append(best_metrics)
            comparator.add_result(metrics_)
        comparator.add_truth(test[1])
        # Force-free memory for Linux
        del train
        del validation
        del test
        gc.collect()
        comparator.mark_end_of_fold()
    comparator.finalize()
    ident = print_and_save_k_cross_results(results,
                                           best_results,
                                           conf=conf,
                                           description=id_generator.generate_id('kfold-run'))
    if conf.get('run.test-separately'):
        comparator.compare()
    return None, [ident]


def _separate_datasets(train, test, validation):
    train_x, train_y = train
    test_x, test_y = test
    val_x, val_y = validation
    return [
        ([train_x_part, train_y], [test_x_part, test_y], [val_x_part, val_y])
        for train_x_part, test_x_part, val_x_part in zip(train_x, test_x, val_x)
    ]


def print_and_save_k_cross_results(results,
                                   best_results, *,
                                   conf: Config,
                                   description: str):
    ident = dump_metrics(results, conf=conf, description=description)
    metric_list = []
    # metric_list = ['accuracy', 'f-score']
    for key in metric_list:
        stat_data = [metrics_[key] for metrics_ in best_results]
        print('-' * 72)
        print(key.capitalize())
        print('    * Mean:', statistics.mean(stat_data))
        try:
            print('    * Geometric Mean:', statistics.geometric_mean(stat_data))
        except statistics.StatisticsError:
            pass
        try:
            print('    * Standard Deviation:', statistics.stdev(stat_data))
        except statistics.StatisticsError:
            pass
        print('    * Median:', statistics.median(stat_data))
    return ident


def train_and_test_model(model: tf.keras.Model,
                         dataset_train,
                         dataset_val,
                         dataset_test,
                         epochs,
                         output_mode: OutputMode,
                         label_mapping,
                         test_issue_keys,
                         extra_model_params=None,
                         *,
                         validation_keys=None,
                         training_keys=None,
                         conf: Config):
    train_x, train_y = dataset_train
    test_x, test_y = dataset_test

    if extra_model_params is None:
        extra_model_params = {}

    class_weight = None
    class_balancer = conf.get('run.class-balancer')
    if class_balancer == 'class-weights':
        _, val_y = dataset_val
        labels = []
        labels.extend(train_y)
        labels.extend(test_y)
        labels.extend(val_y)
        if type(labels[0]) is numpy.ndarray:
            counts = Counter([np.argmax(y, axis=0) for y in labels])
        else:
            counts = Counter(labels)
        class_weight = dict()
        for key, value in counts.items():
            class_weight[key] = (1 / value) * (len(labels) / 2.0)
    elif class_balancer == 'upsample':
        train_y, training_keys, train_x = upsampling.upsample(
            conf,
            conf.get('run.upsampler'),
            train_y,
            training_keys,
            *train_x
        )

    callbacks = []

    if conf.get('run.use-early-stopping'):
        attributes = conf.get('run.early-stopping-attribute')
        min_deltas = conf.get('run.early-stopping-min-delta')
        for attribute, min_delta in zip(attributes, min_deltas):
            monitor = keras.callbacks.EarlyStopping(
                monitor=f'val_{attribute}',
                patience=conf.get('run.early-stopping-patience'),
                min_delta=min_delta,
                restore_best_weights=True,
                mode=EARLY_STOPPING_GOALS[attribute]
            )
            callbacks.append(monitor)
        epochs = 1000   # Just a large amount of epochs
        import warnings
        warnings.warn('--epochs is ignored when using early stopping')
        conf.set('run.epochs', 1000)
    #print('Training data shape:', train_y.shape, train_x.shape)

    logger = PredictionLogger(
        model=model,
        training_data=(train_x, train_y),
        validation_data=dataset_val,
        testing_data=(test_x, test_y),
        label_mapping=output_mode.label_encoding,
        max_epochs=epochs,
        use_early_stopping=conf.get('run.use-early-stopping'),
        early_stopping_attributes=conf.get('run.early-stopping-attribute'),
        early_stopping_min_deltas=conf.get('run.early-stopping-min-delta'),
        early_stopping_patience=conf.get('run.early-stopping-patience')
    )
    callbacks.append(logger)

    model.fit(x=train_x, y=train_y,
              batch_size=conf.get('run.batch_size'),
              epochs=epochs if epochs > 0 else 1,
              shuffle=True,
              validation_data=dataset_val,
              callbacks=callbacks,
              verbose=1,    # Less  console spam
              class_weight=class_weight,
              **extra_model_params)

    from . import kw_analyzer
    if kw_analyzer.model_is_convolution(conf) and kw_analyzer.doing_one_run(conf) and kw_analyzer.enabled(conf):
        print('Analyzing keywords', logger.get_main_model_metrics_at_stopping_epoch())
        kw_analyzer.analyze_keywords(model,
                                     test_x,
                                     test_y,
                                     test_issue_keys,
                                     'test',
                                     conf)
        kw_analyzer.analyze_keywords(model,
                                     dataset_val[0],
                                     dataset_test[1],
                                     validation_keys,
                                     'validation',
                                     conf)
        kw_analyzer.analyze_keywords(model,
                                     train_x,
                                     train_y,
                                     training_keys,
                                     'train',
                                     conf)


    # logger.rollback_model_results(monitor.get_best_model_offset())
    return (
        model,
        logger.get_model_results_for_all_epochs(),
        logger.get_main_model_metrics_at_stopping_epoch()
    )


def dump_metrics(runs, *, conf: Config, description: str):
    db: issue_db_api.IssueRepository = conf.get('system.storage.database-api')
    model = db.get_model_by_id(conf.get('run.model-id'))
    return model.add_test_run(runs, description=description).run_id

##############################################################################
##############################################################################
# Ensemble learning
##############################################################################


def run_ensemble(factory,
                 training_data,
                 testing_data,
                 label_mapping, *,
                 conf: Config):
    match (strategy := conf.get('run.ensemble-strategy')):
        case 'stacking':
            return run_stacking_ensemble(factory,
                                         training_data,
                                         testing_data,
                                         label_mapping,
                                         conf=conf)
        case 'voting':
            return run_voting_ensemble(factory,
                                       training_data,
                                       testing_data,
                                       label_mapping,
                                       conf=conf)
        case _:
            raise ValueError(f'Unknown ensemble mode {strategy}')


def run_stacking_ensemble(factory,
                          training_data,
                          testing_data,
                          label_mapping, *,
                          __voting_ensemble_hook=None,
                          conf: Config):
    if conf.get('run.k-cross') > 0 and not conf.get('run.quick_cross'):
        warnings.warn('Absence of --quick-cross is ignored when running with stacking')
    if __voting_ensemble_hook is not None:
        ensemble_mode = 'voting'
    else:
        ensemble_mode = 'stacking'
    id_generator = run_identifiers.IdentifierFactory()

    # stream = split_data_quick_cross(conf.get('run.k-cross'),
    #                                 labels,
    #                                 *datasets,
    #                                 issue_keys=issue_keys,
    #                                 max_train=conf.get('run.max-train'))
    if conf.get('run.k-cross') > 0:
        splitter = splitting.QuickCrossFoldSplitter(
            conf,
            k=conf.get('run.k-cross'),
            max_train=conf.get('run.max-train'),
        )
    elif conf.get('run.cross-project'):
        splitter = splitting.CrossProjectSplitter(
            conf,
            val_split_size=conf.get('run.split-size'),
            max_train=conf.get('run.max-train'),
        )
    else:
        splitter = splitting.SimpleSplitter(
            conf,
            val_split_size=conf.get('run.split-size'),
            test_split_size=conf.get('run.split-size'),
            max_train=conf.get('run.max-train'),
        )
    if __voting_ensemble_hook is None:
        meta_factory, input_conversion_method = stacking.build_stacking_classifier(conf)
    else:
        meta_factory, input_conversion_method = None, False
    number_of_models = len(conf.get('run.classifier'))
    sub_results = [[] for _ in range(number_of_models)]
    best_sub_results = [[] for _ in range(number_of_models)]
    results = []
    best_results = []
    voting_result_data = []
    stream = splitter.split(training_data, testing_data)
    version_id = None
    for train, test, validation, training_keys, validation_keys, test_issue_keys in stream:
        # Step 1) Train all models and get their predictions
        #           on the training and validation set.
        models = factory()
        predictions_train = []
        predictions_val = []
        predictions_test = []
        model_number = 0
        trained_sub_models = []
        for model, model_train, model_test, model_validation in zip(models, train[0], test[0], validation[0], strict=True):
            trained_sub_model, sub_model_results, best_sub_model_results = train_and_test_model(
                model,
                dataset_train=(model_train, train[1]),
                dataset_val=(model_validation, validation[1]),
                dataset_test=(model_test, test[1]),
                epochs=conf.get('run.epochs'),
                output_mode=OutputMode.from_string(conf.get('run.output-mode')),
                label_mapping=label_mapping,
                test_issue_keys=test_issue_keys,
                training_keys=training_keys,
                validation_keys=validation_keys,
                conf=conf
            )
            sub_results[model_number].append(sub_model_results)
            best_sub_results[model_number].append(best_sub_model_results)
            model_number += 1
            predictions_train.append(model.predict(model_train))
            predictions_val.append(model.predict(model_validation))
            predictions_test.append(model.predict(model_test))
            if conf.get('run.store-model'):
                trained_sub_models.append(trained_sub_model)
        if __voting_ensemble_hook is None:
            # Step 2) Generate new feature vectors from the predictions
            train_features = stacking.transform_predictions_to_stacking_input(OutputMode.from_string(conf.get('run.output-mode')),
                                                                              predictions_train,
                                                                              input_conversion_method)
            val_features = stacking.transform_predictions_to_stacking_input(OutputMode.from_string(conf.get('run.output-mode')),
                                                                            predictions_val,
                                                                            input_conversion_method)
            test_features = stacking.transform_predictions_to_stacking_input(OutputMode.from_string(conf.get('run.output-mode')),
                                                                             predictions_test,
                                                                             input_conversion_method)
            # Step 3) Train and test the meta-classifier.
            meta_model = meta_factory()
            epoch_model, epoch_results, best_epoch_results = train_and_test_model(
                meta_model,
                dataset_train=(train_features, train[1]),
                dataset_val=(val_features, validation[1]),
                dataset_test=(test_features, test[1]),
                epochs=conf.get('run.epochs'),
                output_mode=OutputMode.from_string(
                    conf.get('run.output-mode')),
                label_mapping=label_mapping,
                test_issue_keys=test_issue_keys,
                training_keys=training_keys,
                validation_keys=validation_keys,
                conf=conf
            )
            results.append(epoch_results)
            best_results.append(best_epoch_results)

            if conf.get('run.store-model'):     # only ran in single-shot mode
                version_id = model_manager.save_stacking_model(
                    input_conversion_method.to_json(),
                    epoch_model,
                    *trained_sub_models,
                    conf=conf
                )

        else:   # We're being used by the voting ensemble
            voting_result = {
                'classes': [
                    [(list(k) if isinstance(k, tuple) else k), v]
                    for k, v in label_mapping.items()
                ],
                'loss': None,
                'predictions': collections.defaultdict(list),
                'truth': {
                    'training': __voting_ensemble_hook[0](train[1], predictions_train, conf=conf),
                    'validation': __voting_ensemble_hook[0](validation[1], predictions_val, conf=conf),
                    'testing': __voting_ensemble_hook[0](test[1], predictions_test, conf=conf)
                },
                # Voting does not use early stopping, so set to defaults.
                'early_stopping_settings': {
                    'use_early_stopping': False,
                    'attributes': [],
                    'min_deltas': [],
                    'patience': -1,
                    'stopped_early': False,
                    'early_stopping_epoch': -1
                }
            }
            voting_result_data.append(voting_result)

            if conf.get('run.store-model'):
                version_id = model_manager.save_voting_model(
                    *trained_sub_models,
                    conf=conf
                )

    results_ids = []
    it = enumerate(zip(sub_results, best_sub_results))
    for model_number, (sub_model_results, best_sub_model_results) in it:
        print(f'Model {model_number} results:')
        ident = print_and_save_k_cross_results(
            sub_model_results,
            best_sub_model_results,
            conf=conf,
            description=id_generator.generate_id(f'{ensemble_mode}-sub-model-{model_number}'))
        results_ids.append(ident)
        print('=' * 72)
        print('=' * 72)
    if __voting_ensemble_hook is None:
        print('Total Stacking Ensemble Results:')
        ident = print_and_save_k_cross_results(
            results,
            best_results,
            conf=conf,
            description=id_generator.generate_id(f'{ensemble_mode}-final-model')
        )
        results_ids.append(ident)
    else:   # Voting ensemble
        ident = __voting_ensemble_hook[1](
            voting_result_data,
            description=id_generator.generate_id(f'{ensemble_mode}-final-model'),
            conf=conf
        )
        results_ids.append(ident)
    return version_id, results_ids


def run_voting_ensemble(factory,
                        training_data,
                        testing_data,
                        label_mapping, *,
                        conf :Config):
   return run_stacking_ensemble(
       factory, training_data, testing_data, label_mapping,
       __voting_ensemble_hook=(_get_voting_predictions, _save_voting_data),
       conf=conf
   )
    

def _save_voting_data(data, description, *, conf: Config):
    db: issue_db_api.IssueRepository = conf.get('system.storage.database-api')
    model = db.get_model_by_id(conf.get('run.model-id'))
    return model.add_test_run(data, description=description).run_id


def _get_voting_predictions(truth, predictions, *, conf: Config):
    output_mode = OutputMode.from_string(conf.get('run.output-mode'))
    final_predictions = voting_util.get_voting_predictions(output_mode,
                                                           predictions,
                                                           conf.get('run.voting-mode'))
    return final_predictions
