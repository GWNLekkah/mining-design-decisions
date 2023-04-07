"""
This code contains the core of the training algorithms.

Yes, it is a mess... but a relatively easy-to-oversee mess,
and it works.
"""

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

from .database import DatabaseAPI
from .model_io import OutputMode

from . config import conf
from . import stacking
from .metrics.metric_logger import PredictionLogger
from . import metrics
from . import data_splitting as splitting
from . import model_manager
from . import voting_util


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
               testing_data):
    max_train = conf.get('run.max-train')
    if max_train > 0:
        warnings.warn('The --max-train parameter is ignored in single runs.')
    spitter = splitting.SimpleSplitter(val_split_size=conf.get('run.split-size'),
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
                                                             validation_keys=val_keys)
        # Save model can only be true if not testing separately,
        # which means the loop only runs once.
        if conf.get('run.store-model'):
            model_manager.save_single_model(trained_model)
        dump_metrics([metrics_])
        comparator.add_result(metrics_)
    comparator.add_truth(test[1])
    comparator.finalize()
    if conf.get('run.test-separately'):
        comparator.compare()


def run_cross(model_factory,
              epochs: int,
              output_mode: OutputMode,
              label_mapping: dict,
              training_data,
              testing_data):
    results = []
    best_results = []
    # if quick_cross:
    #     stream = split_data_quick_cross(k,
    #                                     labels,
    #                                     *features,
    #                                     issue_keys=issue_keys,
    #                                     max_train=max_train)
    # else:
    #     stream = split_data_cross(k, labels, *features, issue_keys=issue_keys)
    if conf.get('run.quick-cross'):
        splitter = splitting.QuickCrossFoldSplitter(
            k=conf.get('run.k-cross'),
            max_train=conf.get('run.max-train'),
        )
    elif conf.get('run.cross-project'):
        splitter = splitting.CrossProjectSplitter(
            val_split_size=conf.get('run.split-size'),
            max_train=conf.get('run.max-train'),
        )
    else:
        splitter = splitting.CrossFoldSplitter(
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
                                                             validation_keys=validation_keys)
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
    print_and_save_k_cross_results(results, best_results)
    if conf.get('run.test-separately'):
        comparator.compare()


def _separate_datasets(train, test, validation):
    train_x, train_y = train
    test_x, test_y = test
    val_x, val_y = validation
    return [
        ([train_x_part, train_y], [test_x_part, test_y], [val_x_part, val_y])
        for train_x_part, test_x_part, val_x_part in zip(train_x, test_x, val_x)
    ]


def print_and_save_k_cross_results(results, best_results, filename_hint=None):
    dump_metrics(results, filename_hint)
    output_mode = OutputMode.from_string(conf.get('run.output-mode'))
    if conf.get('run.include-detection-performance') and output_mode != OutputMode.Detection:
        metric_list = ['accuracy', 'f-score', 'classification-as-detection-accuracy', 'classification-as-detection-f-score']
    else:
        metric_list = ['accuracy', 'f-score']
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
                         training_keys=None):
    train_x, train_y = dataset_train
    test_x, test_y = dataset_test

    if extra_model_params is None:
        extra_model_params = {}

    class_weight = None
    class_balancer = conf.get('run.class-balancer')
    if class_balancer == 'class-weight':
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
        train_x, train_y = upsample(train_x, train_y)
        val_x, val_y = dataset_val
        val_x, val_y = upsample(val_x, val_y)
        dataset_val = splitting.make_dataset(val_y, val_x)

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
              verbose=2,    # Less  console spam
              class_weight=class_weight,
              **extra_model_params)

    from . import kw_analyzer
    if kw_analyzer.model_is_convolution() and kw_analyzer.doing_one_run() and kw_analyzer.enabled():
        print('Analyzing keywords', logger.get_main_model_metrics_at_stopping_epoch())
        kw_analyzer.analyze_keywords(model,
                                     test_x,
                                     test_y,
                                     test_issue_keys,
                                     'test')
        kw_analyzer.analyze_keywords(model,
                                     dataset_val[0],
                                     dataset_test[1],
                                     validation_keys,
                                     'validation')
        kw_analyzer.analyze_keywords(model,
                                     train_x,
                                     train_y,
                                     training_keys,
                                     'train')


    # logger.rollback_model_results(monitor.get_best_model_offset())
    return (
        model,
        logger.get_model_results_for_all_epochs(),
        logger.get_main_model_metrics_at_stopping_epoch()
    )


def upsample(features, labels):
    counts = Counter([np.argmax(label, axis=0) for label in labels])
    upper = max(counts.values())
    for key, value in counts.items():
        indices = [idx for idx, label in enumerate(labels) if np.argmax(label, axis=0) == key]
        new_samples = random.choices(indices, k=(upper - len(indices)))
        features = numpy.concatenate([features, features[new_samples]])
        labels = numpy.concatenate([labels, labels[new_samples]])
    return features, labels


def dump_metrics(runs, filename_hint=None):
    # if conf.get('system.peregrine'):
    #     data = pathlib.Path(conf.get('system.peregrine.data'))
    #     directory = data / f'{conf.get("system.storage.file_prefix")}_results'
    # else:
    #     directory = pathlib.Path('.')
    # if not directory.exists():
    #     directory.mkdir(exist_ok=True)
    # if filename_hint is None:
    #     filename_hint = ''
    # else:
    #     filename_hint = '_' + filename_hint
    # filename = f'{conf.get("system.storage.file_prefix")}_run_results_{datetime.datetime.now().timestamp()}{filename_hint}.json'
    # with open(directory / filename, 'w') as file:
    #     json.dump(runs, file)
    # with open(directory / f'{conf.get("system.storage.file_prefix")}_most_recent_run.txt', 'w') as file:
    #     file.write(filename)
    db: DatabaseAPI = conf.get('system.storage.database-api')
    db.save_training_results(runs)

##############################################################################
##############################################################################
# Ensemble learning
##############################################################################


def run_ensemble(factory, training_data, testing_data, label_mapping):
    match (strategy := conf.get('run.ensemble-strategy')):
        case 'stacking':
            run_stacking_ensemble(factory,
                                  training_data,
                                  testing_data,
                                  label_mapping)
        case 'voting':
            run_voting_ensemble(factory,
                                training_data,
                                testing_data,
                                label_mapping)
        case _:
            raise ValueError(f'Unknown ensemble mode {strategy}')


def run_stacking_ensemble(factory,
                          training_data,
                          testing_data,
                          label_mapping,
                          *, __voting_ensemble_hook=None):
    if conf.get('run.k-cross') > 0 and not conf.get('run.quick_cross'):
        warnings.warn('Absence of --quick-cross is ignored when running with stacking')

    # stream = split_data_quick_cross(conf.get('run.k-cross'),
    #                                 labels,
    #                                 *datasets,
    #                                 issue_keys=issue_keys,
    #                                 max_train=conf.get('run.max-train'))
    if conf.get('run.k-cross') > 0:
        splitter = splitting.QuickCrossFoldSplitter(
            k=conf.get('run.k-cross'),
            max_train=conf.get('run.max-train'),
        )
    elif conf.get('run.cross-project'):
        splitter = splitting.CrossProjectSplitter(
            val_split_size=conf.get('run.split-size'),
            max_train=conf.get('run.max-train'),
        )
    else:
        splitter = splitting.SimpleSplitter(
            val_split_size=conf.get('run.split-size'),
            test_split_size=conf.get('run.split-size'),
            max_train=conf.get('run.max-train'),
        )
    if __voting_ensemble_hook is None:
        meta_factory, input_conversion_method = stacking.build_stacking_classifier()
    else:
        meta_factory, input_conversion_method = None, False
    number_of_models = len(conf.get('run.classifier'))
    sub_results = [[] for _ in range(number_of_models)]
    best_sub_results = [[] for _ in range(number_of_models)]
    results = []
    best_results = []
    voting_result_data = []
    stream = splitter.split(training_data, testing_data)
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
                validation_keys=validation_keys
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
            train_features = stacking.transform_predictions_to_stacking_input(predictions_train,
                                                                              input_conversion_method)
            val_features = stacking.transform_predictions_to_stacking_input(predictions_val,
                                                                            input_conversion_method)
            test_features = stacking.transform_predictions_to_stacking_input(predictions_test,
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
                validation_keys=validation_keys
            )
            results.append(epoch_results)
            best_results.append(best_epoch_results)

            if conf.get('run.store-model'):     # only ran in single-shot mode
                model_manager.save_stacking_model(
                    input_conversion_method.to_json(),
                    epoch_model,
                    *trained_sub_models
                )

        else:   # We're being used by the voting ensemble
            voting_results = {
                'test': __voting_ensemble_hook[0](test[1], predictions_test),
                'train': __voting_ensemble_hook[0](train[1], predictions_train),
                'val': __voting_ensemble_hook[0](validation[1], predictions_val)
            }
            voting_result_data.append(voting_results)

            if conf.get('run.store-model'):
                model_manager.save_voting_model(
                    *trained_sub_models
                )

    if __voting_ensemble_hook is None:
        it = enumerate(zip(sub_results, best_sub_results))
        for model_number, (sub_model_results, best_sub_model_results) in it:
            print(f'Model {model_number} results:')
            print_and_save_k_cross_results(sub_model_results,
                                           best_sub_model_results,
                                           f'sub_model_{model_number}')
            print('=' * 72)
            print('=' * 72)
        print('Total Stacking Ensemble Results:')
        print_and_save_k_cross_results(results,
                                       best_results,
                                       'stacking_ensemble_total')
    else:   # Voting ensemble
        __voting_ensemble_hook[1](voting_result_data)


def run_voting_ensemble(factory,
                        training_data,
                        testing_data,
                        label_mapping):
    run_stacking_ensemble(factory,
                          training_data,
                          testing_data,
                          label_mapping,
                          __voting_ensemble_hook=(_get_voting_predictions, _save_voting_data))
    

def _save_voting_data(data):
    # filename = f'voting_ensemble_{time.time()}.json'
    # with open(filename, 'w') as file:
    #     json.dump(data, file)
    # with open('most_recent_run.txt', 'w') as file:
    #     file.write(filename)
    db: DatabaseAPI = conf.get('system.storage.database-api')
    db.save_training_results(data)


def _get_voting_predictions(truth, predictions):
    output_mode = OutputMode.from_string(conf.get('run.output-mode'))
    final_predictions = voting_util.get_voting_predictions(output_mode,
                                                           predictions)
    if output_mode == OutputMode.Detection:
        accuracy, other_metrics = metrics.compute_confusion_binary(truth,
                                                                   final_predictions,
                                                                   output_mode.label_encoding)
        return {
            'accuracy': accuracy,
            **other_metrics.as_dictionary()
        }
    else:
        reverse_mapping = {key.index(1): key for key in output_mode.label_encoding}
        final_predictions = numpy.array([reverse_mapping[pred] for pred in final_predictions])
        accuracy, class_metrics = metrics.compute_confusion_multi_class(truth,
                                                                        final_predictions,
                                                                        output_mode.label_encoding)
        return {
            'accuracy': accuracy,
            **{cls: metrics_for_class.as_dictionary()
               for cls, metrics_for_class in class_metrics.items()}
        }
