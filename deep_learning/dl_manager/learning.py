"""
This code contains the core of the training algorithms.

Yes, it is a mess... but a relatively easy-to-oversee mess,
and it works.
"""

##############################################################################
##############################################################################
# Imports
##############################################################################

import collections
import datetime
import gc
import json
import pathlib
import random
import statistics
import time
import warnings

import numpy
import keras.callbacks

import tensorflow as tf
import numpy as np

from collections import Counter

from .classifiers import OutputEncoding
from .feature_generators.generator import OutputMode

from . import kfold
from . import custom_kfold
from . config import conf
from . import stacking
from . import boosting
from .metrics import MetricLogger
from . import metrics
from . import data_splitting as splitting
from . import model_manager


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
# Data Preparation
##############################################################################


# def shuffle_raw_data(*x):
#     c = list(zip(*x))
#     random.shuffle(c)
#     return map(numpy.asarray, zip(*c))
#
#
# def make_dataset(labels, *features):
#     if len(features) == 1:
#         return features[0], labels
#     return list(features), labels
#
#
# def index_list(array, indices):
#     return [array[index] for index in indices]
#
#
# def split_data_single(test_split, validation_split, labels, *features, issue_keys, test_project=None):
#     labels, *features, issue_keys = shuffle_raw_data(labels, *features, issue_keys)
#
#     if test_project != 'None':
#         indices = [index for index, issue_key in enumerate(issue_keys) if issue_key.split('-')[0] == test_project]
#
#         testing_set = make_dataset(
#             np.array([label for index, label in enumerate(labels) if index in indices]),
#             *(
#                 np.array([feature for index, feature in enumerate(feature_set) if index in indices])
#                 for feature_set in features
#             )
#         )
#         test_issue_keys = [issue_key for index, issue_key in enumerate(issue_keys) if index in indices]
#
#         size_test = int(test_split * (len(labels) - len(indices)))
#         size_train = int(len(labels) - len(indices) - size_test)
#
#         training_set = make_dataset(
#             np.array([label for index, label in enumerate(labels) if index not in indices][:size_train]),
#             *(
#                 np.array([feature for index, feature in enumerate(feature_set) if index not in indices][:size_train])
#                 for feature_set in features
#             )
#         )
#
#         validation_set = make_dataset(
#             np.array([label for index, label in enumerate(labels) if index not in indices][size_train:]),
#             *(
#                 np.array([feature for index, feature in enumerate(feature_set) if index not in indices][size_train:])
#                 for feature_set in features
#             )
#         )
#     else:
#         size_train = int((1 - test_split - validation_split) * len(labels))
#         size_test = int(test_split * len(labels))
#
#         training_set = make_dataset(
#             labels[:size_train],
#             *(
#                 feature_set[:size_train] for feature_set in features
#             )
#         )
#         testing_set = make_dataset(
#             labels[size_train:size_train + size_test],
#             *(
#                 feature_set[size_train:size_train + size_test]
#                 for feature_set in features
#             )
#         )
#         test_issue_keys = issue_keys[size_train:size_train + size_test]
#         validation_set = make_dataset(
#             labels[size_train + size_test:],
#             *(
#                 feature_set[size_train + size_test:]
#                 for feature_set in features
#             )
#         )
#
#     return training_set, testing_set, validation_set, test_issue_keys
#
#
# def split_data_cross(k, labels, *features, issue_keys):
#     extended_labels = [(label, issue_key.split('-')[0]) for label, issue_key in zip(labels, issue_keys)]
#     labels, *features, extended_labels, issue_keys = shuffle_raw_data(labels, *features, extended_labels, issue_keys)
#     outer_splitter = kfold.StratifiedKFold(k)
#     outer_stream = enumerate(outer_splitter.split(features[0], extended_labels),
#                              start=1)
#     for outer_index, (inner, test) in outer_stream:
#         testing_set = make_dataset(
#             labels[test],
#             *(feature_set[test] for feature_set in features)
#         )
#         inner_splitter = kfold.StratifiedKFold(k - 1)
#         inner_stream = enumerate(
#             inner_splitter.split(features[0][inner], labels[inner]),
#             start=1
#         )
#         for inner_index, (train, validation) in inner_stream:
#             training_set = make_dataset(
#                 labels[inner][train],
#                 *(feature_set[inner][train] for feature_set in features)
#             )
#             validation_set = make_dataset(
#                 labels[inner][validation],
#                 *(feature_set[inner][validation] for feature_set in features)
#             )
#             # aux = numpy.asarray(extended_labels)
#             # plot_dataset_label_distributions(
#             #     [tuple(x) for x in extended_labels],
#             #     [tuple(x) for x in aux[inner][train].tolist()],
#             #     [tuple(x) for x in aux[test].tolist()],
#             #     [tuple(x) for x in aux[inner][validation].tolist()],
#             # )
#             yield outer_index, inner_index, training_set, testing_set, validation_set, issue_keys[test]
#
#
# def split_data_quick_cross(k, labels, *features, issue_keys, max_train=-1):
#     if conf.get('run.cross-is-cross-project'):
#         yield from split_data_cross_project(labels, *features, issue_keys=issue_keys, max_train=max_train)
#         return
#     projects = [key.split('-')[0] for key in issue_keys]
#     extended_labels = [(str(label), project)
#                        for label, project in zip(labels, projects)]
#     labels, *features, extended_labels, issue_keys = shuffle_raw_data(
#         labels, *features, extended_labels, issue_keys
#     )
#     extended_labels = [tuple(label) for label in extended_labels]
#     test_project = conf.get('run.test-project')
#     test_study = conf.get('run.test-study')
#     use_index_conversion = False
#     if test_project != 'None':
#         test_project_indices = [idx for idx, label in enumerate(extended_labels) if label[1] == test_project]
#         idx_to_original_idx = [idx for idx, label in enumerate(extended_labels) if label[1] != test_project]
#         small_extended_labels = [label for label in extended_labels if label[1] != test_project]
#         stream = custom_kfold.stratified_kfold(k, small_extended_labels)
#         use_index_conversion = True
#     elif test_study != 'None':
#         test_project_indices = idx_to_original_idx = []
#         small_extended_labels = []
#         for idx in range(len(issue_keys)):
#             if issue_keys[idx].split('-')[2] == test_study:
#                 test_project_indices.append(idx)
#             else:
#                 idx_to_original_idx.append(idx)
#                 small_extended_labels.append(extended_labels[idx])
#         stream = custom_kfold.stratified_kfold(k, small_extended_labels)
#         use_index_conversion = True
#     else:
#         stream = custom_kfold.stratified_kfold(k, extended_labels)
#     round_number = 0
#     for training_indices, validation_indices, testing_indices in stream:
#         if use_index_conversion:
#             training_indices.extend(testing_indices)
#             # Your IDE probably will not detect it, but if this branch is used,
#             # all these variables will be defined.
#             testing_indices = test_project_indices
#             training_indices = [idx_to_original_idx[idx] for idx in training_indices]
#             validation_indices = [idx_to_original_idx[idx] for idx in validation_indices]
#         round_number += 1
#         if max_train > 0:
#             training_indices_layer_2 = custom_kfold.stratified_trim(
#                 max_train, index_list(extended_labels, training_indices)
#             )
#             training_set = make_dataset(
#                 labels[training_indices][training_indices_layer_2],
#                 *(feature_set[training_indices][training_indices_layer_2]
#                   for feature_set in features)
#             )
#         else:
#             training_set = make_dataset(
#                 labels[training_indices],
#                 *(feature_set[training_indices] for feature_set in features)
#             )
#         validation_set = make_dataset(
#             labels[validation_indices],
#             *(feature_set[validation_indices] for feature_set in features)
#         )
#         testing_set = make_dataset(
#             labels[testing_indices],
#             *(feature_set[testing_indices] for feature_set in features)
#         )
#         # plot_dataset_label_distributions(
#         #     extended_labels,
#         #     index_list(extended_labels, training_indices),
#         #     index_list(extended_labels, testing_indices),
#         #     index_list(extended_labels, validation_indices)
#         # )
#         yield (round_number,
#                1,
#                training_set,
#                testing_set,
#                validation_set,
#                issue_keys[testing_indices])
#
#
# def split_data_cross_project(labels, *features, issue_keys, max_train):
#     labels, *features, issue_keys = shuffle_raw_data(
#         labels, *features, issue_keys
#     )
#     projects = []
#     for issue_key in issue_keys.tolist():
#         project, number, study = issue_key.split('-')
#         if project == 'HBASE':
#             project = 'HADOOP'
#         projects.append(project)
#     # Get indices per project
#     bins = collections.defaultdict(list)
#     for index, project in enumerate(projects):
#         bins[project].append(index)
#     # Iterate over bins to get testing project
#     output_mode = OutputMode.from_string(conf.get('run.output-mode'))
#     if output_mode.output_encoding == OutputEncoding.OneHot:
#         split_labels = labels.argmax(axis=1)
#     else:
#         split_labels = labels
#     round_number = 0
#     for test_project, testing_indices in bins.items():
#         round_number += 1
#         # Combine all remaining projects
#         remaining_projects = set(bins) - {test_project}
#         remaining_indices = []
#         for remaining_project in remaining_projects:
#             remaining_indices.extend(bins[remaining_project])
#         # Split remaining projects into training and validation set
#         training_indices, validation_indices = custom_kfold.stratified_split(numpy.asarray(remaining_indices),
#                                                                              split_labels[remaining_indices],
#                                                                              conf.get('run.split-size'))
#         training_set = make_dataset(
#             labels[training_indices],
#             *(feature_set[training_indices] for feature_set in features)
#         )
#         validation_set = make_dataset(
#             labels[validation_indices],
#             *(feature_set[validation_indices] for feature_set in features)
#         )
#         testing_set = make_dataset(
#             labels[testing_indices],
#             *(feature_set[testing_indices] for feature_set in features)
#         )
#         yield (round_number,
#                1,
#                training_set,
#                testing_set,
#                validation_set,
#                issue_keys[testing_indices])


def plot_dataset_label_distributions(original, train, test, validation):
    import pandas
    import matplotlib.pyplot as pyplot
    label_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    sets = {
        'Original': original,
        'Training': train,
        'Test': test,
        'Validation': validation
    }
    for dataset_label, dataset in sets.items():
        for label in dataset:
            label_counts[label][dataset_label] += 1
    classes = sorted(label_counts)
    header = ['Class'] + list(sets)
    rows = []
    class_names = []
    for cls in classes:
        frequencies = [label_counts[cls][dataset_name] / len(sets[dataset_name])
                       for dataset_name in header[1:]]
        cls_name = '-'.join(cls)
        row = (cls_name,) + tuple(frequencies)
        rows.append(row)
        class_names.append(cls_name)
    df = pandas.DataFrame(rows, columns=header)
    df.plot(x='Class', y=header[1:], kind="bar")
    pyplot.show()


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
               split_size: float,
               max_train: int,
               labels,
               output_mode: OutputMode,
               label_mapping: dict,
               *features,
               issue_keys,
               test_project=None):
    if max_train > 0:
        warnings.warn('The --max-train parameter is ignored in single runs.')
    # train, test, validation, test_issue_keys = split_data_single(split_size,
    #                                                              split_size,
    #                                                              labels,
    #                                                              *features,
    #                                                              issue_keys=issue_keys,
    #                                                              test_project=test_project)
    spitter = splitting.SimpleSplitter(val_split_size=conf.get('run.split-size'),
                                       test_split_size=conf.get('run.split-size'),
                                       test_study=_coerce_none(conf.get('run.test-study')),
                                       test_project=_coerce_none(conf.get('run.test-project')),
                                       max_train=conf.get('run.max-train'))
    # Split returns an iterator; call next() to get data splits
    train, test, validation, test_issue_keys = next(spitter.split(labels, issue_keys, *features))
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
                                                             test_issue_keys)
        # Save model can only be true if not testing separately,
        # which means the loop only runs once.
        if conf.get('run.store-model'):
            model_manager.save_single_model(conf.get('run.target-model-path'), trained_model)
        dump_metrics([metrics_])
        comparator.add_result(metrics_)
    comparator.add_truth(test[1])
    comparator.finalize()
    if conf.get('run.test-separately'):
        comparator.compare()


def run_cross(model_factory,
              epochs: int,
              k: int,
              max_train: int,
              quick_cross: bool,
              labels,
              output_mode: OutputMode,
              label_mapping: dict,
              *features,
              issue_keys):
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
            test_study=_coerce_none(conf.get('run.test-study')),
            test_project=_coerce_none(conf.get('run.test-project')),
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
    stream = splitter.split(labels, issue_keys, *features)
    for train, test, validation, test_issue_keys in stream:
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
                                                             test_issue_keys)
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
                         extra_model_params=None):
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

    logger = MetricLogger(model,
                          test_x,
                          test_y,
                          output_mode,
                          label_mapping,
                          test_issue_keys)
    callbacks.append(logger)

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
        warnings.warn('--epochs is ignored when using early stopping')
        conf.set('run.epochs', 1000)
    #print('Training data shape:', train_y.shape, train_x.shape)
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
    if kw_analyzer.model_is_convolution():
        import warnings
        warnings.warn('Not performing keyword analysis. Must be manually uncommented.')
        # kw_analyzer.analyze_keywords(model,
        #                              test_x,
        #                              test_y,
        #                              test_issue_keys)

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
    if conf.get('system.peregrine'):
        data = pathlib.Path(conf.get('system.peregrine.data'))
        directory = data / 'results'
    else:
        directory = pathlib.Path('.')
    if not directory.exists():
        directory.mkdir(exist_ok=True)
    if filename_hint is None:
        filename_hint = ''
    else:
        filename_hint = '_' + filename_hint
    filename = f'run_results_{datetime.datetime.now().timestamp()}{filename_hint}.json'
    with open(directory / filename, 'w') as file:
        json.dump(runs, file)
    with open(directory / 'most_recent_run.txt', 'w') as file:
        file.write(filename)

##############################################################################
##############################################################################
# Ensemble learning
##############################################################################


def run_ensemble(factory, datasets, labels, issue_keys, label_mapping):
    match (strategy := conf.get('run.ensemble-strategy')):
        case 'stacking':
            run_stacking_ensemble(factory,
                                  datasets,
                                  labels,
                                  issue_keys,
                                  label_mapping)
        case 'boosting':
            run_boosting_ensemble(factory,
                                  datasets,
                                  labels,
                                  issue_keys,
                                  label_mapping)
        case 'voting':
            run_voting_ensemble(factory,
                                datasets,
                                labels,
                                issue_keys,
                                label_mapping)
        case _:
            raise ValueError(f'Unknown ensemble mode {strategy}')


def run_stacking_ensemble(factory,
                          datasets,
                          labels,
                          issue_keys,
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
            test_study=_coerce_none(conf.get('run.test-study')),
            test_project=_coerce_none(conf.get('run.test-project')),
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
            test_study=_coerce_none(conf.get('run.test-study')),
            test_project=_coerce_none(conf.get('run.test-project')),
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
    stream = splitter.split(labels, issue_keys, *datasets)
    for train, test, validation, test_issue_keys in stream:
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
                test_issue_keys=test_issue_keys
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
                test_issue_keys=test_issue_keys
            )
            results.append(epoch_results)
            best_results.append(best_epoch_results)

            if conf.get('run.store-model'):     # only ran in single-shot mode
                model_manager.save_stacking_model(
                    conf.get('run.target-model-path'),
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
                    conf.get('run.target-model-path'),
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
                        datasets,
                        labels,
                        issue_keys,
                        label_mapping):
    run_stacking_ensemble(factory,
                          datasets,
                          labels,
                          issue_keys,
                          label_mapping,
                          __voting_ensemble_hook=(_get_voting_predictions, _save_voting_data))
    

def _save_voting_data(data):
    filename = f'voting_ensemble_{time.time()}.json'
    with open(filename, 'w') as file:
        json.dump(data, file)
    with open('most_recent_run.txt', 'w') as file:
        file.write(filename)


def _get_voting_predictions(truth, predictions):
    output_mode = OutputMode.from_string(conf.get('run.output-mode'))
    if output_mode.output_encoding == OutputEncoding.Binary:
        hard_predictions = []
        for pred in predictions:
            hard_pred = pred.flatten()
            hard_pred[hard_pred < 0.5] = 0
            hard_pred[hard_pred >= 0.5] = 1
            hard_predictions.append(hard_pred)
    else:
        hard_predictions = []
        for pred in predictions:
            hard_pred = numpy.argmax(pred, axis=1)
            hard_predictions.append(hard_pred)
    # Step 1: Determine whether there is a majority
    prediction_matrix = numpy.asarray(hard_predictions).transpose().tolist()
    final_predictions = [mode(x) for x in prediction_matrix]
    # Step 2: Break ties using probabilities
    probability_matrix = numpy.asarray(predictions).sum(axis=0)
    probability_classes: list = numpy.argmax(probability_matrix, axis=1).tolist()
    final_predictions = numpy.asarray([
        final_pred if final_pred is not None else probability_classes[index]
        for index, final_pred in enumerate(final_predictions)
    ])
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


def mode(x):
    counter = collections.Counter(x)
    if len(counter) == 1:
        return x[0]
    best_two = counter.most_common(2)
    (value_1, count_1), (value_2, count_2) = best_two
    if count_1 > count_2:
        return value_1
    if count_2 > count_1:
        return value_2
    return None


def run_boosting_ensemble(factory,
                          datasets,
                          labels,
                          issue_keys,
                          label_mapping):
    raise NotImplementedError(
        'The boosting ensemble has been disabled. '
        'The code has to be updated before it can be used again. '
        'Support must be implemented for the new `data_splitting` module. '
        'Additionally, model saving and loading must be implemented. '
        'Currently, the code is outdated, and may not work correctly.'
    )
    if conf.get('run.k-cross') > 0 and not conf.get('run.quick_cross'):
        warnings.warn('Absence of --quick-cross is ignored when running with boosting')
    boosting.check_adaboost_requirements()
    number_of_classifiers = conf.get('run.boosting-rounds')
    output_mode = OutputMode.from_string(conf.get('run.output-mode'))
    stream = split_data_quick_cross(conf.get('run.k-cross'),
                                    labels,
                                    *datasets,
                                    issue_keys=issue_keys,
                                    max_train=conf.get('run.max-train'))
    number_of_classes = output_mode.number_of_classes
    sub_model_results = collections.defaultdict(list)
    best_sub_model_results = collections.defaultdict(list)
    results = []
    for _, _, train, test, validation, test_issue_keys in stream:
        models = []
        alphas = []
        training_labels = train[1]
        weights = boosting.initialize_weights(training_labels)
        for model_number in range(number_of_classifiers):
            model = factory()
            sub_results, best_sub_results = train_and_test_model(model,
                                                                 dataset_train=train,
                                                                 dataset_val=validation,
                                                                 dataset_test=test,
                                                                 epochs=conf.get('run.epochs'),
                                                                 output_mode=OutputMode.from_string(
                                                                     conf.get('run.output-mode')),
                                                                 label_mapping=label_mapping,
                                                                 test_issue_keys=test_issue_keys,
                                                                 extra_model_params={'sample_weight': weights})
            best_sub_model_results[model_number].append(best_sub_results)
            sub_model_results[model_number].append(sub_results)
            predictions = numpy.asarray(model.predict(train[0]))
            # Convert predictions to some format
            if output_mode.output_encoding == OutputEncoding.OneHot:
                predictions = metrics.onehot_indices(predictions)
                training_labels = metrics.onehot_indices(predictions)
            else:
                predictions = metrics.round_binary_predictions(predictions)
            error = boosting.compute_error(training_labels, predictions, weights)
            alpha = boosting.compute_classifier_weight(error, number_of_classes)
            alphas.append(alpha)
            weights = boosting.update_weights(training_labels, predictions, weights, alpha)
            models.append(model)
        # Now, finally, evaluate performance on the test set
        training_predictions = []
        validation_predictions = []
        testing_predictions = []
        for model in models:
            training_predictions.append(model.predict(train[0]))
            validation_predictions.append(model.predict(validation[0]))
            testing_predictions.append(model.predict(test[0]))
        y_pred_train = boosting.compute_final_classifications(alphas,
                                                              number_of_classes,
                                                              *training_predictions)
        y_pred_val = boosting.compute_final_classifications(alphas,
                                                            number_of_classes,
                                                            *validation_predictions)
        y_pred_test = boosting.compute_final_classifications(alphas,
                                                             number_of_classes,
                                                             *testing_predictions)
        # Now, compare with y_true --- shit
        round_result = {'alphas': alphas}
        if output_mode.output_encoding == OutputEncoding.OneHot:
            round_result |= _boosting_eval_multi(train[1],
                                                 y_pred_train,
                                                 output_mode.index_label_encoding,
                                                 'train')
            round_result |= _boosting_eval_multi(validation[1],
                                                 y_pred_val,
                                                 output_mode.index_label_encoding,
                                                 'val')
            round_result |= _boosting_eval_multi(test[1],
                                                 y_pred_test,
                                                 output_mode.index_label_encoding)
        else:   # Detection / Binary
            round_result |= _boosting_eval_detection(train[1],
                                                     y_pred_train,
                                                     output_mode.label_encoding,
                                                     'train')
            round_result |= _boosting_eval_detection(validation[1],
                                                     y_pred_val,
                                                     output_mode.label_encoding,
                                                     'val')
            round_result |= _boosting_eval_detection(test[1],
                                                     y_pred_test,
                                                     output_mode.label_encoding)
        results.append(round_result)
    # Print and save results
    print('=' * 72)
    for model_number, sub_model_data in sub_model_results.items():
        print(f'Model {model_number} results:')
        print_and_save_k_cross_results(sub_model_data,
                                       best_sub_model_results[model_number],
                                       f'boosting_sub_model_{model_number}')
    print('=' * 72)
    print('=' * 72)
    print('Total Boosting Classifier Results:')
    metric_names = sorted(set(results[0].keys()))
    allowed_attributes = ['accuracy', 'f-score']
    for metric_name in metric_names:
        if metric_name not in allowed_attributes:
            continue
        print('=' * 72)
        print(f'{metric_name.capitalize()}:')
        data_points = [run[metric_name] for run in results if run[metric_name]]
        print(f' * Mean: {statistics.mean(data_points)}')
        print(f' * Median: {statistics.median(data_points)}')
        print(f' * Standard Deviation: {statistics.stdev(data_points)}')
    with open('boosting.json', 'w') as file:
        json.dump(results, file)


def _boosting_eval_detection(y_true, y_pred, labels, prefix=None):
    accuracy, metric_set = metrics.compute_confusion_binary(y_true,
                                                            y_pred,
                                                            labels)
    new_prefix = f'{prefix}-' if prefix else ''
    base_dict = {f'{new_prefix}accuracy': accuracy}
    return base_dict | metric_set.as_dictionary(prefix)


def _boosting_eval_multi(y_true, y_pred, labels, prefix=None):
    # y_true_converted = metrics.map_labels_to_names(
    #     metrics.onehot_indices(y_true),
    #     labels
    # )
    # y_pred_converted = metrics.map_labels_to_names(
    #     metrics.onehot_indices(y_pred),
    #     labels
    # )
    accuracy, metrics_per_class = metrics.compute_confusion_multi_class(
        y_true, y_pred, labels
    )
    new_prefix = f'{prefix}-' if prefix else ''
    base_dict = {f'{new_prefix}accuracy': accuracy}
    for cls, metric_set in metrics_per_class.items():
        for metric_name, value in metric_set.as_dictionary().items():
            key = f'{new_prefix}class-{metric_name}'
            base_dict.setdefault(key, {})[cls] = value
    # also compute f-score
    f_scores = [metric_set.f_score
                for metric_set in metrics_per_class.values()]
    base_dict['f-score'] = statistics.mean(f_scores)
    return base_dict
