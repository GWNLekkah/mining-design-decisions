##############################################################################
##############################################################################
# Imports
##############################################################################

import csv
import pathlib

from keras.models import load_model

from .classifiers import OutputEncoding
from .feature_generators import OutputMode
from . import stacking
from . import voting_util
from . import metrics
from .config import conf
from .database import DatabaseAPI


##############################################################################
##############################################################################
# Single Models
##############################################################################


def predict_simple_model(path: pathlib.Path,
                         model_metadata,
                         features,
                         output_mode,
                         issue_ids):
    _check_output_mode(output_mode)
    model = load_model(path / model_metadata['model_path'])
    if len(features) == 1:
        features = features[0]
    predictions = model.predict(features)
    if output_mode.output_encoding == OutputEncoding.Binary:
        canonical_predictions = metrics.round_binary_predictions(predictions)
    else:
        indices = metrics.onehot_indices(predictions)
        canonical_predictions = _predictions_to_canonical(output_mode, indices)
    _store_predictions(canonical_predictions,
                       output_mode,
                       issue_ids,
                       probabilities=predictions)


##############################################################################
##############################################################################
# Stacking
##############################################################################


def predict_stacking_model(path: pathlib.Path,
                           model_metadata,
                           features,
                           output_mode,
                           issue_ids):
    _check_output_mode(output_mode)
    predictions = _ensemble_collect_predictions(path,
                                                model_metadata['child_models'],
                                                features)
    conversion = stacking.InputConversion.from_json(
        model_metadata['input_conversion_strategy']
    )
    new_features = stacking.transform_predictions_to_stacking_input(predictions,
                                                                    conversion)
    meta_model = load_model(path / model_metadata['meta_model'])
    final_predictions = meta_model.predict(new_features)
    if output_mode.output_encoding == OutputEncoding.Binary:
        canonical_predictions = metrics.round_binary_predictions(final_predictions)
    else:
        indices = metrics.onehot_indices(final_predictions)
        canonical_predictions = _predictions_to_canonical(output_mode, indices)
    _store_predictions(canonical_predictions,
                       output_mode,
                       issue_ids,
                       probabilities=final_predictions)


##############################################################################
##############################################################################
# Voting
##############################################################################


def predict_voting_model(path: pathlib.Path,
                         model_metadata,
                         features,
                         output_mode,
                         issue_ids):
    _check_output_mode(output_mode)
    predictions = _ensemble_collect_predictions(path,
                                                model_metadata['child_models'],
                                                features)
    voting_predictions = voting_util.get_voting_predictions(output_mode,
                                                            predictions)
    if output_mode.output_encoding == OutputEncoding.OneHot:
        converted_predictions = _predictions_to_canonical(output_mode,
                                                          voting_predictions)
    else:
        converted_predictions = voting_predictions

    _store_predictions(converted_predictions, output_mode, issue_ids)


##############################################################################
##############################################################################
# Utility functions
##############################################################################


def _predictions_to_canonical(output_mode, voting_predictions):
    if output_mode.output_encoding == OutputEncoding.Binary:
        return voting_predictions
    full_vector_length = output_mode.output_size
    output = []
    for index in voting_predictions:
        vec = [0] * full_vector_length
        vec[index] = 1
        output.append(tuple(vec))
    return output


def _ensemble_collect_predictions(path: pathlib.Path, models, features):
    predictions = []
    for model_path, feature_set in zip(models, features):
        model = load_model(path / model_path)
        predictions.append(model.predict(feature_set))
    return predictions


def _check_output_mode(output_mode):
    #if output_mode == OutputMode.Classification3:
    #    raise ValueError('Support for Classification3 Not Implemented')
    pass


def _store_predictions(predictions, output_mode, issue_ids, *, probabilities=None):
    predictions_by_id = {}
    for i, (pred, issue_id) in enumerate(zip(predictions, issue_ids)):
        match output_mode:
            case OutputMode.Detection:
                predictions_by_id[issue_id] = {
                    'architectural': {
                        'prediction': bool(pred),
                        'probability': float(probabilities[i][0]) if probabilities is not None else None
                    }
                }
            case OutputMode.Classification3:
                predictions_by_id[issue_id] = {
                    'existence': {
                        'prediction': bool(pred[0]),
                        'probability': float(probabilities[i][0]) if probabilities is not None else None
                    },
                    'executive': {
                        'prediction': bool(pred[1]),
                        'probability': float(probabilities[i][1]) if probabilities is not None else None
                    },
                    'property': {
                        'prediction': bool(pred[2]),
                        'probability': float(probabilities[i][2]) if probabilities is not None else None
                    }
                }
            case OutputMode.Classification3Simplified:
                predictions_by_id[issue_id] = {
                    'existence': {
                        'prediction': pred == 0,
                        'probability': float(probabilities[i][0]) if probabilities is not None else None
                    },
                    'executive': {
                        'prediction': pred == 1,
                        'probability': float(probabilities[i][1]) if probabilities is not None else None
                    },
                    'property': {
                        'prediction': pred == 2,
                        'probability': float(probabilities[i][2]) if probabilities is not None else None
                    },
                    'non-architectural': {
                        'prediction': pred == 3,
                        'probability': float(probabilities[i][3]) if probabilities is not None else None
                    }
                }
            case OutputMode.Classification8:
                predictions_by_id[issue_id] = {
                    'non-architectural': {
                        'prediction': pred == 0,
                        'probability': float(probabilities[i][0]) if probabilities is not None else None
                    },
                    'property': {
                        'prediction': pred == 1,
                        'probability': float(probabilities[i][1]) if probabilities is not None else None
                    },
                    'executive': {
                        'prediction': pred == 2,
                        'probability': float(probabilities[i][2]) if probabilities is not None else None
                    },
                    'executive/property': {
                        'prediction': pred == 3,
                        'probability': float(probabilities[i][3]) if probabilities is not None else None
                    },
                    'existence': {
                        'prediction': pred == 4,
                        'probability': float(probabilities[i][4]) if probabilities is not None else None
                    },
                    'existence/property': {
                        'prediction': pred == 5,
                        'probability': float(probabilities[i][5]) if probabilities is not None else None
                    },
                    'existence/executive': {
                        'prediction': pred == 6,
                        'probability': float(probabilities[i][6]) if probabilities is not None else None
                    },
                    'existence/executive/property': {
                        'prediction': pred == 7,
                        'probability': float(probabilities[i][7]) if probabilities is not None else None
                    }
                }
    db: DatabaseAPI = conf.get('system.storage.database-api')
    db.save_predictions('XXX', predictions_by_id)
    if (tag := conf.get('predict.with-tag')) != '':
        db.add_tag(issue_ids, tag)
    # prefix = conf.get('system.storage.file-prefix')
    # with open(f'{prefix}_predictions.csv', 'w') as file:
    #     writer = csv.writer(file)
    #     header = ['Prediction Name']
    #     if probabilities is not None:
    #         match output_mode:
    #             case OutputMode.Detection:
    #                 header += ['Probability Architectural']
    #             case OutputMode.Classification3Simplified:
    #                 header += [
    #                     'Probability Existence',
    #                     'Probability Executive',
    #                     'Probability Property',
    #                     'Probability Non-Architectural',
    #                 ]
    #             case OutputMode.Classification3:
    #                 header += [
    #                     'Probability Existence',
    #                     'Probability Executive',
    #                     'Probability Property'
    #                 ]
    #             case OutputMode.Classification8:
    #                 header += [
    #                     'Probability Non-Architecectural',
    #                     'Probability Property',
    #                     'Probability Executive',
    #                     'Probability Executive/Property',
    #                     'Probability Existence',
    #                     'Probability Existence/Property',
    #                     'Probability Existence/Executive',
    #                     'Probability Existence/Executive/Property',
    #                 ]
    #             case _:
    #                 raise ValueError(output_mode)
    #     writer.writerow(header)
    #     label_encoding = output_mode.label_encoding
    #     for index in range(len(predictions)):
    #         row = [label_encoding[predictions[index]]]
    #         if probabilities is not None:
    #             row += [f'{x:.5f}' for x in probabilities[index]]
    #         writer.writerow(row)
