##############################################################################
##############################################################################
# Imports
##############################################################################

from copy import copy
import pathlib

import numpy
import tensorflow as tf
from keras.models import load_model
from transformers import TFAutoModelForSequenceClassification
from transformers.modeling_tf_outputs import TFSequenceClassifierOutput

import issue_db_api

from .model_io import OutputMode, OutputEncoding
from . import stacking
from . import voting_util
from .config import Config


##############################################################################
##############################################################################
# Utility Functions
##############################################################################


def round_binary_predictions(predictions: numpy.ndarray) -> numpy.ndarray:
    rounded_predictions = copy(predictions)
    rounded_predictions[predictions <= 0.5] = 0
    rounded_predictions[predictions > 0.5] = 1
    return rounded_predictions.flatten().astype(bool)


def round_binary_predictions_no_flatten(predictions: numpy.ndarray) -> numpy.ndarray:
    rounded_predictions = copy(predictions)
    rounded_predictions[predictions <= 0.5] = 0
    rounded_predictions[predictions > 0.5] = 1
    return rounded_predictions


def round_onehot_predictions(predictions: numpy.ndarray) -> numpy.ndarray:
    return (predictions == predictions.max(axis=1)).astype(numpy.int64)


def onehot_indices(predictions: numpy.ndarray) -> numpy.ndarray:
    return predictions.argmax(axis=1)


##############################################################################
##############################################################################
# Single Models
##############################################################################


def predict_simple_model(path: pathlib.Path,
                         model_metadata,
                         features,
                         output_mode,
                         issue_ids,
                         model_id,
                         model_version, *,
                         conf: Config):
    _check_output_mode(output_mode)
    if model_metadata['model-settings']['classifier'][0] == 'Bert':
        model = TFAutoModelForSequenceClassification.from_pretrained(path / model_metadata['model-path'])
        model.classifier.activation = tf.keras.activations.sigmoid
    else:
        model = load_model(path / model_metadata['model-path'])
    if len(features) == 1:
        features = features[0]

    predictions = model.predict(features)
    if type(predictions) is TFSequenceClassifierOutput:
        predictions = predictions['logits']

    if output_mode.output_encoding == OutputEncoding.Binary and output_mode.output_size == 1:
        canonical_predictions = round_binary_predictions(predictions)
    elif output_mode.output_encoding == OutputEncoding.Binary:
        canonical_predictions = round_binary_predictions_no_flatten(predictions)
    else:
        indices = onehot_indices(predictions)
        canonical_predictions = _predictions_to_canonical(output_mode, indices)
    _store_predictions(canonical_predictions,
                       output_mode,
                       issue_ids,
                       model_id,
                       model_version,
                       probabilities=predictions,
                       conf=conf)


##############################################################################
##############################################################################
# Stacking
##############################################################################


def predict_stacking_model(path: pathlib.Path,
                           model_metadata,
                           features,
                           output_mode,
                           issue_ids,
                           model_id,
                           model_version, *,
                           conf: Config):
    _check_output_mode(output_mode)
    predictions = _ensemble_collect_predictions(path,
                                                model_metadata['child-models'],
                                                features)
    conversion = stacking.InputConversion.from_json(
        model_metadata['input-conversion-strategy']
    )
    new_features = stacking.transform_predictions_to_stacking_input(output_mode,
                                                                    predictions,
                                                                    conversion)
    meta_model = load_model(path / model_metadata['meta-model'])
    final_predictions = meta_model.predict(new_features)
    if output_mode.output_encoding == OutputEncoding.Binary:
        canonical_predictions = round_binary_predictions(final_predictions)
    else:
        indices = onehot_indices(final_predictions)
        canonical_predictions = _predictions_to_canonical(output_mode, indices)
    _store_predictions(canonical_predictions,
                       output_mode,
                       issue_ids,
                       model_id,
                       model_version,
                       probabilities=final_predictions,
                       conf=conf)


##############################################################################
##############################################################################
# Voting
##############################################################################


def predict_voting_model(path: pathlib.Path,
                         model_metadata,
                         features,
                         output_mode,
                         issue_ids,
                         model_id,
                         model_version, *,
                         conf: Config):
    _check_output_mode(output_mode)
    predictions = _ensemble_collect_predictions(path,
                                                model_metadata['child-models'],
                                                features)
    voting_predictions = voting_util.get_voting_predictions(output_mode,
                                                            predictions)
    if output_mode.output_encoding == OutputEncoding.OneHot:
        converted_predictions = _predictions_to_canonical(output_mode,
                                                          voting_predictions)
    else:
        converted_predictions = voting_predictions

    _store_predictions(converted_predictions,
                       output_mode,
                       issue_ids,
                       model_id,
                       model_version,
                       conf=conf)


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


def _store_predictions(predictions,
                       output_mode,
                       issue_ids,
                       model_id,
                       model_version,
                       *,
                       probabilities=None,
                       conf: Config):
    predictions_by_id = {}
    for i, (pred, issue_id) in enumerate(zip(predictions, issue_ids)):
        match output_mode:
            case OutputMode.Detection:
                predictions_by_id[issue_id] = {
                    'architectural': {
                        'prediction': bool(pred),
                        'confidence': float(probabilities[i][0]) if probabilities is not None else None
                    }
                }
            case OutputMode.Classification3:
                predictions_by_id[issue_id] = {
                    'existence': {
                        'prediction': bool(pred[0]),
                        'confidence': float(probabilities[i][0]) if probabilities is not None else None
                    },
                    'executive': {
                        'prediction': bool(pred[1]),
                        'confidence': float(probabilities[i][1]) if probabilities is not None else None
                    },
                    'property': {
                        'prediction': bool(pred[2]),
                        'confidence': float(probabilities[i][2]) if probabilities is not None else None
                    }
                }
            case OutputMode.Classification3Simplified:
                predictions_by_id[issue_id] = {
                    'existence': {
                        'prediction': pred == (1, 0, 0, 0),
                        'confidence': float(probabilities[i][0]) if probabilities is not None else None
                    },
                    'executive': {
                        'prediction': pred == (0, 1, 0, 0),
                        'confidence': float(probabilities[i][1]) if probabilities is not None else None
                    },
                    'property': {
                        'prediction': pred == (0, 0, 1, 0),
                        'confidence': float(probabilities[i][2]) if probabilities is not None else None
                    },
                    'non-architectural': {
                        'prediction': pred == (0, 0, 0, 1),
                        'confidence': float(probabilities[i][3]) if probabilities is not None else None
                    }
                }
            case OutputMode.Classification8:
                predictions_by_id[issue_id] = {
                    'non-architectural': {
                        'prediction': pred == 0,
                        'confidence': float(probabilities[i][0]) if probabilities is not None else None
                    },
                    'property': {
                        'prediction': pred == 1,
                        'confidence': float(probabilities[i][1]) if probabilities is not None else None
                    },
                    'executive': {
                        'prediction': pred == 2,
                        'confidence': float(probabilities[i][2]) if probabilities is not None else None
                    },
                    'executive/property': {
                        'prediction': pred == 3,
                        'confidence': float(probabilities[i][3]) if probabilities is not None else None
                    },
                    'existence': {
                        'prediction': pred == 4,
                        'confidence': float(probabilities[i][4]) if probabilities is not None else None
                    },
                    'existence/property': {
                        'prediction': pred == 5,
                        'confidence': float(probabilities[i][5]) if probabilities is not None else None
                    },
                    'existence/executive': {
                        'prediction': pred == 6,
                        'confidence': float(probabilities[i][6]) if probabilities is not None else None
                    },
                    'existence/executive/property': {
                        'prediction': pred == 7,
                        'confidence': float(probabilities[i][7]) if probabilities is not None else None
                    }
                }
    db: issue_db_api.IssueRepository = conf.get('system.storage.database-api')
    model = db.get_model_by_id(model_id)
    version = model.get_version_by_id(model_version)
    version.predictions = predictions_by_id
