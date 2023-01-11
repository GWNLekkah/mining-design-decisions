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


##############################################################################
##############################################################################
# Single Models
##############################################################################


def predict_simple_model(path: pathlib.Path, model_metadata, features, output_mode):
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
                       probabilities=predictions)


##############################################################################
##############################################################################
# Stacking
##############################################################################


def predict_stacking_model(path: pathlib.Path, model_metadata, features, output_mode):
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
                       probabilities=final_predictions)


##############################################################################
##############################################################################
# Voting
##############################################################################


def predict_voting_model(path: pathlib.Path, model_metadata, features, output_mode):
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

    _store_predictions(converted_predictions, output_mode)


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
    if output_mode == OutputMode.Classification3:
        raise ValueError('Support for Classification3 Not Implemented')


def _store_predictions(predictions, output_mode, *, probabilities=None):
    with open('predictions.csv', 'w') as file:
        writer = csv.writer(file)
        header = ['Prediction Name']
        if probabilities is not None:
            match output_mode:
                case OutputMode.Detection:
                    header += ['Probability Architectural']
                case OutputMode.Classification3Simplified:
                    header += [
                        'Probability Existence',
                        'Probability Executive',
                        'Probability Property',
                        'Probability Non-Architectural',
                    ]
                case OutputMode.Classification8:
                    header += [
                        'Probability Non-Architecectural',
                        'Probability Property',
                        'Probability Executive',
                        'Probability Executive/Property',
                        'Probability Existence',
                        'Probability Existence/Property',
                        'Probability Existence/Executive',
                        'Probability Existence/Executive/Property',
                    ]
                case _:
                    raise ValueError(output_mode)
        writer.writerow(header)
        label_encoding = output_mode.label_encoding
        for index in range(len(predictions)):
            row = [label_encoding[predictions[index]]]
            if probabilities is not None:
                row += [f'{x:.5f}' for x in probabilities[index]]
            writer.writerow(row)
