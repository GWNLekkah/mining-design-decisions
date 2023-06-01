##############################################################################
##############################################################################
# Imports
##############################################################################

import collections

import keras.callbacks
from transformers.modeling_tf_outputs import TFSequenceClassifierOutput


##############################################################################
##############################################################################
# Prediction Logger
##############################################################################


class PredictionLogger(keras.callbacks.Callback):

    def __init__(self,
                 model,
                 training_data,
                 validation_data,
                 testing_data,
                 train_ids,
                 val_ids,
                 test_ids,
                 label_mapping,
                 max_epochs: int,
                 use_early_stopping: bool = False,
                 early_stopping_attributes=None,
                 early_stopping_min_deltas=None,
                 early_stopping_patience=None):
        super().__init__()
        self._model = model
        self._train_x, self._train_y = training_data
        self._val_x, self._val_y = validation_data
        self._test_x, self._test_y = testing_data
        self._max_epochs = max_epochs
        self._label_mapping = label_mapping
        self._result = {
            'classes': [
                [(list(k) if isinstance(k, tuple) else k), v]
                for k, v in self._label_mapping.items()
            ],
            'loss': collections.defaultdict(list),
            'predictions': collections.defaultdict(list),
            'truth': {
                'training': self._train_y.tolist(),
                'validation': self._val_y.tolist(),
                'testing': self._test_y.tolist()
            },
            'datasets': {
                'training': train_ids.tolist(),
                'validation': val_ids.tolist(),
                'testing': test_ids.tolist()
            },
            'early_stopping_settings': {
                'use_early_stopping': use_early_stopping,
                'attributes': a if (a := early_stopping_attributes) is not None else [],
                'min_deltas': d if (d := early_stopping_min_deltas) is not None else [],
                'patience': p if (p := early_stopping_patience) is not None else p,
                'stopped_early': use_early_stopping,
                'early_stopping_epoch': -1
            }
        }

    def on_epoch_end(self, epoch, logs=None):
        self._check_for_early_stopping(epoch)
        self._save_predictions(self._train_x,
                               self._train_y,
                               'training',
                               logs,
                               loss_key='loss')
        self._save_predictions(self._val_x,
                               self._val_y,
                               'validation',
                               logs,
                               loss_key='val_loss')
        self._save_predictions(self._test_x,
                               self._test_y,
                               'testing',
                               logs)

    def _save_predictions(self, x, y, label: str, logs, *, loss_key=None):
        z = self._model.predict(x)
        if type(z) is TFSequenceClassifierOutput:
            z = z['logits']
        self._result['predictions'][label].append(z.tolist())
        if loss_key is not None:
            loss = logs[loss_key]
        else:
            loss = 0
            #loss = self._model.compute_loss(x, numpy.asarray(y), z)
        self._result['loss'][label].append(loss)

    def _check_for_early_stopping(self, epoch):
        settings = self._result['early_stopping_settings']
        if settings['use_early_stopping']:
           is_last_epoch = epoch + 1 == self._max_epochs
           if is_last_epoch:
                settings['early_stopping_epoch'] = -1
                settings['stopped_early'] = False
           else:
                settings['early_stopping_epoch'] = epoch + 1

    def get_model_results_for_all_epochs(self):
        return self._result

    def get_main_model_metrics_at_stopping_epoch(self):
        if self._result['early_stopping_settings']['use_early_stopping']:
            offset = self._result['early_stopping_settings']['patience']
        else:
            offset = 0
        # TODO: perhaps some on-demand metric calculation here?
        return {
            'loss': self._result['loss']['testing'][-1 - offset]
        }
