import json

import numpy
import seaborn

from . import ParameterSpec
from .generator import AbstractFeatureGenerator
from ..classifiers import InputEncoding
from .bow_frequency import BOWFrequency
from .bow_normalized import BOWNormalized
from .tfidf import TfidfGenerator
from .. import data_splitting
from ..config import conf
from ..logger import get_logger
log = get_logger('Auto Encoder')

import tensorflow as tf

class AutoEncoder(AbstractFeatureGenerator):
    @staticmethod
    def input_encoding_type() -> InputEncoding:
        return InputEncoding.Vector

    def generate_vectors(self, tokenized_issues: list[list[str]], metadata, args: dict[str, str]):
        ######################################################################
        # Prepare Training data
        log.info('Building Features')
        training_keys, training_data = self.prepare_features()
        shape = training_data['feature_shape']
        features = training_data['features']
        keys = set(self.issue_keys)
        log.info('Removing testing samples')
        features = [vec
                    for vec, key in zip(features, training_keys)
                    if key not in keys]
        training_keys = [key for key in training_keys if key not in keys]
        dataset = data_splitting.DeepLearningData(
            features,
            training_keys,
            features
        )
        ######################################################################
        # Build model
        log.info(f'Number of words in BOW model: {shape}')
        log.info('Building auto encoder network')
        inp = tf.keras.layers.Input(shape=(shape,))
        current = inp
        reg = {
            'kernel_regularizer': tf.keras.regularizers.L2(0.01),
            'bias_regularizer': tf.keras.regularizers.L2(0.01),
            'activity_regularizer': tf.keras.regularizers.L1(0.01),
            'use_bias': True,
            'activation': self.params.get('activation-function', 'elu')
        }
        for i in range(1, 1 + int(self.params.get('number-of-hidden-layers', '1'))):
            x = int(self.params.get(f'hidden-layer-{i}-size', '8'))
            current = tf.keras.layers.Dense(x, **reg)(current)
        middle = tf.keras.layers.Dense(int(self.params['target-feature-size']), name='encoder_layer', **reg)(current)
        current = middle
        for i in reversed(range(1, 1 + int(self.params.get('number-of-hidden-layers', '1')))):
            x = int(self.params.get(f'hidden-layer-{i}-size', '8'))
            current = tf.keras.layers.Dense(x, **reg)(current)
        out = tf.keras.layers.Dense(shape, **(reg | {'activation': self.params.get('activation-function', 'elu')}))(current)
        model = tf.keras.Model(inputs=[inp], outputs=out)
        scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
            0.01,
            200,
            end_learning_rate=0.001,
            power=1.0,
            #cycle=False,
            #name=None
        )
        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                      optimizer=tf.keras.optimizers.Adam(scheduler))
        ######################################################################
        # Train Model
        train, val = dataset.split_fraction(0.9)
        train_x, train_y = train.to_dataset()
        val_x, val_y = val.to_dataset()
        model.fit(x=train_x,
                  y=train_y,
                  validation_data=(val_x, val_y),
                  epochs=30,
                  batch_size=128)
        ######################################################################
        # Build Encoder
        encoder = tf.keras.Model(inputs=model.input,
                                 outputs=model.get_layer('encoder_layer').output)
        ######################################################################
        # Plot Test Data
        log.info('Generating Testing Features')
        with open(conf.get('system.storage.generators')[0]) as file:
            settings = json.load(file)
        features = self.prepare_features(keys=self.issue_keys,
                                         issues=tokenized_issues,
                                         settings=settings['settings'])[1]['features']
        log.info('Mapping testing features')
        as_2d = encoder.predict(features)
        # log.info('Rendering plot')
        # import matplotlib.pyplot as pyplot
        # colors = numpy.asarray(self.colors)
        # pyplot.scatter(as_2d[:, 0][colors == 0], as_2d[:, 1][colors == 0], c='r', alpha=0.5, label='Executive')
        # pyplot.scatter(as_2d[:, 0][colors == 1], as_2d[:, 1][colors == 1], c='g', alpha=0.5, label='Property')
        # pyplot.scatter(as_2d[:, 0][colors == 2], as_2d[:, 1][colors == 2], c='b', alpha=0.5, label='Existence')
        # pyplot.scatter(as_2d[:, 0][colors == 3], as_2d[:, 1][colors == 3], c='y', alpha=0.5, label='Non-Architectural')
        # pyplot.legend(loc='upper left')
        # pyplot.show()
        # raise RuntimeError
        transformed = model.predict(features)
        difference = (features - transformed) ** 2
        avg = difference.sum(axis=0) / 2072
        log.info(f'Loss on test set: {avg.sum() / 2072}')
        var_old = numpy.var(features, axis=1, ddof=1)
        var_new = numpy.var(transformed, axis=1, ddof=1)
        assert len(var_old) == 2179
        log.info(f'Preserved variance: {var_new.sum() / var_old.sum()}')
        import matplotlib.pyplot as pyplot
        seaborn.heatmap(avg.reshape(37, 56), cmap='viridis')
        pyplot.show()
        return {
            'features': as_2d.tolist(),
            'feature_shape': int(self.params['target-feature-size'])
        }

    def prepare_features(self, keys=None, issues=None, settings=None):
        if issues is None:
            with open(self.params['training-data-file']) as file:
                data = json.load(file)
            keys = [issue['key'] for issue in data]
            issues = [
                issue['summary'] + issue['description']
                for issue in data
            ]
        if settings is None:
            params = self.params.copy()
            params['min-doc-count'] = params['bow-min-count']
            for name in self._get_extra_params():
                try:
                    del params[name]
                except KeyError:
                    pass
            match self.params.get('inner-generator', 'BOWNormalized'):
                case 'BOWFrequency':
                    generator = BOWFrequency(**params)
                case 'BOWNormalized':
                    generator = BOWNormalized(**params)
                case 'TfidfGenerator':
                    try:
                        del params['min-doc-count']
                    except KeyError:
                        pass
                    generator = TfidfGenerator(**params)
                case _ as g:
                    raise ValueError(f'Unsupported feature generator for auto-encoder: {g}')
        else:
            match self.params.get('inner-generator', 'BOWNormalized'):
                case 'BOWFrequency':
                    generator = BOWFrequency(pretrained_generator_settings=settings)
                case 'BOWNormalized':
                    generator = BOWNormalized(pretrained_generator_settings=settings)
                case 'TfidfGenerator':
                    generator = TfidfGenerator(pretrained_generator_settings=settings)
                case _ as g:
                    raise ValueError(f'Unsupported feature generator for auto-encoder: {g}')
        return keys, generator.generate_vectors(
            generator.preprocess(issues),
            [[] for _ in range(len(issues))],
            generator.params
        )

    @classmethod
    def get_parameters(cls) -> dict[str, ParameterSpec]:
        return super(AutoEncoder, AutoEncoder).get_parameters() | cls._get_extra_params()

    @staticmethod
    def _get_extra_params():
        layers = {f'hidden-layer-{i}-size': ParameterSpec(description=f'Size of layer {i}', type='int')
                  for i in range(1, 17)}
        return layers | {
            'inner-generator': ParameterSpec(
                description='Feature generator to transform issues to text',
                type='str'
            ),
            'number-of-hidden-layers': ParameterSpec(
                description='Number of hidden layers',
                type='int'
            ),
            'target-feature-size': ParameterSpec(
                description='Target feature size',
                type='int'
            ),
            'training-data-file': ParameterSpec(
                description='File of data to use to train the auto-encoder',
                type='str'
            ),
            'bow-min-count': ParameterSpec(
                description='Minimum document count for bag of words',
                type='int'
            ),
            'activation-function': ParameterSpec(
                description='Activation function to use in the auto encoder',
                type='str'
            )
        }