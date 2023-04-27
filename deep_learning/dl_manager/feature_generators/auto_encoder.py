from ..config import Argument, IntArgument, EnumArgument
from .abstract_auto_encoder import AbstractAutoEncoder
from ..model_io import InputEncoding
from .. import data_splitting
from ..logger import get_logger
log = get_logger('Auto Encoder')

import tensorflow as tf

class AutoEncoder(AbstractAutoEncoder):
    @staticmethod
    def input_encoding_type() -> InputEncoding:
        return InputEncoding.Vector

    def train_encoder(self, tokenized_issues: list[list[str]], metadata, args: dict[str, str]):
        ######################################################################
        # Prepare Training data
        log.info('Building Features')
        training_keys, training_data = self.prepare_features(
            generator_name=self.params.get('inner-generator', 'BOWNormalized')
        )
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
        out = tf.keras.layers.Dense(shape, **(reg | {'activation': self.params.get('activation-function', 'elu')}))(
            current)
        model = tf.keras.Model(inputs=[inp], outputs=out)
        scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
            0.01,
            200,
            end_learning_rate=0.001,
            power=1.0,
            # cycle=False,
            # name=None
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
        # Evaluate encoder
        # with open(conf.get('system.storage.generators')[-1]) as file:
        #     settings = json.load(file)
        # features = self.prepare_features(keys=self.issue_keys,
        #                                  issues=tokenized_issues,
        #                                  settings=settings['settings'],
        #                                  generator_name=settings['generator'])[1]['features']
        # transformed = model.predict(features)
        # For evaluation, compute the amount of preserved variance
        # difference = (features - transformed) ** 2
        # avg = difference.sum(axis=0) / 2072
        # log.info(f'Loss on test set: {avg.sum() / 2072}')
        # var_old = numpy.var(features, axis=1, ddof=1)
        # var_new = numpy.var(transformed, axis=1, ddof=1)
        # assert len(var_old) == 2179
        # log.info(f'Preserved variance: {var_new.sum() / var_old.sum()}')

        ######################################################################
        # return result
        return encoder

    @classmethod
    def get_arguments(cls) -> dict[str, Argument]:
        return super(AutoEncoder, AutoEncoder).get_arguments() | cls.get_extra_params()

    @staticmethod
    def get_extra_params():
        layers = {f'hidden-layer-{i}-size': IntArgument(name=f'hidden-layer-{i}-size',
                                                        description=f'Size of layer {i}',
                                                        minimum=2)
                  for i in range(1, 17)}
        return layers | {
            'number-of-hidden-layers': IntArgument(
                name='number-of-hidden-layers',
                description='Number of hidden layers',
                minimum=0,
                default=0
            ),
            'target-feature-size': IntArgument(
                name='target-feature-size',
                description='Target feature size',
                minimum=0
            ),
            'activation-function': EnumArgument(
                default='linear',
                options=[
                    'linear', 'relu', 'elu', 'leakyrule', 'sigmoid',
                    'tanh', 'softmax', 'softsign', 'selu', 'exp', 'prelu'
                ],
                name=f'activation-function',
                description='Activation to use in the hidden layers'
            )
        }