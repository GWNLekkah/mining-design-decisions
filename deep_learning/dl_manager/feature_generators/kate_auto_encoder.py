import json
import numpy

from . import ParameterSpec
from .abstract_auto_encoder import AbstractAutoEncoder
from ..classifiers import InputEncoding
from .. import data_splitting
from ..config import conf
from ..logger import get_logger

log = get_logger('Kate Auto Encoder')

import tensorflow as tf

from ..keras_extensions.k_competitive_layer import KCompetitive
from ..keras_extensions.dense_tied_layer import Dense_tied

class KateAutoEncoder(AbstractAutoEncoder):
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

        # The model code is based on
        # https://github.com/hugochan/KATE/blob/master/autoencoder/core/ae.py
        tied_layer = tf.keras.layers.Dense(
            int(self.params.get('hidden-layer-size')), activation='tanh', kernel_initializer='glorot_normal'
        )
        hidden = tied_layer(inp)
        encoded = KCompetitive(int(self.params.get('k-competitive')), 'kcomp', name='encoder_layer')(hidden)
        decoded = Dense_tied(shape, activation='sigmoid', tied_to=tied_layer)(encoded)
        model = tf.keras.Model(inputs=[inp], outputs=decoded)
        #encoder_model = tf.keras.Model(inputs=[inp], outputs=encoded)
        # match self.params.get('loss', 'binary_cross_entropy'):
        #     case 'contractive':
        #         selected_loss = contractive_loss(encoder_model)
        #     case 'crossentropy':
        #         selected_loss = 'binary_crossentropy'
        #     case _ as x:
        #         raise ValueError(f'Invalid loss for {self.__class__.__name__}: {x}')
        model.compile(
            optimizer=tf.keras.optimizers.Adadelta(lr=0.2),
            loss='binary_crossentropy'
        )

        ######################################################################
        # Train Model
        train, val = dataset.split_fraction(0.9)
        train_x, train_y = train.to_dataset()
        val_x, val_y = val.to_dataset()
        model.fit(x=train_x,
                  y=train_y,
                  validation_data=(val_x, val_y),
                  epochs=50,
                  batch_size=100,
                  shuffle=True,
                  callbacks=[
                      tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.01),
                      tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=5, verbose=1, mode='auto'),
                  ]
        )

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
    def get_parameters(cls) -> dict[str, ParameterSpec]:
        return super(KateAutoEncoder, KateAutoEncoder).get_parameters() | cls.get_extra_params()

    @staticmethod
    def get_extra_params():
        return {
            'hidden-layer-size': ParameterSpec(
                description='Size of the hidden layer',
                type='int'
            ),
            'k-competitive': ParameterSpec(
                description='Size of the K-Competitive layer',
                type='int'
            ),
            # 'loss': ParameterSpec(
            #     description='Loss to use for training the encoder. Either "contractive" or "crossentropy"',
            #     type='str'
            # )
        }