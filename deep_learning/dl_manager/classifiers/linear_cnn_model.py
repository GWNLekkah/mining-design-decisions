import tensorflow as tf

from .model import AbstractModel, HyperParameter, _fix_hyper_params
from ..model_io import InputEncoding


class LinearConv1Model(AbstractModel):

    def get_model(self, *,
                  embedding=None,
                  embedding_size: int | None = None,
                  embedding_output_size: int | None = None,
                  **kwargs) -> tf.keras.Model:
        inputs, next_layer = self.get_input_layer(
            embedding=embedding,
            embedding_size=embedding_size,
            embedding_output_size=embedding_output_size,
            trainable_embedding=kwargs.get('use-trainable-embedding', False)
        )
        layer_size = int(kwargs.get('fully-connected-layer-size', 32))
        filters = int(kwargs.get('filters', 32))
        num_convolutions = int(kwargs.get('number-of-convolutions', 1))
        convolution_sizes = [int(kwargs.get(f'kernel-{i}-size', 8))
                             for i in range(1, num_convolutions + 1)]
        height = self.input_size
        pooling_sizes = [height - int(kwargs.get(f'kernel-{i}-size', 8))
                         for i in range(1, num_convolutions + 1)]
        convolutions = [
            tf.keras.layers.Conv1D(filters=filters,
                                   kernel_size=kernel_size)(next_layer)
            for kernel_size in convolution_sizes
        ]
        pooling_layers = [
            tf.keras.layers.MaxPooling1D(pool_size=p_size)(hidden)
            for hidden, p_size in zip(convolutions, pooling_sizes)
        ]
        # keras.load_model does not work on a concatenation layer with only
        # a single input layer.
        # This is intended, or will at least not be fixed.
        # For more info, see
        # https://github.com/keras-team/keras/issues/15547
        if len(pooling_layers) == 1:
            concatenated = pooling_layers[0]
        else:
            concatenated = tf.keras.layers.concatenate(pooling_layers, axis=1)
        hidden = tf.keras.layers.Flatten()(concatenated)
        if layer_size > 0:
            hidden = tf.keras.layers.Dense(layer_size)(hidden)
        if (act := kwargs.get('fnn-layer-activation', 'linear')) != 'linear':
            hidden = self.get_activation(act)(hidden)
        outputs = self.get_output_layer()(hidden)
        return tf.keras.Model(inputs=[inputs], outputs=outputs)

    @staticmethod
    def supported_input_encodings() -> list[InputEncoding]:
        return [
            InputEncoding.Vector,
            InputEncoding.Embedding,
        ]

    @staticmethod
    def input_must_support_convolution() -> bool:
        return True

    @classmethod
    @_fix_hyper_params
    def get_hyper_parameters(cls) -> dict[str, HyperParameter]:
        max_convolutions = 11
        num_convolutions = HyperParameter(
            default=1, minimum=1, maximum=max_convolutions
        )
        kernel_sizes = {
            f'kernel_{i}_size': HyperParameter(minimum=1, default=4, maximum=512)
            for i in range(1, max_convolutions + 1)
        }
        return {
            'fully_connected_layer_size': HyperParameter(
                default=32, minimum=1, maximum=16384
            ),
            'number_of_convolutions': num_convolutions,
            'filters': HyperParameter(
                default=32, minimum=1, maximum=64
            ),
            'fnn-layer-activation': HyperParameter(
                minimum=None, maximum=None, default='linear',
                options=[
                    'linear', 'relu', 'elu', 'leakyrule', 'sigmoid',
                    'tanh', 'softmax', 'softsign', 'selu', 'exp', 'prelu'
                ]
            )
            # 'pooling_size': HyperParameter(
            #     default=2, minimum=2, maximum=16
            # ),
        } | kernel_sizes | super().get_hyper_parameters()
