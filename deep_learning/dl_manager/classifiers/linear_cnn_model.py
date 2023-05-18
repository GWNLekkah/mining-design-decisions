import tensorflow as tf

from ..config import Argument, IntArgument, EnumArgument
from .model import AbstractModel
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
            trainable_embedding=kwargs['use-trainable-embedding']
        )
        layer_size = kwargs['fully-connected-layer-size']
        filters = kwargs['filters']
        num_convolutions = kwargs['number-of-convolutions']
        convolution_sizes = [kwargs[f'kernel-{i}-size']
                             for i in range(1, num_convolutions + 1)]
        height = self.input_size
        pooling_sizes = [height - kwargs[f'kernel-{i}-size']
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
        if (act := kwargs['fnn-layer-activation']) != 'linear':
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
    def get_arguments(cls) -> dict[str, Argument]:
        max_convolutions = 11
        num_convolutions = IntArgument(
            default=1, minimum=1, maximum=max_convolutions,
            name='number-of-convolutions',
            description='Number of different convolutions to use'
        )
        kernel_sizes = {
            f'kernel-{i}-size': IntArgument(minimum=1,
                                            default=4,
                                            maximum=512,
                                            name=f'kernel-{i}-size',
                                            description='Size of the i-th convolution kernel.')
            for i in range(1, max_convolutions + 1)
        }
        return {
            'fully-connected-layer-size': IntArgument(
                default=32, minimum=0, maximum=16384,
                name='fully-connected-layer-size',
                description='Size of the fully connected layer. Set to 0 to disable.'
            ),
            'number-of-convolutions': num_convolutions,
            'filters': IntArgument(
                default=32, minimum=1, maximum=64,
                name='filters', description='Number of filters per convolution layer.'
            ),
            'fnn-layer-activation': EnumArgument(
                default='linear',
                options=[
                    'linear', 'relu', 'elu', 'leakyrelu', 'sigmoid',
                    'tanh', 'softmax', 'softsign', 'selu', 'exp', 'prelu'
                ],
                name='fnn-layer-activation',
                description='Activation to use in the fully connected layer.'
            )
            # 'pooling_size': HyperParameter(
            #     default=2, minimum=2, maximum=16
            # ),
        } | kernel_sizes | super().get_arguments()
