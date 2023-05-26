import tensorflow as tf

from ..config import IntArgument, EnumArgument, Argument
from .model import AbstractModel
from ..model_io import InputEncoding


class LinearRNNModel(AbstractModel):

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
        bilayer_size = kwargs['bidirectional-layer-size']
        current = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(bilayer_size)
        )(next_layer)
        n_layers = kwargs['number-of-hidden-layers']
        for i in range(1, n_layers + 1):
            layer_size = kwargs[f'hidden-layer-{i}-size']
            current = tf.keras.layers.Dense(layer_size)(current)
        outputs = self.get_output_layer()(current)
        return tf.keras.Model(inputs=[inputs], outputs=outputs)

    @staticmethod
    def supported_input_encodings() -> list[InputEncoding]:
        return [
            InputEncoding.Vector,
            InputEncoding.Embedding,
        ]

    @staticmethod
    def input_must_support_convolution() -> bool:
        return False

    @classmethod
    def get_arguments(cls) -> dict[str, Argument]:
        max_layers = 5
        num_layers_param = IntArgument(default=1, minimum=0, maximum=max_layers,
                                       name='number-of-hidden-layers',
                                       description='Number of hidden layers in the network.')
        layer_sizes = {
            f'hidden-layer-{i}-size': IntArgument(minimum=2,
                                                  default=32,
                                                  maximum=16384,
                                                  name=f'hidden-layer-{i}-size',
                                                  description='Number of units in the i-th hidden layer.')
            for i in range(1, max_layers + 1)
        }
        activations = {
            f'layer-{i}-activation': EnumArgument(
                default='linear',
                options=[
                    'linear', 'relu', 'elu', 'leakyrelu', 'sigmoid',
                    'tanh', 'softmax', 'softsign', 'selu', 'exp', 'prelu'
                ],
                name=f'layer-{i}-activation',
                description='Activation to use in the i-th hidden layer'
            )
            for i in range(1, max_layers + 1)
        }
        return {
            'bidirectional-layer-size': IntArgument(
                default=64, minimum=1, maximum=4096,
                name='bidirectional-layer-size',
                description='Size of the bidirectional layer.'
            ),
            'number-of-hidden-layers': num_layers_param
        } | layer_sizes | activations | super().get_arguments()
