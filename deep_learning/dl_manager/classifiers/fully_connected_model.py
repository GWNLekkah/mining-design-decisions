import tensorflow as tf

from ..config import Argument, IntArgument, EnumArgument
from .model import AbstractModel
from ..model_io import InputEncoding


class FullyConnectedModel(AbstractModel):

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
        if self.input_encoding == InputEncoding.Embedding:
            current = tf.keras.layers.Flatten()(next_layer)
        else:
            current = next_layer
        n_layers = kwargs['number-of-hidden-layers']
        for i in range(1, n_layers + 1):
            layer_size = kwargs[f'hidden-layer-{i}-size']
            current = tf.keras.layers.Dense(layer_size)(current)
            if (act := kwargs[f'layer-{i}-activation']) != 'linear':
                current = self.get_activation(act)(current)
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
        max_layers = 11
        num_layers_param = IntArgument(default=1, minimum=0, maximum=max_layers,
                                       name='number-of-hidden-layer',
                                       description='number of hidden layers in the model.')
        layer_sizes = {
            f'hidden-layer-{i}-size': IntArgument(minimum=2, default=32, maximum=16384,
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
            'number-of-hidden-layers': num_layers_param,
        } | layer_sizes | activations | super().get_arguments()
