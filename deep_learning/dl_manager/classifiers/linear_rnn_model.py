import tensorflow as tf

from ..config import IntArgument, EnumArgument, Argument, FloatArgument
from .model import AbstractModel
from ..model_io import InputEncoding


class LinearRNNModel(AbstractModel):
    def get_model(
        self,
        *,
        embedding=None,
        embedding_size: int | None = None,
        embedding_output_size: int | None = None,
        **kwargs,
    ) -> tf.keras.Model:
        inputs, current = self.get_input_layer(
            embedding=embedding,
            embedding_size=embedding_size,
            embedding_output_size=embedding_output_size,
            trainable_embedding=kwargs["use-trainable-embedding"],
        )
        n_rnn_layers = kwargs["number-of-rnn-layers"]
        for i in range(1, n_rnn_layers + 1):
            layer_type = kwargs[f"rnn-layer-{i}-type"]
            units = kwargs[f"rnn-layer-{i}-size"]
            activation = kwargs[f"rnn-layer-{i}-activation"]
            recurrent_activation = kwargs[f"rnn-layer-{i}-recurrent-activation"]
            dropout = kwargs[f"rnn-layer-{i}-dropout"]
            recurrent_dropout = kwargs[f"rnn-layer-{i}-recurrent-dropout"]
            return_sequences = True
            if i == n_rnn_layers:
                return_sequences = False
            if layer_type == "SimpleRNN":
                current = tf.keras.layers.Bidirectional(
                    tf.keras.layers.SimpleRNN(
                        units=units,
                        activation=activation,
                        recurrent_activation=recurrent_activation,
                        return_sequences=return_sequences,
                        dropout=dropout,
                        recurrent_dropout=recurrent_dropout,
                    )
                )(current)
            elif layer_type == "GRU":
                current = tf.keras.layers.Bidirectional(
                    tf.keras.layers.GRU(
                        units=units,
                        activation=activation,
                        recurrent_activation=recurrent_activation,
                        return_sequences=return_sequences,
                        dropout=dropout,
                        recurrent_dropout=recurrent_dropout,
                    )
                )(current)
            elif layer_type == "LSTM":
                current = tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(
                        units=units,
                        activation=activation,
                        recurrent_activation=recurrent_activation,
                        return_sequences=return_sequences,
                        dropout=dropout,
                        recurrent_dropout=recurrent_dropout,
                    )
                )(current)

        n_dense_layers = kwargs["number-of-dense-layers"]
        for i in range(1, n_dense_layers + 1):
            layer_size = kwargs[f"dense-layer-{i}-size"]
            current = tf.keras.layers.Dense(layer_size)(current)
        outputs = self.get_output_layer()(current)
        return tf.keras.Model(inputs=[inputs], outputs=outputs)

    def get_keras_tuner_model(
        self,
        *,
        embedding=None,
        embedding_size: int | None = None,
        embedding_output_size: int | None = None,
        **kwargs,
    ):
        def get_model(hp):
            inputs, current = self.get_input_layer(
                embedding=embedding,
                embedding_size=embedding_size,
                embedding_output_size=embedding_output_size,
                trainable_embedding=kwargs["use-trainable-embedding"]["options"][
                    "values"
                ][0],
            )
            n_rnn_layers = self._get_values(hp, "number-of-rnn-layers", **kwargs)
            for i in range(1, n_rnn_layers + 1):
                layer_type = self._get_values(hp, f"rnn-layer-{i}-type", **kwargs)
                units = self._get_values(hp, f"rnn-layer-{i}-size", **kwargs)
                activation = self._get_values(hp, f"rnn-layer-{i}-activation", **kwargs)
                recurrent_activation = self._get_values(
                    hp, f"rnn-layer-{i}-recurrent-activation", **kwargs
                )
                dropout = self._get_values(hp, f"rnn-layer-{i}-dropout", **kwargs)
                recurrent_dropout = self._get_values(
                    hp, f"rnn-layer-{i}-recurrent-dropout", **kwargs
                )
                return_sequences = True
                if i == n_rnn_layers:
                    return_sequences = False
                if layer_type == "SimpleRNN":
                    current = tf.keras.layers.Bidirectional(
                        tf.keras.layers.SimpleRNN(
                            units=units,
                            activation=activation,
                            recurrent_activation=recurrent_activation,
                            return_sequences=return_sequences,
                            dropout=dropout,
                            recurrent_dropout=recurrent_dropout,
                        )
                    )(current)
                elif layer_type == "GRU":
                    current = tf.keras.layers.Bidirectional(
                        tf.keras.layers.GRU(
                            units=units,
                            activation=activation,
                            recurrent_activation=recurrent_activation,
                            return_sequences=return_sequences,
                            dropout=dropout,
                            recurrent_dropout=recurrent_dropout,
                        )
                    )(current)
                elif layer_type == "LSTM":
                    current = tf.keras.layers.Bidirectional(
                        tf.keras.layers.LSTM(
                            units=units,
                            activation=activation,
                            recurrent_activation=recurrent_activation,
                            return_sequences=return_sequences,
                            dropout=dropout,
                            recurrent_dropout=recurrent_dropout,
                        )
                    )(current)
            outputs = self.get_output_layer()(current)
            model = tf.keras.Model(inputs=[inputs], outputs=outputs)

            # Compile model
            model.compile(
                optimizer=self._get_tuner_optimizer(hp, **kwargs),
                loss=self._get_tuner_loss_function(hp, **kwargs),
                metrics=self.get_metric_list(),
            )
            return model

        return get_model

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
        max_layers = 10
        n_rnn_layers = {
            "number-of-rnn-layers": IntArgument(
                default=1,
                minimum=1,
                maximum=max_layers,
                name="number-of-rnn-layers",
                description="Number of RNN layers to use",
            )
        }
        rnn_layer_types = {
            f"rnn-layer-{i}-type": EnumArgument(
                default="LSTM",
                options=["SimpleRNN", "GRU", "LSTM"],
                name=f"rnn-layer-{i}-type",
                description="Type of RNN layer",
            )
            for i in range(1, max_layers + 1)
        }
        rnn_layer_sizes = {
            f"rnn-layer-{i}-size": IntArgument(
                minimum=2,
                default=32,
                maximum=4096,
                name=f"rnn-layer-{i}-size",
                description="Number of units in the i-th rnn layer.",
            )
            for i in range(1, max_layers + 1)
        }
        rnn_layer_activations = {
            f"rnn-layer-{i}-activation": EnumArgument(
                default="tanh",
                options=[
                    "linear",
                    "relu",
                    "elu",
                    "leakyrelu",
                    "sigmoid",
                    "tanh",
                    "softmax",
                    "softsign",
                    "selu",
                    "exp",
                    "prelu",
                ],
                name=f"rnn-layer-{i}-activation",
                description="Activation to use in the i-th rnn layer",
            )
            for i in range(1, max_layers + 1)
        }
        rnn_layer_recurrent_activations = {
            f"rnn-layer-{i}-recurrent-activation": EnumArgument(
                default="sigmoid",
                options=[
                    "linear",
                    "relu",
                    "elu",
                    "leakyrelu",
                    "sigmoid",
                    "tanh",
                    "softmax",
                    "softsign",
                    "selu",
                    "exp",
                    "prelu",
                ],
                name=f"rnn-layer-{i}-recurrent-activation",
                description="Recurrent activation to use in the i-th rnn layer",
            )
            for i in range(1, max_layers + 1)
        }
        rnn_layer_dropouts = {
            f"rnn-layer-{i}-dropout": FloatArgument(
                default=0.0,
                minimum=0.0,
                maximum=1.0,
                name=f"rnn-layer-{i}-dropout",
                description="Dropout for the i-th rnn layer",
            )
            for i in range(1, max_layers + 1)
        }
        rnn_layer_recurrent_dropouts = {
            f"rnn-layer-{i}-recurrent-dropout": FloatArgument(
                default=0.0,
                minimum=0.0,
                maximum=1.0,
                name=f"rnn-layer-{i}-recurrent-dropout",
                description="Recurrent dropout for i-th rnn layer",
            )
            for i in range(1, max_layers + 1)
        }

        n_dense_layers = {
            "number-of-dense-layers": IntArgument(
                default=0,
                minimum=0,
                maximum=max_layers,
                name="number-of-dense-layers",
                description="Number of dense layers to use",
            )
        }
        dense_layer_sizes = {
            f"dense-layer-{i}-size": IntArgument(
                minimum=2,
                default=32,
                maximum=16384,
                name=f"dense-layer-{i}-size",
                description="Number of units in the i-th dense layer.",
            )
            for i in range(1, max_layers + 1)
        }

        return (
            n_rnn_layers
            | rnn_layer_types
            | rnn_layer_sizes
            | rnn_layer_activations
            | rnn_layer_recurrent_activations
            | rnn_layer_dropouts
            | rnn_layer_recurrent_dropouts
            | n_dense_layers
            | dense_layer_sizes
            | super().get_arguments()
        )
