import tensorflow as tf
import keras_tuner

from ..config import IntArgument, EnumArgument, Argument, FloatArgument
from .model import (
    AbstractModel,
    get_tuner_values,
    get_activation,
    get_tuner_activation,
    get_tuner_optimizer,
)
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
            activation = get_activation(f"rnn-layer-activation", **kwargs)
            recurrent_activation = get_activation(
                f"rnn-layer-recurrent-activation", **kwargs
            )
            dropout = kwargs[f"rnn-layer-{i}-dropout"]
            recurrent_dropout = kwargs[f"rnn-layer-{i}-recurrent-dropout"]

            # Regularization
            kernel_regularizer = tf.keras.regularizers.L1L2(
                l1=kwargs[f"rnn-layer-kernel-l1"],
                l2=kwargs[f"rnn-layer-kernel-l2"],
            )
            recurrent_regularizer = tf.keras.regularizers.L1L2(
                l1=kwargs[f"rnn-layer-recurrent-l1"],
                l2=kwargs[f"rnn-layer-recurrent-l2"],
            )
            bias_regularizer = tf.keras.regularizers.L1L2(
                l1=kwargs[f"rnn-layer-bias-l1"],
                l2=kwargs[f"rnn-layer-bias-l2"],
            )
            activity_regularizer = tf.keras.regularizers.L1L2(
                l1=kwargs[f"rnn-layer-activity-l1"],
                l2=kwargs[f"rnn-layer-activity-l2"],
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
                        kernel_regularizer=kernel_regularizer,
                        recurrent_regularizer=recurrent_regularizer,
                        bias_regularizer=bias_regularizer,
                        activity_regularizer=activity_regularizer,
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
                        kernel_regularizer=kernel_regularizer,
                        recurrent_regularizer=recurrent_regularizer,
                        bias_regularizer=bias_regularizer,
                        activity_regularizer=activity_regularizer,
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
                        kernel_regularizer=kernel_regularizer,
                        recurrent_regularizer=recurrent_regularizer,
                        bias_regularizer=bias_regularizer,
                        activity_regularizer=activity_regularizer,
                    )
                )(current)

        # FNN layers
        n_dense_layers = kwargs["number-of-dense-layers"]
        for i in range(1, n_dense_layers + 1):
            current = tf.keras.layers.Dense(
                units=kwargs[f"dense-layer-{i}-size"],
                activation=get_activation(f"fnn-layer-activation", **kwargs),
                kernel_regularizer=tf.keras.regularizers.L1L2(
                    l1=kwargs[f"fnn-layer-kernel-l1"],
                    l2=kwargs[f"fnn-layer-kernel-l2"],
                ),
                bias_regularizer=tf.keras.regularizers.L1L2(
                    l1=kwargs[f"fnn-layer-bias-l1"],
                    l2=kwargs[f"fnn-layer-bias-l2"],
                ),
                activity_regularizer=tf.keras.regularizers.L1L2(
                    l1=kwargs[f"fnn-layer-activity-l1"],
                    l2=kwargs[f"fnn-layer-activity-l2"],
                ),
            )(current)
            current = tf.keras.layers.Dropout(kwargs[f"fnn-layer-{i}-dropout"])(current)
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
            n_rnn_layers = get_tuner_values(hp, "number-of-rnn-layers", **kwargs)
            activation = get_tuner_values(hp, "rnn-layer-activation", **kwargs)
            activation_alpha = get_tuner_values(
                hp, "rnn-layer-activation-alpha", **kwargs
            )
            recurrent_activation = get_tuner_values(
                hp, "rnn-layer-recurrent-activation", **kwargs
            )
            recurrent_activation_alpha = get_tuner_values(
                hp, "rnn-layer-recurrent-activation-alpha", **kwargs
            )
            kernel_l1 = get_tuner_values(hp, f"rnn-layer-kernel-l1", **kwargs)
            kernel_l2 = get_tuner_values(hp, f"rnn-layer-kernel-l2", **kwargs)
            recurrent_l1 = get_tuner_values(hp, f"rnn-layer-recurrent-l1", **kwargs)
            recurrent_l2 = get_tuner_values(hp, f"rnn-layer-recurrent-l2", **kwargs)
            bias_l1 = get_tuner_values(hp, f"rnn-layer-bias-l1", **kwargs)
            bias_l2 = get_tuner_values(hp, f"rnn-layer-bias-l2", **kwargs)
            activity_l1 = get_tuner_values(hp, f"rnn-layer-activity-l1", **kwargs)
            activity_l2 = get_tuner_values(hp, f"rnn-layer-activity-l2", **kwargs)
            for i in range(1, n_rnn_layers + 1):
                layer_type = get_tuner_values(hp, f"rnn-layer-{i}-type", **kwargs)
                units = get_tuner_values(hp, f"rnn-layer-{i}-size", **kwargs)
                dropout = get_tuner_values(hp, f"rnn-layer-{i}-dropout", **kwargs)
                recurrent_dropout = get_tuner_values(
                    hp, f"rnn-layer-{i}-recurrent-dropout", **kwargs
                )
                # Regularization
                kernel_regularizer = tf.keras.regularizers.L1L2(
                    l1=kernel_l1,
                    l2=kernel_l2,
                )
                recurrent_regularizer = tf.keras.regularizers.L1L2(
                    l1=recurrent_l1,
                    l2=recurrent_l2,
                )
                bias_regularizer = tf.keras.regularizers.L1L2(
                    l1=bias_l1,
                    l2=bias_l2,
                )
                activity_regularizer = tf.keras.regularizers.L1L2(
                    l1=activity_l1,
                    l2=activity_l2,
                )

                return_sequences = True
                if i == n_rnn_layers:
                    return_sequences = False
                if layer_type == "SimpleRNN":
                    current = tf.keras.layers.Bidirectional(
                        tf.keras.layers.SimpleRNN(
                            units=units,
                            activation=get_tuner_activation(
                                activation, activation_alpha
                            ),
                            recurrent_activation=get_tuner_activation(
                                recurrent_activation, recurrent_activation_alpha
                            ),
                            return_sequences=return_sequences,
                            dropout=dropout,
                            recurrent_dropout=recurrent_dropout,
                            kernel_regularizer=kernel_regularizer,
                            recurrent_regularizer=recurrent_regularizer,
                            bias_regularizer=bias_regularizer,
                            activity_regularizer=activity_regularizer,
                        )
                    )(current)
                elif layer_type == "GRU":
                    current = tf.keras.layers.Bidirectional(
                        tf.keras.layers.GRU(
                            units=units,
                            activation=get_tuner_activation(
                                activation, activation_alpha
                            ),
                            recurrent_activation=get_tuner_activation(
                                recurrent_activation, recurrent_activation_alpha
                            ),
                            return_sequences=return_sequences,
                            dropout=dropout,
                            recurrent_dropout=recurrent_dropout,
                            kernel_regularizer=kernel_regularizer,
                            recurrent_regularizer=recurrent_regularizer,
                            bias_regularizer=bias_regularizer,
                            activity_regularizer=activity_regularizer,
                        )
                    )(current)
                elif layer_type == "LSTM":
                    current = tf.keras.layers.Bidirectional(
                        tf.keras.layers.LSTM(
                            units=units,
                            activation=get_tuner_activation(
                                activation, activation_alpha
                            ),
                            recurrent_activation=get_tuner_activation(
                                recurrent_activation, recurrent_activation_alpha
                            ),
                            return_sequences=return_sequences,
                            dropout=dropout,
                            recurrent_dropout=recurrent_dropout,
                            kernel_regularizer=kernel_regularizer,
                            recurrent_regularizer=recurrent_regularizer,
                            bias_regularizer=bias_regularizer,
                            activity_regularizer=activity_regularizer,
                        )
                    )(current)
            # FNN layers
            n_dense_layers = get_tuner_values(hp, "number-of-dense-layers", **kwargs)
            activation = get_tuner_values(hp, "fnn-layer-activation", **kwargs)
            activation_alpha = get_tuner_values(
                hp, "fnn-layer-activation-alpha", **kwargs
            )
            kernel_l1 = get_tuner_values(hp, f"fnn-layer-kernel-l1", **kwargs)
            kernel_l2 = get_tuner_values(hp, f"fnn-layer-kernel-l2", **kwargs)
            bias_l1 = get_tuner_values(hp, f"fnn-layer-bias-l1", **kwargs)
            bias_l2 = get_tuner_values(hp, f"fnn-layer-bias-l2", **kwargs)
            activity_l1 = get_tuner_values(hp, f"fnn-layer-activity-l1", **kwargs)
            activity_l2 = get_tuner_values(hp, f"fnn-layer-activity-l2", **kwargs)
            for i in range(1, n_dense_layers + 1):
                units = get_tuner_values(hp, f"dense-layer-{i}-size", **kwargs)
                current = tf.keras.layers.Dense(
                    units=units,
                    activation=get_tuner_activation(activation, activation_alpha),
                    kernel_regularizer=tf.keras.regularizers.L1L2(
                        l1=kernel_l1,
                        l2=kernel_l2,
                    ),
                    bias_regularizer=tf.keras.regularizers.L1L2(
                        l1=bias_l1,
                        l2=bias_l2,
                    ),
                    activity_regularizer=tf.keras.regularizers.L1L2(
                        l1=activity_l1,
                        l2=activity_l2,
                    ),
                )(current)
                current = tf.keras.layers.Dropout(
                    get_tuner_values(hp, f"fnn-layer-{i}-dropout", **kwargs)
                )(current)
            outputs = self.get_output_layer()(current)
            model = tf.keras.Model(inputs=[inputs], outputs=outputs)

            # Compile model
            model.compile(
                optimizer=get_tuner_optimizer(hp, **kwargs),
                loss=self._get_tuner_loss_function(hp, **kwargs),
                metrics=self.get_metric_list(),
            )
            return model

        class TunerRNN(keras_tuner.HyperModel):
            def __init__(self):
                self.batch_size = None

            def build(self, hp):
                self.batch_size = get_tuner_values(hp, "batch-size", **kwargs)
                return get_model(hp)

            def fit(self, hp, model, *args, **kwargs_):
                return model.fit(*args, batch_size=self.batch_size, **kwargs_)

        input_layer, _ = self.get_input_layer(
            embedding=embedding,
            embedding_size=embedding_size,
            embedding_output_size=embedding_output_size,
            trainable_embedding=kwargs["use-trainable-embedding"]["options"]["values"][
                0
            ],
        )

        return TunerRNN(), input_layer.shape

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
            f"rnn-layer-activation": EnumArgument(
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
                    "swish",
                ],
                name=f"rnn-layer-activation",
                description="Activation to use in the rnn layers",
            )
        }
        rnn_layer_recurrent_activations = {
            f"rnn-layer-recurrent-activation": EnumArgument(
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
                    "swish",
                ],
                name=f"rnn-layer-recurrent-activation",
                description="Recurrent activation to use in the rnn layers",
            )
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
        activation_alpha = {
            f"rnn-layer-activation-alpha": FloatArgument(
                default=0.0,
                name=f"rnn-layer-activation-alpha",
                description=f"Alpha value for the elu activation of the i-th layer",
            ),
            f"rnn-layer-recurrent-activation-alpha": FloatArgument(
                default=0.0,
                name=f"rnn-layer-recurrent-activation-alpha",
                description=f"Alpha value for the elu activation of the i-th layer",
            ),
        }
        regularizers = {}
        for goal in ["kernel", "recurrent", "bias", "activity"]:
            for type_ in ["l1", "l2"]:
                regularizers |= {
                    f"rnn-layer-{goal}-{type_}": FloatArgument(
                        default=0.0,
                        minimum=0.0,
                        maximum=1.0,
                        name=f"rnn-layer-{goal}-{type_}",
                        description=f"{type_} {goal} regularizer for the layers",
                    )
                }
        # FNN params
        fnn_params = (
            {
                "number-of-dense-layers": IntArgument(
                    default=0,
                    minimum=0,
                    maximum=max_layers,
                    name="number-of-dense-layers",
                    description="Number of dense layers to use",
                )
            }
            | {
                f"dense-layer-{i}-size": IntArgument(
                    minimum=2,
                    default=32,
                    maximum=16384,
                    name=f"dense-layer-{i}-size",
                    description="Number of units in the i-th dense layer.",
                )
                for i in range(1, max_layers + 1)
            }
            | {
                f"fnn-layer-activation": EnumArgument(
                    default="linear",
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
                        "swish",
                    ],
                    name=f"fnn-layer-activation",
                    description="Activation to use in the hidden FNN layers",
                )
            }
            | {
                f"fnn-layer-activation-alpha": FloatArgument(
                    default=0.0,
                    name=f"fnn-layer-activation-alpha",
                    description=f"Alpha value for the elu activation",
                )
            }
            | {
                f"fnn-layer-{i}-dropout": FloatArgument(
                    default=0.0,
                    minimum=0.0,
                    maximum=1.0,
                    name=f"fnn-layer-{i}-dropout",
                    description=f"Dropout for the i-th FNN layer",
                )
                for i in range(1, max_layers + 1)
            }
        )
        for goal in ["kernel", "bias", "activity"]:
            for type_ in ["l1", "l2"]:
                fnn_params |= {
                    f"fnn-layer-{goal}-{type_}": FloatArgument(
                        default=0.0,
                        minimum=0.0,
                        maximum=1.0,
                        name=f"fnn-layer-{goal}-{type_}",
                        description=f"{type_} {goal} regularizer for the layers",
                    )
                }
        return (
            n_rnn_layers
            | rnn_layer_types
            | rnn_layer_sizes
            | rnn_layer_activations
            | rnn_layer_recurrent_activations
            | rnn_layer_dropouts
            | rnn_layer_recurrent_dropouts
            | fnn_params
            | activation_alpha
            | regularizers
            | super().get_arguments()
        )
