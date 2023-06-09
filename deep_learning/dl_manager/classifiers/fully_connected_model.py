import tensorflow as tf
import keras_tuner

from ..config import Argument, IntArgument, EnumArgument, FloatArgument
from .model import (
    AbstractModel,
    get_tuner_values,
    get_activation,
    get_tuner_activation,
    get_tuner_optimizer,
    get_non_increasing_next_value,
)
from ..model_io import InputEncoding


class FullyConnectedModel(AbstractModel):
    def get_model(
        self,
        *,
        embedding=None,
        embedding_size: int | None = None,
        embedding_output_size: int | None = None,
        **kwargs,
    ) -> tf.keras.Model:
        inputs, next_layer = self.get_input_layer(
            embedding=embedding,
            embedding_size=embedding_size,
            embedding_output_size=embedding_output_size,
            trainable_embedding=kwargs["use-trainable-embedding"],
        )
        if self.input_encoding == InputEncoding.Embedding:
            current = tf.keras.layers.Flatten()(next_layer)
        else:
            current = next_layer
        n_layers = kwargs["number-of-hidden-layers"]
        for i in range(1, n_layers + 1):
            current = tf.keras.layers.Dense(
                units=kwargs[f"hidden-layer-{i}-size"],
                activation=get_activation(f"layer-{i}-activation", **kwargs),
                kernel_regularizer=tf.keras.regularizers.L1L2(
                    l1=kwargs[f"layer-{i}-kernel-l1"],
                    l2=kwargs[f"layer-{i}-kernel-l2"],
                ),
                bias_regularizer=tf.keras.regularizers.L1L2(
                    l1=kwargs[f"layer-{i}-bias-l1"],
                    l2=kwargs[f"layer-{i}-bias-l2"],
                ),
                activity_regularizer=tf.keras.regularizers.L1L2(
                    l1=kwargs[f"layer-{i}-activity-l1"],
                    l2=kwargs[f"layer-{i}-activity-l2"],
                ),
            )(current)
            current = tf.keras.layers.Dropout(kwargs[f"layer-{i}-dropout"])(current)

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
            inputs, next_layer = self.get_input_layer(
                embedding=embedding,
                embedding_size=embedding_size,
                embedding_output_size=embedding_output_size,
                trainable_embedding=kwargs["use-trainable-embedding"],
            )
            if self.input_encoding == InputEncoding.Embedding:
                current = tf.keras.layers.Flatten()(next_layer)
            else:
                current = next_layer
            n_hidden_layers = get_tuner_values(hp, "number-of-hidden-layers", **kwargs)
            activation = get_tuner_activation(hp, f"layer-activation", **kwargs)
            kernel_l1 = get_tuner_values(hp, f"layer-kernel-l1", **kwargs)
            kernel_l2 = get_tuner_values(hp, f"layer-kernel-l2", **kwargs)
            bias_l1 = get_tuner_values(hp, f"layer-bias-l1", **kwargs)
            bias_l2 = get_tuner_values(hp, f"layer-bias-l2", **kwargs)
            activity_l1 = get_tuner_values(hp, f"layer-activity-l1", **kwargs)
            activity_l2 = get_tuner_values(hp, f"layer-activity-l2", **kwargs)
            previous_size = -1
            for i in range(1, n_hidden_layers + 1):
                if previous_size == -1:
                    units = get_tuner_values(hp, f"hidden-layer-{i}-size", **kwargs)
                else:
                    units = get_non_increasing_next_value(
                        hp, f"hidden-layer-{i}-size", previous_size, **kwargs
                    )
                previous_size = units
                current = tf.keras.layers.Dense(
                    units=units,
                    activation=activation,
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
                    get_tuner_values(hp, f"layer-{i}-dropout", **kwargs)
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

        class TunerFNN(keras_tuner.HyperModel):
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
            trainable_embedding=kwargs["use-trainable-embedding"],
        )

        return TunerFNN(), input_layer.shape

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
        num_layers_param = IntArgument(
            default=1,
            minimum=0,
            maximum=max_layers,
            name="number-of-hidden-layers",
            description="number of hidden layers in the model.",
        )
        layer_sizes = {
            f"hidden-layer-{i}-size": IntArgument(
                minimum=2,
                default=32,
                maximum=16384,
                name=f"hidden-layer-{i}-size",
                description="Number of units in the i-th hidden layer.",
            )
            for i in range(1, max_layers + 1)
        }
        activations = {
            f"layer-activation": EnumArgument(
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
                ],
                name=f"layer-activation",
                description="Activation to use in the hidden layers",
            )
        }
        activation_alpha = {
            f"layer-activation-alpha": FloatArgument(
                default=0.0,
                name=f"layer-activation-alpha",
                description=f"Alpha value for the elu activation",
            )
        }
        dropouts = {
            f"layer-{i}-dropout": FloatArgument(
                default=0.0,
                minimum=0.0,
                maximum=1.0,
                name=f"layer-{i}-dropout",
                description=f"Dropout for the i-th layer",
            )
            for i in range(1, max_layers + 1)
        }
        regularizers = {}
        for goal in ["kernel", "bias", "activity"]:
            for type_ in ["l1", "l2"]:
                regularizers |= {
                    f"layer-{goal}-{type_}": FloatArgument(
                        default=0.0,
                        minimum=0.0,
                        maximum=1.0,
                        name=f"layer-{goal}-{type_}",
                        description=f"{type_} {goal} regularizer for the layers",
                    )
                }
        return (
            {
                "number-of-hidden-layers": num_layers_param,
            }
            | layer_sizes
            | activations
            | activation_alpha
            | dropouts
            | regularizers
            | super().get_arguments()
        )
