import tensorflow as tf
import keras_tuner

from ..config import Argument, IntArgument, EnumArgument, FloatArgument
from .model import (
    AbstractModel,
    get_tuner_values,
    get_activation,
    get_tuner_activation,
    get_tuner_optimizer,
)
from ..model_io import InputEncoding


class LinearConv1Model(AbstractModel):
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
        layer_size = kwargs["fully-connected-layer-size"]
        filters = kwargs["filters"]
        num_convolutions = kwargs["number-of-convolutions"]
        convolution_sizes = [
            kwargs[f"kernel-{i}-size"] for i in range(1, num_convolutions + 1)
        ]
        height = self.input_size
        pooling_sizes = [
            height - kwargs[f"kernel-{i}-size"] for i in range(1, num_convolutions + 1)
        ]
        convolutions = []
        for i, kernel_size in enumerate(convolution_sizes):
            activation = get_activation(f"layer-activation", **kwargs)
            convolutions.append(
                tf.keras.layers.Conv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    activation=activation,
                    kernel_regularizer=tf.keras.regularizers.L1L2(
                        l1=kwargs[f"layer-kernel-l1"],
                        l2=kwargs[f"layer-kernel-l2"],
                    ),
                    bias_regularizer=tf.keras.regularizers.L1L2(
                        l1=kwargs[f"layer-bias-l1"],
                        l2=kwargs[f"layer-bias-l2"],
                    ),
                    activity_regularizer=tf.keras.regularizers.L1L2(
                        l1=kwargs[f"layer-activity-l1"],
                        l2=kwargs[f"layer-activity-l2"],
                    ),
                )(next_layer)
            )
        convolutions = [
            tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size)(next_layer)
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
            hidden = tf.keras.layers.Dense(
                layer_size, activation=get_activation("fnn-layer-activation", **kwargs)
            )(hidden)
        outputs = self.get_output_layer()(hidden)
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
                trainable_embedding=kwargs["use-trainable-embedding"]["options"][
                    "values"
                ][0],
            )
            layer_size = get_tuner_values(hp, "fully-connected-layer-size", **kwargs)
            filters = get_tuner_values(hp, "filters", **kwargs)
            num_convolutions = get_tuner_values(hp, "number-of-convolutions", **kwargs)
            convolution_sizes = [
                get_tuner_values(hp, f"kernel-{i}-size", **kwargs)
                for i in range(1, num_convolutions + 1)
            ]
            height = self.input_size
            pooling_sizes = [
                height - convolution_sizes[i - 1]
                for i in range(1, num_convolutions + 1)
            ]
            convolutions = []
            activation = get_tuner_activation(hp, f"layer-activation", **kwargs)
            kernel_l1 = get_tuner_values(hp, f"layer-kernel-l1", **kwargs)
            kernel_l2 = get_tuner_values(hp, f"layer-kernel-l2", **kwargs)
            bias_l1 = get_tuner_values(hp, f"layer-bias-l1", **kwargs)
            bias_l2 = get_tuner_values(hp, f"layer-bias-l2", **kwargs)
            activity_l1 = get_tuner_values(hp, f"layer-activity-l1", **kwargs)
            activity_l2 = get_tuner_values(hp, f"layer-activity-l2", **kwargs)
            for i, kernel_size in enumerate(convolution_sizes):
                convolutions.append(
                    tf.keras.layers.Conv1D(
                        filters=filters,
                        kernel_size=kernel_size,
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
                    )(next_layer)
                )
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
                hidden = tf.keras.layers.Dense(
                    units=layer_size,
                    activation=get_tuner_activation(
                        hp, "fnn-layer-activation", **kwargs
                    ),
                )(hidden)
            outputs = self.get_output_layer()(hidden)
            model = tf.keras.Model(inputs=[inputs], outputs=outputs)

            # Compile model
            model.compile(
                optimizer=get_tuner_optimizer(hp, **kwargs),
                loss=self._get_tuner_loss_function(hp, **kwargs),
                metrics=self.get_metric_list(),
            )
            return model

        class TunerCNN(keras_tuner.HyperModel):
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

        return TunerCNN(), input_layer.shape

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
            default=1,
            minimum=1,
            maximum=max_convolutions,
            name="number-of-convolutions",
            description="Number of different convolutions to use",
        )
        kernel_sizes = {
            f"kernel-{i}-size": IntArgument(
                minimum=1,
                default=4,
                maximum=512,
                name=f"kernel-{i}-size",
                description="Size of the i-th convolution kernel.",
            )
            for i in range(1, max_convolutions + 1)
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
                    "swish",
                ],
                name=f"layer-activation",
                description="Activation to use in the cnn layers",
            )
        }
        activation_alpha = {
            f"layer-activation-alpha": FloatArgument(
                default=0.0,
                name=f"layer-activation-alpha",
                description=f"Alpha value for the elu activation of the layers",
            ),
            f"fnn-layer-activation-alpha": FloatArgument(
                default=0.0,
                name=f"fnn-layer-activation-alpha",
                description=f"Alpha value for the elu activation of the fnn layer",
            ),
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
                "fully-connected-layer-size": IntArgument(
                    default=32,
                    minimum=0,
                    maximum=16384,
                    name="fully-connected-layer-size",
                    description="Size of the fully connected layer. Set to 0 to disable.",
                ),
                "number-of-convolutions": num_convolutions,
                "filters": IntArgument(
                    default=32,
                    minimum=1,
                    maximum=64,
                    name="filters",
                    description="Number of filters per convolution layer.",
                ),
                "fnn-layer-activation": EnumArgument(
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
                    name="fnn-layer-activation",
                    description="Activation to use in the fully connected layer.",
                )
                # 'pooling_size': HyperParameter(
                #     default=2, minimum=2, maximum=16
                # ),
            }
            | kernel_sizes
            | activations
            | activation_alpha
            | regularizers
            | super().get_arguments()
        )
