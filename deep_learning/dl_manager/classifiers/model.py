"""
This module defines a base class for all future models.
"""

##############################################################################
##############################################################################
# Imports
##############################################################################

import abc

import numpy
import tensorflow as tf
import tensorflow_addons as tfa

from ..config import (
    Argument,
    BoolArgument,
    StringArgument,
    EnumArgument,
    FloatArgument,
    ArgumentConsumer,
    IntArgument,
    NestedArgument,
)
from ..model_io import InputEncoding, OutputEncoding


##############################################################################
##############################################################################
# Tuner functions
##############################################################################


class CustomHinge(tf.keras.losses.Hinge):
    def call(self, y_true, y_pred):
        return super().call(y_true, 2 * y_pred - 1)


def get_tuner_values(hp, arg, **kwargs):
    arg_values = kwargs[arg]
    if arg_values["type"] == "range":
        return hp.Int(
            arg,
            min_value=arg_values["options"]["start"],
            max_value=arg_values["options"]["stop"],
            step=arg_values["options"]["step"],
            sampling=arg_values["options"]["sampling"],
        )
    elif arg_values["type"] == "values":
        return hp.Choice(arg, arg_values["options"]["values"])
    elif arg_values["type"] == "floats":
        return hp.Float(
            arg,
            min_value=arg_values["options"]["start"],
            max_value=arg_values["options"]["stop"],
            step=arg_values["options"]["step"],
            sampling=arg_values["options"]["sampling"],
        )


def get_tuner_activation(activation, activation_alpha):
    if activation == "elu":
        return tf.keras.layers.ELU(alpha=activation_alpha)
    elif activation == "leakyrelu":
        return tf.keras.layers.LeakyReLU(alpha=activation_alpha)
    elif activation == "prelu":
        return tf.keras.layers.PReLU()
    return activation


def get_tuner_learning_rate_scheduler(hp, **kwargs):
    initial_learning_rate = get_tuner_values(hp, "learning-rate-start", **kwargs)
    end_learning_rate = get_tuner_values(hp, "learning-rate-stop", **kwargs)
    # if kwargs["learning-rate-start"]["type"] in ["range", "floats"]:
    #     old_max = kwargs["learning-rate-stop"]["options"]["stop"]
    #     kwargs["learning-rate-stop"]["options"]["stop"] = initial_learning_rate
    #     end_learning_rate = get_tuner_values(hp, "learning-rate-stop", **kwargs)
    #     kwargs["learning-rate-stop"]["options"]["stop"] = old_max
    # elif kwargs["learning-rate-start"]["type"] == "values":
    #     old_values = kwargs["learning-rate-stop"]["options"]["values"]
    #     available_values = []
    #     for value in kwargs["learning-rate-stop"]["options"]["values"]:
    #         if value <= initial_learning_rate:
    #             available_values.append(value)
    #     kwargs["learning-rate-stop"]["options"]["values"] = available_values
    #     end_learning_rate = get_tuner_values(hp, "learning-rate-stop", **kwargs)
    #     kwargs["learning-rate-stop"]["options"]["values"] = old_values

    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=get_tuner_values(hp, "learning-rate-steps", **kwargs),
        end_learning_rate=end_learning_rate,
        power=get_tuner_values(hp, "learning-rate-power", **kwargs),
        cycle=False,
        name=None,
    )
    return lr_schedule


def get_tuner_optimizer(hp, **kwargs):
    optimizer = get_tuner_values(hp, "optimizer", **kwargs)
    params = kwargs["optimizer-params"]
    if optimizer not in params:
        raise ValueError(
            f"Mismatch between params and optimizer: {optimizer}, {params}"
        )
    params = params[optimizer][0]
    learning_rate = get_tuner_learning_rate_scheduler(hp, **kwargs)
    match optimizer:
        case "adam":
            return tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=1 - get_tuner_values(hp, "beta-1", **params),
                beta_2=1 - get_tuner_values(hp, "beta-2", **params),
                epsilon=get_tuner_values(hp, "epsilon", **params),
                weight_decay=get_tuner_values(hp, "weight-decay", **params),
            )
        case "nadam":
            return tf.keras.optimizers.Nadam(
                learning_rate=learning_rate,
                beta_1=1 - get_tuner_values(hp, "beta-1", **params),
                beta_2=1 - get_tuner_values(hp, "beta-2", **params),
                epsilon=get_tuner_values(hp, "epsilon", **params),
                weight_decay=get_tuner_values(hp, "weight-decay", **params),
            )
        case "adamw":
            return tf.keras.optimizers.experimental.AdamW(
                learning_rate=learning_rate,
                beta_1=1 - get_tuner_values(hp, "beta-1", **params),
                beta_2=1 - get_tuner_values(hp, "beta-2", **params),
                epsilon=get_tuner_values(hp, "epsilon", **params),
                weight_decay=get_tuner_values(hp, "weight-decay", **params),
            )
        case "sgd":
            return tf.keras.optimizers.SGD(
                learning_rate=learning_rate,
                momentum=params["momentum"],
                nesterov=params["use-nesterov"],
            )
        case _ as x:
            raise ValueError(f"Invalid optimiser: {x}")


##############################################################################
##############################################################################
# Functions for all models
##############################################################################


def get_activation(key, **kwargs):
    activation = kwargs[key]
    if activation == "elu":
        return tf.keras.layers.ELU(alpha=kwargs[f"{key}-alpha"])
    elif activation == "leakyrelu":
        return tf.keras.layers.LeakyReLU(alpha=kwargs[f"{key}-alpha"])
    elif activation == "prelu":
        return tf.keras.layers.PReLU()
    return activation


##############################################################################
##############################################################################
# Main class
##############################################################################


class AbstractModel(abc.ABC, ArgumentConsumer):
    def __init__(
        self,
        input_size: int | tuple[int],
        input_encoding: InputEncoding,
        number_of_outputs: int,
        output_encoding: OutputEncoding,
    ):
        self.__n_inputs = input_size
        self.__input_encoding = input_encoding
        self.__n_outputs = number_of_outputs
        self.__output_encoding = output_encoding
        # match self.__input_encoding:
        #    case InputEncoding.Matrix:
        #        self.__check_input_size_type(tuple)
        #    case _:
        #        self.__check_input_size_type(int)

    def __check_input_size_type(self, expected_type):
        if not isinstance(self.input_size, expected_type):
            message = (
                f"Invalid input size type for input encoding "
                f"{self.input_encoding}: "
                f"{self.input_size.__class__.__name__}"
            )
            raise ValueError(message)

    # ================================================================
    # Attributes

    @property
    def input_size(self) -> int | tuple[int]:
        return self.__n_inputs

    @property
    def input_encoding(self) -> InputEncoding:
        return self.__input_encoding

    @property
    def number_of_outputs(self) -> int:
        return self.__n_outputs

    @property
    def output_encoding(self) -> OutputEncoding:
        return self.__output_encoding

    # ================================================================
    # Abstract Methods

    @abc.abstractmethod
    def get_model(
        self,
        *,
        embedding=None,
        embedding_size: int | None = None,
        embedding_output_size: int | None = None,
        **kwargs,
    ) -> tf.keras.Model:
        """Build and return the (not compiled) model.

        Note that input and output layers must also be
        added in this method. This can be done using the
        auxiliary functions get_input_layer and
        get_output_layer.
        """

    @classmethod
    @abc.abstractmethod
    def get_arguments(cls) -> dict[str, Argument]:
        """Return the names of all the hyper-parameters,
        possibly with a suggestion for the range of possible values.

        Remember to call super() when implementing
        """
        result = {
            "optimizer": StringArgument(
                default="adam",
                description="Optimizer to use. Special case: use sgd_XXX to specify SGD with momentum XXX",
                name="optimizer",
            ),
            "optimizer-params": NestedArgument(
                name="optimizer-params",
                description="Hyper-parameters for the optimizer",
                spec={
                    "adam": {
                        "beta-1": FloatArgument(
                            name="beta-1",
                            description="Beta-1 value for the Adam optimizer",
                            default=0.9,
                        ),
                        "beta-2": FloatArgument(
                            name="beta-2",
                            description="Beta-2 value for the Adam optimizer",
                            default=0.999,
                        ),
                        "epsilon": FloatArgument(
                            name="epsilon",
                            description="Epsilon value for the Adam optimizer",
                            default=1e-07,
                        ),
                        "weight-decay": FloatArgument(
                            name="weight-decay",
                            description="Weight decay",
                            default=0.0,
                        ),
                    },
                    "nadam": {
                        "beta-1": FloatArgument(
                            name="beta-1",
                            description="Beta-1 value for the Nadam optimizer",
                            default=0.9,
                        ),
                        "beta-2": FloatArgument(
                            name="beta-2",
                            description="Beta-2 value for the Nadam optimizer",
                            default=0.999,
                        ),
                        "epsilon": FloatArgument(
                            name="epsilon",
                            description="Epsilon value for the Nadam optimizer",
                            default=1e-07,
                        ),
                        "weight-decay": FloatArgument(
                            name="weight-decay",
                            description="Weight decay",
                            default=0.0,
                        ),
                    },
                    "adamw": {
                        "beta-1": FloatArgument(
                            name="beta-1",
                            description="Beta-1 value for the Nadam optimizer",
                            default=0.9,
                        ),
                        "beta-2": FloatArgument(
                            name="beta-2",
                            description="Beta-2 value for the Nadam optimizer",
                            default=0.999,
                        ),
                        "epsilon": FloatArgument(
                            name="epsilon",
                            description="Epsilon value for the Nadam optimizer",
                            default=1e-07,
                        ),
                        "weight-decay": FloatArgument(
                            name="weight-decay",
                            description="Weight decay",
                            default=0.0,
                        ),
                    },
                    "sgd": {
                        "momentum": FloatArgument(
                            name="momentum",
                            description="Momentum value for the SGD optimizer",
                            minimum=0.0,
                            maximum=1.0,
                            default=0.0,
                        ),
                        "use-nesterov": BoolArgument(
                            name="use-nesterov",
                            description="Whether to use Nesterov momentum in the SGD optimizer",
                            default=False,
                        ),
                    },
                },
            ),
            "loss": EnumArgument(
                default="crossentropy",
                description="Loss to use in the training process",
                options=["crossentropy", "hinge"],
                name="loss",
            ),
            "learning-rate-start": FloatArgument(
                default=0.005,
                minimum=0.0,
                name="learning-rate-start",
                description="Initial learning rate for the learning process",
            ),
            "learning-rate-stop": FloatArgument(
                default=0.0005,
                minimum=0.0,
                name="learning-rate-stop",
                description='Learnign rate after "learning-rate-steps" steps',
            ),
            "learning-rate-steps": IntArgument(
                default=470,
                minimum=1,
                name="learning-rate-steps",
                description="Amount of decay steps requierd to go from start to stop LR",
            ),
            "learning-rate-power": FloatArgument(
                default=1.0,
                minimum=0.0,
                name="learning-rate-power",
                description="Degree of the polynomial to use for the learning rate.",
            ),
            "batch-size": IntArgument(
                default=32,
                minimum=1,
                name="batch-size",
                description="Batch size used during training",
            ),
        }
        result |= {
            "use-trainable-embedding": BoolArgument(
                default=False,
                name="use-trainable-embedding",
                description="Whether to make the word-embedding trainable.",
            )
        }
        return result

    @staticmethod
    @abc.abstractmethod
    def supported_input_encodings() -> list[InputEncoding]:
        """List of supported input encodings."""

    @staticmethod
    @abc.abstractmethod
    def input_must_support_convolution() -> bool:
        pass

    # ================================================================
    # Auxiliary Methods for Model Creation

    def get_input_layer(
        self,
        *,
        embedding=None,
        embedding_size: int | None = None,
        embedding_output_size: int | None = None,
        trainable_embedding: bool = False,
    ) -> (tf.keras.layers.Layer, tf.keras.layers.Layer):
        match self.__input_encoding:
            case InputEncoding.Vector:
                if self.input_must_support_convolution():
                    inputs = tf.keras.layers.Input(shape=(self.__n_inputs, 1))
                else:
                    inputs = tf.keras.layers.Input(shape=(self.__n_inputs,))
                return inputs, inputs
            case InputEncoding.Matrix:
                if self.input_must_support_convolution():
                    inputs = tf.keras.layers.Input(shape=tuple(self.__n_inputs) + (1,))
                else:
                    inputs = tf.keras.layers.Input(shape=self.__n_inputs)
                return inputs, inputs
            case InputEncoding.Embedding:
                assert embedding is not None
                assert embedding_size is not None
                if self.input_must_support_convolution():
                    shape = (self.__n_inputs,)
                    inputs = tf.keras.layers.Input(shape=shape)
                else:
                    shape = (self.__n_inputs,)
                    inputs = tf.keras.layers.Input(shape=shape)
                return inputs, tf.keras.layers.Embedding(
                    embedding_size,
                    embedding_output_size,
                    weights=[numpy.asarray(embedding)],
                    input_shape=shape,
                    trainable=trainable_embedding,
                )(inputs)
            case InputEncoding.Text:
                inputs = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
                return inputs, inputs

    def get_output_layer(self) -> tf.keras.layers.Layer:
        match self.__output_encoding:
            case OutputEncoding.Binary:
                return tf.keras.layers.Dense(self.__n_outputs, activation="sigmoid")
            case OutputEncoding.OneHot:
                return tf.keras.layers.Dense(self.__n_outputs, activation="softmax")

    # ================================================================
    # Optimizer Configuration

    def get_learning_rate_scheduler(self, **kwargs):
        if abs(kwargs["learning-rate-start"] - kwargs["learning-rate-stop"]) <= 1e-10:
            return kwargs["learning-rate-start"]
        lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=kwargs["learning-rate-start"],
            decay_steps=kwargs["learning-rate-steps"],
            end_learning_rate=kwargs["learning-rate-stop"],
            power=kwargs["learning-rate-power"],
            cycle=False,
            name=None,
        )
        return lr_schedule

    def get_optimizer(self, **kwargs) -> tf.keras.optimizers.Optimizer:
        optimizer = kwargs["optimizer"]
        params = kwargs["optimizer-params"]
        if optimizer not in params:
            raise ValueError(
                f"Mismatch between params and optimizer: {optimizer}, {params}"
            )
        params = params[optimizer][0]
        learning_rate = self.get_learning_rate_scheduler(**kwargs)
        match optimizer:
            case "adam":
                return tf.keras.optimizers.Adam(
                    learning_rate=learning_rate,
                    beta_1=params["beta-1"],
                    beta_2=params["beta-2"],
                    epsilon=params["epsilon"],
                    weight_decay=params["weight-decay"],
                )
            case "nadam":
                return tf.keras.optimizers.Nadam(
                    learning_rate=learning_rate,
                    beta_1=params["beta-1"],
                    beta_2=params["beta-2"],
                    epsilon=params["epsilon"],
                    weight_decay=params["weight-decay"],
                )
            case "adamw":
                return tf.keras.optimizers.experimental.AdamW(
                    learning_rate=learning_rate,
                    beta_1=params["beta-1"],
                    beta_2=params["beta-2"],
                    epsilon=params["epsilon"],
                    weight_decay=params["weight-decay"],
                )
            case "sgd":
                return tf.keras.optimizers.SGD(
                    learning_rate=learning_rate,
                    momentum=params["momentum"],
                    nesterov=params["use-nesterov"],
                )
            case _ as x:
                raise ValueError(f"Invalid optimiser: {x}")

    # ================================================================
    # Model Building Functionality

    def get_compiled_model(
        self,
        *,
        embedding=None,
        embedding_size: int | None = None,
        embedding_output_size: int | None = None,
        **kwargs,
    ):
        model = self.get_model(
            embedding=embedding,
            embedding_size=embedding_size,
            embedding_output_size=embedding_output_size,
            **kwargs,
        )
        model.compile(
            optimizer=self.get_optimizer(**kwargs),
            loss=self.__get_loss_function(**kwargs),
            metrics=self.get_metric_list(),
        )
        return model

    def get_metric_list(self):
        return [
            tf.keras.metrics.TruePositives(thresholds=0.5, name="true_positives"),
            tf.keras.metrics.TrueNegatives(thresholds=0.5, name="true_negatives"),
            tf.keras.metrics.FalsePositives(thresholds=0.5, name="false_positives"),
            tf.keras.metrics.FalseNegatives(thresholds=0.5, name="false_negatives"),
            self.__get_accuracy(),
            # Precision and recall use thresholds=0.5 by default
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tfa.metrics.F1Score(
                num_classes=self.__n_outputs,
                threshold=0.5
                if self.__output_encoding != OutputEncoding.OneHot
                else None,
                name="f_score_tf_macro",
                average="macro",
            ),  # one_hot=self.__output_encoding == OutputEncoding.OneHot
        ]

    def get_loss(self, **kwargs):
        return self.__get_loss_function(**kwargs)

    def __get_loss_function(self, **kwargs):
        loss = kwargs["loss"]
        match self.__output_encoding:
            case OutputEncoding.OneHot:
                if loss == "crossentropy":
                    return tf.keras.losses.CategoricalCrossentropy()
                elif loss == "hinge":
                    return tf.keras.losses.CategoricalHinge()
                else:
                    raise ValueError(f"Invalid loss: {loss}")
            case OutputEncoding.Binary:
                if loss == "crossentropy":
                    return tf.keras.losses.BinaryCrossentropy()
                elif loss == "hinge":
                    return tf.keras.losses.Hinge()
                else:
                    raise ValueError(f"Invalid loss: {loss}")

    def __get_accuracy(self):
        match self.__output_encoding:
            case OutputEncoding.OneHot:
                return tf.keras.metrics.CategoricalAccuracy(name="accuracy")
            case OutputEncoding.Binary:
                return tf.keras.metrics.BinaryAccuracy(name="accuracy")

    get_accuracy = __get_accuracy

    ##############################################################################
    ##############################################################################
    # Tuner functions
    ##############################################################################
    def _get_tuner_loss_function(self, hp, **kwargs):
        loss = get_tuner_values(hp, "loss", **kwargs)
        match self.__output_encoding:
            case OutputEncoding.OneHot:
                if loss == "crossentropy":
                    return tf.keras.losses.CategoricalCrossentropy()
                elif loss == "hinge":
                    return tf.keras.losses.CategoricalHinge()
                else:
                    raise ValueError(f"Invalid loss: {loss}")
            case OutputEncoding.Binary:
                if loss == "crossentropy":
                    return tf.keras.losses.BinaryCrossentropy()
                elif loss == "hinge":
                    return CustomHinge()
                else:
                    raise ValueError(f"Invalid loss: {loss}")
