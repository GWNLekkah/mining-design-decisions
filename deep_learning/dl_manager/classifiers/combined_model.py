from abc import ABC

import tensorflow as tf

from ..config import Config
from .fully_connected_model import FullyConnectedModel
from ..model_io import InputEncoding, OutputMode


class CombinedModel(FullyConnectedModel):
    pass


combined_models = {"CombinedModel": CombinedModel}


def get_simple_combinator(conf: Config):
    match conf.get("run.combination-strategy"):
        case "add":
            return tf.keras.layers.add
        case "subtract":
            return tf.keras.layers.subtract
        case "average":
            return tf.keras.layers.average
        case "multiply":
            return tf.keras.layers.multiply
        case "max":
            return tf.keras.layers.maximum
        case "min":
            return tf.keras.layers.minimum
        case "dot":
            return tf.keras.layers.dot
        case "concat":
            return tf.keras.layers.concatenate
        case _:
            return None


def calculate_input_size(conf: Config, n):
    output_mode = OutputMode.from_string(conf.get("run.output-mode"))
    match conf.get("run.combination-strategy"):
        case "add":
            return output_mode.output_size
        case "subtract":
            return output_mode.output_size
        case "average":
            return output_mode.output_size
        case "multiply":
            return output_mode.output_size
        case "max":
            return output_mode.output_size
        case "min":
            return output_mode.output_size
        case "dot":
            return 1
        case "concat":
            return output_mode.output_size * n
        case _:
            return None


def combine_models(models, conf: Config, **params) -> tf.keras.Model:
    """Replacement for AbstractModel.get_model(), which
    combines multiple models into one.
    """
    if len(models) < 2:
        raise ValueError("Expected at least 2 models")
    if params["number-of-hidden-layers"] < 1:
        raise ValueError("Expected at least 1 hidden layer")

    combiner = get_simple_combinator(conf)
    if combiner is None:
        raise NotImplementedError
    input_size = calculate_input_size(conf, len(models))

    output_mode = OutputMode.from_string(conf.get("run.output-mode"))
    model_builder = CombinedModel(
        input_size=input_size,
        input_encoding=InputEncoding.Vector,
        output_encoding=output_mode.output_encoding,
        number_of_outputs=output_mode.output_size,
    )

    hidden = combiner([model.output for model in models])
    for i in range(1, params["number-of-hidden-layers"] + 1):
        layer_size = params[f"hidden-layer-{i}-size"]
        hidden = tf.keras.layers.Dense(layer_size)(hidden)
        if (act := params[f"layer-{i}-activation"]) != "linear":
            hidden = model_builder.get_activation(act)(hidden)

    outputs = model_builder.get_output_layer()(hidden)

    combined_model = tf.keras.Model(
        inputs=[model.inputs for model in models], outputs=outputs
    )
    combined_model.compile(
        optimizer=model_builder.get_optimizer(**params),
        loss=model_builder.get_loss(**params),
        metrics=model_builder.get_metric_list(),
    )
    return combined_model


def tuner_combine_models(models, conf: Config, **params) -> tf.keras.Model:
    """
    Creates a combined model suitable for Keras Tuner.
    """
    if len(models) < 2:
        raise ValueError("Expected at least 2 models")
    if params["number-of-hidden-layers"]["type"] == "values":
        if min(params["number-of-hidden-layers"]["options"]["values"]) < 1:
            raise ValueError("Expected at least 1 hidden layer")
    else:
        if min(params["number-of-hidden-layers"]["options"]["start"]) < 1:
            raise ValueError("Expected at least 1 hidden layer")

    for idx, model in enumerate(models):
        for layer in model.layers:
            layer._name = f"{layer.name}_{idx}"

    def get_model(hp):
        combiner = get_simple_combinator(conf)
        if combiner is None:
            raise NotImplementedError
        input_size = calculate_input_size(conf, len(models))

        output_mode = OutputMode.from_string(conf.get("run.output-mode"))
        model_builder = CombinedModel(
            input_size=input_size,
            input_encoding=InputEncoding.Vector,
            output_encoding=output_mode.output_encoding,
            number_of_outputs=output_mode.output_size,
        )

        hidden = combiner([model.output for model in models])
        n_hidden_layers = model_builder._get_values(
            hp, "number-of-hidden-layers", **params
        )
        for i in range(1, n_hidden_layers + 1):
            layer_size = model_builder._get_values(
                hp, f"hidden-layer-{i}-size", **params
            )
            activation = model_builder._get_values(
                hp, f"layer-{i}-activation", **params
            )
            hidden = tf.keras.layers.Dense(units=layer_size, activation=activation)(
                hidden
            )

        outputs = model_builder.get_output_layer()(hidden)

        combined_model = tf.keras.Model(
            inputs=[model.inputs for model in models], outputs=outputs
        )
        combined_model.compile(
            optimizer=model_builder._get_tuner_optimizer(hp, **params),
            loss=model_builder._get_tuner_loss_function(hp, **params),
            metrics=model_builder.get_metric_list(),
        )
        return combined_model

    return get_model, [model.inputs for model in models]
