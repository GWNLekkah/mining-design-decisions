import tensorflow as tf


from ..config import Config


def get_simple_combinator(conf: Config):
    match conf.get('run.combination-strategy'):
        case 'add':
            return tf.keras.layers.add
        case 'subtract':
            return tf.keras.layers.subtract
        case 'multiply':
            return tf.keras.layers.multiply
        case 'max':
            return tf.keras.layers.maximum
        case 'min':
            return tf.keras.layers.minimum
        case 'dot':
            return tf.keras.layers.dot
        case 'concat':
            return tf.keras.layers.concatenate
        case _:
            return None


def combine_models(parent_model, *models,
                   fully_connected_layers=(None,),
                   conf: Config) -> tf.keras.Model:
    """Replacement for AbstractModel.get_model(), which
    combines multiple models into one.
    """
    assert len(models) >= 2
    instances = models
    combiner = get_simple_combinator(conf)
    if combiner is None:
        raise NotImplementedError
    hidden = combiner([model.output for model in instances])
    for layer_size in fully_connected_layers:
        if layer_size is None:
            layer_size = 1
        hidden = tf.keras.layers.Dense(layer_size)(hidden)
    outputs = parent_model.get_output_layer()(hidden)
    model = tf.keras.Model(inputs=[instance.inputs for instance in instances],
                           outputs=outputs)
    model.compile(optimizer=parent_model.get_optimizer(),
                  loss=parent_model.get_loss(),
                  metrics=parent_model.get_metric_list())
    return model
