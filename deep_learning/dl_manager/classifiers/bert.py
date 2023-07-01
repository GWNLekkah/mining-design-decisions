import keras_tuner
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification

from .model import AbstractModel, get_tuner_values, get_tuner_optimizer
from ..config import Argument, IntArgument
from ..model_io import InputEncoding


class Bert(AbstractModel):
    def get_model(
        self,
        *,
        embedding=None,
        embedding_size: int | None = None,
        embedding_output_size: int | None = None,
        **kwargs,
    ) -> tf.keras.Model:
        model = TFAutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=self.number_of_outputs
        )
        # We can freeze the first layers of Bert
        num_frozen_layers = kwargs["number-of-frozen-layers"]
        for idx in range(num_frozen_layers):
            model.bert.encoder.layer[idx].trainable = False
        model.classifier.activation = tf.keras.activations.sigmoid
        return model

    def get_keras_tuner_model(
        self,
        *,
        embedding=None,
        embedding_size: int | None = None,
        embedding_output_size: int | None = None,
        **kwargs,
    ):
        def get_model(hp):
            model = TFAutoModelForSequenceClassification.from_pretrained(
                "bert-base-uncased", num_labels=self.number_of_outputs
            )
            # Freeze layers
            num_frozen_layers = get_tuner_values(
                hp, "number-of-frozen-layers", **kwargs
            )
            for idx in range(num_frozen_layers):
                model.bert.encoder.layer[idx].trainable = False
            model.classifier.activation = tf.keras.activations.sigmoid

            # Compile model
            model.compile(
                optimizer=get_tuner_optimizer(hp, **kwargs),
                loss=self._get_tuner_loss_function(hp, **kwargs),
                metrics=self.get_metric_list(),
            )

            return model

        class TunerBert(keras_tuner.HyperModel):
            def __init__(self):
                self.batch_size = None

            def build(self, hp):
                self.batch_size = get_tuner_values(hp, "batch-size", **kwargs)
                return get_model(hp)

            def fit(self, hp, model, *args, **kwargs_):
                return model.fit(*args, batch_size=self.batch_size, **kwargs_)

        return TunerBert(), None

    @staticmethod
    def supported_input_encodings() -> list[InputEncoding]:
        return [InputEncoding.Text]

    @staticmethod
    def input_must_support_convolution() -> bool:
        return False

    @classmethod
    def get_arguments(cls) -> dict[str, Argument]:
        return {
            "number-of-frozen-layers": IntArgument(
                default=10,
                minimum=0,
                maximum=12,
                name="number-of-frozen-layers",
                description="Number of layers to freeze.",
            )
        } | super().get_arguments()
