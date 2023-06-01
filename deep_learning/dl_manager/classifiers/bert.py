import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification

from ..config import Argument
from .model import AbstractModel
from ..model_io import InputEncoding

class Bert(AbstractModel):
    def get_model(self, *,
                  embedding=None,
                  embedding_size: int | None = None,
                  embedding_output_size: int | None = None,
                  **kwargs) -> tf.keras.Model:
        model = TFAutoModelForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=self.number_of_outputs
        )
        # We freeze the first 10 layers of Bert
        for idx in range(10):
            model.bert.encoder.layer[idx].trainable = False
        model.classifier.activation = tf.keras.activations.sigmoid
        return model

    @staticmethod
    def supported_input_encodings() -> list[InputEncoding]:
        return [
            InputEncoding.Text
        ]

    @staticmethod
    def input_must_support_convolution() -> bool:
        return False

    @classmethod
    def get_arguments(cls) -> dict[str, Argument]:
        return {} | super().get_arguments()
