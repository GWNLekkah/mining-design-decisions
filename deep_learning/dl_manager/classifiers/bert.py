import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification

from .model import AbstractModel, HyperParameter, InputEncoding, _fix_hyper_params


class Bert(AbstractModel):
    def get_model(self, *,
                  embedding=None,
                  embedding_size: int | None = None,
                  embedding_output_size: int | None = None,
                  **kwargs) -> tf.keras.Model:
        model = TFAutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
        # We freeze the first 10 layers of Bert
        for idx in range(10):
            model.bert.encoder.layer[idx].trainable = False
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
    @_fix_hyper_params
    def get_hyper_parameters(cls) -> dict[str, HyperParameter]:
        return {} | super(Bert, Bert).get_hyper_parameters()
