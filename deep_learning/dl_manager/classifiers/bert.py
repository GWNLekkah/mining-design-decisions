import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from .model import AbstractModel, HyperParameter, InputEncoding, _fix_hyper_params


class Bert(AbstractModel):
    def get_model(self, *,
                  embedding=None,
                  embedding_size: int | None = None,
                  embedding_output_size: int | None = None,
                  **kwargs) -> tf.keras.Model:
        inputs, next_layer = self.get_input_layer()
        preprocessing_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
                                             name='preprocessing')
        encoder_inputs = preprocessing_layer(inputs)
        encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/2',
                                 trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        hidden = outputs['pooled_output']
        hidden = tf.keras.layers.Dropout(0.1)(hidden)
        hidden = tf.keras.layers.Dense(8)(hidden)
        outputs = self.get_output_layer()(hidden)
        return tf.keras.Model(inputs=[inputs], outputs=outputs)

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
