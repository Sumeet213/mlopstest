from transformers import TFBertModel
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model


class BaseModelPreparer:
    def __init__(self, base_model_name, intents):
        self.base_model_name = base_model_name
        self.intents = intents

    def prepare_base_model(self):
        # Load the BERT model
        bert_model = TFBertModel.from_pretrained(self.base_model_name)

        # Define the model
        input_ids = tf.keras.Input(shape=(512,), dtype=tf.int32)
        sequence_output = bert_model(input_ids)[0]
        clf_output = sequence_output[:, 0, :]
        # Using sigmoid activation for multi-label classification
        out = Dense(len(self.intents), activation='sigmoid')(clf_output)

        model = Model(inputs=input_ids, outputs=out)

        return model
