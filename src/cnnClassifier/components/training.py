import tensorflow as tf
from tensorflow import keras
from keras import optimizers
from keras.optimizers import Adam


class Trainer:
    def __init__(self, model, train_dataset, val_dataset):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train(self, epochs):
        print(f"MAYBE HERE???????????//")
        # Compile the model with a optimizer, loss, and metrics
        self.model.compile(optimizer=Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0),
                           # Using binary_crossentropy for multi-label classification
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

        # Train the model
        history = self.model.fit(
            self.train_dataset, epochs=epochs, validation_data=self.val_dataset, verbose=2)

        return "test"
