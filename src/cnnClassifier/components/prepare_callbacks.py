import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


class CallbacksPreparer:
    def __init__(self, tensorboard_log_dir, checkpoint_filepath):
        self.tensorboard_log_dir = tensorboard_log_dir
        self.checkpoint_filepath = checkpoint_filepath

    def prepare_callbacks(self):
        # Prepare the TensorBoard callback
        tensorboard_callback = TensorBoard(
            log_dir=self.tensorboard_log_dir, histogram_freq=1)

        # Prepare the ModelCheckpoint callback
        checkpoint_callback = ModelCheckpoint(
            filepath=self.checkpoint_filepath, save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True)

        return [tensorboard_callback, checkpoint_callback]
