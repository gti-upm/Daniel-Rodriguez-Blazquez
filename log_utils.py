import logz
import keras


class MyCallback(keras.callbacks.Callback):
    """
    Customized callback class.

    # Arguments
       filepath: Path to save model.
       period: Frequency in epochs with which model is saved.
       batch_size: Number of images per batch.
    """

    def __init__(self, filepath):
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs={}):
        # Save training and validation losses
        logz.log_tabular('train_loss', logs.get('loss'))
        logz.log_tabular('val_loss', logs.get('val_loss'))
        logz.dump_tabular()
