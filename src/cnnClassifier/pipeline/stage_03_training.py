from src.cnnClassifier.config.configuration import ConfigurationManager
from src.cnnClassifier.components.prepare_callbacks import CallbacksPreparer
from src.cnnClassifier.components.training import Trainer
from src.cnnClassifier import logger

STAGE_NAME = "Training"


class ModelTrainingPipeline:
    def __init__(self, model, train_inputs, val_inputs, train_labels, val_labels):
        self.model = model
        self.train_inputs = train_inputs
        self.val_inputs = val_inputs
        self.train_labels = train_labels
        self.val_labels = val_labels

    def main(self):
        config = ConfigurationManager()
        prepare_callbacks_config = config.get_callbacks_preparation_config().to_dict()

        """ prepare_callbacks = CallbacksPreparer(**prepare_callbacks_config)
        callback_list = prepare_callbacks.prepare_callbacks() """

        training_config = config.get_training_config().to_dict()
        print(f"training_config")
        trainer = Trainer(self.model, self.train_inputs, self.val_inputs)

        training = trainer.train(training_config['epochs'])


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
