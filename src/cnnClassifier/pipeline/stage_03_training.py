from src.cnnClassifier.config.configuration import ConfigurationManager
from src.cnnClassifier.components.prepare_callbacks import CallbacksPreparer
from src.cnnClassifier.components.training import Trainer
from src.cnnClassifier import logger

STAGE_NAME = "Training"


class ModelTrainingPipeline:
    def __init__(self, model, train_dataset, val_dataset):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def main(self):
        config = ConfigurationManager()
        prepare_callbacks_config = config.get_callbacks_preparation_config().to_dict()

        """ prepare_callbacks = CallbacksPreparer(**prepare_callbacks_config)
        callback_list = prepare_callbacks.prepare_callbacks() """

        training_config = config.get_training_config().to_dict()
        print(f"training_config")
        trainer = Trainer(self.model, self.train_dataset, self.val_dataset)

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
