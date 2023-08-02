from src.cnnClassifier.config.configuration import ConfigurationManager
from src.cnnClassifier.components.prepare_base_model import BaseModelPreparer
from cnnClassifier import logger

STAGE_NAME = "Prepare Base Model stage"


class PrepareBaseModelTrainingPipeline:
    def __init__(self, base_model_config, train_inputs, val_inputs, train_labels, val_labels):
        self.base_model_config = base_model_config
        self.train_inputs = train_inputs
        self.val_inputs = val_inputs
        self.train_labels = train_labels
        self.val_labels = val_labels

    def main(self):
        print(f"something3")
        config = ConfigurationManager()
        prepare_base_model_config = config.get_base_model_preparation_config().to_dict()
        print(f"something1")

        base_model_preparer = BaseModelPreparer(**prepare_base_model_config)
        print(f"something")
        model = base_model_preparer.prepare_base_model()
        print(f"numbers")

        return model


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
