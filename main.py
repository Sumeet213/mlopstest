from cnnClassifier import logger
from src.cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from src.cnnClassifier.pipeline.stage_03_training import ModelTrainingPipeline
from src.cnnClassifier.pipeline.stage_04_evaluation import EvaluationPipeline
from src.cnnClassifier.config.configuration import ConfigurationManager

if __name__ == "__main__":
    try:
        # Data ingestion stage
        STAGE_NAME = "Data Ingestion stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_ingestion = DataIngestionTrainingPipeline()
        train_inputs, val_inputs, train_labels, val_labels = data_ingestion.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        # Prepare base model stage
        STAGE_NAME = "Prepare base model"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        config = ConfigurationManager()
        base_model_config = config.get_base_model_preparation_config()
        prepare_base_model = PrepareBaseModelTrainingPipeline(base_model_config, train_inputs, val_inputs, train_labels, val_labels)
        model = prepare_base_model.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        # Training stage
        STAGE_NAME = "Training"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model_trainer = ModelTrainingPipeline(model, train_inputs, val_inputs, train_labels, val_labels)
        model_trainer.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        # Evaluation stage
        STAGE_NAME = "Evaluation stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model_evalution = EvaluationPipeline(model, val_inputs, val_labels)
        model_evalution.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

    except Exception as e:
        logger.exception(e)
        raise e
