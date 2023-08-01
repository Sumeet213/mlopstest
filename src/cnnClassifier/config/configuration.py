class DataIngestionConfig:
    def __init__(self, source_file, intents_dict, threshold):
        self.source_file = source_file
        self.intents_dict = intents_dict
        self.threshold = threshold


class BaseModelPreparationConfig:
    def __init__(self, base_model_name, intents):
        self.base_model_name = base_model_name
        self.intents = intents


class CallbacksPreparationConfig:
    def __init__(self, tensorboard_log_dir, checkpoint_filepath):
        self.tensorboard_log_dir = tensorboard_log_dir
        self.checkpoint_filepath = checkpoint_filepath


class TrainingConfig:
    def __init__(self, epochs):
        self.epochs = epochs


class ConfigurationManager:
    def __init__(self):
        pass

    def get_data_ingestion_config(self):
        # Replace with your actual values
        source_file = '/kaggle/input/part1-file/blogtext.csv'
        intents_dict = {
            "informative_dict": ["information", "facts", "detail", "data", "knowledge", "clarify", "explain", "report", "reveal", "show", "statistics", "study", "update", "news", "research", "analysis", "discover", "learn", "reference", "insight", "summary", "overview", "guidance", "resource", "announcement", "result", "finding", "document", "outline", "intel", "source", "evidence", "proof", "understand", "notification", "illuminate", "dissemination", "briefing", "demonstration", "exposition", "discovery", "observation", "deliver", "awareness", "breaking", "fresh", "recent", "verify", "comprehend", "grasp", "master", "glean", "survey", "uncover", "realize", "establish", "conclude", "digest", "assimilate", "catch", "perceive", "discern", "deduce", "note", "learned", "erudite", "scholarly", "informed", "enlightened", "absorb", "knowledgeable", "educated", "lettered", "well-read", "bookish", "literate", "savvy", "brainy", "cerebral", "cultured", "sophisticated", "grounded", "apprehend", "fathom", "sense", "detect", "identify", "recognition", "notice", "spot", "locate", "find", "disclose", "unveil", "expose", "unearth", "excavate", "dredge", "quarry", "mine", "bring to light"],
            # Add more intents as needed
        }
        threshold = 3
        return DataIngestionConfig(source_file, intents_dict, threshold)

    def get_base_model_preparation_config(self):
        # Replace with your actual values
        base_model_name = 'bert-base-uncased'
        intents = list(self.get_data_ingestion_config().intents_dict.keys())
        return BaseModelPreparationConfig(base_model_name, intents)

    def get_callbacks_preparation_config(self):
        # Replace with your actual values
        tensorboard_log_dir = './logs'
        checkpoint_filepath = './checkpoint'
        return CallbacksPreparationConfig(tensorboard_log_dir, checkpoint_filepath)

    def get_training_config(self):
        # Replace with your actual values
        epochs = 3
        return TrainingConfig(epochs)
