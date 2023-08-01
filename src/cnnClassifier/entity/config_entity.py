from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfigEntity:
    source_file: Path
    intents_dict: dict
    threshold: int


@dataclass(frozen=True)
class BaseModelPreparationConfigEntity:
    base_model_name: str
    intents: list


@dataclass(frozen=True)
class CallbacksPreparationConfigEntity:
    tensorboard_log_dir: Path
    checkpoint_filepath: Path


@dataclass(frozen=True)
class TrainingConfigEntity:
    epochs: int
