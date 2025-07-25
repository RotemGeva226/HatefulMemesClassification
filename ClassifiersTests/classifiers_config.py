import os
from dataclasses import dataclass
from datetime import datetime

BASE_DIR = os.getcwd()
CURR_TIME = datetime.now().strftime('%Y%m%d-%H%M%S')

@dataclass
class EmbeddingsConfig:
    train_path: str = os.path.join(BASE_DIR, "3", "train")
    validation_path: str =  os.path.join(BASE_DIR, "3", "val")
    test_path: str = os.path.join(BASE_DIR, "3", "test")

@dataclass
class ClassifierConfig:
    name: str
    params: dict

@dataclass
class ExperimentConfig:
    embeddings: EmbeddingsConfig
    classifier: ClassifierConfig


# EXAMPLE CONFIG
classifier_config = ExperimentConfig(
    embeddings=EmbeddingsConfig(),
    classifier=ClassifierConfig(
        name="mlp",
        params={
            "hidden_layer_sizes": [512, 256],
            "early_stopping": True,
            "n_iter_no_change": 10,
            "max_iter": 300,
        }
    ),
)
