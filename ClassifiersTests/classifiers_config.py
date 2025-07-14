import os
from dataclasses import dataclass
from datetime import datetime

BASE_DIR = os.getcwd()
CURR_TIME = datetime.now().strftime('%Y%m%d-%H%M%S')

@dataclass
class EmbeddingsConfig:
    train_path: str = "llava_embeddings_mean_pool_last_hidden_train"
    validation_path: str = "llava_embeddings_mean_pool_last_hidden_val"
    test_path: str = "llava_embeddings_mean_pool_last_hidden_test"

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
    embeddings=EmbeddingsConfig(
        train_path=os.path.join(BASE_DIR, "llava_embeddings_mean_pool_last_hidden_train"),
        validation_path=os.path.join(BASE_DIR, "llava_embeddings_mean_pool_last_hidden_val"),
        test_path=os.path.join(BASE_DIR, "llava_embeddings_mean_pool_last_hidden_test")
    ),
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
