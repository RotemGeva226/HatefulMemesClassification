from datetime import datetime
from pathlib import Path
import torch
from utils import BASE_PATH

RAW_DATA_DIR = BASE_PATH / "data"
CURR_TIME = datetime.now().strftime('%Y%m%d-%H%M%S')

class Config:
    def __init__(self):
        self.config = {
            "mode" : "test",
            "batch_size":1,
            "epochs": 1,
            "learning_rate": 5e-5,
            "patience": 5,
            "num_workers": 4,
            "weight_decay" : 1e-3,
            "project_name": "llava-meme-classifier",
            "model_id": "llava-hf/llava-1.5-7b-hf",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "train_path": RAW_DATA_DIR / "train.jsonl",
            "validation_path": RAW_DATA_DIR / "dev.jsonl",
            "test_path": RAW_DATA_DIR / "test.jsonl",
            "checkpoint_path": "checkpoint_20250621-152520.pth",
            "run_name": "bs1-lr2e-05-20250621-152520",
            "time": CURR_TIME
        }
        self.validate_config()

    def validate_config(self):
        if self.config["mode"] == "test":
            missing = []
            if not self.config.get("checkpoint_path"):
                missing.append("checkpoint_path")
            if not self.config.get("run_name"):
                missing.append("run_name")
            if not self.config.get("test_path"):
                missing.append("test_path")
            elif not Path(self.config["test_path"]).exists():
                raise ValueError(f"test_path does not exist: {self.config['test_path']}")

            if missing:
                raise ValueError(f"Missing required config entries in test mode: {', '.join(missing)}")
