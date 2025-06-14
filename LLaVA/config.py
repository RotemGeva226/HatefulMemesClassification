from datetime import datetime
from pathlib import Path
import torch

BASE_PATH = Path(r"C:\Users\rotem.geva\PycharmProjects\HatefulMemesClassification")
RAW_DATA_DIR = BASE_PATH / "raw_data"
CURR_TIME = datetime.now().strftime('%Y%m%d-%H%M%S')

config = {
    "batch_size":1,
    "epochs": 3,
    "learning_rate": 2e-5,
    "patience": 3,
    "project_name": "llava-meme-classifier",
    "model_id": "llava-hf/llava-1.5-7b-hf",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "train_path": RAW_DATA_DIR / "train.jsonl",
    "validation_path": RAW_DATA_DIR / "dev.jsonl",
    "time": CURR_TIME
}