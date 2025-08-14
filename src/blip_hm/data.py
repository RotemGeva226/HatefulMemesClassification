from dataclasses import dataclass
from typing import Optional, Any, Dict
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor

@dataclass
class DataConfig:
    csv_path: str
    image_root: str
    text_col: str = "text"
    image_col: str = "image"
    label_col: str = "label"
    processor_name: Optional[str] = None

class HatefulMemesDataset(Dataset):
    def __init__(self, cfg: DataConfig, processor: Optional[Any] = None):
        self.df = pd.read_csv(cfg.csv_path)
        self.image_root = cfg.image_root
        self.text_col = cfg.text_col
        self.image_col = cfg.image_col
        self.label_col = cfg.label_col
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, str(row[self.image_col]))
        text = str(row[self.text_col])
        label = int(row[self.label_col])

        image = Image.open(img_path).convert("RGB")

        if self.processor is not None:
            encoded = self.processor(
                images=image,
                text=text,
                return_tensors="pt",
                padding="max_length",
                truncation=True
            )
            item = {k: v.squeeze(0) for k, v in encoded.items()}
        else:
            item = {"image": image, "text": text}

        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item

def build_processor(processor_name: Optional[str]):
    if processor_name is None:
        return None
    return AutoProcessor.from_pretrained(processor_name)

def make_loader(csv_path: str, image_root: str, text_col: str, image_col: str, label_col: str,
                processor_name: Optional[str], batch_size: int, num_workers: int, shuffle: bool):
    processor = build_processor(processor_name)
    dcfg = DataConfig(csv_path, image_root, text_col, image_col, label_col, processor_name)
    ds = HatefulMemesDataset(dcfg, processor=processor)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
