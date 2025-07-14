import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
import matplotlib.pyplot as plt
from utils import BASE_PATH

RAW_DATA_DIR = BASE_PATH / "data"

class LLaVAMemesDataset(Dataset):
    def __init__(self, dataframe, image_dir=RAW_DATA_DIR):
        self.df = dataframe
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row["img"])
        image = Image.open(image_path).convert("RGB")
        meme_text = row["text"].strip()
        prompt = (
            f"A chat between a curious user and an AI assistant specialized that detects hate speech in memes.\n"
            f"USER: <image>\n{meme_text}\n"
            f"Is this meme hateful? Answer only 'Yes' or 'No'.\n"
            f"ASSISTANT:"
        )
        label = torch.tensor(row["label"], dtype=torch.long)
        return row.id, image, prompt, label

    def check_sample(self, idx=0):
        """Visual check for a sample"""
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row['img'])
        image = Image.open(image_path).convert("RGB")
        caption = row['text'].strip()
        prompt = f"<image>\nUSER: {caption}\nASSISTANT:"
        label = row['label']

        # Show prompt
        print("📝 Prompt:")
        print(prompt)
        print(f"🏷️ Label: {label}")

        # Show image
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Image: {row['img']}")
        plt.show()


if __name__ == "__main__":
    file_path = RAW_DATA_DIR / "raw_data.csv"
    df = pd.read_csv(file_path)
    dataset = LLaVAMemesDataset(df, RAW_DATA_DIR)
    dataset.check_sample(12)
