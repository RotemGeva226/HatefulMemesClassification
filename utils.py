import os
import pandas as pd
import json
import kagglehub
import torch
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent

def parse_jsonl_to_df(jsonl_path: str) -> None:
    """Parse and outputs json file to pandas dataframe"""
    base_directory = os.path.dirname(jsonl_path)
    output_filepath = os.path.join(base_directory, 'train_data.csv')

    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    df.to_csv(output_filepath, index=False)


def download_dataset(dataset_path: str = "parthplc/facebook-hateful-meme-dataset") -> None:
    """Downloads a specified dataset from Kaggle Hub"""
    path = kagglehub.dataset_download(dataset_path)
    print(f"Dataset downloaded to: {path}")

def collate_fn(batch):
    images,prompts,labels = zip(*batch)
    return list(images), list(prompts), torch.tensor(labels)
def fix_duplicated_in_dev_file(json_path: str) -> None:
    df = pd.read_json(json_path, lines=True)
    last_column = df.columns[-1]
    df_no_duplicates = df.drop_duplicates(subset=[last_column], keep='first')
    output_path = os.path.splitext(json_path)[0] + '_dedup.jsonl'
    df_no_duplicates.to_json(output_path, orient='records', lines=True)
    print(f"Saved deduplicated data to {output_path}")

if __name__ == "__main__":
    train_json_path = r"C:\Users\rotem.geva\PycharmProjects\HatefulMemesClassification\raw_data\train.jsonl"
    parse_jsonl_to_df(train_json_path)

