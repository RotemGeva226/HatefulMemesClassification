import pandas as pd
import torch
from torch.utils.data import DataLoader
from LLaVAMemesDataset import LLaVAMemesDataset
from LLaVA import LLaVA
from utils import collate_fn
from config import Config
from tqdm import tqdm
from utils import BASE_PATH
import numpy as np
import os


"""This module handles the generation of embeddings for LLaVA memes dataset."""

def create_embeddings_dataset(model, loader, embeddings_folder):
    base_dir = r"C:\Users\rotem.geva\PycharmProjects\HatefulMemesClassification\ClassifiersTests\3"
    embedding_dir = os.path.join(base_dir, embeddings_folder)
    rows = []
    for idx, images, prompts, labels in tqdm(loader, desc="Samples"):
        embeddings_pooled = model.generate_embeddings(images, prompts)
        embeddings_np = embeddings_pooled.cpu().numpy()
        labels_np = labels.cpu().numpy()

        for i, emb, label in zip(idx, embeddings_np, labels_np):
            embedding_path = os.path.join(embedding_dir, f"{i}.npy")
            np.save(embedding_path, emb)

            rows.append({
                "id": i,
                "label": label,
                "embedding_path": embedding_path
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(embedding_dir, "embeddings_metadata.csv"), index=False)

def save_embedding(embedding: torch.Tensor, sample_id: str):
    embedding_np = embedding.cpu().numpy()
    dest_dir = os.path.join(BASE_PATH, "ClassifiersTests", "2", "train")
    np.save(f"{dest_dir}/{sample_id}.npy", embedding_np)
    print("Saved embedding for sample:", sample_id)

if __name__ == "__main__":
    config_wrapper = Config()

    # Prepare dataframe
    df = pd.read_json(config_wrapper.config["test_path"], lines=True)

    # Load dataset
    dataset = LLaVAMemesDataset(df)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        num_workers=config_wrapper.config["num_workers"],
        collate_fn=collate_fn
    )

    # Init model
    model = LLaVA(
        model_id=config_wrapper.config["model_id"],
        device=config_wrapper.config["device"]
    )

    create_embeddings_dataset(model, loader, "test")
