import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import json
import kagglehub
import torch
from pathlib import Path
import umap

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
    idx,images,prompts,labels = zip(*batch)
    return list(idx),list(images), list(prompts), torch.tensor(labels)

def fix_duplicated_in_data_file(json_path: str) -> None:
    df = pd.read_json(json_path, lines=True)
    last_column = df.columns[-1]
    df_no_duplicates = df.drop_duplicates(subset=[last_column], keep='first')
    output_path = os.path.splitext(json_path)[0] + '_dedup.jsonl'
    df_no_duplicates.to_json(output_path, orient='records', lines=True)
    print(f"Saved deduplicated data to {output_path}")


def plot_tsne(X, y, plot_3d=False, perplexity=40):
    n_components = 3 if plot_3d else 2
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, metric='cosine')
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))

    if plot_3d:
        ax = plt.subplot(111, projection='3d')
        ax.scatter(X_tsne[y == 0, 0], X_tsne[y == 0, 1], X_tsne[y == 0, 2], label="Non-Hateful", alpha=0.5)
        ax.scatter(X_tsne[y == 1, 0], X_tsne[y == 1, 1], X_tsne[y == 1, 2], label="Hateful", alpha=0.5, color='red')
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
    else:
        plt.scatter(X_tsne[y == 0, 0], X_tsne[y == 0, 1], label="Non-Hateful", alpha=0.5)
        plt.scatter(X_tsne[y == 1, 0], X_tsne[y == 1, 1], label="Hateful", alpha=0.5, color='red')
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")

    plt.title(f"{'3D' if plot_3d else '2D'} t-SNE of LLaVA Embeddings")
    plt.legend()
    plt.grid(True)
    plt.show()

def save_embedding(embedding: torch.Tensor, sample_id: str):
    embedding_np = embedding.cpu().numpy()
    np.save(f"llava_embeddings_max_pool/{sample_id}.npy", embedding_np)
def plot_umap(X, y, plot_3d=False, n_neighbors=100, min_dist=0.5):
    n_components = 3 if plot_3d else 2
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42, metric='cosine')
    X_umap = reducer.fit_transform(X)

    plt.figure(figsize=(10, 8))

    if plot_3d:
        ax = plt.subplot(111, projection='3d')
        ax.scatter(X_umap[y == 0, 0], X_umap[y == 0, 1], X_umap[y == 0, 2], label="Non-Hateful", alpha=0.5)
        ax.scatter(X_umap[y == 1, 0], X_umap[y == 1, 1], X_umap[y == 1, 2], label="Hateful", alpha=0.5, color='red')
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
    else:
        plt.scatter(X_umap[y == 0, 0], X_umap[y == 0, 1], label="Non-Hateful", alpha=0.5)
        plt.scatter(X_umap[y == 1, 0], X_umap[y == 1, 1], label="Hateful", alpha=0.5, color='red')
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")

    plt.title(f"{'3D' if plot_3d else '2D'} UMAP of LLaVA Embeddings")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    train_json_path = r"C:\Users\rotem.geva\PycharmProjects\HatefulMemesClassification\raw_data\train.jsonl"
    parse_jsonl_to_df(train_json_path)

