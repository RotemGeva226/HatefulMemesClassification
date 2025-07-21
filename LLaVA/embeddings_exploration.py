import os
import numpy as np
import utils
from tqdm import tqdm
import pandas as pd
from config import Config
from pathlib import Path

def find_label(id: str):
    """Find the label for a given id."""
    label = source_file[source_file["id"] == int(id)]["label"].values[0]
    return label

config_wrapper = Config()

# Directory containing embedding files
base_dir = Path(__file__).resolve().parent.parent
embedding_dir = os.path.join(base_dir, "ClassifiersTests","llava_embeddings_mean_pool_last_hidden_train")
source_file = pd.read_json(config_wrapper.config["train_path"], lines=True)

# List to store embeddings
embeddings = []
labels = []

# Iterate through the directory
for file_name in tqdm(os.listdir(embedding_dir), desc="Loading embeddings"):
    file_path = Path(os.path.join(embedding_dir, file_name))
    if file_name.endswith(".npy"):
        embedding = np.load(file_path)
        embeddings.append(embedding)
        label = find_label(file_path.stem)  # Extract id from file name
        labels.append(label)  # Use file name as label
X = np.array(embeddings)
y = np.array(labels)

utils.plot_tsne(X,y, perplexity=70)
utils.plot_umap(X,y, n_neighbors=100, min_dist=0.01)
