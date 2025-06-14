import pandas as pd
from torch.utils.data import DataLoader
from LLaVAMemesDataset import LLaVAMemesDataset
from LLaVA import LLaVA
from Trainer import Trainer
from utils import collate_fn
from config import config


# Prepare dataframe
df_train = pd.read_json(config["train_path"], lines=True)
df_val = pd.read_json(config["validation_path"], lines=True)

# Load train dataset
train_dataset = LLaVAMemesDataset(df_train)
train_loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    collate_fn=collate_fn
)

# Load val dataset
val_dataset = LLaVAMemesDataset(df_val)
val_loader = DataLoader(
    val_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    collate_fn=collate_fn
)

# Init model
model = LLaVA(
    model_id=config["model_id"],
    device=config["device"]
)

# Train
trainer = Trainer(model, train_loader, val_loader, config)
trainer.train()