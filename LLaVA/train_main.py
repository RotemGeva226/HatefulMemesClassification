import pandas as pd
from torch.utils.data import DataLoader
from LLaVAMemesDataset import LLaVAMemesDataset
from LLaVA import LLaVA
from Trainer import Trainer
from utils import collate_fn
from config import Config

config_wrapper = Config()

# Prepare dataframe
df_train = pd.read_json(config_wrapper.config["train_path"], lines=True)
df_val = pd.read_json(config_wrapper.config["validation_path"], lines=True)

# Load train dataset
train_dataset = LLaVAMemesDataset(df_train)
train_loader = DataLoader(
    train_dataset,
    batch_size=config_wrapper.config["batch_size"],
    shuffle=True,
    pin_memory=True,
    num_workers=config_wrapper.config["num_workers"],
    collate_fn=collate_fn
)

# Load val dataset
val_dataset = LLaVAMemesDataset(df_val)
val_loader = DataLoader(
    val_dataset,
    batch_size=config_wrapper.config["batch_size"],
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

# Train
trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=None,
                  config=config_wrapper.config)
trainer.train()