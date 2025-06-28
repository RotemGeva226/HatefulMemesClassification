import pandas as pd
from torch.utils.data import DataLoader
from LLaVAMemesDataset import LLaVAMemesDataset
from LLaVA import LLaVA
from Trainer import Trainer
from utils import collate_fn
from config import Config

config_wrapper = Config()

# Prepare dataframe
df_test = pd.read_json(config_wrapper.config["test_path"], lines=True)


# Load test dataset
test_dataset = LLaVAMemesDataset(df_test)
test_loader = DataLoader(
    test_dataset,
    batch_size=config_wrapper.config["batch_size"],
    shuffle=True,
    pin_memory=True,
    num_workers=config_wrapper.config["num_workers"],
    collate_fn=collate_fn
)

# Load model
model = LLaVA(
    model_id=config_wrapper.config["model_id"],
    device=config_wrapper.config["device"]
)

# Init trainer
trainer = Trainer(model, None, None, test_loader, config_wrapper.config)

# Load the best checkpoint
trainer.load_checkpoint(config_wrapper.config["checkpoint_path"])

# Train
trainer.test()