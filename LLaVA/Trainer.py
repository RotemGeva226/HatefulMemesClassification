import torch
import wandb
from tqdm import tqdm
from torch import nn

class Trainer:
    def __init__(self, model, train_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.device = config["device"]
        self.optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=config["learning_rate"])
        wandb.init(project=config["project_name"],
                   name=f"bs{config['batch_size']}-lr{config['learning_rate']}-{config['time']}",
                   config=config)

    def train(self):
        self.model.train()

        for epoch in range(self.config["epochs"]):
            total_loss = 0

            for images, prompts, labels in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}"):
                labels = labels.to(self.device)
                logits = self.model(images, prompts)
                loss = nn.functional.cross_entropy(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                wandb.log({"batch_loss": loss.item()})
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            wandb.log({"epoch": epoch + 1, "epoch_loss": avg_loss})
            print(f"[Epoch {epoch + 1}] Loss: {avg_loss:.4f}")