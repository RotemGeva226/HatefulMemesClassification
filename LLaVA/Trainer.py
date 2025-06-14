import torch
import wandb
from tqdm import tqdm
from torch import nn
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class Trainer:
    def __init__(self, model, train_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.device = config["device"]
        self.optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=config["learning_rate"])
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.compute_class_weights().to(self.device))
        wandb.init(project=config["project_name"],
                   name=f"bs{config['batch_size']}-lr{config['learning_rate']}-{config['time']}",
                   config=config)

    def compute_class_weights(self):
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array([0, 1]),
            y=self.train_loader.dataset.df['label']
        )
        # Get only the pos_weight (weight for class 1)
        pos_weight = class_weights[1] / class_weights[0]  # Or just class_weights[1]
        return torch.tensor(pos_weight, dtype=torch.float32)

    @staticmethod
    def compute_metrics(logits, labels):
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()  # Probabilities for class 1
        preds = (probs >= 0.5).astype(int)  # Convert to binary predictions
        labels_np = labels.cpu().numpy()
        auc = roc_auc_score(labels_np, probs)
        acc = accuracy_score(labels_np, preds)
        f1 = f1_score(labels_np, preds)
        return {"auroc": auc, "accuracy": acc, "f1": f1}

    @staticmethod
    def log_metrics(epoch, loss, metrics, prefix="train"):
        log_dict = {
            f"{prefix}_loss": loss,
            f"{prefix}_auroc": metrics["auroc"],
            f"{prefix}_accuracy": metrics["accuracy"],
            f"{prefix}_f1": metrics["f1"],
        }
        wandb.log(log_dict, step=epoch+1)

    def train(self):
        self.model.train()

        for epoch in range(self.config["epochs"]):
            total_loss = 0
            all_logits = []
            all_labels = []

            for images, prompts, labels in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}"):
                labels = labels.to(self.device)
                logits = self.model(images, prompts)
                loss = self.criterion(logits.view(-1), labels.float().view(-1))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                wandb.log({"batch_loss": loss.item()})
                total_loss += loss.item()

                all_logits.append(logits.detach())
                all_labels.append(labels.detach())

            avg_loss = total_loss / len(self.train_loader)

            all_logits = torch.cat(all_logits, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            metrics = self.compute_metrics(all_logits, all_labels)
            self.log_metrics(epoch, avg_loss, metrics)
            print(f"[Epoch {epoch + 1}] Loss: {avg_loss:.4f}")