import torch
import wandb
from tqdm import tqdm
from torch import nn
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

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

    @staticmethod
    def compute_metrics(logits, labels):
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        labels_np = labels.cpu().numpy()
        auc = roc_auc_score(labels_np, probs)
        acc = accuracy_score(labels_np, preds)
        f1 = f1_score(labels_np, preds)
        return {"auroc": auc, "accuracy": acc, "f1": f1}

    @staticmethod
    def log_metrics(epoch, loss, metrics):
        wandb.log({
            "epoch": epoch + 1,
            "epoch_loss": loss,
            "epoch_auroc": metrics["auroc"],
            "epoch_accuracy": metrics["accuracy"],
            "epoch_f1": metrics["f1"]
        })

    def train(self):
        self.model.train()

        for epoch in range(self.config["epochs"]):
            total_loss = 0
            all_logits = []
            all_labels = []

            for images, prompts, labels in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}"):
                labels = labels.to(self.device)
                logits = self.model(images, prompts)
                loss = nn.functional.cross_entropy(logits, labels)

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