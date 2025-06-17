import torch
from tqdm import tqdm
from torch import nn
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from WandbLogger import WandbLogger

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config["device"]
        self.logger = WandbLogger(config)

        self.optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=config["learning_rate"])
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.compute_class_weights().to(self.device))
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.patience = config['patience']
        self.checkpoint_path = f"checkpoint_{config['time']}.pth"

    def compute_class_weights(self):
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array([0, 1]),
            y=self.train_loader.dataset.df['label']
        )
        # Get only the pos_weight (weight for class 1)
        pos_weight = class_weights[1] / class_weights[0]  # Or just class_weights[1]
        return torch.tensor(pos_weight, dtype=torch.float32)

    def check_early_stopping(self, val_loss, epoch):
        if val_loss < self.best_val_loss:
            print(f"✅ Validation loss improved ({self.best_val_loss:.4f} → {val_loss:.4f}). Saving checkpoint.")
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            self.save_checkpoint(epoch, val_loss)
            self.logger.set_summary("best_val_loss", val_loss)
        else:
            self.epochs_without_improvement += 1
            print(f"⚠️ No improvement. Patience: {self.epochs_without_improvement}/{self.patience}")
            if self.epochs_without_improvement >= self.patience:
                raise StopIteration

    def save_checkpoint(self, epoch=None, loss=None):
        path = self.checkpoint_path
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
            "best_val_loss": self.best_val_loss,
            "epochs_without_improvement": self.epochs_without_improvement,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        torch.cuda.empty_cache()
        checkpoint = torch.load(path, map_location='cpu')

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)

        epoch = checkpoint.get("epoch", 0)
        loss = checkpoint.get("loss", None)

        self.model.to(self.device)
        self.model.eval()

        print(f"Checkpoint loaded from {path}, resuming at epoch {epoch}")
        return epoch, loss

    def train(self, accumulation_steps=32):
        self.model.train()
        scaler = torch.cuda.amp.GradScaler()
        batch_counter = 0

        try:
            for epoch in range(self.config["epochs"]):
                total_loss = 0.0
                all_logits = []
                all_labels = []
                gradient_accumulated = 0

                self.optimizer.zero_grad()

                for step, (images, prompts, labels) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")):
                    labels = labels.to(self.device)

                    with torch.cuda.amp.autocast():
                        logits = self.model(images, prompts)
                        loss = self.criterion(logits.view(-1), labels.float().view(-1))
                        loss = loss / accumulation_steps  # scale loss

                    scaler.scale(loss).backward()
                    gradient_accumulated += 1

                    if gradient_accumulated == accumulation_steps:
                        scaler.step(self.optimizer)
                        scaler.update()
                        self.optimizer.zero_grad()
                        gradient_accumulated = 0
                        print(f"[Step {step}] Optimizer stepped")

                    total_loss += loss.item() * accumulation_steps  # Unscale to original loss

                    all_logits.append(logits.detach())
                    all_labels.append(labels.detach())

                    if (step + 1) % accumulation_steps == 0:
                        batch_counter += 1
                        self.logger.log_batch_loss(loss.item() * accumulation_steps, batch_counter)

                # Final step if leftover gradients were accumulated but not stepped
                if gradient_accumulated > 0:
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                    print(f"[Epoch {epoch}] Final optimizer step for leftover gradients")

                avg_loss = total_loss / len(self.train_loader)

                all_logits = torch.cat(all_logits, dim=0)
                all_labels = torch.cat(all_labels, dim=0)

                metrics = self.compute_metrics(all_logits, all_labels)
                self.logger.log_epoch_metrics(epoch, avg_loss, metrics)
                print(f"[Epoch {epoch + 1}] Loss: {avg_loss:.4f}")

                val_loss = self.validate(epoch)
                self.check_early_stopping(val_loss, epoch)

        except StopIteration:
            print("⏹️ Early stopping triggered. Training halted.")

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        all_logits = []
        all_labels = []

        for images, prompts, labels in tqdm(self.val_loader, desc=f"Validation Epoch {epoch + 1}"):
            labels = labels.to(self.device)
            logits = self.model(images, prompts)

            loss = self.criterion(logits.view(-1), labels.float().view(-1))
            total_loss += loss.item()

            all_logits.append(logits.detach())
            all_labels.append(labels.detach())

        avg_loss = total_loss / len(self.val_loader)
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        metrics = self.compute_metrics(all_logits, all_labels)
        self.logger.log_epoch_metrics(epoch, avg_loss, metrics, prefix="val")
        print(f"[Validation Epoch {epoch + 1}] Loss: {avg_loss:.4f}")

        return avg_loss

    @staticmethod
    def compute_metrics(logits, labels):
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()  # Probabilities for class 1
        preds = (probs >= 0.5).astype(int)  # Convert to binary predictions
        labels_np = labels.cpu().numpy()
        auc = roc_auc_score(labels_np, probs)
        acc = accuracy_score(labels_np, preds)
        f1 = f1_score(labels_np, preds)
        return {"auroc": auc, "accuracy": acc, "f1": f1}