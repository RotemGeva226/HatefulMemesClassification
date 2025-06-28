import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


class MetricAccumulator:
    def __init__(self):
        self.all_probs = []
        self.all_labels = []

    def update(self, logits, labels):
        """Update accumulator with batch predictions and labels"""
        # Convert logits to probabilities
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        probs = np.atleast_1d(probs)
        labels_np = np.atleast_1d(labels_np)

        # Store as numpy arrays to save memory
        self.all_probs.extend(probs.tolist())
        self.all_labels.extend(labels_np.tolist())

    def compute(self):
        """Compute final metrics from accumulated data"""
        # Concatenate all accumulated predictions and labels
        all_probs = np.array(self.all_probs)
        all_labels = np.array(self.all_labels)

        # Convert probabilities to binary predictions
        preds = (all_probs >= 0.5).astype(int)

        # Compute metrics
        auc = roc_auc_score(all_labels, all_probs)
        acc = accuracy_score(all_labels, preds)
        f1 = f1_score(all_labels, preds)

        return {"auroc": auc, "accuracy": acc, "f1": f1}

    def reset(self):
        """Reset accumulator for new evaluation"""
        self.all_probs = []
        self.all_labels = []