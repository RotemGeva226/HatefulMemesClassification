import os, json, random
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, accuracy_score, classification_report
)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out

def compute_metrics(labels, probs, threshold=0.5):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs) if len(set(labels.tolist())) > 1 else float("nan")
    report = classification_report(labels, preds, digits=4, zero_division=0, output_dict=True)
    per_class = {
        "0": {"precision": report["0"]["precision"], "recall": report["0"]["recall"], "f1": report["0"]["f1-score"]},
        "1": {"precision": report["1"]["precision"], "recall": report["1"]["recall"], "f1": report["1"]["f1-score"]},
    }
    return {"accuracy": acc, "roc_auc": auc, "per_class": per_class, "report": report}

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
