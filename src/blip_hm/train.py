import os, argparse, yaml
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import make_loader
from .model import build_model
from .utils import set_seed, to_device, compute_metrics, save_json

def validate(model, loader: DataLoader, device: str, threshold: float):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    all_probs, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            logits = model(input_ids=batch.get("input_ids"),
                           pixel_values=batch.get("pixel_values"),
                           attention_mask=batch.get("attention_mask"))
            loss = loss_fn(logits.view(-1), batch["labels"].float())
            probs = torch.sigmoid(logits).view(-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy().astype(int)
            total_loss += loss.item() * labels.shape[0]
            all_probs.extend(probs.tolist()); all_labels.extend(labels.tolist())
    avg_loss = total_loss / max(1, len(loader.dataset))
    metrics = compute_metrics(all_labels, all_probs, threshold=threshold)
    return avg_loss, metrics

def train_loop(cfg: Dict[str, Any]):
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.get("seed", 42))

    dcfg = cfg["data"]
    mcfg = cfg["model"]
    tcfg = cfg["train"]
    ecfg = cfg.get("eval", {})
    threshold = float(ecfg.get("threshold", 0.5))

    # Data
    train_loader = make_loader(dcfg["train_csv"], dcfg["image_root"], dcfg["text_col"], dcfg["image_col"], dcfg["label_col"],
                               mcfg.get("processor_name"), tcfg["batch_size"], tcfg["num_workers"], shuffle=True)
    val_loader = make_loader(dcfg["val_csv"], dcfg["image_root"], dcfg["text_col"], dcfg["image_col"], dcfg["label_col"],
                             mcfg.get("processor_name"), tcfg["batch_size"], tcfg["num_workers"], shuffle=False)

    # Model & Optim
    model = build_model(mcfg["backbone_name"], mcfg["freeze_backbone"], mcfg["hidden_dim"], mcfg["dropout"]).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])

    pos_weight = tcfg.get("pos_weight", None)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pos_weight)], device=device)) if pos_weight is not None else nn.BCEWithLogitsLoss()

    best_val_auc = -float("inf")
    patience = int(tcfg.get("early_stopping_patience", 3))
    wait = 0
    hist = {"train_losses": [], "val_losses": [], "val_aucs": [], "val_accs": []}

    for epoch in range(1, int(tcfg["epochs"]) + 1):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{tcfg['epochs']}"):
            batch = to_device(batch, device)
            logits = model(input_ids=batch.get("input_ids"),
                           pixel_values=batch.get("pixel_values"),
                           attention_mask=batch.get("attention_mask"))
            loss = loss_fn(logits.view(-1), batch["labels"].float())
            optim.zero_grad(); loss.backward(); optim.step()
            running += loss.item() * batch["labels"].shape[0]

        train_loss = running / max(1, len(train_loader.dataset))
        val_loss, val_metrics = validate(model, val_loader, device, threshold)
        val_auc = float(val_metrics["roc_auc"]); val_acc = float(val_metrics["accuracy"])

        hist["train_losses"].append(train_loss); hist["val_losses"].append(val_loss)
        hist["val_aucs"].append(val_auc); hist["val_accs"].append(val_acc)
        print(f"[Epoch {epoch}] TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | ValAUC={val_auc:.4f} | ValAcc={val_acc:.4f}")

        # Early stopping on AUC
        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc; wait = 0
            os.makedirs(cfg["logging"]["checkpoint_dir"], exist_ok=True)
            torch.save({"model_state": model.state_dict(), "cfg": cfg}, os.path.join(cfg["logging"]["checkpoint_dir"], "best.ckpt"))
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping (patience={patience})")
                break

    os.makedirs(cfg["logging"]["output_dir"], exist_ok=True)
    save_json(hist, os.path.join(cfg["logging"]["output_dir"], "training_log.json"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train_loop(cfg)
