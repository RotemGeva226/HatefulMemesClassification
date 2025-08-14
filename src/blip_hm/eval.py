import os, argparse, yaml
import torch
import torch.nn as nn
from .data import make_loader
from .model import build_model
from .utils import to_device, compute_metrics, save_json

@torch.no_grad()
def evaluate(model, loader, device: str, threshold: float):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    all_probs, all_labels = [], []
    total_loss = 0.0
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

def run_eval(cfg, split: str):
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    dcfg = cfg["data"]; mcfg = cfg["model"]; tcfg = cfg["train"]; ecfg = cfg.get("eval", {})
    threshold = float(ecfg.get("threshold", 0.5))

    if split == "val":
        csv_path = dcfg["val_csv"]
    elif split == "test":
        csv_path = dcfg.get("test_csv")
        if not csv_path:
            print("No test_csv provided; skipping test.")
            return None
    else:
        raise ValueError("split must be 'val' or 'test'")

    loader = make_loader(csv_path, dcfg["image_root"], dcfg["text_col"], dcfg["image_col"], dcfg["label_col"],
                         mcfg.get("processor_name"), tcfg["batch_size"], tcfg["num_workers"], shuffle=False)

    model = build_model(mcfg["backbone_name"], mcfg["freeze_backbone"], mcfg["hidden_dim"], mcfg["dropout"]).to(device)
    ckpt_path = os.path.join(cfg["logging"]["checkpoint_dir"], "best.ckpt")
    if os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"], strict=False)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("WARNING: best.ckpt not found; evaluating untrained model.")

    avg_loss, metrics = evaluate(model, loader, device, threshold)
    pc = metrics["per_class"]
    print(f"\n[{split.upper()}] Loss={avg_loss:.4f}  Acc={metrics['accuracy']:.4f}  AUC={metrics['roc_auc']:.4f}")
    print(f" Class 0: P={pc['0']['precision']:.4f} R={pc['0']['recall']:.4f} F1={pc['0']['f1']:.4f}")
    print(f" Class 1: P={pc['1']['precision']:.4f} R={pc['1']['recall']:.4f} F1={pc['1']['f1']:.4f}")

    out = {
        "split": split,
        "loss": float(avg_loss),
        "accuracy": float(metrics["accuracy"]),
        "roc_auc": float(metrics["roc_auc"]),
        "per_class": metrics["per_class"],
        "report": metrics["report"],
        "threshold": threshold
    }
    os.makedirs(cfg["logging"]["output_dir"], exist_ok=True)
    out_path = os.path.join(cfg["logging"]["output_dir"], f"metrics_{split}.json")
    save_json(out, out_path)
    print(f"Saved metrics JSON → {out_path}")
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    run_eval(cfg, args.split)
