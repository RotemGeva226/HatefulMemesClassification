import os, argparse, json
import matplotlib.pyplot as plt

def main(log_path: str):
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"Training log not found: {log_path}")
    with open(log_path, "r") as f:
        hist = json.load(f)

    plt.figure(figsize=(12,5))
    # Loss
    plt.subplot(1,2,1)
    plt.plot(hist.get("train_losses", []), label="Train Loss")
    plt.plot(hist.get("val_losses", []), label="Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss over Epochs")
    plt.grid(True); plt.legend()

    # Validation score
    val_scores = hist.get("val_aucs", []) or hist.get("val_accs", [])
    title = "Validation AUC over Epochs" if hist.get("val_aucs", []) else "Validation Accuracy over Epochs"

    plt.subplot(1,2,2)
    if val_scores:
        plt.plot(val_scores, label=title.split()[1])
        plt.title(title)
    else:
        plt.title("Validation Score (no series available)")
    plt.xlabel("Epoch"); plt.ylabel("Score"); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default="reports/training_log.json")
    args = parser.parse_args()
    main(args.log_path)
