#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flamingo Binary Hate Speech Classifier (Academic-ready version)
---------------------------------------------------------------
Implementation for multimodal hate speech classification using the
Flamingo architecture (frozen CLIP vision encoder + GPT-2 language model).
Includes prompt-based fine-tuning, selective parameter updates, class
imbalance handling, Focal Loss, logging, graphs, and final evaluation.
"""

import os, json, re, torch
from sklearn.metrics import roc_curve, roc_auc_score
from collections import Counter
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torchvision import transforms as T
from open_flamingo import create_model_and_transforms
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ==============================
# CONFIGURATION
# ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
THRESHOLD = 0.6
EPOCHS = 25
USEFOCAL= False
# ==============================
# UTILS
# ==============================
def print_trainable(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {train:,} / {total:,}  ({100*train/total:.2f} %)")

# ==============================
# LOSS FUNCTION
# ==============================
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma, self.weight = gamma, weight
    def forward(self, logits, labels):
        ce = torch.nn.functional.cross_entropy(
            logits, labels, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1-pt)**self.gamma * ce).mean()

# ==============================
# DATASET
# ==============================
class HateDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, vision_tform, img_dir="data"):
        self.recs = [json.loads(l) for l in open(jsonl_path, encoding="utf-8")]
        self.tok  = tokenizer
        self.vt   = vision_tform
        self.dir  = Path(img_dir)
    @staticmethod
    def clean(txt):
        txt = re.sub(r"http\S+|@\w+|#\w+", "", txt)
        txt = re.sub(r"[^\x00-\x7F]", "", txt)
        return txt.strip().lower()
    def __len__(self): return len(self.recs)
    def __getitem__(self, idx):
        r = self.recs[idx]
        img = self.vt(Image.open(self.dir / r["img"]).convert("RGB"))
        prompt = self.build_prompt(self.clean(r['text']))
        enc = self.tok(prompt, truncation=True, return_tensors=None)
        return img, enc, r["label"]
    def build_prompt(self, caption):
        examples = [
            ("These parasites are ruining our country.", "Yes"),
            ("This government is a joke.", "No"),
            ("Send them back to where they came from.", "Yes"),
            ("I'm so done with Mondays...", "No"),
            ("Women should just stay in the kitchen.", "Yes"),
            ("I love when women are confident and powerful 💪", "No"),
            ("Black people are always up to no good.", "Yes"),
            ("People from all races deserve respect and dignity.", "No")
        ]
        prompt = (
            "Task: Determine whether the following meme contains hate speech.\n\n"
            "Definition:\n"
            "Hate speech is any content—textual, visual, or a combination—that attacks, dehumanizes, mocks, "
            "or incites discrimination or violence against people based on attributes such as race, ethnicity, "
            "national origin, religion, gender, sexual orientation, or disability.\n\n"
        )
        for ex_text, ex_label in examples:
            prompt += f"<image>\nCaption: {ex_text}\nQuestion: Does this meme contain hate speech?\nAnswer: {ex_label}\n\n"
        prompt += f"<image>\nCaption: {caption}\nQuestion: Does this meme contain hate speech?\nAnswer:"
        return prompt

# ==============================
# COLLATE FUNCTION
# ==============================
def collate_fn(batch, tokenizer):
    imgs, encs, labels = zip(*batch)
    imgs = torch.stack(imgs).to(DEVICE, dtype=torch.float16)
    enc  = tokenizer.pad(encs, return_tensors="pt").to(DEVICE)
    labels = torch.tensor(labels, device=DEVICE)
    return imgs, enc, labels

# ==============================
# MODEL
# ==============================
class FlamingoClassifier(torch.nn.Module):
    def __init__(self, flamingo, tok, weight=None, use_focal=True):
        super().__init__()
        self.flamingo = flamingo
        self.tok = tok
        self.no_id  = tok.encode(" No",  add_special_tokens=False)[0]
        self.yes_id = tok.encode(" Yes", add_special_tokens=False)[0]
        if use_focal:
            self.crit = FocalLoss(gamma=2.0, weight=weight)
        else:
            self.crit = torch.nn.CrossEntropyLoss(weight=weight)
    def forward(self, images, input_ids, attention_mask, labels=None):
        images = images.unsqueeze(1).unsqueeze(2).float()
        out = self.flamingo(images, input_ids, attention_mask)
        last_logits = out.logits[:, -1, :]
        logits = last_logits[:, [self.no_id, self.yes_id]]
        loss = self.crit(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

# ==============================
# MAIN
# ==============================
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    TRAIN_JSON, VAL_JSON = "data/train.jsonl", "data/dev.jsonl"

    backbone, vision_tform, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="laion2b_s32b_b82k",
        lang_encoder_path="gpt2",
        tokenizer_path="gpt2",
        cross_attn_every_n_layers=4,
        decoder_layers_attr_name="transformer.h",
    )
    tokenizer.pad_token = tokenizer.eos_token

    n_layers = backbone.lang_encoder.config.n_layer
    for n, p in backbone.lang_encoder.named_parameters():
        p.requires_grad_(any(n.startswith(f"transformer.h.{i}") for i in range(n_layers-4, n_layers)))
    print_trainable(backbone)

    train_vt = T.Compose([
        T.RandomResizedCrop(224, scale=(0.9,1.0)),
        T.ColorJitter(0.1,0.1,0.1,0.05),
        vision_tform,
    ])

    BATCH = 32
    train_ds = HateDataset(TRAIN_JSON, tokenizer, train_vt)
    val_ds   = HateDataset(VAL_JSON, tokenizer, vision_tform)
    lbls = [r["label"] for r in train_ds.recs]
    cnt  = Counter(lbls)
    weights = [1/cnt[l] for l in lbls]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler,
                              collate_fn=lambda b: collate_fn(b, tokenizer))
    val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                              collate_fn=lambda b: collate_fn(b, tokenizer))

    class_weight = torch.tensor([1.0, cnt[0]/cnt[1]], device=DEVICE)
    model = FlamingoClassifier(backbone, tokenizer, weight=class_weight, use_focal=USEFOCAL).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    steps = len(train_loader)*EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    best_auc, bad, patience = 0, 0, 5
    accum = 4
    train_losses, val_rs, val_accuracies = [], [], []

    for ep in range(1,EPOCHS+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"Epoch {ep}")
        total_loss = 0
        for i,(imgs,enc,labels) in enumerate(pbar):
            loss = model(images=imgs, **enc, labels=labels)["loss"] / accum
            loss.backward()
            total_loss += loss.item()
            if (i+1)%accum==0 or (i+1)==len(train_loader):
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        model.eval()
        preds, golds, probs = [], [], []
        with torch.no_grad():
            for imgs, enc, labels in val_loader:
                outs = model(images=imgs, **enc)
                logits = outs["logits"]
                prob_yes = torch.softmax(logits, dim=-1)[:, 1]
                preds += (prob_yes >= THRESHOLD).long().cpu().tolist()
                golds += labels.cpu().tolist()
                probs += prob_yes.cpu().tolist()

        acc = accuracy_score(golds, preds)
        _, r, _, _ = precision_recall_fscore_support(golds, preds, average="binary", zero_division=0)
        auc = roc_auc_score(golds, probs)
        val_rs.append(r)
        val_accuracies.append(acc)

        fpr, tpr, _ = roc_curve(golds, probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Validation ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"roc_curve_epoch{ep}.png")

        if auc > best_auc:
            Path("flamingo_blanace_focal").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), "flamingo_blanace_focal/best_model.pt")
            best_auc, bad = auc, 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    TEST_JSON = "data/test.jsonl"
    if Path(TEST_JSON).exists():
        test_ds = HateDataset(TEST_JSON, tokenizer, vision_tform)
        test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False,
                                 collate_fn=lambda b: collate_fn(b, tokenizer))
        model.load_state_dict(torch.load("flamingo_blanace_focal/best_model.pt"))
        test_preds, test_golds, test_probs = [], [], []
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, enc, labels in test_loader:
                outs = model(images=imgs, **enc)
                logits = outs["logits"]
                prob_yes = torch.softmax(logits, dim=-1)[:, 1]
                test_preds += logits.argmax(-1).cpu().tolist()
                test_golds += labels.cpu().tolist()
                test_probs += prob_yes.cpu().tolist()
                preds = (prob_yes >= THRESHOLD).long()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        test_acc = accuracy_score(test_golds, test_preds)
        test_p, test_r, test_f, _ = precision_recall_fscore_support(test_golds, test_preds, average="binary",
                                                                    zero_division=0)
        test_auc = roc_auc_score(test_golds, test_probs)

        fpr, tpr, _ = roc_curve(test_golds, test_probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {test_auc:.4f}")
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Test ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig("roc_curve_test.png")

        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Hate", "Hate"])
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")

if __name__ == "__main__":
    main()
