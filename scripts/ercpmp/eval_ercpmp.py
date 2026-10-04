"""Evaluate the final malignancy classifier on the held-out test split (val is not read).
Writes metrics, confusion matrices and precision-recall curves to <checkpoint dir>/test_eval.
Reports image level and patient level (mean probability over a patient's or video's images)
results, for all datasets combined and for each dataset separately.

Usage: python eval_ercpmp.py [--data DIR ...] [--ckpt FILE]
  default: --data data/malignancy_prepared --ckpt models/malignancy/malignancy_effb0.pth"""
import argparse, csv, os
import numpy as np
import torch
import timm
import albumentations as A
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, precision_recall_curve, roc_auc_score)

ap = argparse.ArgumentParser()
ap.add_argument("--data", nargs="+", default=[r"C:\polyp-project\data\malignancy_prepared"])
ap.add_argument("--ckpt", default=r"C:\polyp-project\models\malignancy\malignancy_effb0.pth")
args = ap.parse_args()
OUT = os.path.join(os.path.dirname(args.ckpt), "test_eval")
os.makedirs(OUT, exist_ok=True)

dev = torch.device("cuda")
ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
meta = ck["meta"]; classes = meta["classes"]
model = timm.create_model(meta["arch"], pretrained=False, num_classes=len(classes)).to(dev)
model.load_state_dict(ck["state_dict"]); model.eval()
resize = A.Resize(*meta["input_hw"])
mean, std = np.array(meta["mean"], np.float32), np.array(meta["std"], np.float32)

probs, ys, patients, names, dsets = [], [], [], [], []
for root in args.data:
    for r in csv.DictReader(open(os.path.join(root, "manifest.csv"))):
        if r["split"] != "test":
            continue
        img = resize(image=np.array(Image.open(os.path.join(root, r["path"])).convert("RGB")))["image"]
        x = torch.from_numpy(((img.astype(np.float32) / 255 - mean) / std).transpose(2, 0, 1))[None].to(dev)
        with torch.no_grad():
            probs.append(torch.softmax(model(x).float(), 1)[0].cpu().numpy())
        ys.append(classes.index(r["class"])); names.append(r["path"]); dsets.append(r["dataset"])
        patients.append(f'{r["dataset"]}:{r["patient"]}')
P, Y, D, PT = np.array(probs), np.array(ys), np.array(dsets), np.array(patients)


def patient_level(mask):
    """Average the probabilities of all images of one patient (or video)."""
    pids = sorted(set(PT[mask]))
    return (np.array([P[mask & (PT == pid)].mean(0) for pid in pids]),
            np.array([Y[mask & (PT == pid)][0] for pid in pids]))


def report(tag, probs_, y_):
    pred = probs_.argmax(1)
    labels = list(range(len(classes)))
    txt = classification_report(y_, pred, labels=labels, target_names=classes, digits=3, zero_division=0)
    lines = [f"== {tag}: {len(y_)} samples ==", txt,
             f"Accuracy           {(pred == y_).mean():.4f}",
             f"Balanced accuracy  {balanced_accuracy_score(y_, pred):.4f}"]
    for ci, c in enumerate(classes):
        if 0 < (y_ == ci).sum() < len(y_):
            lines.append(f"{c:<22} AP {average_precision_score(y_ == ci, probs_[:, ci]):.3f}   "
                         f"ROC AUC {roc_auc_score(y_ == ci, probs_[:, ci]):.3f}")
        else:
            lines.append(f"{c:<22} AP/AUC undefined ({(y_ == ci).sum()} positives)")
    cm = confusion_matrix(y_, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(labels, classes, rotation=25, ha="right"); ax.set_yticks(labels, classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"Confusion matrix ({tag})")
    for i in labels:
        for j in labels:
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"confusion_{tag}.png"), dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for ci, c in enumerate(classes):
        if 0 < (y_ == ci).sum() < len(y_):
            pr, rc, _ = precision_recall_curve(y_ == ci, probs_[:, ci])
            ax.plot(rc, pr, label=f"{c} (AP {average_precision_score(y_ == ci, probs_[:, ci]):.2f}, n={(y_ == ci).sum()})")
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_title(f"Precision-recall, one vs rest ({tag})"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"pr_curves_{tag}.png"), dpi=120); plt.close(fig)
    return "\n".join(lines) + "\n\n"


summary = (f"Test split of {', '.join(sorted(set(dsets)))} (never used for training or checkpoint selection; "
           f"val split untouched)\nModel trained on: {meta.get('dataset', '?')}\nClasses: {', '.join(classes)}\n\n")
groups = [("all", np.ones(len(Y), bool))] if len(set(dsets)) > 1 else []
groups += [(d, D == d) for d in sorted(set(dsets))]
for gname, mask in groups:
    suffix = "" if len(groups) == 1 else f"_{gname}"
    summary += report(f"image_level{suffix}", P[mask], Y[mask])
    summary += report(f"patient_level{suffix}", *patient_level(mask))
open(os.path.join(OUT, "test_metrics.txt"), "w").write(summary)
with open(os.path.join(OUT, "per_image.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["dataset", "image", "patient", "true", "pred"] + [f"p_{c}" for c in classes])
    for d, n, pid, y, p in zip(dsets, names, patients, Y, P):
        w.writerow([d, n, pid, classes[y], classes[p.argmax()]] + [f"{v:.4f}" for v in p])
print(summary)
