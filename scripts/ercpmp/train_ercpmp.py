"""Fine tune EfficientNet-B0 (ImageNet) to classify polyps as benign or malignant.
Classes: benign, malignant.
Trains 25 epochs on the train split of every dataset given and keeps the final epoch checkpoint.
The test split is scored after every epoch for the learning curves only; it is never
used to pick a checkpoint. The val split is not read.

Usage: python train_ercpmp.py [--data DIR ...] [--out DIR]
  default: --data data/malignancy_prepared --out models/malignancy"""
import argparse, csv, json, os, random, time
import numpy as np
import torch
import timm
import albumentations as A
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score

ap = argparse.ArgumentParser()
ap.add_argument("--data", nargs="+", default=[r"C:\polyp-project\data\malignancy_prepared"])
ap.add_argument("--out", default=r"C:\polyp-project\models\malignancy")
ap.add_argument("--name", default="malignancy_effb0.pth")
ap.add_argument("--weighting", choices=["inverse", "sqrt"], default="inverse",
                help="class weights: inverse frequency, or its square root (gentler for very rare classes)")
args = ap.parse_args()

EPOCHS, BATCH, LR, SEED = 25, 16, 3e-4, 42
H, W = 256, 384  # ERCPMP frames are 368x256 (landscape)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if not torch.cuda.is_available():
    raise SystemExit("GPU not available to PyTorch. Stopping (no CPU fallback).")
dev = torch.device("cuda")
print(f"torch {torch.__version__} | GPU: {torch.cuda.get_device_name(0)}")

classes = json.load(open(os.path.join(args.data[0], "split.json")))["classes"]
for d in args.data[1:]:
    assert json.load(open(os.path.join(d, "split.json")))["classes"] == classes, f"class list differs in {d}"
resize = A.Resize(H, W)

def load(part):
    xs, ys, ds = [], [], []
    for root in args.data:
        for r in csv.DictReader(open(os.path.join(root, "manifest.csv"))):
            if r["split"] != part:
                continue
            xs.append(resize(image=np.array(Image.open(os.path.join(root, r["path"])).convert("RGB")))["image"])
            ys.append(classes.index(r["class"])); ds.append(r["dataset"])
    return xs, np.array(ys), np.array(ds)

tr_x, tr_y, tr_d = load("train")
te_x, te_y, te_d = load("test")
for tag, y, d in (("train", tr_y, tr_d), ("test", te_y, te_d)):
    for name in sorted(set(d)):
        print(f"{tag} {name}: {dict(zip(classes, np.bincount(y[d == name], minlength=len(classes)).tolist()))}")

aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Affine(scale=(0.85, 1.15), translate_percent=0.05, rotate=(-20, 20), p=0.7),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=10, p=0.3),
])
mean, std = np.array(MEAN, np.float32), np.array(STD, np.float32)

def to_tensor(batch):
    return torch.from_numpy(np.stack([((b.astype(np.float32) / 255 - mean) / std).transpose(2, 0, 1) for b in batch])).to(dev)

def batches(xs, ys, train):
    order = np.random.permutation(len(xs)) if train else np.arange(len(xs))
    for i in range(0, len(order), BATCH):
        idx = order[i:i + BATCH]
        imgs = [aug(image=xs[j])["image"] if train else xs[j] for j in idx]
        yield to_tensor(imgs), torch.from_numpy(ys[idx]).to(dev)

model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=len(classes), drop_rate=0.3).to(dev)
# class weights against the heavy imbalance; sqrt keeps a very rare class from dominating
counts = np.bincount(tr_y, minlength=len(classes)).astype(np.float32)
w = counts.sum() / (len(classes) * np.maximum(counts, 1))
if args.weighting == "sqrt":
    w = np.sqrt(w)
weights = torch.tensor(w / w.mean(), device=dev)
print("class weights:", dict(zip(classes, np.round(w / w.mean(), 2).tolist())))
criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
scaler = torch.amp.GradScaler("cuda")

def evaluate(xs, ys):
    model.eval(); tot, preds = 0.0, []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for x, y in batches(xs, ys, False):
            logits = model(x).float()
            tot += criterion(logits, y).item() * len(x); preds.append(logits.argmax(1).cpu().numpy())
    p = np.concatenate(preds)
    return tot / len(xs), (p == ys).mean(), balanced_accuracy_score(ys, p), f1_score(ys, p, average="macro", zero_division=0)

os.makedirs(args.out, exist_ok=True)
log_txt = open(os.path.join(args.out, "train_log.txt"), "w", buffering=1)
log_csv = open(os.path.join(args.out, "train_log.csv"), "w", newline="", buffering=1)
writer = csv.writer(log_csv)
writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "test_bal_acc", "test_macro_f1", "lr", "epoch_sec"])
log_txt.write(f"datasets: {[os.path.basename(d) for d in args.data]}\n")
hist = []
start = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train(); t0 = time.time(); tot_loss = correct = 0.0
    for x, y in batches(tr_x, tr_y, True):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x); loss = criterion(logits, y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        tot_loss += loss.item() * len(x); correct += (logits.argmax(1) == y).sum().item()
    lr = opt.param_groups[0]["lr"]; sched.step()
    tr_loss, tr_acc = tot_loss / len(tr_x), correct / len(tr_x)
    te_loss, te_acc, te_bal, te_f1 = evaluate(te_x, te_y)
    sec = time.time() - t0
    hist.append((epoch, tr_loss, tr_acc, te_loss, te_acc, te_bal, te_f1))
    line = (f"Epoch {epoch}/{EPOCHS} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | test loss {te_loss:.3f} "
            f"acc {te_acc:.3f} bal_acc {te_bal:.3f} macroF1 {te_f1:.3f} | lr {lr:.1e} | {sec:.0f}s")
    print(line); log_txt.write(line + "\n")
    writer.writerow([epoch] + [f"{v:.4f}" for v in (tr_loss, tr_acc, te_loss, te_acc, te_bal, te_f1)] + [f"{lr:.2e}", f"{sec:.1f}"])

torch.save({
    "state_dict": model.state_dict(),
    "meta": {"arch": "efficientnet_b0", "classes": classes, "input_hw": [H, W], "mean": MEAN, "std": STD,
             "epochs": EPOCHS, "train_images": len(tr_x), "class_weighting": args.weighting,
             "datasets": sorted(set(tr_d.tolist())), "dataset": " + ".join(sorted(set(tr_d.tolist())))},
}, os.path.join(args.out, args.name))

h = np.array(hist)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(h[:, 0], h[:, 1], label="train"); ax[0].plot(h[:, 0], h[:, 3], label="test"); ax[0].set_title("Loss")
ax[1].plot(h[:, 0], h[:, 2], label="train"); ax[1].plot(h[:, 0], h[:, 4], label="test"); ax[1].set_title("Accuracy")
ax[2].plot(h[:, 0], h[:, 5], label="test balanced acc"); ax[2].plot(h[:, 0], h[:, 6], label="test macro F1"); ax[2].set_title("Test balanced accuracy / macro F1")
for a in ax: a.set_xlabel("epoch"); a.grid(alpha=.3); a.legend()
fig.suptitle(f"Training on {' + '.join(sorted(set(tr_d.tolist())))}")
fig.tight_layout(); fig.savefig(os.path.join(args.out, "train_test_curves.png"), dpi=120)

done = f"Done in {(time.time() - start) / 60:.1f} min. Saved {os.path.join(args.out, args.name)} (final epoch)"
print(done); log_txt.write(done + "\n")
