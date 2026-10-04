"""Evaluate the final checkpoint once on the 200 held-out test images.
Read only on data; writes metrics and example overlays to models\test_eval."""
import csv, os
import numpy as np
import torch
import segmentation_models_pytorch as smp
from PIL import Image

DATA = r"C:\polyp-project\data\kvasir_prepared\test"
CKPT = r"C:\polyp-project\models\polyp_unet_r34.pth"
OUT = r"C:\polyp-project\models\test_eval"
os.makedirs(OUT, exist_ok=True)

dev = torch.device("cuda")
ck = torch.load(CKPT, map_location=dev, weights_only=False)
meta = ck["meta"]
model = smp.Unet(meta["encoder"], encoder_weights=None, in_channels=3, classes=1).to(dev)
model.load_state_dict(ck["state_dict"]); model.eval()
mean, std, thr = np.array(meta["mean"], np.float32), np.array(meta["std"], np.float32), meta["threshold"]

names = sorted(os.listdir(os.path.join(DATA, "images")))
rows = []
for n in names:
    img = np.array(Image.open(os.path.join(DATA, "images", n)).convert("RGB"))
    gt = np.array(Image.open(os.path.join(DATA, "masks", n))) > 127
    x = torch.from_numpy(((img.astype(np.float32) / 255 - mean) / std).transpose(2, 0, 1))[None].to(dev)
    with torch.no_grad():
        pr = (torch.sigmoid(model(x))[0, 0] > thr).cpu().numpy()
    tp = (pr & gt).sum(); fp = (pr & ~gt).sum(); fn = (~pr & gt).sum()
    dice = 2 * tp / max(2 * tp + fp + fn, 1); iou = tp / max(tp + fp + fn, 1)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    rows.append((n, dice, iou, prec, rec, img, gt, pr))

d = np.array([r[1] for r in rows]); i = np.array([r[2] for r in rows])
p = np.array([r[3] for r in rows]); rc = np.array([r[4] for r in rows])

with open(os.path.join(OUT, "per_image.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["image", "dice", "iou", "precision", "recall"])
    for r in rows: w.writerow([r[0]] + [f"{v:.4f}" for v in r[1:5]])

# Overlays: green = ground truth outline area, red = prediction (yellow = overlap)
def overlay(img, gt, pr):
    o = img.astype(np.float32).copy()
    o[gt & ~pr] = o[gt & ~pr] * 0.4 + np.array([0, 255, 0]) * 0.6
    o[pr & ~gt] = o[pr & ~gt] * 0.4 + np.array([255, 0, 0]) * 0.6
    o[pr & gt] = o[pr & gt] * 0.4 + np.array([255, 255, 0]) * 0.6
    return np.concatenate([img, o.astype(np.uint8)], axis=1)

order = np.argsort(d)
picks = {"worst": order[:3], "median": order[len(order) // 2 - 1: len(order) // 2 + 2], "best": order[-3:]}
for tag, idx in picks.items():
    for k, j in enumerate(idx):
        r = rows[j]
        Image.fromarray(overlay(r[5], r[6], r[7])).save(os.path.join(OUT, f"{tag}_{k + 1}_dice{r[1]:.2f}_{r[0]}"))

summary = (f"Test set: {len(rows)} images (never seen in training)\n"
           f"Mean Dice      {d.mean():.4f}  (median {np.median(d):.4f})\n"
           f"Mean IoU       {i.mean():.4f}  (median {np.median(i):.4f})\n"
           f"Mean precision {p.mean():.4f}\nMean recall    {rc.mean():.4f}\n"
           f"Images with Dice < 0.5: {(d < 0.5).sum()}   Dice >= 0.9: {(d >= 0.9).sum()}\n"
           f"Overlay legend: yellow = correct, red = false positive, green = missed\n")
open(os.path.join(OUT, "test_metrics.txt"), "w").write(summary)
print(summary)
