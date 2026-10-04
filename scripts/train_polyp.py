"""Fine tune U-Net (ResNet34, ImageNet) on the 800 Kvasir-SEG training images.
No validation, no test evaluation. Keeps the final epoch checkpoint."""
import csv, os, random, time
import numpy as np
import torch
import albumentations as A
import segmentation_models_pytorch as smp
from PIL import Image

DATA = r"C:\polyp-project\data\kvasir_prepared\train"
OUT = r"C:\polyp-project\models"
EPOCHS, BATCH, LR, SIZE, SEED = 25, 8, 1e-4, 352, 42
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

if not torch.cuda.is_available():
    raise SystemExit("GPU not available to PyTorch. Stopping (no CPU fallback).")
dev = torch.device("cuda")
print(f"torch {torch.__version__} | GPU: {torch.cuda.get_device_name(0)}")

# Load all training pairs into memory (small dataset, avoids Windows worker issues)
names = sorted(os.listdir(os.path.join(DATA, "images")))
imgs = [np.array(Image.open(os.path.join(DATA, "images", n)).convert("RGB")) for n in names]
masks = [(np.array(Image.open(os.path.join(DATA, "masks", n))) > 127).astype(np.float32) for n in names]
print(f"Loaded {len(imgs)} training images")

aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(scale=(0.9, 1.1), translate_percent=0.05, rotate=(-15, 15), p=0.5),
    A.RandomBrightnessContrast(p=0.5),
])
mean, std = np.array(MEAN, np.float32), np.array(STD, np.float32)

def batch_iter():
    order = np.random.permutation(len(imgs))
    for i in range(0, len(order), BATCH):
        xb, yb = [], []
        for j in order[i:i + BATCH]:
            r = aug(image=imgs[j], mask=masks[j])
            xb.append(((r["image"].astype(np.float32) / 255 - mean) / std).transpose(2, 0, 1))
            yb.append(r["mask"][None])
        yield torch.from_numpy(np.stack(xb)).to(dev), torch.from_numpy(np.stack(yb)).to(dev)

model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(dev)
dice_loss, bce = smp.losses.DiceLoss(mode="binary"), torch.nn.BCEWithLogitsLoss()
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
scaler = torch.amp.GradScaler("cuda")

os.makedirs(OUT, exist_ok=True)
log_txt = open(os.path.join(OUT, "train_log.txt"), "w", buffering=1)
log_csv = open(os.path.join(OUT, "train_log.csv"), "w", newline="", buffering=1)
writer = csv.writer(log_csv); writer.writerow(["epoch", "loss", "train_dice", "lr", "epoch_sec", "eta_sec"])

n_batches = (len(imgs) + BATCH - 1) // BATCH
start = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train(); t0 = time.time(); tot_loss = inter = union = 0.0
    for b, (x, y) in enumerate(batch_iter(), 1):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)
            loss = dice_loss(logits, y) + bce(logits, y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        tot_loss += loss.item() * len(x)
        p = (torch.sigmoid(logits.float()) > 0.5).float()
        inter += (p * y).sum().item(); union += (p.sum() + y.sum()).item()
        print(f"\r  epoch {epoch}/{EPOCHS} batch {b}/{n_batches} loss {loss.item():.3f}", end="", flush=True)
    lr = opt.param_groups[0]["lr"]; sched.step()
    sec = time.time() - t0; eta = sec * (EPOCHS - epoch)
    loss_m, dice_m = tot_loss / len(imgs), 2 * inter / max(union, 1)
    line = (f"Epoch {epoch}/{EPOCHS} | loss {loss_m:.3f} | train Dice {dice_m:.3f} | lr {lr:.1e} "
            f"| {sec:.0f}s/epoch | ETA {int(eta // 60)}m {int(eta % 60)}s")
    print("\r" + line + " " * 10)
    log_txt.write(line + "\n"); writer.writerow([epoch, f"{loss_m:.4f}", f"{dice_m:.4f}", f"{lr:.2e}", f"{sec:.1f}", f"{eta:.0f}"])

torch.save({
    "state_dict": model.state_dict(),
    "meta": {"arch": "Unet", "encoder": "resnet34", "input_size": SIZE, "mean": MEAN, "std": STD,
             "threshold": 0.5, "epochs": EPOCHS, "train_images": len(imgs)},
}, os.path.join(OUT, "polyp_unet_r34.pth"))
done = f"Done in {(time.time() - start) / 60:.1f} min. Saved {os.path.join(OUT, 'polyp_unet_r34.pth')}"
print(done); log_txt.write(done + "\n")
print("Note: train Dice is measured on training images, not the real test score.")
