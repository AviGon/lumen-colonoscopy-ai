"""Split Kvasir-SEG 80/20 and resize to 352x352. Originals are not modified."""
import json, os, random
from PIL import Image
import numpy as np

SRC = r"C:\polyp-project\data\Kvasir-SEG"
DST = r"C:\polyp-project\data\kvasir_prepared"
SIZE, SEED = 352, 42

files = sorted(os.listdir(os.path.join(SRC, "images")))
random.Random(SEED).shuffle(files)
n_train = int(len(files) * 0.8)
split = {"train": sorted(files[:n_train]), "test": sorted(files[n_train:])}

for part, names in split.items():
    for sub in ("images", "masks"):
        os.makedirs(os.path.join(DST, part, sub), exist_ok=True)
    for f in names:
        stem = os.path.splitext(f)[0]
        img = Image.open(os.path.join(SRC, "images", f)).convert("RGB")
        img.resize((SIZE, SIZE), Image.BILINEAR).save(os.path.join(DST, part, "images", stem + ".png"))
        m = Image.open(os.path.join(SRC, "masks", f)).convert("L")
        m = np.array(m.resize((SIZE, SIZE), Image.NEAREST))
        Image.fromarray(np.where(m > 127, 255, 0).astype(np.uint8)).save(os.path.join(DST, part, "masks", stem + ".png"))

json.dump({"seed": SEED, "size": SIZE, "source": SRC, **split}, open(os.path.join(DST, "split.json"), "w"), indent=2)
print({k: len(v) for k, v in split.items()})
