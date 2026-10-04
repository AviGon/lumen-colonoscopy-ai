"""Build the benign vs malignant dataset.

Sources
  GastroVision (CC BY 4.0, osf.io/84e7f): "Colon polyps" -> benign, "Colorectal cancer" -> malignant.
  ERCPMP (already prepared): adenocarcinoma -> malignant, every other diagnosis -> benign.
  ERCPMP contributes both classes so the image source does not predict the label.

GastroVision has no patient IDs and filenames are random, so near-duplicate images (same lesion,
consecutive frames) are grouped with a perceptual hash and each group stays in one split.
Split 70/15/15 by group, stratified by class. ERCPMP keeps its existing patient level split.
Also reports how often the green on-screen overlay box appears per class (a possible shortcut).
The val split is written but must not be used until it is explicitly released."""
import csv, io, json, os, random, shutil, zipfile
import numpy as np
from PIL import Image

GV_ZIP = r"C:\polyp-project\data\GastroVision\raw\Gastrovision.zip"
ERCPMP = r"C:\polyp-project\data\ercpmp_prepared"
DST = r"C:\polyp-project\data\malignancy_prepared"
SEED, VAL_FRAC, TEST_FRAC, DUP_BITS = 42, 0.15, 0.15, 6
CLASSES = ["benign", "malignant"]
GV_CLASS = {"Colon polyps": "benign", "Colorectal cancer": "malignant"}


def dhash(img):
    """64 bit difference hash of the image centre (ignores borders, overlays and text)."""
    w, h = img.size
    g = img.crop((w // 8, h // 8, w - w // 8, h - h // 8)).convert("L").resize((9, 8), Image.BILINEAR)
    a = np.asarray(g, np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())


def green_box(img):
    """True if the teal/green on-screen thumbnail box covers a noticeable area."""
    a = np.asarray(img.resize((200, 160)), np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return float(((g > 130) & (r < 90) & (b > 90) & (g - r > 80)).mean()) > 0.01


def groups_by_hash(hashes):
    """Union-find over images whose hashes differ by at most DUP_BITS bits."""
    parent = list(range(len(hashes)))
    find = lambda i: i if parent[i] == i else (parent.__setitem__(i, parent[parent[i]]) or find(parent[i]))
    bits = np.unpackbits(np.stack(hashes), axis=1)
    for i in range(len(hashes)):
        close = np.nonzero((bits[i + 1:] != bits[i]).sum(1) <= DUP_BITS)[0] + i + 1
        for j in close:
            parent[find(j)] = find(i)
    return [find(i) for i in range(len(hashes))]


if os.path.exists(DST):
    shutil.rmtree(DST)
rows, rng = [], random.Random(SEED)

# GastroVision
z = zipfile.ZipFile(GV_ZIP)
box_stats = {}
for folder, cls in GV_CLASS.items():
    names = sorted(n for n in z.namelist() if n.startswith(f"Gastrovision/{folder}/") and not n.endswith("/"))
    imgs = [Image.open(io.BytesIO(z.read(n))).convert("RGB") for n in names]
    box_stats[cls] = (sum(map(green_box, imgs)), len(imgs))
    gid = groups_by_hash([dhash(im) for im in imgs])
    groups = sorted(set(gid)); rng.shuffle(groups)
    n_test, n_val = round(len(groups) * TEST_FRAC), round(len(groups) * VAL_FRAC)
    part_of = {g: "test" if k < n_test else "val" if k < n_test + n_val else "train" for k, g in enumerate(groups)}
    print(f"GastroVision {cls}: {len(imgs)} images in {len(groups)} near-duplicate groups")
    for n, im, g in zip(names, imgs, gid):
        part = part_of[g]
        os.makedirs(os.path.join(DST, part, cls), exist_ok=True)
        fname = "gv_" + os.path.basename(n)
        im.save(os.path.join(DST, part, cls, fname), quality=95)
        rows.append({"path": f"{part}/{cls}/{fname}", "patient": f"gvgroup{gid.index(g)}_{cls}",
                     "diagnosis": folder, "class": cls, "split": part, "dataset": "gastrovision"})

# ERCPMP, reusing its patient level split
for r in csv.DictReader(open(os.path.join(ERCPMP, "manifest.csv"))):
    cls = "malignant" if r["class"] == "adenocarcinoma" else "benign"
    os.makedirs(os.path.join(DST, r["split"], cls), exist_ok=True)
    fname = "ercpmp_" + os.path.basename(r["path"])
    shutil.copy2(os.path.join(ERCPMP, r["path"]), os.path.join(DST, r["split"], cls, fname))
    rows.append({"path": f"{r['split']}/{cls}/{fname}", "patient": r["patient"], "diagnosis": r["diagnosis"],
                 "class": cls, "split": r["split"], "dataset": "ercpmp"})

with open(os.path.join(DST, "manifest.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, ["path", "patient", "diagnosis", "class", "split", "dataset"]); w.writeheader(); w.writerows(rows)
json.dump({"classes": CLASSES, "seed": SEED, "sources": {
               "gastrovision": "osf.io/84e7f, CC BY 4.0: Colon polyps -> benign, Colorectal cancer -> malignant",
               "ercpmp": "adenocarcinoma -> malignant, all other diagnoses -> benign; ERCPMP patient split kept"},
           "near_duplicate_hash_bits": DUP_BITS,
           "note": "Split by near-duplicate group (GastroVision) or patient (ERCPMP). val is held out."},
          open(os.path.join(DST, "split.json"), "w"), indent=2)

print("\nGreen overlay box (GastroVision):", {c: f"{k}/{n} = {k / n:.0%}" for c, (k, n) in box_stats.items()})
print(f"\n{'split':<6} {'dataset':<13} {'benign':>7} {'malignant':>10}")
for part in ("train", "val", "test"):
    for ds in ("gastrovision", "ercpmp"):
        sel = [r for r in rows if r["split"] == part and r["dataset"] == ds]
        print(f"{part:<6} {ds:<13} {sum(r['class'] == 'benign' for r in sel):>7} {sum(r['class'] == 'malignant' for r in sel):>10}")
