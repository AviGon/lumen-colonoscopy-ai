"""Build the ERCPMP pathology dataset: 3 classes, patient level train/val/test split.

Labels come from the ERCPMP spreadsheet (per patient), cross-checked against the
diagnosis in each filename. Patients with no usable diagnosis are excluded.
Split 70/15/15 by patient, stratified by class, so no patient appears in two splits.
Copies images unchanged to data/ercpmp_prepared/<split>/<class>/. Originals are not modified.
The val split is written but must not be used until it is explicitly released."""
import collections, csv, json, os, random, re, shutil
import openpyxl

SRC_IMAGES = r"C:\polyp-project\data\ERCPMP\images"
SRC_XLSX = r"C:\polyp-project\data\ERCPMP\raw\ERCPMP_v5_Morphology_Pathological_Data.xlsx"
DST = r"C:\polyp-project\data\ercpmp_prepared"
SEED, VAL_FRAC, TEST_FRAC = 42, 0.15, 0.15
CLASSES = ["adenoma", "hyperplastic_serrated", "adenocarcinoma"]
CLASS_OF = {
    "tubular": "adenoma", "t + v": "adenoma", "tubulovillous": "adenoma", "villous": "adenoma",
    "hyperplastic": "hyperplastic_serrated", "serrated": "hyperplastic_serrated",
    "serrated, hyperplastic": "hyperplastic_serrated", "traditional serrated adenoma": "hyperplastic_serrated",
    "adenocarcinoma": "adenocarcinoma",
}

ws = openpyxl.load_workbook(SRC_XLSX, read_only=True).worksheets[0]
sheet = {str(r[0]): str(r[12]).strip() for r in list(ws.iter_rows(values_only=True))[2:] if r[0] is not None}

by_patient = collections.defaultdict(list)
for name in sorted(os.listdir(SRC_IMAGES)):
    if name.lower().endswith(".jpg"):
        by_patient[name.split("_")[0]].append(name)

labels, excluded, mismatches = {}, [], []
for pid, names in sorted(by_patient.items()):
    sheet_dx = sheet.get(pid, "None")
    m = re.match(r"^\d+_\d+_([A-Za-z]+)", names[0])
    file_dx = m.group(1).lower() if m and m.group(1) != "JNet" else None
    cls = CLASS_OF.get(sheet_dx.lower()) or CLASS_OF.get(file_dx or "")
    if file_dx and CLASS_OF.get(file_dx) and CLASS_OF.get(sheet_dx.lower()) and CLASS_OF[file_dx] != CLASS_OF[sheet_dx.lower()]:
        mismatches.append((pid, sheet_dx, file_dx))
    if cls is None:
        excluded.append((pid, sheet_dx, file_dx, len(names)))
    else:
        labels[pid] = (cls, sheet_dx if CLASS_OF.get(sheet_dx.lower()) else file_dx)

rng = random.Random(SEED)
split = {"train": [], "val": [], "test": []}
for cls in CLASSES:
    pids = sorted(p for p, (c, _) in labels.items() if c == cls)
    rng.shuffle(pids)
    n = len(pids)
    n_test = max(1, round(n * TEST_FRAC)) if n >= 3 else 0
    n_val = max(1, round(n * VAL_FRAC)) if n >= 3 else 0
    split["test"] += pids[:n_test]; split["val"] += pids[n_test:n_test + n_val]; split["train"] += pids[n_test + n_val:]

if os.path.exists(DST):
    shutil.rmtree(DST)
rows = []
for part, pids in split.items():
    for pid in sorted(pids):
        cls, dx = labels[pid]
        os.makedirs(os.path.join(DST, part, cls), exist_ok=True)
        for name in by_patient[pid]:
            shutil.copy2(os.path.join(SRC_IMAGES, name), os.path.join(DST, part, cls, name))
            rows.append({"path": f"{part}/{cls}/{name}", "patient": pid, "diagnosis": dx, "class": cls, "split": part, "dataset": "ercpmp"})

with open(os.path.join(DST, "manifest.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, ["path", "patient", "diagnosis", "class", "split", "dataset"]); w.writeheader(); w.writerows(rows)
json.dump({"seed": SEED, "classes": CLASSES, "class_of_diagnosis": CLASS_OF,
           "patients": {k: sorted(v) for k, v in split.items()},
           "excluded_patients": [{"patient": p, "sheet": s, "filename": f, "images": n} for p, s, f, n in excluded],
           "note": "Split by patient. val is held out and must not be used until released."},
          open(os.path.join(DST, "split.json"), "w"), indent=2)

print(f"{len(by_patient)} patients with images, {sum(map(len, by_patient.values()))} images")
print(f"excluded (no usable diagnosis): {len(excluded)} patients, {sum(e[3] for e in excluded)} images -> {[e[0] for e in excluded]}")
print(f"sheet vs filename class mismatches: {mismatches or 'none'}")
print(f"\n{'split':<6} {'class':<22} patients  images")
for part in split:
    for cls in CLASSES:
        pr = [r for r in rows if r["split"] == part and r["class"] == cls]
        print(f"{part:<6} {cls:<22} {len({r['patient'] for r in pr}):>8}  {len(pr):>6}")
