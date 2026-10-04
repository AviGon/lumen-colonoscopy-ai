"""Lumen · Colonoscopy AI: local web app for the polyp review pipeline.

Upload an image and pick a pipeline:
  - colonoscopy_segmentation: U-Net polyp segmentation, size and shape features (Kvasir-SEG)
  - colonoscopy_malignancy: EfficientNet-B0 benign vs malignant classifier (GastroVision + ERCPMP)
Either writes reports/<case_id>/ together with TASK.md, the instructions an OpenSwarm
agent follows to write report_DRAFT.md.
The page polls until that report appears and then offers it for download.

Run:  .venv\\Scripts\\python -m uvicorn app:app --port 8000
"""

from __future__ import annotations

import io
import json
import re
import threading
import time
from datetime import date
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image

from polyp_inference import run_inference
from polyp_pathology import run_pathology
from report_pdf import build_pdf

PROJECT_ROOT = Path(__file__).parent.resolve()
CASES_DIR = PROJECT_ROOT / "cases"
REPORTS_DIR = PROJECT_ROOT / "reports"
WEB_DIR = PROJECT_ROOT / "web"
REPORT_NAME = "report_DRAFT.md"
PROCEDURES = {
    "colonoscopy_segmentation": {
        "label": "Polyp Detection & Sizing",
        "description": "Locates the polyp, outlines it and estimates its size in millimetres.",
        "result": "features.json",
        "artifacts": ("input.png", "overlay.png", "mask.png", "prob_heatmap.png", "features.json"),
        "report_image": "overlay.png",
        "report_caption": "Endoscopic image (left) and AI outline of the polyp (right).",
    },
    "colonoscopy_malignancy": {
        "label": "Malignancy Risk Assessment",
        "description": "Assesses whether the lesion looks benign (a polyp) or malignant (colorectal cancer).",
        "result": "pathology.json",
        "artifacts": ("input.png", "gradcam.png", "pathology.json"),
        "report_image": "input.png",
        "report_caption": "Endoscopic image.",
        "ckpt": PROJECT_ROOT / "models" / "malignancy" / "malignancy_effb0.pth",
        "template": "REPORT_TEMPLATE_MALIGNANCY.md", "rules": "REPORT_RULES_MALIGNANCY.md",
        "report_kind": "malignancy risk assessment report",
        "literature": "endoscopic features that distinguish benign colorectal polyps from colorectal cancer "
                      "(e.g. ulceration, depression, irregular or fold-converging margins, friability, "
                      "luminal narrowing) and features predicting invasive cancer in polyps",
    },
}
ALIASES = {"colonoscopy": "colonoscopy_segmentation"}  # cases made before the pathology pipeline
ALL_ARTIFACTS = {a for p in PROCEDURES.values() for a in p["artifacts"]}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
REPORT_SETTLE_SECONDS = 3  # report counts as finished once unchanged this long
CASE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")

app = FastAPI(title="Lumen · Colonoscopy AI")
_model_lock = threading.Lock()  # one GPU inference at a time


def _case_dir(case_id: str) -> Path:
    if not CASE_ID_RE.match(case_id):
        raise HTTPException(400, "Invalid case id.")
    path = REPORTS_DIR / case_id
    if _procedure(path) is None:
        raise HTTPException(404, "Case not found.")
    return path


def _procedure(case: Path) -> str | None:
    """Pipeline key of a case folder, or None if it is not a case."""
    meta_path = case / "case.json"
    if meta_path.exists():
        key = json.loads(meta_path.read_text(encoding="utf-8")).get("procedure")
        key = ALIASES.get(key, key)
        return key if key in PROCEDURES else None
    for key, proc in PROCEDURES.items():
        if (case / proc["result"]).exists():
            return key
    return None


def _new_case_id(filename: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(filename).stem).strip("-")[:60] or "image"
    case_id, n = stem, 2
    while (REPORTS_DIR / case_id).exists() or (CASES_DIR / case_id).exists():
        case_id, n = f"{stem}-{n}", n + 1
    return case_id


def _report_status(case: Path) -> str:
    report = case / REPORT_NAME
    if not report.exists() or report.stat().st_size == 0:
        return "waiting"
    if time.time() - report.stat().st_mtime < REPORT_SETTLE_SECONDS:
        return "writing"
    return "ready"


def _task_text(case: Path, procedure: str) -> str:
    if "ckpt" in PROCEDURES[procedure]:
        return _classifier_task_text(case, PROCEDURES[procedure])
    return f"""# Task: write the polyp report for case `{case.name}`

You are writing a draft polyp report for a physician. Write it as a clinical note.
Work only inside this folder: `{case}`

## Inputs (already produced by the segmentation model, do not modify)
- `input.png`: the colonoscopy image.
- `overlay.png`: the image (left) and the AI outline of the polyp (right, yellow fill, red border).
- `features.json`: measurements, including `regions[i].size_estimate_mm`.

## Steps
1. Read `{REPORTS_DIR / 'REPORT_RULES.md'}` and follow every rule. In particular: no file names, model
   names, machine learning metrics, pixel values or descriptions of where the lesion sits in the image.
2. Copy the structure of `{REPORTS_DIR / 'REPORT_TEMPLATE.md'}` and fill every `{{placeholder}}`.
   Take the size numbers from `features.json`.
3. Look at `input.png` and `overlay.png` to write the description, the Paris classification and the
   points to verify. Use the outline only to judge whether the lesion is fully in view and how reliable
   the size estimate is; describe any such issue in clinical terms.
4. Search PubMed for literature that fits this lesion (morphology, size category, notable features) and
   cite it. Use real, verifiable references only.
5. Case `{case.name}`, date `{date.today().isoformat()}`.
6. Delete the template's HTML comment block.

## Output
Write the finished report to `{case / REPORT_NAME}`, in Markdown, once and in full. The web app turns
it into a PDF with the image. Do not create or change any other files.
"""


def _classifier_task_text(case: Path, proc: dict) -> str:
    kind = proc["report_kind"]
    return f"""# Task: write the {kind} for case `{case.name}`

You are writing a draft {kind} for a physician. Write it as a clinical note.
Work only inside this folder: `{case}`

## Inputs (already produced by the classification model, do not modify)
- `input.png`: the colonoscopy image.
- `gradcam.png`: the image (left) and the area the AI relied on (right, red = most important).
- `pathology.json`: the AI assessment (`predicted_class`), its probability and the margin over the other class(es).

## Steps
1. Read `{REPORTS_DIR / proc['rules']}` and follow every rule. In particular: likelihood in words only,
   the fixed reliability sentence, and no file names, model names, probability numbers or machine
   learning metrics.
2. Copy the structure of `{REPORTS_DIR / proc['template']}` and fill every `{{placeholder}}`.
3. Look at `input.png` to write the description and assessment, and at `gradcam.png` only to check
   whether the AI relied on the lesion itself.
4. Search PubMed for literature on {proc['literature']}, as relevant to this image, and cite it.
   Use real, verifiable references only.
5. Case `{case.name}`, date `{date.today().isoformat()}`.
6. Delete the template's HTML comment block.

## Output
Write the finished report to `{case / REPORT_NAME}`, in Markdown, once and in full. The web app turns
it into a PDF with the image. Do not create or change any other files.
"""


def _summary(case: Path) -> dict:
    procedure = _procedure(case)
    proc = PROCEDURES[procedure]
    result = json.loads((case / proc["result"]).read_text(encoding="utf-8"))
    meta_path = case / "case.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return {
        "case_id": case.name,
        "procedure": procedure,
        "procedure_label": proc["label"],
        "beta": proc.get("beta", False),
        "artifacts": [a for a in proc["artifacts"] if a.endswith(".png")],
        "original_name": meta.get("original_name", Path(result.get("image", case.name)).name),
        "created": meta.get("created", time.strftime("%Y-%m-%d %H:%M", time.localtime(case.stat().st_mtime))),
        "status": _report_status(case),
        "has_task": (case / "TASK.md").exists(),
        "task_path": str(case / "TASK.md"),
        "result": result,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/procedures")
def list_procedures() -> list[dict]:
    return [{"key": k, "label": p["label"], "description": p["description"], "beta": p.get("beta", False),
             "available": "ckpt" not in p or p["ckpt"].exists()}
            for k, p in PROCEDURES.items() if not p.get("hidden")]


@app.post("/api/cases")
def create_case(image: UploadFile = File(...), procedure: str = Form(...)) -> dict:
    procedure = ALIASES.get(procedure, procedure)
    if procedure not in PROCEDURES or PROCEDURES[procedure].get("hidden"):
        raise HTTPException(400, f"Unsupported procedure: {procedure}")
    proc = PROCEDURES[procedure]
    if "ckpt" in proc and not proc["ckpt"].exists():
        raise HTTPException(503, "This model has not been trained yet.")
    data = image.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is larger than 25 MB.")
    try:
        Image.open(io.BytesIO(data)).verify()
    except Exception:
        raise HTTPException(400, "File is not a readable image.")

    original_name = Path(image.filename or "image.png").name
    case_id = _new_case_id(original_name)
    upload_dir = CASES_DIR / case_id
    upload_dir.mkdir(parents=True)
    suffix = Path(original_name).suffix.lower() or ".png"
    upload_path = upload_dir / f"original{suffix}"
    upload_path.write_bytes(data)

    case = REPORTS_DIR / case_id
    with _model_lock:
        if "ckpt" in proc:
            run_pathology(str(upload_path), str(case), ckpt=str(proc["ckpt"]))
        else:
            run_inference(str(upload_path), str(case))
    (case / "case.json").write_text(json.dumps({
        "procedure": procedure, "original_name": original_name,
        "created": time.strftime("%Y-%m-%d %H:%M"), "upload": str(upload_path),
    }, indent=2), encoding="utf-8")
    (case / "TASK.md").write_text(_task_text(case, procedure), encoding="utf-8")
    return _summary(case)


@app.get("/api/cases")
def list_cases() -> list[dict]:
    cases = [p for p in REPORTS_DIR.iterdir() if p.is_dir() and _procedure(p)] if REPORTS_DIR.exists() else []
    cases.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [{k: v for k, v in _summary(p).items() if k != "result"} for p in cases]


@app.get("/api/cases/{case_id}")
def get_case(case_id: str) -> dict:
    return _summary(_case_dir(case_id))


@app.get("/api/cases/{case_id}/files/{name}")
def get_file(case_id: str, name: str) -> FileResponse:
    if name not in ALL_ARTIFACTS | {REPORT_NAME, "TASK.md"}:
        raise HTTPException(404, "Unknown file.")
    path = _case_dir(case_id) / name
    if not path.exists():
        raise HTTPException(404, "File not available yet.")
    return FileResponse(path, headers={"Cache-Control": "no-store"})


@app.get("/api/cases/{case_id}/report.pdf")
def download_report(case_id: str, inline: bool = False) -> Response:
    case = _case_dir(case_id)
    if _report_status(case) != "ready":
        raise HTTPException(409, "Report is not ready yet.")
    proc = PROCEDURES[_procedure(case)]
    pdf = build_pdf((case / REPORT_NAME).read_text(encoding="utf-8"),
                    case / proc["report_image"], proc["report_caption"])
    disposition = "inline" if inline else "attachment"
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'{disposition}; filename="{case_id}_polyp_report.pdf"'})
