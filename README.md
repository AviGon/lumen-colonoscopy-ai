#Winners of HealthLink Life Sciences Hackathon 2026 - Track 1 at UCSD!!

# Lumen · Colonoscopy AI

**Every polyp, measured and explained.**

Lumen is a colonoscopy assistant. It outlines and sizes polyps, flags lesions that look malignant, and uses a swarm of AI agents (OpenSwarm) to draft a referenced clinical report. The gastroenterologist reviews and signs the report instead of writing it.

> 🏥 Built for the **HealthLink Hackathon**, **Track 1: Using Open Swarm to work on Colonoscopy**.

![Lumen web app: polyp detection and sizing](images/Screenshot%202026-10-03%20232350.png)

> ⚠️ **Research prototype, not a medical device.** The models are not clinically validated. Everything Lumen produces is a non-diagnostic draft for physician review and must not be used on its own for diagnosis or treatment decisions.

---

## Team

| Name |
| --- |
| Simran Moorjani |
| Avinash Gondela |
| Vibhawari Wusirika |
| Romy Bornstein |

---

## The problem

| | What goes wrong | Evidence |
|---|---|---|
| **Missed lesions** | About **26%** of adenomas are missed during colonoscopy. | Zhao et al., *Gastroenterology* 2019 |
| **Fragmented evidence** | Images, lesion assessments and literature must be brought together into one clinical report. | — |
| **Report overload** | Physicians spend nearly **2 hours** on desk and EHR work for every hour with patients. | Sinsky et al., *Ann Intern Med* 2016 |

## What Lumen does

The physician uploads a still image from the colonoscopy and picks one of two analyses. Within a quick amount of time:

1. **Measure.** A U-Net outlines the polyp and estimates its size in millimetres, with a range and a size category (<5 mm, 5–9 mm, ≥10 mm).
2. **Assess.** An EfficientNet classifier says whether the lesion looks benign or malignant and shows a Grad-CAM heatmap of where it looked.
3. **Report.** OpenSwarm agents read the model outputs, search PubMed and draft a clinical report: description, Paris classification, referenced assessment, points to verify and next steps. The app converts it to a PDF with the image embedded.
4. **Sign.** The physician reviews, edits and signs. Nothing is final without them.

## How it works

```
 ┌──────────────┐   upload    ┌──────────────────────┐   model outputs    ┌──────────────────────────┐
 │  Web app     │ ──────────▶ │  FastAPI  (app.py)   │ ─────────────────▶ │  reports/<case_id>/      │
 │  web/        │             │  • U-Net segmentation│   features.json /  │  input, overlay, gradcam │
 │  index.html  │             │  • EfficientNet-B0   │   pathology.json   │  + TASK.md for agents    │
 └──────▲───────┘             └──────────────────────┘                    └────────────┬─────────────┘
        │                                                                              │
        │  polls; shows report + PDF download                                          ▼
        │                                                                 ┌──────────────────────────┐
        └──────────────────────────────────────────────────────────────── │  OpenSwarm agents        │
                               report_DRAFT.md  ──▶  report_pdf.py        │  + MCP server (server.py)│
                                                                          │  PubMed search           │
                                                                          └──────────────────────────┘
```

- **`app.py`** — FastAPI backend. It saves each upload under `cases/<case_id>/`, runs the chosen model, writes outputs and a `TASK.md` with instructions for the agents to `reports/<case_id>/`, then watches for the agents' `report_DRAFT.md`.
- **`server.py`** — MCP server (stdio) that gives the agents their tools: polyp segmentation and PubMed abstract search.
- **`reports/REPORT_TEMPLATE*.md` / `REPORT_RULES*.md`** — the template and rules the agents follow for each pipeline. The rules require plain clinical language, real and verifiable citations, and no raw model numbers.
- **`report_pdf.py`** — turns the Markdown report into a PDF with the endoscopic image.

## Results

All metrics are on held-out test images that were never used for training or checkpoint selection. Splits are made by patient, or by groups of near-duplicate frames when a dataset has no patient IDs.

| Model | Task | Data | Test result |
|---|---|---|---|
| U-Net (ResNet34 encoder) | Polyp segmentation | Kvasir-SEG (200 test images) | **Dice 0.906** |
| EfficientNet-B0 | Benign vs malignant | GastroVision + ERCPMP (150 patients / groups) | **ROC AUC 0.91**, malignant recall 82%, benign recall 89% (patient level) |

Evaluation plots (confusion matrices, PR curves, loss curves and the best, median and worst segmentation examples) are in [`images/`](images/). Full test metrics are in [`models/test_eval/`](models/test_eval/) (segmentation) and [`models/malignancy/test_eval/`](models/malignancy/test_eval/) (malignancy).

## Getting started

### Requirements

- Python 3.10+
- An NVIDIA GPU is recommended. The CPU works for inference, but slowly.
- [Git LFS](https://git-lfs.com/) to download the model weights

```powershell
git lfs install
git clone https://github.com/vwusirika-jpg/lumen-colonoscopy-ai.git
cd lumen-colonoscopy-ai

python -m venv .venv
.\.venv\Scripts\Activate.ps1

# PyTorch first (CUDA 12.8 build shown; pick the right one for your GPU at pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### Run the web app

```powershell
python -m uvicorn app:app --port 8000
```

Open http://localhost:8000, click **New analysis**, upload a colonoscopy image and choose a pipeline:

| Option | Model | Output |
|---|---|---|
| Polyp Detection & Sizing | U-Net / ResNet34 (`polyp_inference.py`) | mask, overlay, probability heatmap, size and shape features (`features.json`) |
| Malignancy Risk Assessment | EfficientNet-B0 (`polyp_pathology.py`) | benign / malignant probabilities and Grad-CAM (`pathology.json`) |

After the analysis, the page shows a prompt to paste into OpenSwarm. The agents follow `reports/<case_id>/TASK.md`, and once they write `report_DRAFT.md` the page shows the report with a **Download PDF** button.

### Run the MCP server (agent tools)

```powershell
python server.py
```

The server uses stdio transport. Register it with your MCP client (e.g. OpenSwarm) using the virtual environment's Python and the path to `server.py`.

| Tool | Purpose |
|---|---|
| `segment_polyp(image_path, threshold=0.5)` | Runs the U-Net and writes `input.png`, `mask.png`, `overlay.png`, `prob_heatmap.png` and `features.json`. Returns area, size estimate in mm, shape, position and confidence. |
| `search_pubmed(query, max_results=5)` | Queries NCBI E-utilities and returns metadata plus plain-text abstracts, which the agents cite in both reports. Set `NCBI_EMAIL` to send a contact email to NCBI. |

## Training from scratch

The datasets are not included in this repository because of their size. Download them into `data/`:

| Dataset | Used for | License / source |
|---|---|---|
| [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) | Segmentation | Simula, research and education use |
| [ERCPMP-v5](https://data.mendeley.com/datasets/7grhw5tv7n/6) | Malignancy | CC BY 4.0 |
| [GastroVision](https://osf.io/84e7f/) | Malignancy | CC BY 4.0 |

```powershell
# 1. Polyp segmentation (Kvasir-SEG)
python scripts\prepare_kvasir.py
python scripts\train_polyp.py          # -> models\polyp_unet_r34.pth
python scripts\eval_test.py            # -> models\test_eval\

# 2. Malignancy classifier (GastroVision + ERCPMP)
python scripts\ercpmp\prepare_ercpmp.py       # ERCPMP patient-level split (needed by the next step)
python scripts\ercpmp\prepare_malignancy.py   # -> data\malignancy_prepared
python scripts\ercpmp\train_ercpmp.py         # -> models\malignancy\malignancy_effb0.pth
python scripts\ercpmp\eval_ercpmp.py          # -> models\malignancy\test_eval\
python scripts\ercpmp\plot_loss.py models\malignancy
```

The scripts have default paths that point to `C:\polyp-project`. Edit the constants at the top of each script if you cloned the repository somewhere else. Training needs a CUDA GPU. Each prepare script writes a `val` split that is held out and never read by training or evaluation.

## Repository layout

```
app.py                  FastAPI web app
server.py               MCP server with the agents' tools
polyp_inference.py      U-Net inference + polyp feature extraction
polyp_pathology.py      Malignancy classifier inference + Grad-CAM
report_pdf.py           Markdown report -> PDF
web/index.html          Front end
reports/                Report templates and rules for the agents (case outputs are git-ignored)
scripts/                Segmentation scripts; scripts/ercpmp/ holds the malignancy scripts
models/                 Segmentation weights + logs; models/malignancy/ holds the malignancy model (weights via Git LFS)
images/                 Screenshots and evaluation figures
PITCH_README.md         Pitch narrative
```

---

*Lumen is a hackathon research prototype and is not a medical device. It assists physicians and does not provide a diagnosis.*
