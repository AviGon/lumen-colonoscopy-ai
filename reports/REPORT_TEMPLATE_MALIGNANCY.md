<!--
MALIGNANCY RISK TEMPLATE. Copy to reports/<case_id>/report_DRAFT.md and fill every {placeholder}.
Values come from reports/<case_id>/pathology.json; the endoscopic image is added to the PDF automatically.
Follow REPORT_RULES_MALIGNANCY.md: clinical language only, likelihood in words, no probability numbers,
file names, model names or ML metrics. Delete this comment block in the final report.
-->
# Malignancy Risk Assessment Report

**AI-assisted draft for physician review. Not a diagnosis.**

| | |
|---|---|
| Case | {case_id} |
| Date | {date} |
| Examination | Colonoscopy, single still image |

## AI assessment

**{Looks benign / Looks malignant}**, {high / moderate / low (uncertain)} likelihood.

This assessment comes from an AI tool that has not been clinically validated. It is designed to distinguish established colorectal tumours from benign polyps, may miss some cancers and may flag some benign polyps, and cannot detect early cancer within a benign-looking polyp; it must not guide management on its own. Histopathology is the reference standard.

## Description

{Shape and size impression, surface, colour relative to surrounding mucosa, margins, ulceration, depression, friability or bleeding, fold convergence, luminal narrowing, debris, image quality. Clinical language only; no NICE.}

## Assessment

{Which endoscopic features in this image support or argue against malignancy, with citations: features of benign polyps (smooth surface, regular margins, preserved surface pattern) versus features concerning for cancer (ulceration, depression, irregular or raised margins, friability, fold convergence, luminal narrowing). Note what cannot be judged from one still image (consistency, lifting, extent). Use real, verifiable references.}

**Impression.** {One to three sentences: the endoscopic appearance, whether it agrees with the AI assessment, histology unknown.}

## Points to verify

- {Malignancy concern per rule 3, if applicable}
- Histology is required; the AI assessment is not a diagnosis.
- {AI assessment not based on the lesion, per rule 5, if applicable}
- {Limits of this assessment: single view, lesion not fully in view, image quality}

## Suggested next steps

- {If malignancy is suspected: biopsy or resection with histology, tattooing, and staging per local practice.}
- {If benign appearing: resection per standard practice with histology.}
- Surveillance or further management to follow histology [ref].

## Physician sign-off

- [ ] Report reviewed
- [ ] Histology: ________ (AI assessment: {looks benign / looks malignant}, {likelihood} likelihood)
- [ ] AI assessment agrees with histology: Yes / No
- Physician: ____________  Date: ________

---

### References
{Lesion specific references found during the literature search, numbered, with PMID where available}
