<!--
REPORT TEMPLATE. Copy to reports/<case_id>/report_DRAFT.md and fill every {placeholder}.
Numbers come from reports/<case_id>/features.json; the endoscopic image is added to the PDF automatically.
Follow REPORT_RULES.md: clinical language only, no file names, model names, ML metrics,
pixel values or position-in-image descriptions. Delete this comment block in the final report.
-->
# Polyp Report

**AI-assisted draft for physician review. Not a diagnosis.**

| | |
|---|---|
| Case | {case_id} |
| Date | {date} |
| Examination | Colonoscopy, single white light still image |

## Findings

| | |
|---|---|
| Polyp seen | {Yes / No} |
| Number of lesions | {n} |
| Estimated size | **~{size_typical} mm** (range {size_low} to {size_high} mm) |
| Likely size category | {<6 mm / 6-9 mm / >=10 mm}{, cannot be determined from this image alone, if the range spans categories} |
| Lesion fully in view | {Yes / No, extends beyond the field of view} |
| Morphology (Paris) | {e.g. 0-Is, sessile; with qualifier if the base is not visible} |

Size was estimated from the image without a reference object (such as open forceps or a snare) and must be confirmed during the procedure.

## Description

{Shape, surface, colour relative to surrounding mucosa, lobulation, relation to folds, ulceration or depression, debris, bleeding, image quality (prep, glare, blur). Clinical language only.}

## Assessment

**Morphology.** {Paris type with reasoning, cite [1, 2].}

{Optional: relevant literature for this lesion type, with citations. No NICE.}

**Size.** {Estimated size and category, and why it matters: a lesion of 10 mm or more, if adenomatous, is an advanced adenoma and changes the surveillance interval [ref Gupta]. If the category is uncertain or the lesion is not fully in view, say so. Visual size estimates are known to be imprecise [ref ICCR].}

**Impression.** {One to three sentences: morphology, estimated size, notable features. Histology unknown.}

## Points to verify

- {Size: no reference object in view; confirm with an instrument.}
- {Anything that limits this assessment: lesion not fully in view, base or stalk not visible, single view, image quality, possible additional lesions.}

## Suggested next steps

- Confirm morphology and measure size with a reference instrument.
- {Lesion specific steps}
- Resection and histopathology per standard practice; surveillance interval to follow histology and size [ref Gupta].

## Physician sign-off

- [ ] Report reviewed
- [ ] Morphology confirmed: ________
- [ ] Size: ______ mm (AI estimate: ~{size_typical} mm, range {size_low} to {size_high} mm)
- Physician: ____________  Date: ________

---

### References
1. The Paris endoscopic classification of superficial neoplastic lesions: esophagus, stomach, and colon. *Gastrointest Endosc* 2003;58(6 Suppl):S3 to S43.
2. Endoscopy Campus, Paris classification. https://www.endoscopy-campus.com/en/classifications/paris-classification-early-cancer/
{Lesion specific references found during the literature search}
{n}. ICCR, Endoscopic polyp size and classification. https://www.iccr-cancer.org/docs/ICCR-Poly-EndoscopicPS.pdf
{n}. Gupta S et al. Recommendations for follow-up after colonoscopy and polypectomy: US Multi-Society Task Force on Colorectal Cancer. *Gastroenterology* 2020;158:1131 to 1153.
