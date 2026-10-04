# Lumen · Colonoscopy AI

**Every polyp, measured and explained.**
Lumen is colonoscopy AI that outlines and sizes polyps, flags lesions that look malignant, and drafts the clinical report, so the gastroenterologist reviews and signs instead of typing.

---

## The hook: meet Dr. Amy

It's 7:40 a.m. Amy, a gastroenterologist, has 14 colonoscopies before lunch. Every polyp she finds needs two answers in seconds: *how big is it, and could it be cancer?* After every case comes a report. By the end of the day, the reports are still waiting.

*(Amy is an illustrative persona.)*

## The problem

| | What goes wrong | Evidence |
|---|---|---|
| **Missed lesions** | About **26%** of adenomas are missed during colonoscopy. | Zhao et al., *Gastroenterology* 2019 |
| **Sizing by eye** | A polyp of **10 mm or more** moves a patient to a 3-year surveillance interval, yet size is estimated visually. | Gupta et al., US Multi-Society Task Force, 2020 |
| **Report overload** | Physicians spend nearly **2 hours** on desk and EHR work for every hour with patients. | Sinsky et al., *Ann Intern Med* 2016 |

For Amy, that means rushed size calls, uncertain risk calls, and an evening of documentation.

## The solution

Amy uploads a still image and picks an analysis. In about **3 seconds**:

1. **Measure.** Lumen outlines the polyp and estimates its size in millimetres, with a range and a size category.
2. **Assess.** Lumen says whether the lesion looks benign or malignant.
3. **Report.** AI agents draft a clinical report: description, Paris classification, PubMed-referenced assessment, points to verify and next steps, delivered as a PDF with the image.
4. **Sign.** Amy reviews, edits and signs. Nothing is final without her.

**Proof, on held-out test images split by patient**

| Measure | Result |
|---|---|
| Polyp outline vs expert annotation (Dice) | **0.91** |
| Malignancy ROC AUC | **0.91** |
| Cancers caught / benign polyps cleared | **82% / 89%** (about 4 in 5 and 9 in 10) |
| Time to answer | **~3 seconds** on one laptop GPU |

## The value

| For | What changes |
|---|---|
| **Amy** | Objective size and a risk flag at a glance; a drafted, referenced report to sign, not write. |
| **Her clinic** | Consistent, structured reports and fewer after-hours charting hours. |
| **Her patients** | The right surveillance interval, and faster action on lesions that look malignant. |

## The market

| Signal | Size | Source |
|---|---|---|
| US colonoscopies per year | **~15 million** | 2012 US estimate |
| Active US gastroenterologists | **15,678** | AAMC Physician Specialty Data Report, 2022 |
| New colorectal cancers worldwide per year | **1.93 million** | GLOBOCAN 2022 |

**Bottom-up:** ~15M US procedures × **[$__]** per AI-assisted report = **[$__]** US opportunity, before international markets.

## Why us, why now

- **Working product today:** two trained models, a web app and an agent-written PDF report, running end to end.
- **Built with evidence:** patient-level splits, a held-out validation set, and honest limits stated in every report.
- **Agents make reporting cheap:** literature search and clinical drafting that used to need a person now take minutes.

## Closing

Back to Amy: 14 cases, 14 measured polyps, 14 drafted reports. **She signs instead of types.**

**Next:** clinical validation with endoscopists, support for narrow-band imaging, and multi-polyp detection.

**The ask:** [your ask, e.g. $__ to run a validation study with __ clinics]

[Founder name] · [email]

---

*Lumen is a research prototype and not a medical device. It assists physicians and does not provide a diagnosis.*
