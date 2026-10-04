# Rules for every generated malignancy risk assessment report

The report is read by a physician. Write it as a clinical note, in plain clinical language.

1. Do NOT include a "Surface and vascular pattern (NICE)" section, and do not cite the NICE papers (Hewett 2012, Hayashi 2013).
2. State the AI assessment in words, never as probability numbers. Use `pathology.json`:
   - `predicted_class` "benign": "Looks benign"; "malignant": "Looks malignant".
   - Likelihood from `top_probability`: >= 0.85 "high", >= 0.70 "moderate", otherwise "low (uncertain)".
3. If the assessment is "Looks malignant", or `probabilities.malignant` is 0.30 or more, list first under "Points to verify": "Features concerning for malignancy: assess carefully for invasion, biopsy or resect with histological confirmation, and consider tattooing and staging per local practice."
4. Always include this sentence, unchanged, directly under the AI assessment:
   "This assessment comes from an AI tool that has not been clinically validated. It is designed to distinguish established colorectal tumours from benign polyps, may miss some cancers and may flag some benign polyps, and cannot detect early cancer within a benign-looking polyp; it must not guide management on its own. Histopathology is the reference standard."
5. Look at `gradcam.png`. If the highlighted area is not on the lesion (e.g. on folds, glare, instruments or background), add to "Points to verify": "The AI assessment was not based on the lesion itself and should be disregarded." Do not otherwise mention Grad-CAM or heat maps.
6. Do NOT include technical or machine learning details: no file names or paths, no image source, no model, dataset or software names, no accuracy, AUC, sensitivity or specificity figures, no probability numbers, no pixel or colour values.
7. Do NOT describe where the lesion sits within the image frame (e.g. "upper left of the image").
8. Pre-fill the physician sign-off line with the AI assessment in words, e.g. `Histology: ________ (AI assessment: looks benign, high likelihood)`.

Always start from REPORT_TEMPLATE_MALIGNANCY.md (same folder); it already follows all rules above.
