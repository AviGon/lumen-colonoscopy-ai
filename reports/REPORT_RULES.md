# Rules for every generated polyp report

The report is read by a physician. Write it as a clinical note, in plain clinical language.

1. Do NOT include a "Surface and vascular pattern (NICE)" section, and do not cite the NICE papers (Hewett 2012, Hayashi 2013).
2. Always give the approximate size in mm from `features.json` -> `regions[i].size_estimate_mm`: low to high range, most likely value and likely category (<6, 6-9, >=10 mm), with a short note that it was estimated from the image without a reference object and must be confirmed with an instrument.
3. If the range spans more than one clinical size category, say the category cannot be determined from this image alone, and list it under "Points to verify".
4. Pre-fill the physician sign-off size line with the AI estimate, e.g. `Size: ______ mm (AI estimate: ~10 mm, range 5 to 20 mm)`.
5. Do NOT include technical or machine learning details: no file names or paths, no image source, no model or software names, no accuracy or Dice figures, no pixel values, no circularity, solidity, colour values or probability numbers. Translate anything relevant into clinical terms (e.g. "lesion extends beyond the field of view", "lesion borders are well defined").
6. Do NOT describe where the lesion sits within the image frame (e.g. "upper left of the image"). Anatomical location in the colon is unknown from a still image; say so only if relevant.

Always start from REPORT_TEMPLATE.md (same folder); it already follows all rules above.
