# Recorded experiment evidence

This directory is versioned so GitHub readers can inspect measured results without training a model.
The snapshot was prepared for the portfolio on 2026-10-21 from the existing frozen milestone-3 run.

- `metrics.json`: complete validation selection report, with portable config paths.
- `test_metrics.json`: historical test metrics tied to the frozen selection hash.
- `selection.json`: original data/config/model/validation-report hashes and frozen choices.
- `manifest.json`: hashes of the public snapshot files and original selection identity.
- CSVs: aggregate CV scores, comparisons, costs, calibration, importance and group/time-block errors.
- Four PNGs: precision–recall, calibration, permutation importance and temporal diagnostics.

The validation report retains `test_evaluated: false` because it was created before
the separate test command. The subsequent test results are in `test_metrics.json`.
Public `metrics.json` replaces machine-specific paths with those in the config;
its hash therefore differs from `selection.json`'s original validation-report hash.
Other original hashes identify the local source artifacts, which are not included.
`selection.json` is copied byte-for-byte so its hash still matches the test report.
Git attributes preserve the exact snapshot bytes across Windows/Linux checkouts.

Regenerate after a complete verified run:

```powershell
.\.venv\Scripts\python.exe tools/export_portfolio.py
```

This command checks the frozen data/config/model/report and the test selection identity.
It copies only the listed aggregate outputs and changes no model decision.
It does not rerun training or write raw data, row-level predictions or joblib files here.
The manifest detects changes in the snapshot; it is not a signed proof of training history.

Test was already inspected during baseline/EDA. Cost scenarios are illustrative,
and distribution shift limits interpretation; read [the model card](../model_card.md).

Source data: Moro, Rita & Cortez (2014), [UCI Bank Marketing](https://doi.org/10.24432/C5K306),
CC BY 4.0. Tables/figures are derived analyses, including target encoding, exclusion of
post-call duration, chronological splits and model evaluation. Retain this attribution.
