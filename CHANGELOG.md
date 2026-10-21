# Changelog

Changes are grouped by version. Dates describe repository preparation, not a GitHub release or deployment.

## 1.0.0 — 2026-10-21 (public portfolio preparation)

### Added

- Executed EDA notebook, data-quality review and chronological distribution-shift findings (milestone 1).
- Controlled experiment runner: expanding train-only TimeSeriesSplit, hyperparameter search, three feature sets and class-weight comparisons (milestone 3).
- Validation calibration diagnostics, precision–recall figures, permutation importance, group/time-block errors and illustrative cost-based threshold selection.
- Frozen model/threshold provenance using SHA-256 and a separate test-evaluation command with no refitting.
- Behavioral and integration tests using synthetic data; GitHub Actions matrix for Python 3.11/3.13 on Ubuntu/Windows and wheel building.
- Public aggregate result snapshots with integrity manifest and an exporter that validates frozen provenance before publishing evidence.
- Portfolio report, model card, CV wording, release guide, recorded dependency versions and MIT license (milestone 4).

### Changed

- Replaced the starter README with a recruiter-facing overview, measured results, portable setup, full reproduction commands and CSV inference contract.
- Marked all four portfolio milestones complete with evidence and explicit operational limitations.
- Updated package metadata to 1.0.0 and added source/changelog links.
- Expanded Git ignores for environment files, build outputs, keys and serialized models.
- Preserved the temporal baseline, `duration` exclusion, train-fitted preprocessing and CSV ranking interface from milestone 2.

### Evaluation caveats

- Test was already viewed during baseline/EDA; it is not a new independent confirmation.
- Selected model validation AP is 0.2077 and top-20% lift is 1.983×; historical test lift is 1.168× under a substantially changed base rate.
- Costs are illustrative. A fixed threshold does not guarantee the validation contact budget on new data. Feature availability and probability reliability require further validation.

## 0.1.0 — Initial scaffold

- Configurable UCI Bank Marketing baseline with Dummy, Logistic Regression and Random Forest.
- Chronological or stratified splits, validation AP model selection, validation F1 threshold and test metrics.
- UCI data downloader, model serialization, synthetic example CSV and ranked batch inference.
