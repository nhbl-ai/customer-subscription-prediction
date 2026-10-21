# Model card — Customer subscription ranking

## Scope

Educational offline ranking of bank term-deposit subscriptions on UCI Bank Marketing.
The selected experiment is `scheduled_assumptions/logistic_regression`, raw score,
`C=0.1`, `class_weight=None`; threshold `0.1531420788149499`.
No deployed service, prospective business trial or operational approval is claimed.

## Data and prediction point

Source: Moro, Rita & Cortez (2014), [UCI Bank Marketing](https://doi.org/10.24432/C5K306),
CC BY 4.0. Use `bank-additional-full.csv`: 41,188 rows ordered by date according to UCI,
covering May 2008–November 2010. Target `y`: no → 0, yes → 1.
Input CSV SHA-256: `74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8`.

Prediction is after scheduling and before the upcoming call. `duration` is excluded.

| Feature group | Availability assumption |
| --- | --- |
| Customer profile: age, job, marital, education, default, housing, loan | Profile is current and available before scoring |
| campaign | Must be the upcoming contact's sequence number, not a later campaign total; unverified |
| contact, month, day_of_week | Must match the planned call channel/date already known at scoring |
| pdays, previous, poutcome | Must describe only earlier contacts, frozen before the upcoming call |
| emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed | Must use releases available at scoring, accounting for publication/revision lag; unverified |

Feature ablations test sensitivity to these assumptions, not whether the assumptions hold.
The historical CSV cannot establish release-time availability. There is no customer ID
or complete timestamp to verify repeat-customer isolation or label latency.

## Training and selection

- Ordered outer split: 28,831 train / 6,178 validation / 6,179 historical test.
- Numeric median imputation/scaling and categorical imputation/one-hot encoding inside a Pipeline.
- Three expanding train-only CV folds, no row gap in the recorded configuration; average precision tuning objective.
- Three feature groups; Dummy/LR/RF; validation AP selects the candidate, validation Brier selects score form.
- Validation utility selects the threshold with illustrative cost/value 1:10 and a 20% contact limit. Utility is label-based, not causal profit.
- No train+validation refit. Separate test command loads the saved model and checks SHA-256 provenance.

## Evaluation

| Metric | Validation | Historical test |
| --- | ---: | ---: |
| Positive rate | 10.65% | 38.45% |
| Average precision | 0.2077 | 0.4508 |
| Precision top 20% | 21.12% | 44.90% |
| Lift top 20% | 1.983× | 1.168× |
| Precision at fixed threshold | 23.12% | 45.21% |
| Recall at fixed threshold | 37.39% | 58.46% |
| Brier score | 0.09185 | 0.27497 |
| Selected by fixed threshold | 17.22% | 49.72% |

The 20% ranking metrics and fixed-threshold selection measure different policies.
The validation budget is not guaranteed by a fixed threshold under shift. Score ties
are sorted by original row order, without an inferred priority among tied customers.

Test had already been viewed during baseline/EDA. Validation is reused for model,
calibration and threshold decisions. These estimates are exploratory; there are no
confidence intervals or independent future-period confirmation.

## Limitations and intended use

Scores support a reproducible offline portfolio example. They are not established as
calibrated future probabilities. No claim is made for other markets or current campaigns.
Group diagnostics are descriptive and incomplete as a fairness assessment. Age,
occupation and other profile features can affect rankings; appropriateness for an actual
application needs a separate review with representative data.

Before real use, verify feature availability and label timing, evaluate a new period,
measure uncertainty and group outcomes, assess a batch budget policy, and validate
actual costs/benefits. Monitor score distributions, realized rates, ranking performance,
calibration and contact volumes by period once delayed labels arrive.

Only load trusted joblib artifacts. Inference retains input columns in the exported CSV;
the user controls access and retention of any customer data supplied locally.

## Artifacts and reproducibility

[Recorded aggregate results](results/README.md), [experiment config](../configs/milestone3.json),
[findings](milestone3_findings.md) and [environment snapshot](../requirements-lock.txt).
Runtime data, models and row-level predictions are Git-ignored. The model is rebuilt
locally using the documented training procedure; no binary artifact is distributed.
