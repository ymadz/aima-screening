# Findings and Evaluation

## Audit Scope

This audit verified the generated `data/processed/dataset.csv`, the LightGBM implementation in `src/train_classifier.py`, the saved out-of-fold predictions, the ITA fairness evaluation, and the aggregated LightGBM gain ranking.

## Dataset

| Property | Verified value |
| --- | ---: |
| Rows | 184 |
| Unique image IDs | 184 |
| Unique country-subject pairs | 184 |
| Numeric subject IDs | 122, repeated across countries |
| Non-anemic labels | 115 |
| Anemic labels | 69 |
| Model inputs | 583 |
| Missing values | 0 |

The 583 inputs are 576 MobileNet embeddings, six LAB statistics, and one ITA angle. The label is derived from workbook hemoglobin values using `< 12 g/dL` for females and `< 13 g/dL` for males.

## Leakage Verification

`train_classifier.py` excludes the following columns before constructing `X`:

```text
image_id, subject_id, country, gender, age, hgb, label
```

The independent audit confirmed:

- Feature count: 583.
- Excluded columns present in `X`: none.
- All 583 feature columns are numeric.
- Dataset null count: 0.
- `hgb` and `label` are not model inputs.

The country-plus-subject key is unique for all 184 rows. Numeric `subject_id` alone is not globally unique because India and Italy both use local numbering; this does not create duplicate rows in the dataset.

## Cross-Validation Verification

The configured splitter is `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`. The independent audit confirmed:

- Every row appears in exactly one validation fold.
- Every train/validation split has zero row overlap.
- Validation folds contain 6 or 7 anemic and 11 or 12 non-anemic samples.
- The saved OOF file preserves dataset row order and contains 184 predictions.
- No training fold uses its validation rows.

The executed baseline metrics were:

| Metric | Mean result |
| --- | ---: |
| Accuracy | 80.41% |
| Sensitivity | 76.90% |
| Specificity | 82.73% |
| ROC-AUC | 0.8723 |

These are cross-validation estimates, not performance on an independent external test set.

## Probability Separation Diagnostic

OOF probability means were substantially separated:

| True class | Count | Mean predicted probability |
| --- | ---: | ---: |
| Non-anemic | 115 | 0.2252 |
| Anemic | 69 | 0.6095 |

The mean difference was `0.3843`. A Welch two-sample t-test gave `t = 11.2293`, `p = 9.41e-22`. A one-sided Mann-Whitney test for higher anemic probabilities gave `p = 2.78e-17`. This confirms that the OOF probabilities are directionally aligned with the target in this sample.

## Feature-Gain Ranking

Gains below are summed across the ten independently trained CV models and also saved in `results/feature_gain_importance.csv`. The ranking is an association diagnostic, not a causal or clinical importance claim.

| Rank | Feature | Total gain | Mean gain/fold |
| ---: | --- | ---: | ---: |
| 1 | `mean_b` | 3360.600 | 336.060 |
| 2 | `ita_angle` | 1480.582 | 148.058 |
| 3 | `emb_431` | 1270.447 | 127.045 |
| 4 | `emb_505` | 786.869 | 78.687 |
| 5 | `std_a` | 745.031 | 74.503 |
| 6 | `emb_308` | 431.853 | 43.185 |
| 7 | `emb_450` | 337.093 | 33.709 |
| 8 | `emb_532` | 323.771 | 32.377 |
| 9 | `emb_544` | 321.255 | 32.126 |
| 10 | `emb_547` | 303.655 | 30.365 |
| 11 | `emb_487` | 288.349 | 28.835 |
| 12 | `emb_108` | 284.081 | 28.408 |
| 13 | `emb_285` | 278.654 | 27.865 |
| 14 | `emb_478` | 232.954 | 23.295 |
| 15 | `emb_252` | 207.553 | 20.755 |
| 16 | `emb_211` | 194.616 | 19.462 |
| 17 | `emb_10` | 163.378 | 16.338 |
| 18 | `emb_28` | 136.876 | 13.688 |
| 19 | `emb_344` | 135.469 | 13.547 |
| 20 | `emb_392` | 131.624 | 13.162 |

Three of the top 20 features are directly colorimetric (`mean_b`, `ita_angle`, and `std_a`), while the remaining ranked features are MobileNet embeddings. This verifies that both modalities contribute to the LightGBM decision process.

## Fairness Evaluation

The executed ITA cohort evaluation found:

| Cohort | Count | Accuracy | Sensitivity | Specificity | MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Very Light | 2 | 100.00% | 100.00% | 100.00% | 0.1122 |
| Brown / Dark | 182 | 80.22% | 76.47% | 82.46% | 0.2891 |

No subjects fell into the Light, Intermediate, or Tan/Kayumanggi bands. One-way ANOVA over absolute probability errors gave:

- `F = 1.0478`
- `p = 0.3074`
- `alpha = 0.05`
- Decision: fail to reject the null hypothesis

This p-value does not establish demographic neutrality. With 182 Dark subjects and only 2 Very Light subjects, the comparison is severely underpowered and cannot support a robust fairness conclusion. A broader, deliberately balanced skin-tone dataset is required.

## Reproduction

```bash
source aima_env/bin/activate
python src/dataset_builder.py
python src/train_classifier.py
python src/evaluate.py
```

The audit does not replace prospective clinical validation, external testing, calibration analysis, subgroup power analysis, or ethics review.
