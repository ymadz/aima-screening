# Experiment 4 local eyelid color model card

Completed exploratory study,2026-10-12; manual anatomical masks only. Purpose: investigate missed anemia and test deterministic color handling/features/classical models on the same119 verified subjects (19 anemia/100 non-anemia). No clinical deployment model or external test was fitted.

Inputs: aligned native RGB annotation TIFF plus identical approved anatomical mask. Original and ICC-managed renderings; glare retained. Unadjusted adult15–65 WHO labels (M<13,F<12 g/dL), local non-pregnancy owner-attested outside the Sheet. No Hb/sex/age/quality/export/identity predictor enters classification. The full-resolution color features have6/18/21 dimensions. Classification uses regularized logistic or conservative LightGBM, none/balanced weights, nested5×3 subject CV and train-only scaling/selection/thresholds.

Primary joint inner-AP policy:7TP/12FN/83TN/17FP, sensitivity 36.8%/specificity 83%, AUC .657/AP .352. Predefined original-rendering robust18/unweighted C=1 LR comparison:10TP/9FN/84TN/16FP, sensitivity 52.6%/specificity 84%, AUC .709/AP .385. The latter gains3 original false negatives without losing the 7 original true positives, but its paired sensitivity interval includes zero and the managed-color counterpart detects7/19. Do not promote an outer candidate as a confirmed winner.

Limitations:19 positives; prior experiments informed design on these subjects; conditional uncertainty omits refitting/overlapping training dependence; mixed rendering, unknown tone mapping and clipping; no color-reference acquisition; no accepted glare exclusion; score calibration not studied. Inner specificity constraints do not guarantee outer specificity. Fold 5 still misses all 3 positives for the primary and robust matched rules. Anatomical approval is not clinical validation.

Optional fixed managed robust18 Ridge Hb feasibility after classification:MAE 1.3274 g/dL, RMSE 1.7022, R² .1341, compared with training-mean MAE 1.3750. Separate research output; not a clinical Hb measurement.

All720 classifier and 10 regression models reproduce without fitting, all fresh source vectors match exactly and 810 artifact checksums pass. Final90-test suite:87 pass/3 unavailable-data skips. Patient outputs/models stay ignored and private. The corrected `src/eyelid_color_verify.py` supersedes the sealed original metadata reader; it never retrains or modifies the run. See [complete results and reproduction](../experiments/eyelid-exp04-results-2026-10-12.md) and [experiment registry](../experiments/registry.md).
