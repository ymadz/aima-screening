# Eyelid anemia experiment registry

Current completed research experiments; all local scores are exploratory internal subject CV, with no untouched classifier holdout. The 119 local subjects and exact five outer folds are shared. The separate segmentation reservation is unchanged. No historical output/model is replaced.

| Experiment | Branch | Verified base / training commit | Scope and status | Report |
| --- | --- | --- | --- | --- |
| 1 | archive/eyelid-exp01-cielab-baseline | preservation4bf6e25 | Six masked LAB + fixed logistic; completed and privately archived. Selected thresholds below were separately analyzed in Exp3. | [baseline](local-eyelid-classifier.md), [preservation](eyelid-exp01-preservation.md) |
| 2 | experiment/eyelid-exp02-hybrid | base4bf6e25; completedf4b2cb5 | Frozen ImageNet features + LAB + training PCA + conservative LightGBM; completed. | [results](eyelid-exp02-results-2026-10-11.md) |
| 3 | experiment/eyelid-exp03-public-transfer | basef4b2cb5; training4908020 | Public-pretrained MobileNet adapted to local; acceptedv3 completed, negative transfer result preserved. | [results](eyelid-exp03-results-2026-10-11.md) |
| 4 | experiment/eyelid-exp04-color-optimization | base66afa8e; training2a0f971 | Original/managed color,6/18/21 features, bounded classical grid; completedv1. Joint policy no sensitivity gain; robust18 matched comparison promising but unconfirmed. | [methods](eyelid-exp04-color.md), [results](eyelid-exp04-results-2026-10-12.md) |
| 4B | experiment/eyelid-exp04b-glare-regional | base0b50504; reviewed training1388ec7 | Owner-reviewed glare exclusion and ROI-relative regional color; strict subset/full cohort119, complete. No sensitivity gain over robust10/19 at84%. | [methods](eyelid-exp04b-methods.md), [results](eyelid-exp04b-results-2026-10-12.md) |

| Selected rule | Sensitivity | Specificity | ROC-AUC | AP | TP/FN/TN/FP |
| --- | ---: | ---: | ---: | ---: | --- |
| Historical LAB / nested threshold | 36.8% | 84.0% | .719 | .388 | 7/12/84/16 |
| Exp2 hybrid | 36.8% | 81.0% | .576 | .247 | 7/12/81/19 |
| Exp3 transfer | 5.3% | 84.0% | .464 | .144 | 1/18/84/16 |
| Exp4 joint primary policy | 36.8% | 83.0% | .657 | .352 | 7/12/83/17 |
| Exp4 original/robust18/C=1 LR matched comparison | 52.6% | 84.0% | .709 | .385 | 10/9/84/16 |
| Exp4B glare-excluded robust18 unweighted | 52.6% | 83.0% | .710 | .392 | 10/9/83/17 |
| Exp4B glare-excluded robust18 weighted | 52.6% | 84.0% | .721 | .392 | 10/9/84/16 |
| Exp4B full-eight nested policy | 47.4% | 78.0% | .667 | .356 | 9/10/78/22 |

Thresholds maximize inner sensitivity under an80% inner specificity constraint. Achieved outer specificity can differ. Configurations select by inner AP; the robust matched comparison is not an outer-selected replacement winner. Its paired sensitivity improvement interval is0–31.6 percentage points, conditional on saved scores; confirm using new independent subjects. No clinical validity claim.
