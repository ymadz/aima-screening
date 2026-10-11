# Experiment 2 completed comparison — 2026-10-11

Experiment 2 completed once on the same 119 manually annotated, eligible eyelid subjects (19 anemia/100 non-anemia) and exact five subject folds as Experiment 1. The hybrid detected more positive cases, but specificity and ranking metrics declined. This study does not establish that the added complexity improves anemia screening. No result-driven retuning or refitting followed the comparison.

| Metric | Experiment 1, fixed 0.5 | Hybrid, fixed 0.5 | Hybrid, inner-selected |
| --- | --- | --- | --- |
| Accuracy | 84.87% | 75.63% | 73.95% |
| Sensitivity | 10.53% | 26.32% | 36.84% |
| Specificity | 99.00% | 85.00% | 81.00% |
| Balanced accuracy | 54.76% | 55.66% | 58.92% |
| Precision | 66.67% | 25.00% | 26.92% |
| F1 | 0.181818 | 0.256410 | 0.311111 |
| ROC-AUC | 0.719474 | 0.575789 | 0.575789 |
| Average precision | 0.387866 | 0.247161 | 0.247161 |

Confusion matrices use true rows/predicted columns ordered non-anemia/anemia: Experiment 1 `[[99,1],[17,2]]`, hybrid fixed `[[85,15],[14,5]]`, hybrid selected `[[81,19],[12,7]]`. The hybrid detected 5/19 positives at 0.5 and 7/19 under selected thresholds, versus 2/19 in Experiment 1; false positives rose from 1/100 to 15/100 and 19/100.

## Architecture and prespecified development

Experiment 1: unchanged six full-resolution masked LAB mean/population-standard-deviation values → training-only StandardScaler → fixed L2 logistic regression. Experiment 2: the same six LAB values plus 576 frozen ImageNet1K_V1 MobileNetV3-Small embeddings → training-only embedding scaling/PCA8 → 14 fused features → conservative LightGBM. Every image uses an approved manual anatomical mask. The embedding path mean-fills outside tissue before aspect-preserving 224-square letterboxing and ImageNet normalization; LAB remains pixel-identical to its baseline cache. Neural weights and batch-normalization statistics are frozen.

The five original outer partitions are unchanged. Three inner subject folds select between exactly two prespecified shallow-tree candidates (unweighted/balanced), using inner OOF AP. The selected candidate's inner scores choose maximum sensitivity at specificity ≥0.80; ties favor specificity then the higher threshold. No outer labels inform these decisions. All 30 inner and five outer pipelines are saved. Balanced weighting is selected in folds 1, 2 and 4; unweighted in 3 and 5. Selected thresholds are 0.545872, 0.549735, 0.183141, 0.582275 and 0.256952. The specificity constraint is an inner research objective, not a clinical target or outer guarantee.

## Uncertainty and consistency

At fixed 0.5 the paired sensitivity delta is +0.157895 (conditional 95% interval −0.052632 to +0.368421). With training-selected thresholds the delta is +0.263158 (+0.052632 to +0.526316), but this comparison changes the operating rule. Corresponding specificity deltas are −0.14 (−0.22 to −0.07) and −0.18 (−0.26 to −0.11). Balanced-accuracy delta intervals include zero for both rules. ROC-AUC delta is −0.143684 (−0.325007 to +0.035276); AP delta is −0.140705 (−0.349086 to +0.112157).

Hybrid sensitivity Wilson intervals are 0.118064–0.487916 (fixed) and 0.191495–0.589605 (selected). ROC-AUC interval is 0.424184–0.720796 and AP interval 0.163599–0.447485. These 2,000 paired stratified subject bootstraps are conditional on saved OOF scores, fix prevalence and omit model-refit/overlapping-training uncertainty. They do not support clinical validity or external performance claims.

Fold ROC-AUC values are 0.5750, 0.7125, 0.7875, 0.5250 and 0.5000. Fixed-threshold sensitivity ranges 0–0.50 and specificity 0.60–1.00. Selected-threshold specificity ranges 0.65–0.95. Improvement is not consistent across folds. Only 3–4 positive subjects appear in each outer validation fold.

## Preserved artifacts and verification

Experiment 1 was archived before Experiment 2: code milestone `4bf6e2577d02b5e3338e1937140b7eaff31369b4`, preservation branch `archive/eyelid-exp01-cielab-baseline`, annotated tag `eyelid-exp01-cielab-baseline-2026-10-11`. The private owner-only archive is `~/Library/Application Support/AIMA/experiment-archives/eyelid-exp01-cielab-baseline_2026-10-11_v1`; all 97 checksum-covered files and ten saved models verify without retraining. All 41 original output/readiness file hashes still match the initial audit.

Experiment 2 native worktree: `~/Documents/aima-screening-exp02-hybrid`, branch `experiment/eyelid-exp02-hybrid`. The run records training-code commit `01cb856842a637ff8f4ba3ece246e18e904975d7`. A subsequent verifier-only native-thread correction is committed as `9d88cc750dc1d176e21d947a2f6bd40f054ac60c`; completed models and copied original source remain unchanged.

Private output: `~/Documents/aima-screening/data/processed/local_eyelid_classification_exp02_hybrid_2026-10-11_v1/`. It has 87 files (86 checksum-covered plus the checksum manifest), about 15.5 MB, with owner-only permissions. It contains weights, cached LAB/embeddings, all pipelines, inner assignments/selections, OOF rows, configuration/source/code/environment provenance, metrics/intervals, aggregate PNG/SVG, logs, report and verification. No patient image, ROI or mask was duplicated.

Validation: 58 tests passed and three public-data-dependent tests were skipped in the isolated worktree; all ten new hybrid tests passed. The original checkout's earlier full 51-test run passed without skips. Relevant verifier regression tests passed after its runtime correction. Compilation and whitespace checks passed. Saved-model verification reproduces every inner score, candidate/threshold decision and all 119 outer probabilities, checks training-only scaling/PCA statistics, and independently matches pooled/per-fold metrics. Standalone verification checks authoritative source hashes and original artifact preservation. Independent sklearn recalculation matches every requested metric under both rules; fresh embeddings from one real subject per outer fold exactly match the cache.

## Limits and next decision

Mixed, uncalibrated TIFF rendering persists. Mask shape, interpolation boundaries and letterbox proportions may affect embeddings. PCA may discard useful low-variance information; this experiment has no ablation identifying which component explains the observed results. Class weighting varies by outer fold and scores are uncalibrated, which also limits pooled probability/ranking interpretation. Inner model and threshold selection share OOF scores, making their selection metrics optimistic. All subjects are development subjects, with no untouched classifier holdout.

The extra complexity is not justified by this run's poorer discrimination and specificity. Both models remain research prototypes. Priorities are a consistent acquisition/rendering protocol, more positive cases, new untouched validation data and prespecified ablations/calibration. The verified data/model plumbing can support a future experiment, but public-anemia-to-local transfer is not ready to be treated as the next validated milestone without a separately approved domain/label/preprocessing/split design. It was not implemented or run here.

See [methods and reproduction commands](eyelid-exp02-hybrid.md), [preservation details](eyelid-exp01-preservation.md), and [Git cleanup/handoff](../audits/git-cleanup-and-experiment-handoff-2026-10-11.md).
