# Experiment 6: complementarity and conservative ensemble protocol

Owner requested this isolated study on 2026-10-13. Branch
`experiment/eyelid-exp06-ensemble`, worktree
`/Users/madz/Documents/aima-screening-exp06-ensemble`, base `0943363`.
No merging or changes to earlier worktrees, references, labels or masks.

The initial read-only historical comparison found eight shared anemia
detections, two LAB-only, four CP-only and five shared misses. All 16 LAB false
positives were also CP false positives; CP added six. Pearson score correlation
was 0.801841 and binary agreement 107/119. These distinct positive errors
justify a small feasibility test; neither union performance nor this exploratory
decision gate establishes independent model selection.

Freeze the following before computing any new ensemble outer outcomes:

- Base models: Experiment 4 original-render robust18 unweighted C=1 logistic
  comparison and Experiment 5 CP-classification C frozen features + robust18
  balanced C=0.1 logistic candidate. Keep both experiments' primary selected
  policies separate.
- Reuse saved matched three-fold inner cross-fitted scores and pipelines within
  each unchanged outer training fold. Verify all inner identities, partitions,
  labels, class ordering and reproduced scores. No base refitting is needed.
- Fixed equal raw-probability averaging; candidate LAB weights 0.50, 0.75,
  0.25. Weighted policy selects maximum inner AP; exact ties prefer that order.
  Never select a weight using pooled outer scores.
- Thresholds maximize inner sensitivity subject to specificity >=0.80
  (primary), >=0.85 or >=0.90 (descriptive secondary), with ties favoring
  specificity then higher cutoff. Report fixed 0.5 as well. Apply the same
  training-only targets to the two base models for comparable trade-offs.
- No calibration, stacking, new features, new public data or neural fitting.
  LAB and class-balanced CP scores have different uncalibrated scales.
- Report pooled and per-fold metrics, confusion counts, Wilson class-rate
  intervals and 2,000 stratified paired conditional bootstrap draws, seed
  20261011. These omit refit/selection and overlapping-training uncertainty.
- Audit existing anatomical ROI consistency, provenance-linked repeated
  photographs, verified Hb severity and historical rendering/glare findings
  descriptively. Never tune on named misses, alter masks, relabel or exclude
  subjects from these findings.

Private versioned output:
`data/processed/local_eyelid_classification_exp06_ensemble_v1/` under the main
source checkout. Metadata, image/model sources and previous predictions are
read-only references; patient tables and contact sheets remain ignored.

The equal-average rule and chosen base candidates were proposed after earlier
results were examined. Every outcome is exploratory internal cross-validation,
with 119 reused subjects (19 positive) and no independent clinical validation.
