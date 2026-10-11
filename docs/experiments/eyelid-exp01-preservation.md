# Experiment 1 preservation

The owner requested preservation followed by an isolated hybrid experiment on 2026-10-11. The historical output and readiness bundle remain unchanged. `src/preserve_local_experiment.py` verifies the saved ten fold models, exact OOF scores, fixed threshold, scaler statistics, configuration/source hashes and class/fold coverage without fitting. It creates a new owner-only local archive with exact copies, a committed implementation snapshot, frozen metadata, dependency inventory, reproduction instructions and complete checksums. Images, ROIs and masks remain external path/hash references.

The verified baseline uses 119 subjects (19 anemia/100 non-anemia), six unchanged masked LAB statistics, training-fold scaling and fixed five-fold logistic regression. At threshold 0.5 its confusion matrix is [[99, 1], [17, 2]], ROC-AUC 0.719474 and AP 0.387866. Stored annotation TIFF RGB, rather than raw sensor pixels, supplies the features. Mixed rendering histories and only 19 positive cases limit interpretation. All subjects are development subjects; there is no classifier holdout.

The code milestone will be identified by branch `archive/eyelid-exp01-cielab-baseline` and annotated tag `eyelid-exp01-cielab-baseline-2026-10-11`. The private archive is planned at `~/Library/Application Support/AIMA/experiment-archives/eyelid-exp01-cielab-baseline_2026-10-11_v1`. Successful archive verification is required before the hybrid worktree is created. Git refs preserve code; the private archive independently preserves ignored results.

## Feature cleanup preceding preservation

All original uncommitted changes were inspected, including the initial empty index and 26 changed/untracked paths. Separate commits group implemented work by purpose:

| Commit | Purpose | Files/features |
| --- | --- | --- |
| 58881dd | Frozen local labels and folds | local_metadata, local_who2024_cohort, cohort tests |
| d80c8c3 | Reviewed annotation manifests | annotation manifest, review registration, tests |
| b14eee9 | Provenance-linked TIFF rendering | renderer and tifffile dependency |
| 0cf891e | Manual-mask proposal prototype | segmentation prototype and joblib dependency |
| e9b091b | Public segmentation local development evaluation | pilot and tests |
| 47d8fc7 | Public classification reproduction | reproducibility runner |
| 31c2946 | Experiment 1 implementation | validation, readiness, classifier, 31 tests, experiment guide, README status |
| cdcd91f | Documentation organization | core/model/workflow/audit/experiment/policy moves, navigation, PRD status correction |

The documentation reorganization belonged to the completed “AIMA Thesis Q&A” chat; its ownership was verified and the owner explicitly approved its commit. Four untracked documents in that chat's separate worktree were left untouched. No source data, masks, clinical rows, private model checkpoints or patient predictions were staged. Full validation passed 51 tests and compilation; the fresh read-only audit checked 560 source/TIFF/ROI/mask hashes, 140 approved eyelids and the unchanged 119-subject cohort. Original Experiment 1 checksums and saved-model reproduction passed.
