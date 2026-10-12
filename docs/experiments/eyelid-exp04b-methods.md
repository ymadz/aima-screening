# Experiment 4B: glare review and limited regional color follow-up

Owner-authorized follow-up on 2026-10-12. Start from verified Experiment4 commit `0b50504` on `experiment/eyelid-exp04b-glare-regional`, in `../aima-screening-exp04b-glare`. Experiments1-4 and manual anatomy remain immutable. All patient-derived files stay in separate ignored, owner-only storage in the main repository.

## Glare proposals and review

The canonical annotation manifest has 140 approved eyelid anatomical inputs, 12 unvalidated pilot glare rules, 10 unreviewed glare candidates and 118 without glare masks. None is accepted glare ground truth. Missing masks are unreviewed/missing annotation; empty generated masks do not establish absence of glare.

`src/eyelid_glare_masks.py` defines a frozen conservative appearance rule: minimum RGB channel180/range65 candidate, brighter core220/range35, minimum-channel local contrast15 against a mask-normalized Gaussian neighborhood (sigma1.5% of square root of ROI area), components3 pixels to5% ROI. Proposals never escape anatomy. Broad bright tissue, large components, removal above3% or contrast without a conservative core trigger uncertainty. These are review thresholds, not validated clinical thresholds. Bright red-channel clipping or low redness alone does not trigger removal. Pale tissue can still be misidentified and chromatic/wet glare missed.

Every eligible image receives a binary same-size proposal, including explicit empty proposals. Statuses distinguish APPROVED_GLARE, REVIEWED_NO_GLARE, AUTO_GENERATED_PENDING_REVIEW, UNCERTAIN and INVALID_MASK. Only explicit hash-linked human review can confer the first two. All119 need review initially. No automatic mask is accepted by this implementation.

Private panels show an unaltered source eyelid crop, approved anatomy edge, cyan proposal and retained-tissue preview, using pseudonyms without filenames or Hb/labels. Display thumbnails are resized for review; scientific features use native pixels. Paginated sheets group uncertain cases, likely reflections and empty pending proposals. No approved no-glare group is invented. The separate identity key remains private.

## Frozen bounded comparison

Keep the119 subjects (19/100), original five outer folds and three inner folds, seed20261011. Rendering A reuses exact Experiment4 stored TIFF RGB assumed sRGB, including mixed export-history limitations. The unchanged18 robust features are the comparator. Add four contrasts between ROI-relative distance-to-boundary core/rim: median L/a/b and median natural-log((R+1)/(G+1)). Define core distance at or above median occupied distance and rim below, from original anatomy before exclusions. These geometry-relative sampling regions are invariant to flips/90-degree rotations, but are not anatomical landmarks or proof of homologous tissue.

Evaluate four candidates: robust18 or regional22, each L2 logistic C1 unweighted/balanced, lbfgs, max_iter2000, tolerance0.0001, seed20261010, training-only StandardScaler. Class weighting addresses19/100 imbalance as a controlled ablation; weighted scores are uncalibrated. Select pooled inner AP with frozen-order ties. Evaluate0.5 and an inner-selected cutoff maximizing sensitivity subject to inner specificity>=0.80, then higher specificity/cutoff ties. No outer outcomes tune masks, regions, features, models or rules. The specificity constraint is a research choice and does not guarantee outer specificity or clinical acceptability.

Glare-excluded robust/regional feature perturbations are diagnostic only until accepted review evidence exists. Strict verified-subset and full-cohort exclusion classifiers are blocked if reviews are absent. Continue approved-anatomy regional comparisons without changing membership. Do not choose features to rescue specific missed subjects. Coefficient signs/ranks and training scaler variation are descriptive stability checks, never feature-selection inputs.

Report all requested classification metrics, counts, per-fold metrics, Wilson binomial intervals and shared2000 stratified conditional bootstrap intervals/paired differences. These omit refit uncertainty and dependence from overlapping training folds. All follow-up results are exploratory repeated-development evidence, never external testing or clinical validation. Nine robust misses receive private Hb severity, geometry, rendering, score and regional/proposal diagnostics; unvalidated proposal burden cannot prove a glare-error association.
