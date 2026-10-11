# Experiment 3 results — 2026-10-11

Public-to-local transfer did not improve anemia detection in this prespecified experiment. The separately nested LAB threshold analysis detected 7/19 anemia subjects at 84% specificity; the inner-selected transfer model detected 1/19 at the same specificity. Experiments 1 and 2 remain unchanged. These are internal development results.

## Readiness and provenance

Public: 218 original JPEG/manual-palpebral pairs, no exact source/decoded duplicates. Excluding 19 subjects outside ages 15–65, one missing Hb and 15 pregnancy-dependent female labels leaves 183 (64 anemia, 119 non-anemia). Public pregnancy is unrecorded; only labels consistent across pregnancy thresholds are retained. The owner reconfirmed local non-pregnancy; this remains an attestation outside the Sheet. Local membership, laboratory values, reviewed masks and the exact five outer folds remain the original 119 subjects (19 anemia, 100 non-anemia). All source, registration and owner duplicate-review gates passed. No unresolved training blocker remains.

The [original study](https://iris.cnr.it/retrieve/670c5fcb-ffd7-47c2-b3e5-81d55c27f886/1-s2.0-S0933365722002299-main.pdf) documents Hb in g/dL, Samsung S6 capture with controlled LED illumination and a single <=10.5 cutoff. That historical cutoff is explicitly replaced for this independent experiment by [WHO adult cutoffs](https://www.ncbi.nlm.nih.gov/books/NBK602183/?report=reader): male <13, female <12 g/dL, with strict boundaries and no inferred altitude/smoking adjustment. Public ancillary/trailing workbook cells remain in the private audit. The legacy 184-row public baseline is not changed.

The original-JPEG pHash screen flags four pairs and the aligned-raster screen flags none. Conservative grouping uses both screens. Public training has 146 subjects (54 anemia/92 non-anemia), validation 37 (10/27), stratified by country and label. No flagged pair crosses the split. Cross-domain source/similarity checks passed. Similarity flags do not prove duplicate identity.

## Implementation and actual training

Worktree: `/Users/madz/Documents/aima-screening-exp03-transfer`; branch `experiment/eyelid-exp03-public-transfer`; base `f4b2cb5`; recorded training commit `49080204add7e7b81245e7fb6fa5cd5adb1f807b`. Configurations and source hashes were frozen before the accepted run.

Neural input uses actual ICC-to-sRGB conversion, including 11 pixel-identical Display P3 DNG previews whose genuine profile supersedes the misleading TIFF tag. Public originals are EXIF-oriented and registered to their manual masks. Both domains use zero-margin mask bounding boxes, ImageNet-mean background before resize, aspect-preserving 224 letterboxing and fixed ImageNet normalization. Glare stays; augmentation is training-only horizontal flip. No enhancement, learned color normalization or automated segmentation is used. Public JPEGs have no ICC but all 218 declare EXIF sRGB. Camera calibration and mixed local export histories remain limitations.

MobileNetV3-Small starts from hash-verified ImageNet1K_V1 weights. Global pooling supplies 576 values to dropout 0.2 and a binary linear head. Public training adapts the head/final two blocks, keeping earlier weights and all BatchNorm statistics fixed. It completed 12 epochs and selected epoch 8 using public validation only. Descriptive public validation ROC-AUC was 0.826 and AP 0.693; that set selected the checkpoint and threshold, so this is selection performance rather than independent testing.

Local A adapts 577 head parameters; B adapts 351,121 head/late-block parameters. AdamW uses weight decay 0.001 and training-only BCE positive weights. Three inner folds use maximum 12 epochs, validation stopping with patience 4, and training-only thresholds. Outer refits use median inner best epochs and all outer-training subjects, receiving no outer validation labels. Inner strategy choices are A, A, B, B, A. Public-only inference fits no local parameters; its selected threshold uses public validation only. Scores are uncalibrated.

The new LAB comparison fits 15 inner and five outer logistic models using the unchanged six historical LAB features, C=1, no weighting and training-only StandardScaler. Its outer probabilities reproduce Experiment 1 within 1e-12; only the independently selected thresholds differ. From the original fixed rule, true positives rise from 2 to 7, while false positives rise from 1 to 16. It matches the selected hybrid's 7 true positives with three fewer false positives (16 versus 19); that modest difference does not establish superiority in future populations.

## Uncertainty and interpretation

The Wilson 95% sensitivity interval is 19.1–59.0% for nested LAB and 0.9–24.6% for selected transfer. Both have 84% specificity, interval 75.6–89.9%. Selected-transfer ROC-AUC interval is 0.347–0.578; LAB is 0.582–0.842. Paired selected-transfer minus nested-LAB sensitivity is −31.6 percentage points (conditional interval −52.6 to −10.5), balanced accuracy −15.8 points (−27.7 to −4.8), ROC-AUC −0.256 (−0.453 to −0.055), and specificity 0 points (−10 to +9).

These intervals condition on saved scores and reuse identical 2,000 stratified subject draws, seed 20261011. They omit refit uncertainty and dependence from overlapping CV training sets. Original artifact intervals remain unchanged; unified ranking intervals are recomputed under this common seed. Individual transfer ranking intervals include 0.5. No external or clinical validity follows.

Eligible public prevalence is 35.0%, local 16.0%. Public sex counts are 126M/57F versus local 57M/62F; ages 19–65 versus 16–61, mean 39.0 versus 31.9. Country confounding is substantial: public India has 54 positives/81 subjects, Italy 10/102. Controlled illumination/magnification differ from local capture. Public-only local ROC-AUC 0.451 and zero detected anemia at both reported thresholds show failure to generalize in this run.

At 0.5, both transfer strategies detect 4/19 with 41–42 false positives; nested LAB detects 7/19 with 16 false positives. Training-selected transfer performs worse still. This tested design does not justify its additional complexity. LAB uses six statistics and a small linear model; hybrid adds an ImageNet forward pass, PCA and LightGBM; transfer adds public label training and 41 neural fits/checkpoints. The accepted training/evaluation invocation took 206.8 seconds on deterministic single-thread CPU, excluding preparation, prefix-cache generation and reproduction. Sealed artifacts occupy 568,930,222 bytes. No device latency benchmark was conducted.

Investigate LAB with training-selected thresholds and a larger independent cohort with more anemia cases. Its 36.8% sensitivity still misses 12/19 and does not support screening deployment. Preserve this negative transfer result. A future matched-preprocessing ImageNet-only local neural ablation could separate pretraining effects from input-policy differences; it must be a new experiment. Neural preprocessing differs from the historical stored-RGB LAB/hybrid policy, so failure cannot be attributed solely to public pretraining.

## Artifacts and reproduction

Accepted private run: `/Users/madz/Documents/aima-screening/data/processed/local_eyelid_classification_exp03_transfer_2026-10-11_v3/`. Public checkpoint: `public/best.pt`; local checkpoints: `fold_<1..5>/<A|B>/outer/best.pt`. All 41 selected neural and 20 LAB models, provenance/audits/configuration/environment/seed/folds/scores/threshold records/metrics/figures/copied implementation and 216 checksum-covered files are present, plus the checksum manifest. Owner-only artifacts stay Git-ignored. Full per-fold metrics and intervals are in `metrics.json`.

Attempts v1 (EXIF mismatch before training) and v2 (public grouping) are preserved as invalid. V3 reused only verified image tensors and fixed ImageNet prefix maps, with no trained weights from v2. No outer comparison metric guided the integrity correction.

The initial post-training verifier rejected a 1.58e-6 float32 difference caused by differing convolution batch geometry. The corrected independent verifier freshly recomputes every public/local prefix map and flipped view using the same geometry: all match exactly. It reproduces every saved prediction, selection, threshold, frozen parameter and five local source tensors without refitting. The recorded training implementation and sealed outputs remain unchanged. Independent sklearn calculations match all 72 pooled/per-fold metric sets.

Run from the isolated worktree; this command supersedes the initial verifier command retained in the sealed historical guide:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
/Users/madz/Documents/aima-screening/aima_env/bin/python -m src.eyelid_transfer_reproduce \
  --source-root /Users/madz/Documents/aima-screening \
  --output /Users/madz/Documents/aima-screening/data/processed/local_eyelid_classification_exp03_transfer_2026-10-11_v3
```

Verification reports `reproduced_without_refitting`, 41 neural checkpoints, 20 LAB checkpoints, 216 files and unchanged historical experiments. Proofs are saved under `/Users/madz/Documents/aima-screening/data/processed/exp03_readiness_audit_2026-10-11_v1/`. The original 41 Experiment 1/readiness files, 86 checksum-covered Experiment 2 files, 97 archive files and preservation references remain unchanged. No data/models were committed or uploaded.

## Unified local metrics

Sensitivity equals recall. Fixed means 0.5; selected thresholds use inner training information (or public validation for public-only). All rows use the same 119 subjects and five outer folds.

| Model/rule | Accuracy | Sensitivity | Specificity | Balanced acc. | Precision | F1 | ROC-AUC | AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Exp1 LAB, fixed | 84.9% | 10.5% | 99.0% | 54.8% | 66.7% | 0.182 | 0.719 | 0.388 |
| Exp2 hybrid, fixed | 75.6% | 26.3% | 85.0% | 55.7% | 25.0% | 0.256 | 0.576 | 0.247 |
| Exp2 hybrid, selected | 73.9% | 36.8% | 81.0% | 58.9% | 26.9% | 0.311 | 0.576 | 0.247 |
| New LAB, selected | 76.5% | 36.8% | 84.0% | 60.4% | 30.4% | 0.333 | 0.719 | 0.388 |
| Public only, fixed | 78.2% | 0.0% | 93.0% | 46.5% | 0.0% | 0.000 | 0.451 | 0.141 |
| Public only, public-selected | 84.0% | 0.0% | 100.0% | 50.0% | undefined | 0.000 | 0.451 | 0.141 |
| Transfer A, fixed | 52.9% | 21.1% | 59.0% | 40.0% | 8.9% | 0.125 | 0.475 | 0.147 |
| Transfer A, selected | 69.7% | 5.3% | 82.0% | 43.6% | 5.3% | 0.053 | 0.475 | 0.147 |
| Transfer B, fixed | 52.1% | 21.1% | 58.0% | 39.5% | 8.7% | 0.123 | 0.446 | 0.140 |
| Transfer B, selected | 71.4% | 5.3% | 84.0% | 44.6% | 5.9% | 0.056 | 0.446 | 0.140 |
| Inner-selected A/B, fixed | 51.3% | 15.8% | 58.0% | 36.9% | 6.7% | 0.094 | 0.464 | 0.144 |
| Inner-selected A/B, selected | 71.4% | 5.3% | 84.0% | 44.6% | 5.9% | 0.056 | 0.464 | 0.144 |

| Model/rule | TP | FN | TN | FP |
| --- | ---: | ---: | ---: | ---: |
| Exp1 LAB, fixed | 2 | 17 | 99 | 1 |
| Exp2 hybrid, fixed | 5 | 14 | 85 | 15 |
| Exp2 hybrid, selected | 7 | 12 | 81 | 19 |
| New LAB, selected | 7 | 12 | 84 | 16 |
| Public only, fixed | 0 | 19 | 93 | 7 |
| Public only, public-selected | 0 | 19 | 100 | 0 |
| Transfer A, fixed | 4 | 15 | 59 | 41 |
| Transfer A, selected | 1 | 18 | 82 | 18 |
| Transfer B, fixed | 4 | 15 | 58 | 42 |
| Transfer B, selected | 1 | 18 | 84 | 16 |
| Inner-selected A/B, fixed | 3 | 16 | 58 | 42 |
| Inner-selected A/B, selected | 1 | 18 | 84 | 16 |

## Outer selection records

| Fold | Validation n (positive/negative) | Strategy | A/B epochs | A threshold | B threshold | LAB threshold |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 1 | 24 (4/20) | A | 12/8 | 0.595711 | 0.624352 | 0.254296 |
| 2 | 24 (4/20) | A | 1/1 | 0.581870 | 0.475252 | 0.239638 |
| 3 | 24 (4/20) | B | 1/2 | 0.547306 | 0.638352 | 0.239570 |
| 4 | 24 (4/20) | B | 12/9 | 0.791300 | 0.849452 | 0.222395 |
| 5 | 23 (3/20) | A | 8/12 | 0.662639 | 0.837240 | 0.260672 |

Outer training contains 80 negatives and 15 positives (fold 5:16). Public-only selected threshold is 0.704618. Full inner partitions/epochs/probabilities and objective records remain private.
