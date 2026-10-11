# Local fingernail SAM 2.1 pilot — 2026-10-09

## Purpose and boundary

This is a local-only experiment to test whether SAM 2.1 can reduce manual
tracing of the visible nail plate. It does not alter original photographs,
approved annotations, the authoritative annotation manifest, or the manual
eyelid workflow. Every generated nail mask remains an unverified proposal.

The prediction code is `src/sam2_nail_segmentation.py`. It uses a small
ExtraTrees pixel classifier trained from the ten currently approved manual
nail masks to propose a bounding box and prompt points; Meta SAM 2.1 Hiera Tiny
then refines that prompt. The target is the visible nail plate, not the whole
finger or a smaller nail-bed-only area. A largest-connected-component cleanup
is applied after SAM. Raw and cleaned predictions are kept separately.

SAM 2.1 was installed from [Meta's official repository](https://github.com/facebookresearch/sam2) at commit
`2b90b9f5ceec907a1c18123530e92e794ad901a4`. The local checkpoint is
`data/processed/segmentation/sam2_source/checkpoints/sam2.1_hiera_tiny.pt`,
SHA-256 `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`.
Inference was verified on the Mac's MPS device; no patient images were sent to
an external service.

## Small subject-held-out comparison

The pilot used leave-one-subject-out evaluation over ten approved nail
annotations. For each held-out subject, that subject's mask was excluded from
the ExtraTrees prompt model. SAM received the resulting image-only box and
positive/negative points. This is a small internal pilot, not an independent
test set or an acceptance study.

| Method | Dice | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| ExtraTrees prompt mask | 0.737 | 0.591 | 0.638 | 0.943 |
| SAM 2.1 refined, largest component | 0.794 | 0.669 | 0.680 | 0.987 |

On the reproducible rerun, SAM inference averaged 0.903 s/image and the full
prompt-plus-SAM pipeline averaged 1.352 s/image on MPS. A prior run averaged
0.578 s and 1.169 s respectively, so timings are approximate and vary with
runtime warm-up/system state. The evaluation-set mean difference between
predicted and reference-mask CIELAB means was 7.51 in the current OpenCV
8-bit-LAB encoding. That value is only a within-rendering diagnostic: source
TIFFs do not yet have a uniform calibrated color workflow, so it is not a
calibrated or cross-device color-error estimate.

Despite the higher mean Dice, precision is notably lower than recall, which is
consistent with boundary leakage into adjacent skin. Visual review of the
separate candidate batch also found both overinclusive masks and masks that
cover only part of the nail. The simple area QC did not catch every boundary
error. Therefore a “no flag” result does not mean a correct mask.

## Unannotated-image proposals

The full current pending queue produced 116 predictions, one for each
unannotated nail image with no saved ROI. All 116 are from unique subjects.
Twenty-one were flagged by a simple area comparison against the ten training
masks (three smaller than half the reviewed minimum; eighteen larger than 1.5
times the reviewed maximum). The other 95 still require visual review. The
flag is prioritization only; it is not a quality score or an approval.

Private outputs are under:

```text
data/processed/segmentation/sam2_nail_candidate_batch_2026-10-09/
├── predictions.csv          # subject/source linkage, hashes, prompt, status, QC
├── review_queue.csv         # flagged candidates first
├── review_sheet_01.jpg ...  # local-only contact sheets
├── predicted_masks/         # cleaned candidate and raw SAM binary PNGs
├── previews/                # original-plus-mask overlays
├── run_metadata.json        # checkpoint, code, and prompt-model provenance
└── summary.json
```

The row-level output records `subject_id`, original filename and source path,
annotation TIFF path/hash, prediction paths/hashes, prompt details, model
version/checkpoint hash, QC flags, and status `predicted_unverified`. Outputs
remain under ignored `data/processed/`; none were copied into the anatomical
ground-truth folder or registered in the manifest.

Reproduce a fresh evaluation or prediction run with:

```bash
aima_env/bin/python -m src.sam2_nail_segmentation \
  --mode evaluate \
  --output data/processed/segmentation/sam2_nail_lopo_NEXT

aima_env/bin/python -m src.sam2_nail_segmentation \
  --mode predict \
  --limit 116 \
  --output data/processed/segmentation/sam2_nail_candidates_NEXT
```

Use a new output-directory name for each run; existing outputs are not
overwritten. The evaluation uses the current ten approved manual nail masks.
The prediction run excludes pending images that already have a saved Fiji ROI,
even if those ROIs have not yet been structurally registered.

## Annotation status and next step

- Eyelids remain in the manual annotation workflow; this nail experiment does
  not change that decision.
- The 116 candidate masks are not verified. Review each original against its
  overlay, especially the nail-to-skin boundary; correct or reject it in the
  established annotation workflow before registration.
- Start with `review_queue.csv` and `review_sheet_01.jpg`; then continue through
  the remaining sheets. Inspect full-resolution image/mask pairs for final
  decisions; contact sheets are triage, not boundary review.
- The earlier 67-ROI review snapshot included 58 eyelid ROIs and nine nail
  ROIs. The nine nail files use `_nail_plate_mask.roi`, while the documented
  convention is `_nail_plate.roi`. They are correctly excluded from prediction
  candidates by image stem, but were not rasterized or registered because the
  current annotation contract does not define that alternate source-ROI name.
  Preserve them; do not rename or train from them until the naming discrepancy
  is resolved and their image alignment/boundaries are reviewed.
- No percentage requiring manual correction and no correction time are
  claimed: the 116 candidates have not yet completed human review.

No anemia classification or hemoglobin-prediction training was performed.
