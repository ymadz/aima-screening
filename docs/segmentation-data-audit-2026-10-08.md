# Segmentation data audit — 2026-10-08

This is a file- and annotation-readiness audit, not a trained-model result or
clinical validation. Local patient-level paths, images, and predictions remain
in the ignored `data/raw/` and `data/processed/` workspaces. No source image,
manual ROI, or approved mask was modified during this audit.

## Sources and provenance

- Local metadata authority: `data/raw/local/metadata/frozen_2026-10-07/AIMA Patients Metadata - Sheet1 values.csv`, modified `2026-10-07 14:55:39 +0800`, SHA-256 `1e704fb8d575757f16fc4f690cdc3315bc4063a5cf258d6c2e1b5f784d19596d`. Adjacent `provenance.json` records the same source hash. The owner's non-pregnancy attestation is outside the Sheet.
- Public source in this checkout: `data/raw/dataset anemia/{India,Italy}/`, identified by the project as Eyes Defy Anemia. Country workbook IDs match the numbered subject directories. A public subject key must include the country because number alone is not unique across countries.
- Local annotation authority: `data/raw/local/annotations/manifests/annotation_manifest.csv` and its pending queue. The builder reads a registered inventory; a newly saved ROI does not become approved merely by existing on disk.
- The separate `eyelid-seg-v1` rules mentioned in the task were not found in this repository; the current repository annotation README and `docs/local-annotation-workflow.md` are the available anatomical instructions. This is a documentation gap, not permission to change boundaries.

## Public palpebral pairs

| Check | Observation |
|---|---:|
| Original JPEGs / composite subject keys | 218 (India 95, Italy 123) |
| Candidate palpebral PNGs paired one-to-one with JPEGs | 218 |
| Alpha-mask PNGs | 187 (186 8-bit, one 16-bit) |
| Opaque, exact-white-background masked RGB(A) PNGs | 31 |
| PNGs at 800 × 1067 | 217 |
| PNGs at 2988 × 3984 | 1 |
| Exact duplicate original JPEGs across keys | 0 |

The public PNGs are **masked color images**, not conventional binary masks.
For the 187 files containing zero-alpha pixels, `alpha > 0` recovers the
foreground. For the other 31, the exact-white background permits
`any(RGB != 255)`; nonzero alpha alone is unsafe. Italy/15 demonstrates this:
its alpha is 191–255 everywhere, so an `alpha > 0` rule incorrectly labels the
whole frame. Its white-background rule gives a single-component mask occupying
6.43% of the frame, with close color registration to the JPEG.

Two apparent missing masks are spelling variants, not absent annotations:
`India/7/*_papebral.png` and `Italy/95/*_palplebral.png`. India/8 has two
additional `(1)` palpebral/combined files that are byte-for-byte copies of the
normally named PNGs; they are not extra cases. Forniceal and combined variants
are separate annotations and must not enter the primary palpebral task.

Every reconstructed mask was nonempty and smaller than the frame (area range
1.66%–13.03%). The largest connected component retained at least 98.27% of
foreground pixels. Twenty-seven masks contain small additional components;
these are preserved for now rather than silently cleaned. In the 217 8-bit
cases, the mean foreground RGB difference between masked PNG and oriented,
resized JPEG was at most 6.21/255. The 16-bit India/2 PNG cannot be compared
pixel-for-pixel on that scale, but its alpha overlay was visually aligned.
Representative overlays, the spelling variants, Italy/15, the highest RGB
registration-error cases, and masks with many small components were visually
inspected. This supports a reproducible **candidate** binary-mask loader;
it is not an exhaustive clinical boundary review.

No original JPEGs were byte-identical across subjects. A coarse perceptual
hash produced four visually similar cross-key pairs (Hamming distance ≤6);
visual inspection and SIFT matching did not establish that these are duplicate
people. Keep those candidate pairs in the same partition or quarantine them
until resolved so possible near-duplicates cannot cross splits. The exact
pair list belongs in an ignored experiment manifest, not a local patient table.
OpenCV decodes the public PNGs despite repeated embedded iCCP CRC warnings;
Pillow rejects some of them. Do not treat a Pillow decode failure alone as a
missing file. Do not rewrite the public originals.

## Local manual annotation inventory

| Modality | Frozen image references and TIFFs | Registered approved ROI/mask pairs | Saved but unregistered ROIs | No saved anatomical ROI |
|---|---:|---:|---:|---:|
| Eyelid | 140 | 22 | 9 | 109 |
| Fingernail | 135 | 10 | 9 | 116 |
| Total | 275 | 32 | 18 | 225 |

All 275 frozen image references resolve uniquely to originals and registered
annotation TIFFs; registered TIFF hashes match. All 32 approved masks are
nonempty, same-size, 8-bit grayscale PNGs with only 0/255 values, linked to
their corresponding ROI and TIFF. No registered mask or annotation TIFF was
missing. This is structural validation, not independent anatomical approval.
The canonical queue still has 243 pending rows (118 eyelid, 125 nail), including
the 18 unregistered-ROI items. Five subjects have no nail reference in frozen
metadata; this is not a missing-file error. The local metadata check returned
140 unique subjects, 275 uniquely resolved references, and no inconsistent Hb
units; visual identity confirmation remains separate.

The 18 unregistered ROIs consist of nine eyelid and nine nail polygons from
nine paired subjects. They each map uniquely by modality and original image
stem to a pending manifest row. The supplied Fiji rasterizer's dry-run passed
for all 18: in-bounds, nonempty, not whole-image; candidate area spans
0.11%–1.18% for eyelid and 1.28%–2.36% for nail. Contact-sheet overlay review
found plausible target placement. One eyelid ROI is smaller than the previously
approved eyelid-area range and needs closer boundary review. The new nail
files end `_nail_plate_mask.roi`, whereas the documented convention is
`_nail_plate.roi`; do not silently rename or register them. No masks, overlays,
review ledger entries, or manifest statuses were created from these 18 during
this audit.

The approved local eyelid masks cover 0.19%–1.71% of their full TIFFs
(median 0.75%); new eyelid polygons are broadly similar in scale. Public masks
cover a much larger fraction of their tighter eye photos (median 4.96%), so
full-frame area fractions cannot be compared directly. The public and local
photographs differ visibly in capture geometry and color; successful public
validation does **not** establish local transfer accuracy or equivalent
anatomical conventions.

## Code, dependencies, and experiment status

- `src/dataset_builder.py` consumes public palpebral color PNGs for the existing anemia baseline; it does not reconstruct binary masks, handle the two typo names, or train a segmentation network. Preserve that classification workflow separately.
- `src/segmentation_prototype.py` is an experimental per-modality ExtraTrees proposal generator, not the requested U-Net or SAM workflow. The prior project log reports held-out-subject mean Dice 0.694 for eyelid and 0.746 for nail on a small pilot; these numbers were not rerun here and do not establish deployment readiness.
- No U-Net, LinkNet, SAM 2.1, public segmentation train/validation/test split, local fine-tuning, or unified predicted/verified manifest updater was found. `torch` 2.13.0, `torchvision` 0.28.0, OpenCV, Pillow, and tifffile are installed in `aima_env`; `sam2`, `roifile`, and `segmentation_models_pytorch` are not. The current interpreter reports `torch.backends.mps.is_available() == False` despite an MPS-enabled build, so use CPU fallback unless an actual MPS smoke test succeeds.
- A local Fiji environment contains an older SAM 2 tiny checkpoint, **not** a verified SAM 2.1 checkpoint. Do not label it 2.1 or silently substitute it. Meta's official SAM 2 repository lists a separate 2.1 tiny checkpoint and requires Python ≥3.10 and PyTorch ≥2.5.1.
- The annotation TIFFs are suitable for geometry review, but the documented Display-P3-preview/sRGB-tag mismatch in some DNG-derived nail TIFFs blocks unqualified calibrated CIELAB comparisons. Resolve color management before claiming color-feature equivalence. Glare is an independent QC layer; rejected/unreviewed glare candidates are not anatomical truth.

## Gates and next controlled stage

1. **Public mask-loader gate: candidate pass.** Implement a read-only original/PNG loader that verifies subject key, known spelling variants, pixel geometry, alpha-or-white mask extraction, nonempty area, and orientation. Save binary derivatives only in ignored experiment storage. Include tests for Italy/15, India/2, typo names, and the India/8 duplicate.
2. **Public training gate: conditional.** After automated loader tests and an additional contact-sheet review, freeze a reproducible subject-disjoint split. Group or quarantine the four unresolved near-duplicate candidate pairs. Train a compact U-Net on palpebral masks only; score untouched public test subjects once. Do not use Hb labels or forniceal/combined files for segmentation.
3. **Local transfer gate: evaluation only.** Reserve local evaluation subjects before any fine-tuning, compare public-model predictions with the 22 registered eyelid masks, and report Dice/IoU plus anatomical failures. The 18 unregistered ROIs are not ground truth until rasterized, boundary-reviewed, and registered.
4. **SAM and nail gate: blocked on runtime/review.** Confirm a genuine SAM 2.1 checkpoint and local runtime before building assisted correction. Do not mark proposals verified or update the canonical manifest without human review and a safe registration path. Keep nail and eyelid models separate.

The model architecture and methodological choices requested in the current task
are implementation directions, not achieved results. The original U-Net paper
motivates augmentation for small biomedical datasets
([Ronneberger et al.](https://arxiv.org/abs/1505.04597));
SAM 2.1 checkpoint and prompt APIs come from the
[official Meta repository](https://github.com/facebookresearch/sam2);
the MPS availability test follows
[PyTorch's MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html).
