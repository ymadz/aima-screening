# Importing local iPhone photos

Export the original iPhone files to this folder:

`/Users/madz/Documents/aima-screening/data/raw/local/incoming/`

Keep the ID screenshots in the export. They are needed to verify which photos belong to each subject, but they must never be used as model images.

## Export from iPhone to this Mac

1. Connect the iPhone with a cable, unlock it, and choose **Trust** if prompted.
2. Open **Photos** on the Mac. Select the iPhone in the sidebar and import the relevant collection photos **and ID screenshots** into Photos. If the collection is already in iCloud Photos on the Mac, verify that full-resolution originals have downloaded instead.
3. In the Mac Photos library, select the complete collection, including the screenshots before and after each subject's eyelid and nail images.
4. Choose **File > Export > Export Unmodified Original**. Choose **Use File Name** if Photos asks how to name the export. Do not choose a resized JPEG export.
5. In the folder chooser, select the `incoming` folder above. A dated subfolder such as `2026-09-20-iphone-export` is fine.
6. Wait for the export to finish. Confirm the file count against the collection log, open several full-size files, and check that the exported photos retain their original HEIC/JPEG format and dimensions. Keep the iPhone copies until this check is complete and a second backup exists.

If you used Live Photos, a still image and a video may export separately. Keep the still image for the photo inventory; do not count the video as another eyelid or nail photo.

## Map photos to subject IDs

1. Keep the files in `incoming` untouched, with their original iPhone names. This is the import archive.
2. Review the images in capture order. A screenshot of `ZC-2026-01`, the following eyelid and nail photos, and the second screenshot of `ZC-2026-01` form a proposed group.
3. Check that both screenshots show the same ID, that the intervening photos match the collection log, and that no retake or unrelated photo falls outside the pair. Do not assign an image to a subject based on time alone.
4. Flag missing or conflicting screenshots for manual review. Do not put uncertain images into training or testing.
5. Keep verified originals in `data/raw/local/incoming/` with their original filenames and extensions. Do not rename or copy them into a second subject-folder archive. The frozen metadata manifest is the filename-to-`subject_id` mapping; black subject-ID cards remain in `incoming` for identity checks and are never model images.
6. For manual annotation, create a full-size RGB TIFF *copy* of each referenced image under `data/raw/local/annotations/pending/<modality>/images/`. Keep the original DNG/HEIC/JPEG untouched in `incoming`; preserve its exact filename in the canonical manifest. Once an image is annotated and reviewed, its TIFF and Fiji ROI go under `annotations/completed/<modality>/{images,rois}/`, and the corresponding binary mask goes under `annotations/generated_masks/anatomical/<modality>/`. Use the canonical [`annotation manifest and queue`](../../data/raw/local/annotations/README.md) to retain subject and source-file links. Never overwrite a retake; every referenced original stays a separate record.

The `data/raw` folder is ignored by Git. Treat the export and clinical metadata as access-controlled research data, and keep any subject-name-to-study-ID key separately.

Apple's instructions: [transfer photos from iPhone to Mac](https://support.apple.com/en-au/120267) and [export unmodified originals from Photos on Mac](https://support.apple.com/en-qa/guide/photos/pht6e157c5f/mac).

## Current frozen metadata and experimental outputs

The finalized live Sheet was frozen on 2026-10-07 at `data/raw/local/metadata/frozen_2026-10-07/`. Its `provenance.json` records the Sheet URL, retrieval time, and SHA-256. New local work uses the versioned manifests in `data/raw/local/manifest/frozen_2026-10-07/`, the derived cohort under `data/processed/local_cohort_who2024_2026-10-08/`, and the consolidated annotation workspace at `data/raw/local/annotations/`. The older Downloads CSV and `local_supervised_eyelid_cohort_v1.csv` are historical artifacts, not current local training inputs.

`src/local_metadata.py` regenerates the candidate table from the frozen manifest without deriving anemia labels. `src/local_annotation_manifest.py` validates the consolidated manual-annotation inventory and rebuilds its manifest and priority queue; it does not create annotations. `src/segmentation_prototype.py` generates optional eyelid and nail proposals under an ignored experiment directory; these require human review and are not ground truth. The public-data classifier remains a separate workflow (`src/reproduce_public_baseline.py`) and its cross-validation results are not local-model results. See `data/raw/local/annotations/README.md` for current counts and steps.
