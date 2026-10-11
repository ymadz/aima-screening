# Local image annotation workflow

This page documents the stable workflow for local patient-derived images. Per-image filenames, subject links, statuses, and hashes stay in the ignored local data workspace; do not copy those manifests into Git.

## One working folder

All active local annotations belong under `data/raw/local/annotations/`. The old Batch 1, Batch 2, and paired-pilot folders are historical source batches, not separate work queues. Use these local files as the source of truth:

- `data/raw/local/annotations/README.md` — current inventory and task-specific notes.
- `data/raw/local/annotations/manifests/annotation_manifest.csv` — one row per metadata image reference, linked to `subject_id` and original filename.
- `data/raw/local/annotations/manifests/pending_annotation_queue.csv` — complete priority-sorted manual queue.
- `data/raw/local/annotations/manifests/annotation_audit.json` — most recent structural audit totals and caveats.
- `data/raw/local/annotations/manifests/anatomical_review_register.csv` — hash-linked evidence for human-approved anatomical boundaries.

Original files stay unchanged in `data/raw/local/incoming/`. Use the frozen metadata filename references to associate an original with its subject. ID-card images support identity checks but are never model inputs.

## Annotation procedure

1. Start from `pending_annotation_queue.csv` to identify outstanding images. All 275 frozen image references now have annotation TIFFs; the current queue contains only fingernail images.
2. If a future source is raw-only, create and register a full-size RGB TIFF copy under `annotations/pending/<modality>/images/`. Do not rename or alter the original.
3. In Fiji, draw the anatomical ROI: exposed palpebral conjunctiva for eyelid images, visible nail plate for fingernails. Keep glare inside the anatomical ROI and handle glare as a separate, still-pending QC layer.
4. Save the named Fiji ROI under `annotations/completed/<modality>/rois/` after review. Store the matching TIFF under `completed/<modality>/images/`; rasterize a same-size binary 0/255 mask under `generated_masks/anatomical/<modality>/`.
5. Review the overlay against the original image. Check anatomical boundary, subject/source filename, orientation, ROI-to-image alignment, mask dimensions and values, and file hashes before recording approval. Anatomical approval does not imply glare approval.
6. If unsure, mark `needs_review` with a reason; if unusable, mark `rejected` with a reason. Do not guess, overwrite retakes, or generate labels automatically.
7. Tell the project agent when a set is annotated and reviewed. The agent will dry-run and rasterize the Fiji ROIs, verify that reviewed previews match the final masks, then run `src/register_local_annotation_reviews.py` to reconcile `image_ready_inventory.csv` and `anatomical_review_register.csv`. Finally, the agent runs `src/local_annotation_manifest.py` to rebuild the canonical manifest, queue, and audit. The builder does not discover new files automatically. Do not edit the generated queue or audit JSON yourself.

## Before training

Require reviewed anatomical boundaries, one-to-one source/subject/ROI/mask mapping, valid same-size binary masks, complete modality pairing decisions, frozen label provenance, and subject-level train/validation separation. Glare processing must have its own settled and documented rule. Automated segmentation proposals remain experiments until separately evaluated and corrected by a human; they are not ground truth. Do not train from unreviewed, rejected, unmatched, or unresolved files.
