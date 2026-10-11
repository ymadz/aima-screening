# Experiment 3: public-to-local eyelid classification

This independent experiment starts from verified Experiment 2 commit `f4b2cb5`. Experiment 1 and 2 artifacts, tags, branches and worktrees remain immutable. The owner requested public neural pretraining, local head and late-layer adaptation, the same 119-subject outer folds, and a separate nested LAB threshold analysis on 2026-10-11.

## Input and label contract

Read-only `--source-root` points to the authoritative AIMA checkout and `--public-root` to its public archive. Outputs use a new Git-ignored Experiment 3 directory under the source root. No original images, masks or metadata are modified or copied to Git.

The archive contains 218 original images and primary manually segmented palpebral PNGs. Masks are recovered using validated alpha support or the documented exact-white-background exceptions. Subject identity is country plus patient number. Source and decoded duplicates fail closed; conservative perceptual similarity groups stay together in public train/validation. Cross-domain similarity requires resolution before training.

The [source study](https://iris.cnr.it/retrieve/670c5fcb-ffd7-47c2-b3e5-81d55c27f886/1-s2.0-S0933365722002299-main.pdf) documents laboratory Hb in g/dL, Samsung S6 capture with controlled white LED illumination and a common distance, and a historical single cutoff of <=10.5 g/dL. That cutoff is not the local AIMA target. Public labels are derived explicitly from source Hb using unadjusted adult WHO thresholds: men <13, women <12 g/dL, ages 15–65. The [WHO 2024 guideline](https://www.ncbi.nlm.nih.gov/books/NBK602183/?report=reader) has different pregnancy thresholds. Public pregnancy is unrecorded; women with 10.5<=Hb<12 are excluded because their binary status can depend on pregnancy. This conservative harmonization follows the owner's instruction to resolve label compatibility without guessing. The owner's non-pregnancy attestation applies to local AIMA, not public patients. Local membership and labels remain unchanged.

Twenty public records lack usable Hb or fall outside 15–65, and 15 more have pregnancy-dependent labels. The expected eligible public cohort is 183. Workbook ancillary notes and nonempty trailing cells are preserved in the private audit, never silently treated as additional patient rows or clinical fields. No altitude/smoking adjustments are inferred.

ICC profiles are actually converted to sRGB with relative-colorimetric rendering. For a DNG whose oriented embedded JPEG pixels exactly equal its annotation TIFF, the JPEG's real profile supersedes the misleading TIFF tag. Other local TIFFs use their recorded profile. Untagged public JPEGs are explicitly assumed sRGB; this is not camera calibration. Public JPEG EXIF orientation is applied once by Pillow and checked against the independently validated OpenCV raster and mask geometry before resize. Public images are aligned to their mask resolution using area interpolation. No local mask is regenerated.

Both domains then use zero-margin masked tissue bounding boxes, ImageNet-mean background before interpolation, aspect-preserving bilinear antialiased resize and mean letterboxing to 224, and fixed ImageNet normalization. Glare remains. Training-only horizontal flips preserve tissue color; no color enhancement or learned normalization is fitted. Mixed local export histories, unknown pilot settings, missing public profiles, public controlled illumination versus local capture, and geographic/age/sex/prevalence differences remain domain limitations. The original six LAB features remain unchanged for the separately versioned threshold comparator.

## Training status

Input validation is implemented. Neural training and evaluated results will be recorded only after execution and checkpoint verification. No local classifier final test reservation exists: the owner previously chose all 119 subjects for development. The independent segmentation test ledger is preserved and is not converted into a classification holdout. All local performance will be internal subject CV.
