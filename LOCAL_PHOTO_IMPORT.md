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
5. Once verified, **copy** the photos into `data/raw/local/images/<subject_id>/` and name them `ZC-2026-01_eyelid_01.HEIC`, `ZC-2026-01_nail_01.HEIC`, etc. Preserve each original extension. Keep the screenshots in `incoming`; do not include them among the model images.
6. Record each image's original iPhone filename, new path, modality, sequence number, and subject ID in an image manifest. If there are retakes, record every image rather than overwriting one.

The `data/raw` folder is ignored by Git. Treat the export and clinical metadata as access-controlled research data, and keep any subject-name-to-study-ID key separately.

Apple's instructions: [transfer photos from iPhone to Mac](https://support.apple.com/en-au/120267) and [export unmodified originals from Photos on Mac](https://support.apple.com/en-qa/guide/photos/pht6e157c5f/mac).
