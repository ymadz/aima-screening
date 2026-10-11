# AIMA Screening

Research pipeline for screening anemia from conjunctival images and patient metadata.

## Core project documents

- [Architecture essentials](docs/core/ARCHITECTURE_ESSENTIALS.md) — invariants every agent must preserve.
- [PRD](docs/core/PRD.md) — goals, requirements, intended use, and current scope.
- [Architecture](docs/core/ARCHITECTURE.md) — verified data/model flows and implementation status.
- [Project log](docs/core/PROJECT_LOG.md) — dated decisions, experiments, results, and blockers.
- [Agent guide](AGENTS.md) — required reading order, source-of-truth rules, privacy, and verification expectations.
- [Documentation index](docs/README.md) — map of core, model, workflow, experiment, audit, and policy documents.
- Supporting references: [model card](docs/models/MODEL_CARD.md), [findings and evaluation](docs/models/FINDINGS_AND_EVALUATION.md), [local photo import](docs/workflows/LOCAL_PHOTO_IMPORT.md), and [local annotation workflow](docs/workflows/local-annotation-workflow.md).

## Preview

- Live demo: [add live demo link]
- Repository: [github.com/ymadz/aima-screening](https://github.com/ymadz/aima-screening)
- Screenshots or GIF: [add project preview here]

## Overview

AIMA (Chromatic Anemia Screening via Conjunctival and Fingernail Bed Pallor) is a student research project for exploring image-based anemia screening. This repository currently contains the public inner-eyelid dataset pipeline: it extracts color features and MobileNetV3-Small image embeddings, then trains a LightGBM classifier. The project is intended for research and thesis experimentation, not clinical diagnosis.

## Features

- Loads the public inner-eyelid anemia dataset and its country workbooks
- Joins images with subject metadata and measured hemoglobin values
- Derives binary anemia labels from the configured hemoglobin thresholds
- Calculates CIE L\*a\*b\* color statistics from palpebral regions
- Extracts frozen MobileNetV3-Small image embeddings
- Combines image and color features for LightGBM classification
- Runs stratified cross-validation and saves out-of-fold predictions
- Evaluates performance across ITA skin-tone cohorts

## Screenshots

- Dataset or image example: [add screenshot here]
- Training output: [add screenshot here]
- Evaluation results: [add screenshot here]

## Tech Stack

### Language and data tools

- Python 3.10
- pandas
- NumPy
- OpenPyXL

### Image and machine learning tools

- PyTorch and torchvision
- MobileNetV3-Small with ImageNet weights
- OpenCV
- LightGBM
- scikit-learn
- SciPy

### Project tools

- Kaggle CLI for downloading the public dataset
- CSV, Excel, and generated result files

## My Role

I organized the public dataset, built the metadata and image feature pipeline, implemented the LightGBM training workflow, and prepared the evaluation and model documentation for the AIMA thesis project.

## What I Learned

- How to join image files with subject-level clinical metadata
- How to calculate color features in CIE L\*a\*b\* space
- How to use a pretrained CNN as a frozen feature extractor
- How to combine image embeddings with tabular features
- How to run cross-validation and save out-of-fold predictions
- Why subject-level splitting matters when a subject has multiple images
- How dataset quality, label definitions, and acquisition conditions affect results

## Challenges

- Matching image files to the correct subject and workbook record
- Working with a small, imbalanced public dataset
- Handling different image regions and missing image modalities
- Measuring skin-tone cohorts when some cohorts have very few samples
- Keeping hemoglobin labels separate from unverified or missing laboratory results
- Interpreting cross-validation results without treating them as external clinical validation

## Future Improvements

- Add the verified local Zamboanga dataset through a separate import and validation step
- Change training and validation splits to be explicitly subject-based
- Improve masking so black padding is excluded from color statistics
- Validate or remove the fixed-margin ITA calculation
- Add a reproducible DNG-to-rendered-image conversion workflow for local iPhone images
- Compare eyelid-only, fingernail-only, and combined models
- Add an independent local test set and confidence intervals
- Add model calibration and clearer per-subject error analysis
- Add a data-quality and exclusion report to the dataset builder

## Installation

Python 3.10 is recommended.

```bash
git clone https://github.com/ymadz/aima-screening.git
cd aima-screening

python3.10 -m venv aima_env
source aima_env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, create and activate the environment with:

```powershell
py -3.10 -m venv aima_env
aima_env\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset Setup

The current public-data pipeline expects the Kaggle dataset under `data/raw/`:

```text
data/raw/dataset anemia/<country>/<subject number>/
```

Each country folder should contain its workbook, such as `India.xlsx` or `Italy.xlsx`. Keep the Kaggle credential outside this repository and do not commit raw patient images, workbooks, or generated feature tables.

```bash
mkdir -p ~/.kaggle
# Place your Kaggle credential file at ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download -d harshwardhanfartale/eyes-defy-anemia
unzip eyes-defy-anemia.zip -d data/raw/
```

## Running the Pipeline

Build the processed feature table:

```bash
python src/dataset_builder.py
```

Train the LightGBM classifier and save out-of-fold predictions:

```bash
python src/train_classifier.py
```

Run the ITA cohort evaluation:

```bash
python src/evaluate.py
```

The main generated files are:

- `data/processed/dataset.csv`
- `results/cross_validation_predictions.csv`
- `results/feature_gain_importance.csv` (when generated by the analysis workflow)

## Project Status

This is a student research prototype, not a clinical product. The public inner-eyelid baseline has internal 10-fold cross-validation results. A separate local manual-eyelid LAB classifier completed fixed five-fold subject cross-validation on 119 eligible subjects (19 anemia, 100 non-anemia) on 2026-10-10 after source, mask and owner duplicate-review gates passed. At the fixed 0.5 research threshold it detected only 2 of 19 anemia subjects; its 84.9% accuracy largely reflects class imbalance. Local color rendering remains mixed and uncalibrated, and there is no untouched classifier holdout or external clinical validation. See the [local classifier guide](docs/experiments/local-eyelid-classifier.md), [PRD](docs/core/PRD.md) and [architecture](docs/core/ARCHITECTURE.md) for scope and limits.

A second isolated local experiment completed on 2026-10-11: frozen MobileNetV3-Small + unchanged LAB features, training-only PCA and nested LightGBM. It detected 5/19 anemia cases at 0.5 and 7/19 under inner-selected thresholds, with specificity 85%/81% and lower ROC-AUC/AP than Experiment 1. The added complexity is not established as an improvement. See the [completed comparison](docs/experiments/eyelid-exp02-results-2026-10-11.md), [hybrid methods](docs/experiments/eyelid-exp02-hybrid.md), and [Git cleanup/preservation handoff](docs/audits/git-cleanup-and-experiment-handoff-2026-10-11.md).

## Local Annotation Workspace

Local eyelid and fingernail annotation work is consolidated in one ignored, private folder: `data/raw/local/annotations/`. Its canonical `annotation_manifest.csv` links each metadata `subject_id` and original filename to the unchanged source in `data/raw/local/incoming/`, the standardized annotation TIFF, the Fiji ROI, and the binary mask. Use `pending_annotation_queue.csv` for the complete prioritized manual-annotation queue; the Batch 1 and Batch 2 folders are no longer active workspaces. See the [local annotation workflow](docs/workflows/local-annotation-workflow.md).

Manual anatomical ROIs and reviewed masks are the primary ground truth. Automated proposals are optional experiments only. The latest per-image inventory, glare caveats, review checks, and exact queue are in the local `data/raw/local/annotations/README.md` and its manifest CSVs. These local patient-derived assets are ignored by Git and must not be committed or shared outside the approved research workspace. No local model has been trained by the annotation audit.

## Acknowledgements

- Public image and metadata dataset downloaded through Kaggle
- ImageNet pretrained weights provided through torchvision
- Python open-source libraries used for image processing, feature extraction, modeling, and evaluation
- The local AIMA data collection team and study participants
