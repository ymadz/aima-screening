# AIMA Screening

Research pipeline for screening anemia from conjunctival images and patient metadata.

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

In development as a student research prototype. The public inner-eyelid baseline is implemented. Local Zamboanga image import, fingernail modeling, subject-based splitting, and independent local testing are still in progress.

## Acknowledgements

- Public image and metadata dataset downloaded through Kaggle
- ImageNet pretrained weights provided through torchvision
- Python open-source libraries used for image processing, feature extraction, modeling, and evaluation
- The local AIMA data collection team and study participants
