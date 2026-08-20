# AIMA Screening

Development environment for the anemia screening experiments described in Table 8.

## Setup

Python 3.10 is required for the dedicated environment:

```bash
python3.10 -m venv aima_env
source aima_env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate with `aima_env\\Scripts\\activate`.

## Kaggle dataset

Create a Kaggle API token in your account settings and place the downloaded file at `~/.kaggle/kaggle.json` on macOS/Linux. Then run:

```bash
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d harshwardhanfartale/eyes-defy-anemia
unzip eyes-defy-anemia.zip -d data/raw/
```

Keep credentials out of this repository. The raw and processed data directories are ignored by Git.

## Layout   

- `data/raw/`: downloaded Kaggle images
- `data/processed/`: segmented ROIs and extracted features
- `src/preprocess.py`: CIE L*a*b* conversion and ITA calculation
- `src/feature_extractor.py`: MobileNet/InceptionV3 feature extraction
- `src/train_classifier.py`: LightGBM training and 10-fold CV
- `src/evaluate.py`: sensitivity, specificity, MAE, and ANOVA evaluation

## Build the multimodal dataset

The downloaded dataset stores subjects under country and numeric ID folders. The
builder joins those folders to the country workbooks, uses the canonical
`*_palpebral.png` ROI for color and MobileNet features, and uses the matching
full JPG for the cutaneous ITA estimate:

```bash
source aima_env/bin/activate
python src/dataset_builder.py
```

This writes `data/processed/dataset.csv`. Labels are derived from hemoglobin
using `< 12 g/dL` for females and `< 13 g/dL` for males. Rows without a numeric
hemoglobin value or a matching ROI/full image pair are skipped.

## Train the LightGBM baseline

Run the 10-fold stratified cross-validation pipeline after building the dataset:

```bash
python src/train_classifier.py
```

The model uses 583 numeric inputs: ITA, six LAB statistics, and 576 MobileNet
embeddings. Identifiers and clinical metadata are retained for analysis but are
excluded from training to prevent target leakage. Out-of-fold probabilities and
labels are saved to `results/cross_validation_predictions.csv`.

## Evaluate skin-tone cohorts

Run subgroup metrics and the one-way ANOVA fairness check on the out-of-fold
predictions:

```bash
python src/evaluate.py
```

The current dataset contains 182 `Brown / Dark` subjects and only 2 `Very
Light` subjects, with no observations in the other ITA bands. The resulting
ANOVA is therefore underpowered; a p-value above 0.05 means only that the null
hypothesis was not rejected, not that demographic neutrality has been proven.
