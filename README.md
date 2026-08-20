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
