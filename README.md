# AIMA Screening

Multimodal anemia-screening research pipeline using CIE L*a*b* colorimetry,
cutaneous ITA, MobileNetV3-Small visual embeddings, and LightGBM.

## Project layout

```text
aima-screening/
├── aima_env/                 # Local Python 3.10 environment, ignored by Git
├── data/raw/                 # Kaggle images and country workbooks
├── data/processed/           # Fused feature tables
├── results/                  # OOF predictions and audit artifacts
├── src/
│   ├── preprocess.py         # CIE L*a*b* and ITA calculations
│   ├── feature_extractor.py  # MobileNetV3-Small embeddings
│   ├── dataset_builder.py    # Metadata join and multimodal fusion
│   ├── train_classifier.py   # LightGBM 10-fold CV
│   └── evaluate.py           # Cohort metrics and ANOVA
├── requirements.txt
├── MODEL_CARD.md
└── FINDINGS_AND_EVALUATION.md
```

## Environment setup

Python 3.10 is required. On macOS with Homebrew:

```bash
brew install python@3.10 libomp
/opt/homebrew/bin/python3.10 -m venv aima_env
source aima_env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, create the environment with `py -3.10 -m venv aima_env` and activate
with `aima_env\\Scripts\\activate`.

## Dataset setup

Create a Kaggle API token at https://www.kaggle.com/settings. Store the token
outside this repository. The Kaggle CLI accepts either the legacy JSON file or
the current access-token file:

```bash
mkdir -p ~/.kaggle
# Place kaggle.json at ~/.kaggle/kaggle.json, then:
chmod 600 ~/.kaggle/kaggle.json

cd /path/to/aima-screening
source aima_env/bin/activate
kaggle datasets download -d harshwardhanfartale/eyes-defy-anemia
unzip eyes-defy-anemia.zip -d data/raw/
rm eyes-defy-anemia.zip
```

The downloaded data is organized as `data/raw/dataset anemia/<country>/<number>`
with `India.xlsx` and `Italy.xlsx` metadata. Do not commit credentials, raw data,
or generated feature tables.

## Run the pipeline

Build one fused row per valid canonical palpebral ROI:

```bash
python src/dataset_builder.py
```

This joins country and numeric subject folders to the workbooks, derives labels
from hemoglobin (`< 12 g/dL` for females and `< 13 g/dL` for males), computes
ITA from the full JPG skin margin, and writes `data/processed/dataset.csv`.

Train and generate out-of-fold predictions:

```bash
python src/train_classifier.py
```

Evaluate ITA cohorts and run the ANOVA diagnostic:

```bash
python src/evaluate.py
```

Outputs are `results/cross_validation_predictions.csv` and the independently
generated `results/feature_gain_importance.csv`.

## Plan A and Plan B

**Plan A: multimodal fusion.** Use the canonical palpebral ROI, six LAB
statistics, cutaneous ITA, and 576 MobileNet features as the 583 LightGBM
inputs. This is the primary thesis pipeline and the configuration used for the
reported baseline.

**Plan B: controlled fallback.** If a deployment site lacks a matching full JPG
for the ITA skin patch or has incompatible metadata, retain only rows with
verified labels and use the available ROI features, while reporting the missing
modality and evaluating it as a separate experiment. Do not replace missing
labels with guessed values or compare Plan B results directly with Plan A
without a protocol change.

## Current status

The verified dataset contains 184 unique country-subject pairs and 583 numeric
model inputs. The baseline and audit results are summarized in
[FINDINGS_AND_EVALUATION.md](FINDINGS_AND_EVALUATION.md), with model limitations
and deployment claims in [MODEL_CARD.md](MODEL_CARD.md).
