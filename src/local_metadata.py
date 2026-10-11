"""Build an unlabeled local candidate table from the frozen Sheet manifest.

This does not derive anemia labels or approve subjects for supervised training.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FROZEN_DIR = ROOT / "data/raw/local/metadata/frozen_2026-10-07"
FROZEN_CSV = FROZEN_DIR / "AIMA Patients Metadata - Sheet1 values.csv"
FROZEN_PROVENANCE = FROZEN_DIR / "provenance.json"
SUBJECT_MANIFEST = ROOT / "data/raw/local/manifest/frozen_2026-10-07/local_subject_manifest.csv"
OUTPUT_DIR = ROOT / "data/processed/local/frozen_2026-10-07"


def build_local_candidate_table(output_dir: Path = OUTPUT_DIR) -> Path:
    """Save current subject-level metadata without fabricating clinical labels."""
    source_hash = hashlib.sha256(FROZEN_CSV.read_bytes()).hexdigest()
    provenance = json.loads(FROZEN_PROVENANCE.read_text())
    if source_hash != provenance["frozen_csv_sha256"]:
        raise ValueError("Frozen Sheet snapshot differs from recorded SHA-256")

    frame = pd.read_csv(SUBJECT_MANIFEST, dtype={"subject_id": str})
    if len(frame) != 140 or frame["subject_id"].duplicated().any():
        raise ValueError("Frozen subject manifest has unexpected IDs or count")
    if set(frame["metadata_source_sha256"]) != {source_hash}:
        raise ValueError("Subject manifest does not derive from the frozen Sheet")

    columns = [
        "subject_id", "age", "sex", "lab_hb_g_per_l", "lab_hb_g_per_dl",
        "anemia_status", "raw_eyelid_file", "raw_nail_file", "source",
        "cbc_status", "eyelid_reference_status", "nail_reference_status",
        "approved_for_supervised_learning", "metadata_source_path",
        "metadata_source_sha256",
    ]
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "local_metadata_candidates.csv"
    frame[columns].to_csv(output_path, index=False)
    (output_dir / "provenance.json").write_text(
        json.dumps({
            "source_sheet_url": provenance["source_url"],
            "frozen_csv_path": str(FROZEN_CSV),
            "frozen_csv_sha256": source_hash,
            "subject_manifest_path": str(SUBJECT_MANIFEST),
            "rows": len(frame),
            "label_policy": "none; anemia_status is source-only and blank",
        }, indent=2) + "\n"
    )
    return output_path


if __name__ == "__main__":
    print(build_local_candidate_table())
