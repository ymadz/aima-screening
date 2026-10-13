#!/usr/bin/env python3
"""Validate the central AIMA research records without training or exporting data.

The checks are intentionally lightweight: documentation structure, registry
status/ID uniqueness, local-link resolution, canonical-source provenance, and
the presence of non-sensitive report/metric manifests. The script never opens
patient-level CSVs, images, predictions, or checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs/research"
ARTIFACT_ROOT = ROOT

REQUIRED_DOCS = {
    "research log": RESEARCH / "RESEARCH_LOG.md",
    "experiment registry": RESEARCH / "EXPERIMENT_REGISTRY.md",
    "methodology": RESEARCH / "METHODOLOGY.md",
    "results": RESEARCH / "RESULTS.md",
    "limitations": RESEARCH / "LIMITATIONS.md",
    "thesis handoff": RESEARCH / "THESIS_HANDOFF.md",
    "documentation guide": RESEARCH / "RESEARCH_DOCUMENTATION_GUIDE.md",
    "research index": RESEARCH / "README.md",
    "Experiment 7 report": ROOT / "docs/experiments/eyelid-exp07-results.md",
}

ARTIFACTS = {
    "PUB-BASELINE-2026-08": (
        "data/processed/public_baseline_reproduction_2026-10-07/metrics.json",
        "data/processed/public_baseline_reproduction_2026-10-07/configuration.json",
    ),
    "SEG-EXTRATREES-PILOT-2026-10-07": (
        "data/processed/segmentation_prototype_2026-10-07/report.json",
    ),
    "DATA-LOCAL-READINESS-2026-10-08": (
        "data/processed/local_cohort_who2024_2026-10-08/summary.json",
        "data/raw/local/annotations/manifests/annotation_manifest.csv",
    ),
    "SEG-PUBLIC-UNET-LOCAL-2026-10-09": (
        "data/processed/segmentation/local_unet_dev_eval_fullframe_2026-10-09/summary.json",
    ),
    "SEG-SAM2-NAIL-PILOT-2026-10-09": (
        "data/processed/segmentation/sam2_nail_lopo_areaqc_v1/summary.json",
    ),
    "AIMA-EXP01-FIXED-0.5": (
        "data/processed/local_eyelid_classification_2026-10-10_v1/metrics.json",
        "data/processed/local_eyelid_classification_2026-10-10_v1/report.md",
        "data/processed/local_eyelid_classification_2026-10-10_v1/run_manifest.json",
        "data/processed/local_eyelid_classification_2026-10-10_v1/verification.json",
    ),
    "AIMA-EXP01-PRESERVATION": (
        "docs/experiments/eyelid-exp01-preservation.md",
    ),
    "AIMA-LAB-NESTED-THRESHOLD": (
        "data/processed/local_eyelid_classification_exp04_color_optimization_2026-10-12_v1/metrics.json",
    ),
    "AIMA-EXP02": (
        "data/processed/local_eyelid_classification_exp02_hybrid_2026-10-11_v1/metrics.json",
        "data/processed/local_eyelid_classification_exp02_hybrid_2026-10-11_v1/report.md",
        "data/processed/local_eyelid_classification_exp02_hybrid_2026-10-11_v1/checksum_manifest.json",
    ),
    "AIMA-EXP03-PUBLIC-TRANSFER": (
        "data/processed/local_eyelid_classification_exp03_transfer_2026-10-11_v3/metrics.json",
        "data/processed/local_eyelid_classification_exp03_transfer_2026-10-11_v3/report.md",
        "data/processed/local_eyelid_classification_exp03_transfer_2026-10-11_v3/checksum_manifest.json",
    ),
    "AIMA-EXP03-PUBLIC-ONLY-LOCAL": (
        "data/processed/local_eyelid_classification_exp03_transfer_2026-10-11_v3/metrics.json",
    ),
    "AIMA-EXP04": (
        "data/processed/local_eyelid_classification_exp04_color_optimization_2026-10-12_v1/metrics.json",
        "data/processed/local_eyelid_classification_exp04_color_optimization_2026-10-12_v1/candidate_metrics.json",
        "data/processed/local_eyelid_classification_exp04_color_optimization_2026-10-12_v1/checksum_manifest.json",
    ),
    "AIMA-EXP04-PRIMARY": (
        "data/processed/local_eyelid_classification_exp04_color_optimization_2026-10-12_v1/metrics.json",
    ),
    "AIMA-EXP04-ROBUST-COMPARISON": (
        "data/processed/local_eyelid_classification_exp04_color_optimization_2026-10-12_v1/metrics.json",
        "data/processed/local_eyelid_classification_exp04_color_optimization_2026-10-12_v1/candidate_metrics.json",
    ),
    "AIMA-EXP04-HB-REGRESSION": (
        "data/processed/local_eyelid_classification_exp04_color_optimization_2026-10-12_v1/regression/metrics.json",
    ),
    "AIMA-LOCAL-NAIL-PAIRED": (
        "data/raw/local/annotations/manifests/pending_annotation_queue.csv",
    ),
    "ANNOT-LOCAL-QUEUE-2026-10": (
        "data/raw/local/annotations/manifests/pending_annotation_queue.csv",
    ),
}

ARTIFACTS.update({
    "AIMA-EXP07": (
        "data/processed/local_eyelid_classification_exp07_one_class_v2/metrics.json",
        "data/processed/local_eyelid_classification_exp07_one_class_v2/run_manifest.json",
        "data/processed/local_eyelid_classification_exp07_one_class_v2/checksum_manifest.json",
        "data/processed/exp07_verification_2026-10-14_v1/reproduction.json",
        "data/processed/exp07_verification_2026-10-14_v1/public_transfer_vs_local_mahalanobis.json",
    ),
    "AIMA-EXP06": (
        "data/processed/local_eyelid_classification_exp06_ensemble_v1/metrics.json",
        "data/processed/local_eyelid_classification_exp06_ensemble_v1/error_overlap.json",
        "data/processed/local_eyelid_classification_exp06_ensemble_v1/experiment_manifest.json",
        "data/processed/local_eyelid_classification_exp06_ensemble_v1/checksum_manifest.json",
        "data/processed/exp06_verification_2026-10-13_v1/reproduction.json",
    ),
    "AIMA-EXP04B": (
        "data/processed/local_eyelid_exp04b_reviewed_comparison_2026-10-12_v1/metrics.json",
        "data/processed/local_eyelid_exp04b_reviewed_comparison_2026-10-12_v1/checksum_manifest.json",
    ),
    "AIMA-EXP05": (
        "data/processed/local_eyelid_classification_exp05_cp_transfer_2026-10-12_v1/metrics.json",
        "data/processed/local_eyelid_classification_exp05_cp_transfer_2026-10-12_v1/public_only/metrics.json",
        "data/processed/local_eyelid_classification_exp05_cp_transfer_2026-10-12_v1/public_validation/metrics.json",
        "data/processed/local_eyelid_classification_exp05_cp_transfer_2026-10-12_v1/run_manifest.json",
        "data/processed/local_eyelid_classification_exp05_cp_transfer_2026-10-12_v1/checksum_manifest.json",
        "data/processed/exp05_verification_2026-10-13_v1/reproduction.json",
    ),
})

ALLOWED_STATUSES = {
    "COMPLETED_VERIFIED",
    "COMPLETED_UNVERIFIED",
    "IN_PROGRESS",
    "PLANNED",
    "BLOCKED",
}


def fail(errors: list[str], message: str) -> None:
    """Collect a concise failure without printing sensitive file contents."""
    errors.append(message)


def check_required_docs(errors: list[str]) -> int:
    """Check required files and a few mandatory headings."""
    headings = {
        "research log": "# AIMA research log",
        "experiment registry": "# AIMA experiment registry",
        "methodology": "# AIMA implemented methodology",
        "results": "# AIMA verified results",
        "limitations": "# AIMA limitations and interpretation boundaries",
        "thesis handoff": "# AIMA thesis handoff",
        "documentation guide": "# Research documentation guide",
        "research index": "# AIMA research records",
        "Experiment 7 report": "# Experiment 7: normal-only anemia novelty detection",
    }
    count = 0
    for name, path in REQUIRED_DOCS.items():
        if not path.is_file():
            fail(errors, f"missing {name}: {path.relative_to(ROOT)}")
            continue
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            fail(errors, f"empty {name}: {path.relative_to(ROOT)}")
        if headings[name] not in content:
            fail(errors, f"missing required heading in {name}")
        count += 1
    return count


def check_registry(errors: list[str]) -> tuple[int, int]:
    """Check registry IDs/statuses and return entry and artifact counts."""
    registry_path = REQUIRED_DOCS["experiment registry"]
    content = registry_path.read_text(encoding="utf-8")
    ids = re.findall(r"^\|\s*`([^`]+)`", content, flags=re.MULTILINE)
    if len(ids) != len(set(ids)):
        fail(errors, "duplicate experiment ID in registry")
    if not ids:
        fail(errors, "registry contains no experiment IDs")
    statuses = set(re.findall(r"`(COMPLETED_VERIFIED|COMPLETED_UNVERIFIED|IN_PROGRESS|PLANNED|BLOCKED)`", content))
    missing_statuses = ALLOWED_STATUSES - statuses
    if missing_statuses:
        fail(errors, f"registry does not demonstrate status vocabulary: {sorted(missing_statuses)}")
    missing_artifact_rows = set(ARTIFACTS) - set(ids)
    if missing_artifact_rows:
        fail(errors, f"registry is missing artifact-backed IDs: {sorted(missing_artifact_rows)}")
    return len(ids), len(ARTIFACTS)


def check_artifacts(errors: list[str]) -> tuple[int, int]:
    """Check existence of declared report/metric manifests only."""
    checked = 0
    missing = 0
    for experiment_id, relative_paths in ARTIFACTS.items():
        artifact_root = ROOT if experiment_id == "AIMA-EXP07" else ARTIFACT_ROOT
        for relative in relative_paths:
            checked += 1
            if not (artifact_root / relative).is_file():
                missing += 1
                fail(errors, f"{experiment_id} missing artifact manifest: {relative}")
    return checked, missing


def check_local_links(errors: list[str]) -> int:
    """Resolve relative Markdown links without fetching URLs."""
    checked = 0
    pattern = re.compile(r"\[[^]]+\]\(([^)]+)\)")
    for path in REQUIRED_DOCS.values():
        if not path.is_file():
            continue
        for target in pattern.findall(path.read_text(encoding="utf-8")):
            target = target.split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "/")):
                continue
            checked += 1
            if not (path.parent / target).exists():
                fail(errors, f"broken local link in {path.relative_to(ROOT)}: {target}")
    return checked


def check_canonical_provenance(errors: list[str]) -> bool:
    """Verify the frozen CSV against its adjacent provenance without reading rows."""
    source = ARTIFACT_ROOT / "data/raw/local/metadata/frozen_2026-10-07/AIMA Patients Metadata - Sheet1 values.csv"
    provenance_path = source.parent / "provenance.json"
    if not source.is_file() or not provenance_path.is_file():
        fail(errors, "canonical frozen metadata or provenance.json is missing")
        return False
    recorded = json.loads(provenance_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != recorded.get("frozen_csv_sha256"):
        fail(errors, "canonical frozen metadata does not match provenance SHA-256")
        return False
    return True


def check_metric_shapes(errors: list[str]) -> int:
    """Check aggregate metric keys/counts without loading row-level predictions."""
    checks = 0
    public = json.loads((ARTIFACT_ROOT / "data/processed/public_baseline_reproduction_2026-10-07/metrics.json").read_text())
    if public.get("pooled_out_of_fold_metrics", {}).get("roc_auc") is None:
        fail(errors, "public baseline pooled ROC-AUC is missing")
    checks += 1

    exp1 = json.loads((ARTIFACT_ROOT / "data/processed/local_eyelid_classification_2026-10-10_v1/metrics.json").read_text())
    baseline = exp1.get("pooled_oof", {}).get("lab_logistic", {})
    if baseline.get("subjects") != 119 or baseline.get("anemia") != 19 or baseline.get("non_anemia") != 100:
        fail(errors, "Experiment 1 aggregate subject/class counts do not match the registry")
    checks += 1

    exp2 = json.loads((ARTIFACT_ROOT / "data/processed/local_eyelid_classification_exp02_hybrid_2026-10-11_v1/metrics.json").read_text())
    if exp2.get("baseline_fixed_0_5", {}).get("subjects") != 119:
        fail(errors, "Experiment 2 aggregate subject count is missing or changed")
    checks += 1

    exp3 = json.loads((ARTIFACT_ROOT / "data/processed/local_eyelid_classification_exp03_transfer_2026-10-11_v3/metrics.json").read_text())
    if exp3.get("models", {}).get("selected_selected", {}).get("subjects") != 119:
        fail(errors, "Experiment 3 selected aggregate subject count is missing or changed")
    checks += 1

    exp4 = json.loads((ARTIFACT_ROOT / "data/processed/local_eyelid_classification_exp04_color_optimization_2026-10-12_v1/metrics.json").read_text())
    models = exp4.get("models", {})
    for key in ("exp04_joint_selected", "ablation_A_robust_lr_c1_none"):
        if models.get(key, {}).get("subjects") != 119:
            fail(errors, f"Experiment 4 aggregate metric key is missing or changed: {key}")
        checks += 1
    regression = json.loads((ARTIFACT_ROOT / "data/processed/local_eyelid_classification_exp04_color_optimization_2026-10-12_v1/regression/metrics.json").read_text())
    if not regression:
        fail(errors, "Experiment 4 regression metrics are empty")
    checks += 1
    exp4b = json.loads((ARTIFACT_ROOT / "data/processed/local_eyelid_exp04b_reviewed_comparison_2026-10-12_v1/metrics.json").read_text())
    if exp4b.get("models", {}).get("glare_excluded_robust_balanced_selected", {}).get("subjects") != 119:
        fail(errors, "Experiment4B reviewed aggregate cohort differs")
    checks += 1
    exp5_root = ARTIFACT_ROOT / "data/processed/local_eyelid_classification_exp05_cp_transfer_2026-10-12_v1"
    exp5 = json.loads((exp5_root / "metrics.json").read_text())
    for key in ("cp_classification_selected_threshold", "cp_regression_selected_threshold", "joint_selected_threshold"):
        if exp5.get("models", {}).get(key, {}).get("subjects") != 119:
            fail(errors, f"Experiment5 aggregate cohort differs: {key}")
        checks += 1
    run = json.loads((exp5_root / "run_manifest.json").read_text())
    proof = json.loads((ARTIFACT_ROOT / "data/processed/exp05_verification_2026-10-13_v1/reproduction.json").read_text())
    if (run.get("public_models"), run.get("local_neural_models"), run.get("local_classical_models")) != (2, 120, 60) or proof.get("status") != "reproduced_without_refitting":
        fail(errors, "Experiment5 run/model reproduction evidence differs")
    checks += 1
    exp6_root = ARTIFACT_ROOT / "data/processed/local_eyelid_classification_exp06_ensemble_v1"
    exp6 = json.loads((exp6_root / "metrics.json").read_text())
    for key, counts in (("exp04_robust_historical", [[84, 16], [9, 10]]),
                        ("exp05_cp_C_historical", [[78, 22], [7, 12]]),
                        ("equal_spec0.8", [[79, 21], [8, 11]]),
                        ("weighted_spec0.8", [[79, 21], [9, 10]])):
        model = exp6.get("models", {}).get(key, {})
        if model.get("subjects") != 119 or model.get("confusion_matrix_labels_0_1") != counts:
            fail(errors, f"Experiment6 aggregate evidence differs: {key}")
        checks += 1
    proof6 = json.loads((ARTIFACT_ROOT / "data/processed/exp06_verification_2026-10-13_v1/reproduction.json").read_text())
    if proof6.get("status") != "reproduced_without_refitting" or proof6.get("models_reproduced") != 40:
        fail(errors, "Experiment6 reproduction evidence differs")
    checks += 1
    exp7_root = ROOT / "data/processed/local_eyelid_classification_exp07_one_class_v2"
    exp7 = json.loads((exp7_root / "metrics.json").read_text())
    run7 = json.loads((exp7_root / "run_manifest.json").read_text())
    proof7 = json.loads((ROOT / "data/processed/exp07_verification_2026-10-14_v1/reproduction.json").read_text())
    if (run7.get("subjects"), run7.get("class_counts_0_1"), run7.get("models"),
            run7.get("policy_outputs")) != (119, [100, 19], 289, 87):
        fail(errors, "Experiment7 cohort/run counts differ")
    if run7.get("source_commit") != "6513c9c2aeadc57e436ec55b292f362ff1316fa8":
        fail(errors, "Experiment7 execution provenance differs")
    if (proof7.get("status"), proof7.get("subjects"), proof7.get("models"),
            proof7.get("policy_outputs"), proof7.get("fresh_local_features_exact")) != (
            "reproduced_without_refitting", 119, 289, 87, True):
        fail(errors, "Experiment7 independent reproduction evidence differs")
    checks += 3
    models7 = exp7.get("policies", {})
    for key, expected in (("mahalanobis_A_085", [[87, 13], [15, 4]]),
                          ("svm_010_scale_A_085", [[88, 12], [15, 4]]),
                          ("nested_B_085", [[92, 8], [16, 3]]),
                          ("B1_A_085", [[0, 100], [0, 19]]),
                          ("B2_A_085", [[88, 12], [14, 5]]),
                          ("C1_A_085", [[23, 77], [7, 12]]),
                          ("D1_A_085", [[25, 75], [3, 16]]),
                          ("exp04_robust", [[84, 16], [9, 10]])):
        if models7.get(key, {}).get("confusion_matrix_labels_0_1") != expected:
            fail(errors, f"Experiment7 aggregate evidence differs: {key}")
        checks += 1
    references7 = {"exp04_robust", "exp05_cp_C", "exp05_joint_primary", "exp06_equal_spec080"}
    policy_keys = set(models7) - references7
    if len(policy_keys) != 87:
        fail(errors, "Experiment7 operating-policy coverage differs")
    report = REQUIRED_DOCS["Experiment 7 report"].read_text()
    for key in sorted(policy_keys):
        row = models7[key]
        counts = " / ".join(str(row[name]) for name in ("TP", "FP", "TN", "FN"))
        line = (f"| {key} | {counts} | {100*row['sensitivity']:.1f}% | "
                f"{100*row['specificity']:.1f}% | {row['roc_auc']:.6f} | "
                f"{row['average_precision']:.6f} |")
        if line not in report:
            fail(errors, f"Experiment7 report transcription differs: {key}")
        checks += 1
    return checks


def main() -> int:
    """Run all checks and return a shell-friendly status."""
    global ARTIFACT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=ROOT, help="Read-only artifact/canonical source root, separate from this worktree docs")
    ARTIFACT_ROOT = parser.parse_args().source_root.resolve()
    errors: list[str] = []
    doc_count = check_required_docs(errors)
    registry_count, declared_artifact_groups = check_registry(errors)
    artifact_checks, missing_artifacts = check_artifacts(errors)
    link_checks = check_local_links(errors)
    provenance_ok = check_canonical_provenance(errors)
    metric_checks = check_metric_shapes(errors)

    if errors:
        print("RESEARCH_DOCS_FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    print("RESEARCH_DOCS_PASS")
    print(f"documents_checked={doc_count}")
    print(f"registry_entries_checked={registry_count}")
    print(f"artifact_groups_declared={declared_artifact_groups}")
    print(f"artifact_manifests_checked={artifact_checks}")
    print(f"missing_artifacts={missing_artifacts}")
    print(f"local_links_checked={link_checks}")
    print(f"canonical_provenance_checked={provenance_ok}")
    print(f"aggregate_metric_checks={metric_checks}")
    print("patient_level_files_read=false")
    print("training_started=false")
    return 0


if __name__ == "__main__":
    sys.exit(main())
