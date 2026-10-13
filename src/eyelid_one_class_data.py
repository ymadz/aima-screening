"""Read-only source audit and exclusive private Experiment 7 feature preparation."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image

from .cp_transfer_data import audit_public, describe, verify_official_v2
from .eyelid_color_data import ROBUST_NAMES, color_features
from .eyelid_ensemble_run import EXP4, EXP5, checksum_guard
from .eyelid_glare_prepare import validate_fresh_inputs
from .eyelid_one_class import matrix
from .eyelid_transfer_data import public_metadata
from .local_eyelid_hybrid_features import validated_inputs
from .local_eyelid_validation import (load_manual_pair, perceptual_hash,
    require_duplicate_clearance, sha256, source_snapshot, strict_csv)
from .preserve_local_experiment import tree_checksums
from .public_segmentation_data import discover_public_pairs, load_public_pair

EXP3 = "local_eyelid_classification_exp03_transfer_2026-10-11_v3"
EXP6 = "local_eyelid_classification_exp06_ensemble_v1"
OUTPUT_NAME = "local_eyelid_classification_exp07_one_class_v1"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def normal_groups(x: np.ndarray, ids: list[str], labels: np.ndarray,
                  groups: list[str], training_ids: set[str], source: str) -> tuple[np.ndarray, list[str], dict]:
    """Collapse conservative groups, rejecting mixed labels or split membership."""
    x = matrix(x)
    if len(ids) != len(set(ids)) or any(len(a) != len(x) for a in (ids, labels, groups)):
        raise ValueError("Public feature/identity coverage differs")
    collected, names, mixed = [], [], 0
    for group in sorted(set(groups)):
        indices = np.flatnonzero(np.asarray(groups) == group)
        membership = [ids[i] in training_ids for i in indices]
        if any(membership) and not all(membership):
            raise ValueError("Public conservative group crosses historical split")
        if len(set(labels[indices])) != 1:
            mixed += 1
            continue
        if all(membership) and labels[indices[0]] == 0:
            collected.append(x[indices].mean(axis=0))
            names.append(source+"/"+group)
    if len(names) < 6:
        raise ValueError("Insufficient verified public normal groups")
    return np.asarray(collected), names, {"normal_training_groups": len(names),
        "mixed_label_groups_excluded": mixed, "aggregation": "one arithmetic mean per conservative normal group"}


def fresh_local(root: Path, output: Path) -> tuple[dict, list[dict], list[dict]]:
    """Run the canonical repository's full audit with its explicit source root."""
    # The existing auditor binds constants to its own checkout. Running it in the
    # canonical repository avoids rebinding globals or changing historical code.
    program = """import json,sys
from pathlib import Path
from src.local_eyelid_readiness import audit
out=Path(sys.argv[1])
summary,eligibility,inputs,candidates=audit()
for name,value in [('fresh_audit',summary),('fresh_eligibility',eligibility),('fresh_inputs',inputs),('duplicate_candidates',candidates)]:
 (out/(name+'.json')).write_text(json.dumps(value,indent=2)+'\\n')
"""
    subprocess.run([sys.executable, "-c", program, str(output)], cwd=root, check=True)
    return tuple(json.loads((output/(name+".json")).read_text())
                 for name in ("fresh_audit", "fresh_eligibility", "fresh_inputs"))


def prepare(root: Path, output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    code = Path(__file__).resolve().parents[1]
    os.umask(0o077)
    cv2.setNumThreads(1)
    prefix = OUTPUT_NAME.rsplit("1", 1)[0]
    version = output.name.removeprefix(prefix)
    if (output.exists() or not output.is_relative_to(code/"data/processed")
            or not output.name.startswith(prefix) or not version.isdigit() or int(version) < 1):
        raise ValueError("Require a new exclusive versioned Experiment 7 output")
    if subprocess.run(["git", "check-ignore", "-q", str(output/"private.csv")], cwd=code).returncode:
        raise ValueError("Private output must be ignored")
    gate = root/"data/processed/exp07_preservation_2026-10-13_v1/preservation_manifest.json"
    preservation = json.loads(gate.read_text())
    if preservation["status"] != "PASS" or preservation["missing_or_damaged"]:
        raise ValueError("Preservation gate did not pass")
    canonical = root/"data/raw/local/metadata/frozen_2026-10-07/AIMA Patients Metadata - Sheet1 values.csv"
    snapshot = source_snapshot(canonical, canonical.parent/"provenance.json")
    output.mkdir(parents=True, mode=0o700)
    audit_dir = output/"audit"
    audit_dir.mkdir(mode=0o700)
    write_json(audit_dir/"canonical_snapshot_before_read.json", snapshot)
    summary, clinical, fresh = fresh_local(root, audit_dir)
    data = validated_inputs(root, root/"data/processed/local_eyelid_classification_2026-10-10_v1",
                            root/"data/processed/local_eyelid_readiness_2026-10-10_v2")
    validate_fresh_inputs({r["subject_id"]: r for r in fresh}, data)
    candidates = json.loads((audit_dir/"duplicate_candidates.json").read_text())
    review = strict_csv(root/"data/processed/local_eyelid_classification_2026-10-10_v1/duplicate_review_used.csv")
    require_duplicate_clearance(candidates, review)
    if summary["labeled_eyelid_subjects"] != 119 or summary["anemia"] != 19 or summary["prior_metadata_and_label_discrepancies"]:
        raise ValueError("Frozen local cohort changed")
    if set(summary["blockers"]) - {"human_duplicate_adjudication_pending"}:
        raise ValueError("Unresolved source audit blockers")
    # Preserve the audit's historical wording; record the verified release apart.
    write_json(audit_dir/"gate_release.json", {"status": "PASS_WITH_MANUAL_REVIEW",
        "duplicate_clearance": "existing hash-linked owner distinct-source decisions verified",
        "anatomical_approval": "owner boundary review, not clinician validation"})
    hashes = dict(summary["source_files"])
    inventory = root/"data/raw/local/manifest/frozen_2026-10-07/local_image_manifest.csv"
    hashes[str(inventory)] = sha256(inventory)
    registered = strict_csv(inventory)
    incoming = {p.name: p for p in (root/"data/raw/local/incoming").iterdir() if p.is_file()}
    if set(incoming) != {r["original_filename"] for r in registered}:
        raise ValueError("Original incoming inventory changed")
    for record in registered:
        path = incoming[record["original_filename"]]
        if sha256(path) != record["sha256"]:
            raise ValueError("Registered original bytes changed")
        hashes[str(path)] = record["sha256"]
    fingerprints, domains, byte_hashes = [], [], []
    local_x, quality = [], []
    clinical_by_id = {r["subject_id"]: r for r in clinical}
    for sid in data["subjects"]:
        row = data["rows"][sid]
        rgb, mask = load_manual_pair(root/row["annotation_image_path"], root/row["mask_path"])
        local_x.append(color_features(rgb, mask)["robust"])
        fingerprints.append(perceptual_hash(rgb)); domains.append("local")
        byte_hashes.append(row["original_source_sha256"])
        for path_field, hash_field in [("source_image_path", "original_source_sha256"),
                ("annotation_image_path", "annotation_image_sha256"), ("roi_path", "roi_sha256"), ("mask_path", "mask_sha256")]:
            hashes[str(root/row[path_field])] = row[hash_field]
        c = clinical_by_id[sid]
        quality.append({"subject_id": sid, "hb": float(c["normalized_hb_g_per_dl"]),
                        "sex": c["sex"], "age": int(c["age"]), **describe(rgb, mask)})
    local_x = matrix(np.asarray(local_x))
    checksum_guard(root/"data/processed"/EXP4)
    with np.load(root/"data/processed"/EXP4/"features.npz", allow_pickle=False) as old:
        np.testing.assert_array_equal(old["subjects"], data["subjects"])
        np.testing.assert_array_equal(old["labels"], data["labels"])
        np.testing.assert_array_equal(old["folds"], data["folds"])
        np.testing.assert_array_equal(old["A_robust"], local_x)
    public3 = root/"data/processed"/EXP3
    checksum_guard(public3)
    old3 = json.loads((public3/"public_audit.json").read_text())
    metadata, eyes_audit = public_metadata(root/"data/raw/dataset anemia")
    if any(sha256(Path(s["path"])) != s["sha256"] for s in old3["sources"]):
        raise ValueError("Eyes metadata differs from preserved policy")
    labels_by_id = {s: metadata[s]["label"] for s in metadata}
    old_records = {r["subject"]: r for r in old3["records_private"]}
    eyes_x, eyes_ids, eyes_y = [], [], []
    for pair in discover_public_pairs(root/"data/raw/dataset anemia"):
        pixels, mask, evidence = load_public_pair(pair)
        record = old_records[pair.key]
        if record["label"] != labels_by_id[pair.key]:
            raise ValueError("Eyes labels changed")
        for path, entry in [(pair.image_path, "image"), (pair.segmentation_path, "mask")]:
            digest = sha256(path)
            if digest != record[entry]["sha256"]:
                raise ValueError("Eyes source bytes changed")
            hashes[str(path)] = digest
        with Image.open(pair.image_path) as image:
            if image.info.get("icc_profile"):
                raise ValueError("Unregistered public color profile requires new compatibility review")
        fingerprints.append(perceptual_hash(pixels)); domains.append("eyes")
        byte_hashes.append(record["image"]["sha256"])
        if pair.key in old3["training_subjects"] or pair.key in old3["validation_subjects"]:
            eyes_ids.append(pair.key); eyes_y.append(record["label"])
            eyes_x.append(color_features(pixels, mask.astype(np.uint8)*255)["robust"])
    group_by_id = dict(zip([r["subject"] for r in old3["records_private"] if r["label"] is not None], old3["groups"]))
    eyes, eyes_names, eyes_normals = normal_groups(np.asarray(eyes_x), eyes_ids, np.asarray(eyes_y),
        [group_by_id[s] for s in eyes_ids], set(old3["training_subjects"]), "eyes")
    cp5 = root/"data/processed"/EXP5
    checksum_guard(cp5)
    config5 = json.loads((cp5/"preparation_config.json").read_text())
    zip_entries = [(Path(p), h) for p, h in config5["source_hashes"].items() if p.endswith(".zip")]
    if len(zip_entries) != 1:
        raise ValueError("Ambiguous official CP release")
    cp_root = Path(config5["public_root"])
    official = verify_official_v2(cp_root, *zip_entries[0])
    cp_records, cp_audit = audit_public(cp_root)
    hashes.update(cp_audit["source_hashes"])
    hashes[str(zip_entries[0][0])] = zip_entries[0][1]
    cp_x = []
    for row in cp_records:
        with Image.open(row["path"]) as image:
            values = np.asarray(image).copy()
        rgb, alpha = values[:, :, :3], values[:, :, 3]
        mask = (alpha == 255).astype(np.uint8)*255
        cp_x.append(color_features(rgb, mask)["robust"])
        screened = rgb.copy(); screened[alpha == 0] = 128
        fingerprints.append(perceptual_hash(screened)); domains.append("cp")
        byte_hashes.append(row["source_sha256"])
    cp_ids = [r["image_id"] for r in cp_records]
    cp, cp_names, cp_normals = normal_groups(np.asarray(cp_x), cp_ids,
        np.asarray([r["label"] for r in cp_records]), [r["split_group"] for r in cp_records],
        {cp_ids[i] for i in cp_audit["training_indices"]}, "cp")
    fp = np.asarray(fingerprints); cross = []; blocked = set()
    for i in range(len(fp)):
        for j in range(i+1, len(fp)):
            if domains[i] == domains[j]:
                continue
            distance = int(np.count_nonzero(fp[i] != fp[j]))
            if byte_hashes[i] == byte_hashes[j] or distance <= 6:
                cross.append({"a": i, "b": j, "domains": [domains[i], domains[j]],
                              "phash_distance": distance, "exact_bytes": byte_hashes[i] == byte_hashes[j]})
                blocked.update([domains[i], domains[j]])
    status = {"B": "eyes" not in blocked, "C": "cp" not in blocked,
              "D": not ({"eyes", "cp"} & blocked)}
    ages = np.asarray([int(clinical_by_id[s]["age"]) for s in data["subjects"]])
    audit = {"status": "PASS_WITH_MANUAL_REVIEW", "local": {"subjects": 119, "class_counts_0_1": [100, 19],
        "approved_eyelids": summary["approved_eyelids"], "age_min_max": [int(ages.min()), int(ages.max())],
        "references_resolved": summary["resolved_metadata_image_references"], "original_inventory_files": len(registered),
        "exact_robust18_cache_match": True, "label_policy": "unchanged WHO adult 15–65; M<13/F_nonpregnant<12; attestation outside Sheet"},
        "eyes": {"source_subjects": old3["subjects"], "eligible_subjects": len(eyes_ids),
            "conservative_groups": len(set(group_by_id.values())), "class_counts_0_1": np.bincount(eyes_y).tolist(),
            "training_class_counts_0_1": old3["training_class_counts"], "validation_class_counts_0_1": old3["validation_class_counts"],
            **eyes_normals, "label_policy": old3["label_policy"],
            "age_range_eligible": [min(metadata[s]["age"] for s in eyes_ids), max(metadata[s]["age"] for s in eyes_ids)],
            "exclusions": eyes_audit["exclusions"], "color": "218 untagged RGB JPEGs; registered palpebral raster, assumed sRGB",
            "roi": "provided palpebral segmentation only; never forniceal or combined", "license": "original IEEE DataPort terms not independently retrievable; no redistribution authorized"},
        "cp": {"images": cp_audit["unique_images"], "conservative_groups": cp_audit["split_groups"],
            "patient_count": "unknown", "class_counts_0_1": cp_audit["class_counts_0_1"],
            "training_class_counts_0_1": cp_audit["training_class_counts"], "validation_class_counts_0_1": cp_audit["validation_class_counts"],
            **cp_normals, "label_policy": cp_audit["label_policy"], "age": cp_audit["age_sex_metadata"],
            "official_files_verified": official["local_files_exactly_matching_official"], "aliases_excluded": cp_audit["duplicate_alias_files_not_loaded"],
            "license": official["license"], "color": "RGBA PNG, no ICC/EXIF; opaque alpha==255 core assumed sRGB",
            "roi": "source describes palpebral conjunctiva; triangle/entropy extraction can select a different intensity/extent than local manual ROI"},
        "cross_source_candidates": len(cross), "track_gate": status,
        "pooling_justification": "same documented gross palpebral target and identical functions; source-balanced exploratory distributions only, no pixel/anatomical equivalence",
        "limitations": ["unverified CP patient identities", "different adult/pediatric definitions", "mixed uncalibrated acquisition/rendering",
            "public ROI extraction and spatial resolution differ", "exact/perceptual screens cannot prove identity independence", "skin-tone fairness unassessed"]}
    arrays = {"subjects": np.asarray(data["subjects"], dtype=str), "labels": data["labels"], "folds": data["folds"],
              "local": local_x, "eyes": eyes, "eyes_ids": np.asarray(eyes_names, dtype=str),
              "cp": cp, "cp_ids": np.asarray(cp_names, dtype=str), "schema": np.asarray(ROBUST_NAMES, dtype=str)}
    np.savez_compressed(output/"features.npz", **arrays)
    write_json(audit_dir/"dataset_compatibility.json", audit)
    write_json(audit_dir/"cross_source_candidates_private.json", cross)
    write_json(audit_dir/"local_quality_private.json", quality)
    write_json(audit_dir/"cp_public_audit_private.json", cp_audit)
    write_json(audit_dir/"cp_public_records_private.json", cp_records)
    write_json(audit_dir/"eyes_public_audit_private.json", eyes_audit)
    write_json(output/"preparation.json", {"canonical_snapshot_before_read": snapshot, "source_root": str(root),
        "source_hashes": hashes, "preservation_manifest": str(gate), "preservation_manifest_sha256": sha256(gate),
        "historical_checksums": {r["path"]: r["sha256"] for r in preservation["directories"]},
        "feature_schema": list(ROBUST_NAMES), "feature_sha256": sha256(output/"features.npz"),
        "track_gate": status, "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code, text=True).strip()})
    print(json.dumps(audit, indent=2), flush=True)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source_root, args.output)
