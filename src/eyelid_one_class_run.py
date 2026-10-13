"""Nested one-class evaluation, immutable private artifacts and saved-model QA."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from .eyelid_color_data import ROBUST_NAMES, color_features
from .eyelid_ensemble_run import EXP4, EXP5, checksum_guard
from .eyelid_information_audit import severity
from .eyelid_one_class import (BY_ID, CANDIDATES, SEED, TARGETS, TRACKS, adapt_normal,
    assisted_threshold, balanced_sources, calibration, fit_normal, inner_assignments,
    matrix, normal_threshold, normalize)
from .eyelid_one_class_data import EXP6, write_json
from .eyelid_one_class_evaluation import analyze
from .local_eyelid_validation import load_manual_pair, sha256
from .preserve_local_experiment import tree_checksums


def public_sources(name: str, data: dict) -> list[tuple[np.ndarray, list[str]]]:
    sources = []
    if name.startswith(("B", "D")):
        sources.append((data["eyes"], data["eyes_ids"].tolist()))
    if name.startswith(("C", "D")):
        sources.append((data["cp"], data["cp_ids"].tolist()))
    return sources


def fit_scope(name: str, x: np.ndarray, ids: list[str], forbidden: list[str], data: dict):
    """Callers supply only verified normal rows; public rows are also normal-only."""
    y = np.zeros(len(x), dtype=int)
    if name in BY_ID:
        return fit_normal(name, x, y, ids, forbidden)
    sources = public_sources(name, data)
    if name.endswith("3"):
        return balanced_sources([*sources, (x, ids)], forbidden)
    public = balanced_sources(sources, forbidden)
    return adapt_normal(public, x, y, ids, forbidden) if name.endswith("2") else public


def crossfit_outer(name: str, train_x: np.ndarray, train_y: np.ndarray,
                   train_ids: list[str], heldout_ids: list[str], heldout_x: np.ndarray,
                   public_data: dict, save=None) -> dict:
    """No held-out labels are accepted anywhere in fitting or decision selection."""
    train_x, heldout_x = matrix(train_x), matrix(heldout_x)
    if len(set(train_ids)) != len(train_ids) or set(train_ids) & set(heldout_ids):
        raise ValueError("Outer subject partitions overlap")
    if len(train_x) != len(train_y) or len(train_ids) != len(train_y) or len(heldout_ids) != len(heldout_x):
        raise ValueError("Subject/feature coverage differs")
    assignment = inner_assignments(train_y)
    inner_scores = np.full(len(train_y), np.nan)
    scopes = []
    for inner_fold in range(3):
        fitting = np.flatnonzero((assignment != inner_fold) & (train_y == 0))
        validation = np.flatnonzero(assignment == inner_fold)
        excluded = heldout_ids+[train_ids[i] for i in validation]
        model = fit_scope(name, train_x[fitting], [train_ids[i] for i in fitting], excluded, public_data)
        inner_scores[validation] = model.score(train_x[validation])
        scope = {"inner_fold": inner_fold, "local_normal_indices": fitting.tolist(),
                 "validation_indices": validation.tolist(), "forbidden_ids": excluded,
                 "training_ids": model.training_ids, "adaptation_ids": model.adaptation_ids}
        if save is not None:
            scope["model_path"] = save(f"inner_{inner_fold}", model)
        scopes.append(scope)
    fitting = np.flatnonzero(train_y == 0)
    model = fit_scope(name, train_x[fitting], [train_ids[i] for i in fitting], heldout_ids, public_data)
    outer_path = save("outer", model) if save is not None else None
    return {"inner_scores": inner_scores, "outer_scores": model.score(heldout_x),
            "inner_assignments": assignment, "inner_scopes": scopes,
            "outer_model_path": outer_path, "outer_training_ids": model.training_ids,
            "outer_adaptation_ids": model.adaptation_ids}


def public_calibration(name: str, data: dict, output: Path) -> dict:
    """Group-disjoint public-only cross-fitting excludes public validation rows."""
    sources = public_sources(name, data)
    all_ids = [s for _, ids in sources for s in ids]
    scores = np.empty(len(all_ids)); all_x = np.concatenate([x for x, _ in sources])
    scopes = []
    for fold, (training, validation) in enumerate(KFold(3, shuffle=True, random_state=SEED).split(all_ids)):
        training_ids = {all_ids[i] for i in training}; kept = []
        for x, ids in sources:
            rows = [i for i, s in enumerate(ids) if s in training_ids]
            kept.append((x[rows], [ids[i] for i in rows]))
        excluded = [all_ids[i] for i in validation]
        model = balanced_sources(kept, excluded)
        path = f"models/public_{name}_inner_{fold}.joblib"
        joblib.dump(model, output/path)
        scores[validation] = model.score(all_x[validation])
        scopes.append({"model_path": path, "validation_ids": excluded,
                       "training_ids": model.training_ids, "validation_indices": validation.tolist()})
    return {"ids": all_ids, "scores": scores.tolist(), "scopes": scopes}


def source_guard(output: Path) -> dict:
    preparation = json.loads((output/"preparation.json").read_text())
    if sha256(output/"features.npz") != preparation["feature_sha256"]:
        raise ValueError("Prepared features changed")
    for path, digest in preparation["source_hashes"].items():
        if sha256(Path(path)) != digest:
            raise ValueError("Verified source changed")
    for path, expected in preparation["historical_checksums"].items():
        if tree_checksums(Path(path)) != expected:
            raise ValueError("Historical artifact changed")
    if sha256(Path(preparation["preservation_manifest"])) != preparation["preservation_manifest_sha256"]:
        raise ValueError("Preservation checkpoint changed")
    return preparation


def load_features(output: Path) -> dict:
    with np.load(output/"features.npz", allow_pickle=False) as values:
        data = {k: values[k].copy() for k in values.files}
    matrix(data["local"], tuple(data["schema"]))
    if (len(data["subjects"]) != len(set(data["subjects"])) or len(data["subjects"]) != 119
            or np.bincount(data["labels"]).tolist() != [100, 19] or set(data["folds"]) != {1, 2, 3, 4, 5}):
        raise ValueError("Approved local cohort/folds changed")
    return data


def references(root: Path, data: dict) -> dict:
    frames = {}
    for name in (EXP4, EXP5):
        checksum_guard(root/"data/processed"/name)
        frame = pd.read_csv(root/"data/processed"/name/"oof_predictions.csv", dtype={"subject_id": str}, float_precision="round_trip")
        if frame.subject_id.tolist() != data["subjects"].tolist():
            raise ValueError("Historical subject order differs")
        np.testing.assert_array_equal(frame.label, data["labels"])
        np.testing.assert_array_equal(frame.fold, data["folds"])
        frames[name] = frame
    def columns(frame, prefix):
        return frame[prefix+"_probability"].to_numpy(), frame[prefix+"_threshold"].to_numpy()
    result = {"exp04_robust": columns(frames[EXP4], "ablation_A_robust_lr_c1_none"),
              "exp05_cp_C": columns(frames[EXP5], "cp_classification_C_threshold"),
              "exp05_joint_primary": columns(frames[EXP5], "joint_selected_threshold")}
    e6 = root/"data/processed"/EXP6
    checksum_guard(e6)
    keys = pd.read_csv(e6/"identity_key_private.csv", dtype=str)
    frame = pd.read_csv(e6/"oof_predictions_private.csv", float_precision="round_trip")
    if keys.source_subject_id.tolist() != data["subjects"].tolist() or keys.deidentified_id.tolist() != frame.deidentified_id.tolist():
        raise ValueError("Exp6 identity mapping differs")
    np.testing.assert_array_equal(frame.label, data["labels"]); np.testing.assert_array_equal(frame.fold, data["folds"])
    result["exp06_equal_spec080"] = columns(frame, "equal_spec0.8")
    return result


def error_analysis(output: Path, data: dict, scores: dict) -> dict:
    quality = json.loads((output/"audit/local_quality_private.json").read_text())
    if [r["subject_id"] for r in quality] != data["subjects"].tolist():
        raise ValueError("Image-quality identity mapping differs")
    groups = np.asarray([severity(r["hb"], r["sex"]) for r in quality])
    rows = {r["subject_id"]: r for r in json.loads((output/"audit/fresh_inputs.json").read_text())}
    rendering = np.asarray([rows[s].get("render_method_recorded", "not_recorded") for s in data["subjects"]])
    values = {"hb": np.asarray([r["hb"] for r in quality]),
        "brightness": np.asarray([r["brightness_mean"] for r in quality]),
        "tissue_L": np.asarray([r["lab_mean"][0] for r in quality]),
        "clipping": np.asarray([r["any_channel_clipped_fraction"] for r in quality]),
        "bright_neutral_glare_proxy": np.asarray([r["bright_neutral_proxy_fraction"] for r in quality]),
        "tissue_pixels": np.asarray([r["tissue_pixels"] for r in quality]),
        "bbox_aspect": np.asarray([r["bbox_width"]/r["bbox_height"] for r in quality])}
    result = {"scope": "descriptive associations only; no causal inference or patient-specific tuning",
        "unavailable": ["validated exposure or illumination measurements", "standardized skin tone", "individual acquisition settings",
                        "new graded orientation/eyelash/hair-occlusion annotations", "multiple verified local eyelid retakes"],
        "severity_rule": "existing unadjusted adult WHO2024 descriptive bands; does not change labels", "policies": {}}
    for name, (score, threshold) in scores.items():
        prediction = score >= threshold
        associations = {}
        for key, vector in values.items():
            if np.ptp(vector) == 0 or np.ptp(score) == 0:
                associations[key] = {"spearman_rho": None, "p_unadjusted": None}
            else:
                rho, p = spearmanr(vector, score)
                associations[key] = {"spearman_rho": float(rho), "p_unadjusted": float(p)}
        result["policies"][name] = {"severity": {g: {"subjects": int(np.sum(groups == g)),
            "flagged": int(np.sum(prediction & (groups == g))), "not_flagged": int(np.sum(~prediction & (groups == g)))} for g in sorted(set(groups))},
            "rendering_groups": {g: {"subjects": int(np.sum(rendering == g)),
                "median_score": float(np.median(score[rendering == g])),
                "anemia_missed": int(np.sum((rendering == g) & (data["labels"] == 1) & ~prediction)),
                "false_positives": int(np.sum((rendering == g) & (data["labels"] == 0) & prediction))} for g in sorted(set(rendering))},
            "associations": associations,
            "false_negative_subjects_private": data["subjects"][(data["labels"] == 1) & ~prediction].tolist(),
            "false_positive_subjects_private": data["subjects"][(data["labels"] == 0) & prediction].tolist()}
    return result


def figures(output: Path, data: dict, outputs: dict, metrics: dict) -> None:
    names = [c["id"]+"_A_085" for c in CANDIDATES]
    available = [n for n in names if n in outputs]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for name in available:
        m = metrics["policies"][name]
        ax[0].scatter(100*m["specificity"], 100*m["sensitivity"], label=name.removesuffix("_A_085"))
        score = outputs[name][0]
        for y, style in [(0, "-"), (1, "--")]:
            x = np.sort(score[data["labels"] == y]); p = np.arange(1, len(x)+1)/len(x)
            if name == available[0]:
                ax[1].plot(x, p, style, label="Non-anemia" if y == 0 else "Anemia")
    ax[0].scatter(84, 100*10/19, marker="*", s=130, color="black", label="Exp4 robust reference")
    ax[0].set(xlabel="Specificity (%)", ylabel="Sensitivity (%)", title="Predefined local candidates: normal-only 85% target", xlim=(0, 101), ylim=(-2, 102))
    ax[1].set(xlabel="Training-normal standardized anomaly score", ylabel="Empirical cumulative fraction", title="Mahalanobis held-out score distributions")
    for a in ax:
        a.legend(fontsize=7); a.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output/"comparison.png", dpi=170); fig.savefig(output/"comparison.svg")
    plt.close(fig)


def run(output: Path) -> dict:
    os.umask(0o077)
    code = Path(__file__).resolve().parents[1]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=code, text=True).strip():
        raise ValueError("Commit reviewed source before real fitting")
    if (output/"run_manifest.json").exists() or (output/"models").exists():
        raise FileExistsError("Never overwrite or resume a fitted experiment attempt")
    preparation = source_guard(output)
    data = load_features(output); root = Path(preparation["source_root"])
    (output/"models").mkdir(mode=0o700)
    (output/"implementation").mkdir(mode=0o700)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code, text=True).strip()
    files = subprocess.check_output(["git", "ls-files", "src", "tests", "requirements.txt", "docs/experiments/eyelid-exp07-plan.md"], cwd=code, text=True).splitlines()
    for name in files:
        dest = output/"implementation"/name; dest.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copyfile(code/name, dest)
    config = {"source_commit": commit, "protocol_commit": "340d3bef2b8fc14413463f7c9505aa1ab3df63e4",
        "protocol_sha256": sha256(code/"docs/experiments/eyelid-exp07-plan.md"), "candidates": list(CANDIDATES),
        "seed": SEED, "targets": list(TARGETS), "feature_schema": list(ROBUST_NAMES),
        "source_code_sha256": {n: sha256(code/n) for n in files}, "track_gate": preparation["track_gate"],
        "score": "higher anomaly; pooled ranking uses training cross-fitted normal mean/std",
        "policy_A": "normal-only cross-fitted empirical specificity", "policy_B": "training-label-assisted inner sensitivity at specificity constraint",
        "public_adaptation": "retain public correlation; local normal mean and marginal standard deviations",
        "pooling": "equal total source weights; equal group weight within source", "bootstrap_replicates": 2000,
        "clinical_validation": False}
    write_json(output/"experiment_config.json", config)
    versions = {"python": platform.python_version(), "platform": platform.platform(),
        "packages": {n: importlib.metadata.version(n) for n in ("numpy", "scipy", "scikit-learn", "joblib", "Pillow", "opencv-python")}}
    write_json(output/"environment.json", versions)
    started = time.monotonic()
    candidates = [c["id"] for c in CANDIDATES]+[t for t in TRACKS if preparation["track_gate"][t[0]]]
    outputs = {f"{name}_{policy}_{round(100*t):03d}": (np.full(119, np.nan), np.full(119, np.nan))
               for name in candidates for policy in ("A", "B") for t in TARGETS}
    outputs.update({f"nested_B_{round(100*t):03d}": (np.full(119, np.nan), np.full(119, np.nan)) for t in TARGETS})
    raw = {name: np.full(119, np.nan) for name in candidates}
    public_records = {name: public_calibration(name, data, output) for name in candidates if name in ("B1", "C1", "D1")}
    records = {}
    for fold in range(1, 6):
        tr, va = np.flatnonzero(data["folds"] != fold), np.flatnonzero(data["folds"] == fold)
        train_ids = ["local/"+s for s in data["subjects"][tr]]
        heldout_ids = ["local/"+s for s in data["subjects"][va]]
        record = {"training_indices": tr.tolist(), "heldout_indices": va.tolist(), "training_ids": train_ids,
                  "heldout_ids": heldout_ids, "candidates": {}}
        for name in candidates:
            def save(stage, model):
                path = f"models/fold_{fold}_{name}_{stage}.joblib"; joblib.dump(model, output/path); return path
            fitted = crossfit_outer(name, data["local"][tr], data["labels"][tr], train_ids, heldout_ids, data["local"][va], data, save)
            inner, outer = fitted.pop("inner_scores"), fitted.pop("outer_scores")
            reference = np.asarray(public_records[name]["scores"]) if name in public_records else inner[data["labels"][tr] == 0]
            fitted["calibration"] = calibration(reference)
            fitted["inner_scores"] = inner.tolist(); fitted["outer_scores"] = outer.tolist()
            fitted["inner_assignments"] = fitted["inner_assignments"].tolist()
            fitted["inner_AP"] = float(average_precision_score(data["labels"][tr], inner))
            fitted["thresholds"] = {}
            score = normalize(outer, fitted["calibration"]); raw[name][va] = outer
            for target in TARGETS:
                a = normal_threshold(reference, target); b = assisted_threshold(data["labels"][tr], inner, target)
                fitted["thresholds"][str(target)] = {"A": a, "B": b}
                for policy, rule in [("A", a), ("B", b)]:
                    key = f"{name}_{policy}_{round(100*target):03d}"
                    outputs[key][0][va] = score
                    outputs[key][1][va] = normalize(rule["threshold"], fitted["calibration"])
            record["candidates"][name] = fitted
        # Selection sees outer-training cross-fits only; ties retain frozen order.
        selected = max([c["id"] for c in CANDIDATES], key=lambda n: record["candidates"][n]["inner_AP"])
        record["nested_B_selected_candidate"] = selected
        for target in TARGETS:
            source = outputs[f"{selected}_B_{round(100*target):03d}"]
            key = f"nested_B_{round(100*target):03d}"
            outputs[key][0][va] = source[0][va]; outputs[key][1][va] = source[1][va]
        records[str(fold)] = record
        print(f"Completed outer fold {fold}/5: {len(candidates)} predefined candidates/tracks", flush=True)
    write_json(output/"fold_records_private.json", records)
    write_json(output/"public_calibration_private.json", public_records)
    baseline = references(root, data)
    np.savez_compressed(output/"scores_private.npz", **{n+"_score": p for n, (p, _) in outputs.items()},
        **{n+"_threshold": t for n, (_, t) in outputs.items()}, **{n+"_raw_score": p for n, p in raw.items()})
    frame = pd.DataFrame({"subject_id": data["subjects"], "label": data["labels"], "fold": data["folds"]})
    for n, (p, t) in outputs.items():
        frame[n+"_score"] = p; frame[n+"_threshold"] = t; frame[n+"_flag"] = (p >= t).astype(int)
    frame.to_csv(output/"oof_scores_private.csv", index=False)
    metrics = analyze(data["labels"], data["folds"], outputs, baseline)
    metrics["pooled_raw_ranking_descriptive"] = {n: {"roc_auc": float(roc_auc_score(data["labels"], p)),
        "average_precision": float(average_precision_score(data["labels"], p))} for n, p in raw.items()}
    write_json(output/"metrics.json", metrics)
    write_json(output/"error_analysis_private.json", error_analysis(output, data, outputs))
    figures(output, data, outputs, metrics)
    table = [{"policy": n, **{k: v for k, v in m.items() if k not in ("folds", "paired_comparisons", "intervals_95")}} for n, m in metrics["policies"].items()]
    pd.DataFrame(table).to_csv(output/"aggregate_results.csv", index=False)
    source_guard(output)
    write_json(output/"training_events.json", {"outer_folds_completed": 5, "normal_only_detectors": True,
        "nested_choices_by_fold": {f: r["nested_B_selected_candidate"] for f, r in records.items()},
        "public_training_groups": {s: len(data[s+"_ids"]) for s in ("eyes", "cp")},
        "inner_and_outer_model_paths": sorted(str(p.relative_to(output)) for p in (output/"models").glob("*.joblib")),
        "execution_log": str(root/"data/processed/exp07_preservation_2026-10-13_v1"/("exp07_run_"+output.name.rsplit("_", 1)[-1]+".log"))})
    manifest = {"status": "completed_exploratory_internal_subject_cv", "source_commit": commit,
        "subjects": 119, "class_counts_0_1": [100, 19], "candidate_tracks": candidates,
        "policy_outputs": len(outputs), "models": len(list((output/"models").glob("*.joblib"))),
        "runtime_seconds": time.monotonic()-started, "historical_artifacts_unchanged": True,
        "peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "model_bytes": sum(p.stat().st_size for p in (output/"models").glob("*.joblib")),
        "private_output": str(output), "no_full_cohort_local_model": True}
    write_json(output/"run_manifest.json", manifest)
    (output/"REPRODUCTION.md").write_text("# Experiment 7 saved-model reproduction\n\n"
        "From this worktree with the recorded Python3.10 environment and one native CPU thread:\n\n"
        f"```bash\npython -m src.eyelid_one_class_run --verify --output {output}\n```\n\n"
        "Checksums are checked before model loading. The verifier checks normal-only scope, all public/local and inner/outer group exclusions, "
        "scaler/covariance statistics, cross-fitted scores, training-only cutoffs/selections, every held-out score, matched references and metrics. "
        "No model fitting or artifact writes occur in verification. Source hashes and historical outputs are rechecked. "
        "Use --fresh-features to re-extract the 119 local vectors. Retraining requires a new preparation/output version; never overwrite this bundle.\n")
    for path in output.rglob("*"):
        path.chmod(0o700 if path.is_dir() else 0o600)
    write_json(output/"checksum_manifest.json", tree_checksums(output))
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def check_model(model, lookup: dict, labels: dict, forbidden: list[str]) -> None:
    ids = model.training_ids
    if len(set(ids)) != len(ids) or set(ids) & set(forbidden) or any(labels[s] != 0 for s in ids):
        raise ValueError("Saved detector normal-only/group scope violated")
    x = np.asarray([lookup[s] for s in ids]); weights = model.weights
    source_mean = np.average(x, weights=weights, axis=0)
    source_var = np.average((x-source_mean)**2, weights=weights, axis=0)
    if model.adaptation_ids:
        if set(model.adaptation_ids) & (set(ids) | set(forbidden)) or any(labels[s] != 0 for s in model.adaptation_ids):
            raise ValueError("Adaptation includes a forbidden/positive group")
        local = np.asarray([lookup[s] for s in model.adaptation_ids])
        expected_mean, expected_var = local.mean(0), local.var(0)
    else:
        expected_mean, expected_var = source_mean, source_var
    np.testing.assert_allclose(model.scaler.mean_, expected_mean, rtol=0, atol=1e-11)
    np.testing.assert_allclose(model.scaler.var_, expected_var, rtol=0, atol=1e-10)
    if model.candidate == "mahalanobis":
        scale = np.sqrt(source_var); scale[scale < 1e-14] = 1
        z = (x-source_mean)/scale
        cov = z.T @ (z*weights[:, None]); regularized = .8*cov+(.2*np.trace(cov)/18+1e-10)*np.eye(18)
        np.testing.assert_allclose(model.estimator, np.linalg.inv(regularized), rtol=1e-10, atol=1e-10)
    elif BY_ID[model.candidate]["family"] == "svm":
        np.testing.assert_allclose(model.estimator.support_vectors_, model.scaler.transform(x)[model.estimator.support_], rtol=0, atol=1e-14)
        expected = BY_ID[model.candidate]
        if model.estimator.nu != expected["nu"] or model.estimator.gamma != expected["gamma"]:
            raise ValueError("SVM configuration changed")
    elif (model.estimator.random_state != SEED or model.estimator.n_estimators != 256
          or model.estimator.max_samples != min(64, len(ids)) or model.estimator.max_features != 1.0
          or model.estimator.n_jobs != 1 or model.estimator.contamination != "auto"):
        raise ValueError("Forest configuration changed")


def check_expected_scope(name: str, model, local_ids: list[str], data: dict) -> None:
    if name in BY_ID:
        expected_ids, adapted = local_ids, None
        expected_weights = np.full(len(local_ids), 1/len(local_ids))
    else:
        sources = public_sources(name, data)
        if name.endswith("3"):
            sources = [*sources, (np.empty((len(local_ids), 18)), local_ids)]
        expected_ids = [s for _, ids in sources for s in ids]
        expected_weights = np.concatenate([np.full(len(ids), 1/(len(sources)*len(ids))) for _, ids in sources])
        adapted = local_ids if name.endswith("2") else None
    if model.training_ids != expected_ids or model.adaptation_ids != adapted:
        raise ValueError("Detector source/local membership differs from frozen policy")
    np.testing.assert_allclose(model.weights, expected_weights, rtol=0, atol=1e-15)


def verify(output: Path, fresh_features: bool = False) -> dict:
    checksum_guard(output); preparation = source_guard(output); data = load_features(output)
    root = Path(preparation["source_root"])
    config = json.loads((output/"experiment_config.json").read_text())
    code = Path(__file__).resolve().parents[1]
    for name, digest in config["source_code_sha256"].items():
        if sha256(output/"implementation"/name) != digest:
            raise ValueError("Copied implementation changed")
        recorded = subprocess.check_output(["git", "show", config["source_commit"]+":"+name], cwd=code)
        import hashlib
        if hashlib.sha256(recorded).hexdigest() != digest:
            raise ValueError("Implementation is not the recorded Git commit")
        if name.startswith("src/") and sha256(code/name) != digest:
            raise ValueError("Current source differs; use recorded implementation")
    lookup = {"local/"+s: x for s, x in zip(data["subjects"], data["local"])}
    labels = {"local/"+s: int(y) for s, y in zip(data["subjects"], data["labels"])}
    for source in ("eyes", "cp"):
        lookup.update(dict(zip(data[source+"_ids"], data[source])))
        labels.update({s: 0 for s in data[source+"_ids"]})
    count = 0
    records = json.loads((output/"fold_records_private.json").read_text())
    public = json.loads((output/"public_calibration_private.json").read_text())
    for name, record in public.items():
        scores = np.empty(len(record["ids"]))
        for scope in record["scopes"]:
            model = joblib.load(output/scope["model_path"]); count += 1
            check_model(model, lookup, labels, scope["validation_ids"])
            scores[scope["validation_indices"]] = model.score(np.asarray([lookup[s] for s in scope["validation_ids"]]))
        np.testing.assert_allclose(scores, record["scores"], rtol=0, atol=1e-12)
    with np.load(output/"scores_private.npz", allow_pickle=False) as cache:
        outputs = {n.removesuffix("_threshold"): (cache[n.removesuffix("_threshold")+"_score"].copy(), cache[n].copy()) for n in cache.files if n.endswith("_threshold")}
        for fold, record in records.items():
            tr, va = np.asarray(record["training_indices"]), np.asarray(record["heldout_indices"])
            np.testing.assert_array_equal(tr, np.flatnonzero(data["folds"] != int(fold)))
            np.testing.assert_array_equal(va, np.flatnonzero(data["folds"] == int(fold)))
            train_ids = ["local/"+s for s in data["subjects"][tr]]; heldout_ids = ["local/"+s for s in data["subjects"][va]]
            if record["training_ids"] != train_ids or record["heldout_ids"] != heldout_ids:
                raise ValueError("Recorded local membership differs")
            assignments = inner_assignments(data["labels"][tr])
            for name, c in record["candidates"].items():
                inner = np.empty(len(tr))
                np.testing.assert_array_equal(assignments, c["inner_assignments"])
                for scope in c["inner_scopes"]:
                    fit = np.flatnonzero((assignments != scope["inner_fold"]) & (data["labels"][tr] == 0))
                    vv = np.flatnonzero(assignments == scope["inner_fold"])
                    if scope["local_normal_indices"] != fit.tolist() or scope["validation_indices"] != vv.tolist():
                        raise ValueError("Inner normal-only scope differs")
                    forbidden = heldout_ids+[train_ids[i] for i in vv]
                    model = joblib.load(output/scope["model_path"]); count += 1
                    check_expected_scope(name, model, [train_ids[i] for i in fit], data)
                    check_model(model, lookup, labels, forbidden)
                    inner[vv] = model.score(data["local"][tr[vv]])
                np.testing.assert_allclose(inner, c["inner_scores"], rtol=0, atol=1e-12)
                model = joblib.load(output/c["outer_model_path"]); count += 1
                check_expected_scope(name, model, [train_ids[i] for i in np.flatnonzero(data["labels"][tr] == 0)], data)
                check_model(model, lookup, labels, heldout_ids)
                outer = model.score(data["local"][va])
                np.testing.assert_allclose(outer, c["outer_scores"], rtol=0, atol=1e-12)
                np.testing.assert_allclose(outer, cache[name+"_raw_score"][va], rtol=0, atol=1e-12)
                reference = np.asarray(public[name]["scores"]) if name in public else inner[data["labels"][tr] == 0]
                if c["calibration"] != calibration(reference) or c["inner_AP"] != float(average_precision_score(data["labels"][tr], inner)):
                    raise ValueError("Calibration or inner AP differs")
                for target in TARGETS:
                    rules = {"A": normal_threshold(reference, target), "B": assisted_threshold(data["labels"][tr], inner, target)}
                    if rules != c["thresholds"][str(target)]:
                        raise ValueError("Training-only threshold selection differs")
                    for policy in ("A", "B"):
                        key = f"{name}_{policy}_{round(100*target):03d}"
                        np.testing.assert_allclose(outputs[key][0][va], normalize(outer, c["calibration"]), rtol=0, atol=1e-12)
                        np.testing.assert_allclose(outputs[key][1][va], normalize(rules[policy]["threshold"], c["calibration"]), rtol=0, atol=1e-12)
            selected = max([c["id"] for c in CANDIDATES], key=lambda n: record["candidates"][n]["inner_AP"])
            if selected != record["nested_B_selected_candidate"]:
                raise ValueError("Nested selection differs")
            for target in TARGETS:
                for array in (0, 1):
                    np.testing.assert_array_equal(outputs[f"nested_B_{round(100*target):03d}"][array][va], outputs[f"{selected}_B_{round(100*target):03d}"][array][va])
    frame = pd.read_csv(output/"oof_scores_private.csv", dtype={"subject_id": str}, float_precision="round_trip")
    np.testing.assert_array_equal(frame.subject_id, data["subjects"])
    np.testing.assert_array_equal(frame.label, data["labels"]); np.testing.assert_array_equal(frame.fold, data["folds"])
    for name, (score, threshold) in outputs.items():
        np.testing.assert_array_equal(frame[name+"_score"], score); np.testing.assert_array_equal(frame[name+"_threshold"], threshold)
        np.testing.assert_array_equal(frame[name+"_flag"], score >= threshold)
    saved = json.loads((output/"metrics.json").read_text())
    reproduced = analyze(data["labels"], data["folds"], outputs, references(root, data))
    if reproduced != {k: v for k, v in saved.items() if k != "pooled_raw_ranking_descriptive"}:
        raise ValueError("Metrics or paired intervals differ")
    with np.load(output/"scores_private.npz", allow_pickle=False) as cache:
        raw_metrics = {n.removesuffix("_raw_score"): {"roc_auc": float(roc_auc_score(data["labels"], cache[n])),
            "average_precision": float(average_precision_score(data["labels"], cache[n]))} for n in cache.files if n.endswith("_raw_score")}
    if raw_metrics != saved["pooled_raw_ranking_descriptive"]:
        raise ValueError("Descriptive raw-score ranking metrics differ")
    if error_analysis(output, data, outputs) != json.loads((output/"error_analysis_private.json").read_text()):
        raise ValueError("Error analysis differs")
    if fresh_features:
        rows = {r["subject_id"]: r for r in json.loads((output/"audit/fresh_inputs.json").read_text())}
        fresh = []
        for sid in data["subjects"]:
            row = rows[sid]; rgb, mask = load_manual_pair(root/row["annotation_image_path"], root/row["mask_path"])
            fresh.append(color_features(rgb, mask)["robust"])
        np.testing.assert_array_equal(fresh, data["local"])
    checksum_guard(output)
    return {"status": "reproduced_without_refitting", "subjects": 119, "models": count,
        "policy_outputs": len(outputs), "normal_only_scope_and_all_group_exclusions_verified": True,
        "cross_fits_preprocessing_calibration_thresholds_and_nested_selection_verified": True,
        "all_scores_metrics_paired_intervals_and_error_analysis_verified": True,
        "fresh_local_features_exact": fresh_features, "sources_and_previous_artifacts_unchanged": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--fresh-features", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = verify(args.output.resolve(), args.fresh_features) if args.verify else run(args.output.resolve())
    print(json.dumps(result, indent=2))
