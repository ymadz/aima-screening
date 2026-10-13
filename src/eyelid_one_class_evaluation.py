"""Anomaly-score metrics and efficient paired conditional subject uncertainty."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .eyelid_one_class import SEED
from .local_eyelid_hybrid_evaluation import LIMITATION, wilson


def evaluate(y: np.ndarray, scores: np.ndarray, thresholds: np.ndarray | float) -> dict:
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    threshold = np.broadcast_to(thresholds, y.shape)
    if y.shape != scores.shape or set(y) != {0, 1} or not np.isfinite(scores).all() or not np.isfinite(threshold).all():
        raise ValueError("Verified binary labels, finite scores and cutoffs required")
    prediction = scores >= threshold
    tp = int(prediction[y == 1].sum()); fp = int(prediction[y == 0].sum())
    fn = int((y == 1).sum())-tp; tn = int((y == 0).sum())-fp
    return {"sensitivity": tp/(tp+fn), "specificity": tn/(tn+fp),
        "accuracy": (tp+tn)/len(y), "balanced_accuracy": .5*(tp/(tp+fn)+tn/(tn+fp)),
        "precision": tp/(tp+fp) if tp+fp else None, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "confusion_matrix_labels_0_1": [[tn, fp], [fn, tp]],
        "roc_auc": float(roc_auc_score(y, scores)), "average_precision": float(average_precision_score(y, scores)),
        "score_distribution_by_label": {str(label): {"mean": float(scores[y == label].mean()),
            "std": float(scores[y == label].std()), "quantiles_0_10_25_50_75_90_100": np.quantile(scores[y == label], [0, .1, .25, .5, .75, .9, 1]).tolist()} for label in (0, 1)}}


def draws(y: np.ndarray, replicates: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Multiplicity weights for common stratified subject draws, no model refits."""
    rng = np.random.default_rng(SEED)
    count = []
    for label in (0, 1):
        n = int(np.sum(y == label))
        if not n:
            raise ValueError("Both classes required for stratified resampling")
        sampled = rng.integers(0, n, size=(replicates, n))
        count.append(np.asarray([np.bincount(row, minlength=n) for row in sampled], dtype=float))
    return count[0], count[1]


def ranking_samples(y: np.ndarray, score: np.ndarray, counts: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Exact weighted AUC/AP, including ties and repeated resampled subjects."""
    negative, positive = counts
    pair = ((score[y == 1, None] > score[y == 0]).astype(float)
            + .5*(score[y == 1, None] == score[y == 0]))
    auc = ((positive @ pair)*negative).sum(1)/(positive.sum(1)*negative.sum(1))
    weights = np.empty((len(negative), len(y)))
    weights[:, y == 0] = negative; weights[:, y == 1] = positive
    order = np.argsort(-score, kind="stable")
    sorted_score = score[order]
    ends = np.flatnonzero(np.r_[sorted_score[:-1] != sorted_score[1:], True])
    cumulative_positive = np.cumsum(weights[:, order]*(y[order] == 1), axis=1)[:, ends]
    cumulative_total = np.cumsum(weights[:, order], axis=1)[:, ends]
    increments = np.diff(np.c_[np.zeros(len(weights)), cumulative_positive], axis=1)
    precision = np.divide(cumulative_positive, cumulative_total, out=np.zeros_like(cumulative_total), where=cumulative_total > 0)
    ap = (increments*precision).sum(1)/positive.sum(1)
    return auc, ap


def metric_samples(y: np.ndarray, score: np.ndarray, threshold: np.ndarray,
                   counts: tuple[np.ndarray, np.ndarray], ranking: tuple[np.ndarray, np.ndarray]) -> dict:
    negative, positive = counts; prediction = score >= threshold
    sensitivity = (positive @ prediction[y == 1])/positive.sum(1)
    specificity = (negative @ (~prediction[y == 0]))/negative.sum(1)
    return {"sensitivity": sensitivity, "specificity": specificity,
            "roc_auc": ranking[0], "average_precision": ranking[1]}


def analyze(y: np.ndarray, folds: np.ndarray, outputs: dict, references: dict,
            replicates: int = 2000) -> dict:
    """Report all registered policies; no result-driven selection occurs here."""
    counts = draws(y, replicates)
    all_outputs = {**outputs, **references}
    ranking_cache, samples, result = {}, {}, {}
    interval = lambda a: np.quantile(a, [.025, .975]).tolist()
    for name, (score, threshold) in all_outputs.items():
        key = np.asarray(score).tobytes()
        if key not in ranking_cache:
            ranking_cache[key] = ranking_samples(y, score, counts)
        samples[name] = metric_samples(y, score, threshold, counts, ranking_cache[key])
        point = evaluate(y, score, threshold)
        point["intervals_95"] = {"sensitivity": wilson(point["TP"], point["TP"]+point["FN"]),
            "specificity": wilson(point["TN"], point["TN"]+point["FP"]),
            "roc_auc": interval(samples[name]["roc_auc"]), "average_precision": interval(samples[name]["average_precision"])}
        point["folds"] = {str(f): evaluate(y[folds == f], score[folds == f], threshold[folds == f]) for f in sorted(set(folds))}
        result[name] = point
    for name in outputs:
        score, threshold = outputs[name]; pred = score >= threshold
        result[name]["paired_comparisons"] = {}
        for ref, (base_score, base_threshold) in references.items():
            base = base_score >= base_threshold
            result[name]["paired_comparisons"][ref] = {
                "deltas": {m: {"estimate": result[name][m]-result[ref][m],
                    "conditional_interval_95": interval(samples[name][m]-samples[ref][m])} for m in samples[name]},
                "net_additional_anemia_detections": result[name]["TP"]-result[ref]["TP"],
                "net_additional_false_positives": result[name]["FP"]-result[ref]["FP"],
                "both_detected": int(np.sum(pred & base & (y == 1))),
                "new_only_detected": int(np.sum(pred & ~base & (y == 1))),
                "reference_only_detected": int(np.sum(~pred & base & (y == 1))),
                "both_missed": int(np.sum(~pred & ~base & (y == 1)))}
    return {"policies": result, "bootstrap_replicates": replicates, "seed": SEED,
            "confidence": .95, "uncertainty_limit": LIMITATION,
            "comparison_scope": "same subjects/labels/folds; different threshold/model-selection policies are explicitly separate"}
