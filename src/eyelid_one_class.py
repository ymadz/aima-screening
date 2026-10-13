"""Prespecified normal-only detectors and training-scope decision policies."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from .eyelid_color_data import ROBUST_NAMES

SEED = 20261013
TARGETS = (0.80, 0.85, 0.90)
CANDIDATES = (
    {"id": "mahalanobis", "family": "mahalanobis", "shrinkage": 0.2},
    {"id": "svm_005_scale", "family": "svm", "nu": 0.05, "gamma": "scale"},
    {"id": "svm_005_01", "family": "svm", "nu": 0.05, "gamma": 0.1},
    {"id": "svm_010_scale", "family": "svm", "nu": 0.10, "gamma": "scale"},
    {"id": "svm_010_01", "family": "svm", "nu": 0.10, "gamma": 0.1},
    {"id": "isolation_forest", "family": "forest", "trees": 256, "max_samples": 64},
)
BY_ID = {c["id"]: c for c in CANDIDATES}
TRACKS = ("B1", "B2", "B3", "C1", "C2", "C3", "D1", "D2")


def matrix(x: np.ndarray, names: tuple[str, ...] = ROBUST_NAMES) -> np.ndarray:
    """Reject missing values or any change to the fixed ordered feature schema."""
    x = np.asarray(x, dtype=np.float64)
    if tuple(names) != ROBUST_NAMES or x.ndim != 2 or x.shape[1] != 18 or not np.isfinite(x).all():
        raise ValueError("Finite exact robust18 schema required; no imputation")
    return x


def normal_scope(x: np.ndarray, y: np.ndarray, ids: list[str], forbidden: list[str]) -> np.ndarray:
    x = matrix(x)
    y = np.asarray(y)
    if len(x) < 3 or y.shape != (len(x),) or np.any(y != 0):
        raise ValueError("Fitting accepts verified non-anemic examples only")
    if len(ids) != len(x) or len(set(ids)) != len(ids) or any(not s for s in ids):
        raise ValueError("One row per unique namespaced normal group required")
    if set(ids) & set(forbidden):
        raise ValueError("Held-out group entered fitting or adaptation")
    return x


@dataclass
class NormalDetector:
    candidate: str
    scaler: StandardScaler
    estimator: Any
    training_ids: list[str]
    weights: np.ndarray
    schema: tuple[str, ...] = ROBUST_NAMES
    adaptation_ids: list[str] | None = None

    def score(self, x: np.ndarray) -> np.ndarray:
        z = self.scaler.transform(matrix(x, self.schema))
        family = BY_ID[self.candidate]["family"]
        if family == "mahalanobis":
            score = np.einsum("ij,jk,ik->i", z, self.estimator, z)
        elif family == "svm":
            score = -self.estimator.decision_function(z)
        else:
            score = -self.estimator.score_samples(z)
        if not np.isfinite(score).all():
            raise ValueError("Nonfinite anomaly score")
        return score


def fit_normal(candidate: str, x: np.ndarray, y: np.ndarray, ids: list[str],
               forbidden: list[str], weights: np.ndarray | None = None) -> NormalDetector:
    """Every learned transform shares the same checked normal training scope."""
    x = normal_scope(x, y, ids, forbidden)
    setting = BY_ID[candidate]
    w = np.ones(len(x), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (len(x),) or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("Strictly positive normal-group weights required")
    w = w / w.sum()
    if weights is not None and setting["family"] != "mahalanobis":
        raise ValueError("Weighted pooling was prespecified only for Mahalanobis")
    scaler = StandardScaler().fit(x, sample_weight=w)
    z = scaler.transform(x)
    if setting["family"] == "mahalanobis":
        covariance = z.T @ (z * w[:, None])
        target = np.trace(covariance) / x.shape[1]
        alpha = setting["shrinkage"]
        regularized = (1-alpha)*covariance + (alpha*target + 1e-10)*np.eye(x.shape[1])
        estimator = np.linalg.inv(regularized)
    elif setting["family"] == "svm":
        estimator = OneClassSVM(kernel="rbf", nu=setting["nu"], gamma=setting["gamma"]).fit(z)
    else:
        estimator = IsolationForest(n_estimators=setting["trees"], max_samples=min(setting["max_samples"], len(z)),
                                    max_features=1.0, contamination="auto", random_state=SEED, n_jobs=1).fit(z)
    return NormalDetector(candidate, scaler, estimator, list(ids), w)


def adapt_normal(public: NormalDetector, x: np.ndarray, y: np.ndarray,
                 ids: list[str], forbidden: list[str]) -> NormalDetector:
    """Retain public regularized correlation; learn local normal margins only."""
    if public.candidate != "mahalanobis" or set(public.training_ids) & (set(ids) | set(forbidden)):
        raise ValueError("Public/local groups must be disjoint")
    x = normal_scope(x, y, ids, forbidden)
    model = deepcopy(public)
    model.scaler = StandardScaler().fit(x)
    model.adaptation_ids = list(ids)
    return model


def balanced_sources(sources: list[tuple[np.ndarray, list[str]]],
                     forbidden: list[str]) -> NormalDetector:
    """Give each source equal total mass and each conservative group equal mass."""
    if not sources or any(len(x) != len(ids) or not len(ids) for x, ids in sources):
        raise ValueError("Nonempty verified normal sources required")
    x = np.concatenate([matrix(a) for a, _ in sources])
    ids = [s for _, source_ids in sources for s in source_ids]
    weights = np.concatenate([np.full(len(a), 1/(len(sources)*len(a))) for a, _ in sources])
    return fit_normal("mahalanobis", x, np.zeros(len(x), dtype=int), ids, forbidden, weights)


def inner_assignments(y: np.ndarray) -> np.ndarray:
    """Normal splits depend only on normal-row order, never positive features."""
    y = np.asarray(y)
    if y.ndim != 1 or not np.isin(y, [0, 1]).all():
        raise ValueError("Verified binary training labels required")
    normals = np.flatnonzero(y == 0)
    if len(normals) < 6:
        raise ValueError("Insufficient normal support for three cross-fits")
    assignments = np.full(len(y), -1, dtype=int)
    for fold, (_, va) in enumerate(KFold(3, shuffle=True, random_state=SEED).split(normals)):
        assignments[normals[va]] = fold
    positives = np.flatnonzero(y == 1)
    assignments[np.random.default_rng(SEED).permutation(positives)] = np.arange(len(positives)) % 3
    return assignments


def calibration(normal_scores: np.ndarray) -> dict[str, float]:
    scores = np.asarray(normal_scores, dtype=float)
    if scores.ndim != 1 or len(scores) < 3 or not np.isfinite(scores).all():
        raise ValueError("Cross-fitted normal calibration scores required")
    return {"mean": float(scores.mean()), "scale": max(float(scores.std()), 1e-12)}


def normalize(scores: np.ndarray, fitted: dict[str, float]) -> np.ndarray:
    return (np.asarray(scores)-fitted["mean"])/fitted["scale"]


def normal_threshold(normal_scores: np.ndarray, target: float) -> dict[str, float]:
    """Policy A accepts no positive scores, labels, or held-out data."""
    scores = np.asarray(normal_scores, dtype=float)
    calibration(scores)
    if target not in TARGETS:
        raise ValueError("Unregistered specificity target")
    choices = np.unique(np.r_[scores, np.nextafter(scores.max(), np.inf)])
    threshold = next(float(t) for t in choices if float(np.mean(scores < t)) >= target)
    return {"threshold": threshold, "inner_normal_specificity": float(np.mean(scores < threshold)), "target": target}


def assisted_threshold(y: np.ndarray, scores: np.ndarray, target: float) -> dict[str, float]:
    """Policy B sees outer-training cross-fits only; never detector-fit positives."""
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if y.shape != scores.shape or set(y) != {0, 1} or not np.isfinite(scores).all() or target not in TARGETS:
        raise ValueError("Both training classes and finite cross-fits required")
    choices = []
    for t in np.unique(np.r_[scores, np.nextafter(scores.max(), np.inf)]):
        sensitivity = float(np.mean(scores[y == 1] >= t))
        specificity = float(np.mean(scores[y == 0] < t))
        if specificity >= target:
            choices.append((sensitivity, specificity, float(t)))
    sensitivity, specificity, threshold = max(choices)
    return {"threshold": threshold, "inner_sensitivity": sensitivity, "inner_specificity": specificity, "target": target}
