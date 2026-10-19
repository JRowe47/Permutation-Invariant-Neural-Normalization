"""NumPy-only surrogate algebra and metrics; no native model execution."""
import numpy as np


def fisher_vectors(probabilities, signs):
    """Rows have covariance diag(p)-pp^T for independent Rademacher signs.

    v = sqrt(p)*z - p*sum(sqrt(p)*z). The reference p is held constant
    when differentiating v^T logits. Constant-logit directions are removed.
    """
    p, z = np.asarray(probabilities, dtype=np.float64), np.asarray(signs, dtype=np.float64)
    if p.ndim != 1 or z.ndim != 2 or z.shape[1] != p.size or not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1):
        raise ValueError("Expected probabilities and matching sign rows")
    if not np.all((z == -1) | (z == 1)):
        raise ValueError("Expected Rademacher signs")
    weighted = z * np.sqrt(p)
    return weighted - weighted.sum(axis=1, keepdims=True) * p


def quadratic_sketch(rows, directions):
    """Half the mean squared directional projection, with explicit finite inputs."""
    a, d = np.asarray(rows, dtype=np.float64), np.asarray(directions, dtype=np.float64)
    if a.ndim != 2 or not a.shape[0] or d.ndim != 2 or a.shape[1] != d.shape[1] or not np.isfinite(a).all() or not np.isfinite(d).all():
        raise ValueError("Matching nonempty finite row and direction matrices required")
    return .5 * np.mean((d @ a.T) ** 2, axis=1)


def rankdata(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Expected a finite vector")
    order = np.argsort(x, kind="stable"); ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2
        i = j
    return ranks


def spearman(a, b):
    x, y = rankdata(a), rankdata(b)
    if x.shape != y.shape or x.size < 2:
        raise ValueError("Matching ranking vectors required")
    x -= x.mean(); y -= y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denom) if denom else None


def error_metrics(actual, predicted):
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    positive = (a > 1e-10) & (p > 0)
    total = np.abs(a).sum()
    return {"count": len(a), "mean_absolute_error": float(np.mean(np.abs(a-p))),
            "relative_l1_error": float(np.abs(a-p).sum()/total) if total else None,
            "positive_kl_count": int(positive.sum()),
            "median_multiplicative_error": float(np.median(np.maximum(a[positive]/p[positive], p[positive]/a[positive]))) if positive.any() else None}


def paired_bootstrap(values, seed=20261020, repetitions=2000):
    """Resample passage-level differences, never individual layer/group trials."""
    v = np.asarray(values, dtype=float)
    draws = np.random.default_rng(seed).choice(v, size=(repetitions, len(v)), replace=True).mean(axis=1)
    return {"mean": float(v.mean()), "passages": len(v), "percentile_95": np.quantile(draws, [.025,.975]).tolist(),
            "bootstrap_unit": "passage", "seed": seed, "repetitions": repetitions}
