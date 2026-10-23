#!/usr/bin/env python3
"""
evaluate.py - shared evaluation script for DST Group 1, Sprint 1 (Hotel Booking Demand).

Every member exports predictions for the SAME frozen test set and this script scores
all models identically, with uncertainty and paired significance tests.

Inputs
------
  truth CSV       : columns  id, y            (+ optional group/cluster columns, e.g. hotel)
  prediction CSVs : columns  id, prediction   (binary task: P(is_canceled = 1) in [0, 1];
                                               regression task: the point prediction)

Example
-------
  python evaluate.py --truth test_truth.csv \
      --pred RF_Luke=LukeD/test_preds.csv \
      --pred XGB_Thomas=ThomasF/test_preds.csv \
      --pred Logit_Morgan=MorganL/test_preds.csv \
      --cost-fp 2 --cost-fn 1 --group hotel --n-boot 2000 --outdir results

  or, inside a notebook:
      from evaluate import evaluate_binary
      res = evaluate_binary(y_test, {"RF": p_rf, "XGB": p_xgb}, threshold=0.5)
      res.summary()            # metric table with 95% bootstrap CIs
      res.pairwise             # paired differences + tests (DeLong, McNemar, Diebold-Mariano)
      res.plot("figures/")     # reliability (CORP), ROC, PR and Murphy diagrams

What it computes (binary task) - see evaluation_metrics.md for the maths and references
  proper scores : Brier score, log loss, Brier skill score, CORP decomposition (MCB/DSC/UNC)
  discrimination: ROC AUC (with DeLong SE), average precision, max Youden J (= KS statistic)
  calibration   : Cox calibration intercept & slope, ECE (10 equal-width bins), Spiegelhalter z
  decisions     : accuracy, balanced accuracy, TPR, TNR, PPV, NPV, F1, MCC, Cohen's kappa,
                  expected cost at the decision threshold and cost relative to the best trivial rule
  dominance     : Murphy diagram (mean elementary score for every cost ratio theta)
  uncertainty   : (cluster) bootstrap percentile CIs; paired bootstrap for all differences;
                  DeLong test (AUC), McNemar mid-p test (accuracy at theta), Diebold-Mariano
                  test (Brier / log loss), Holm correction across model pairs.

Provenance: drafted with the help of Claude (Anthropic) for Luke D's metrics research;
implementations cross-checked against scikit-learn in test_evaluate.py.
Dependencies: numpy, pandas, scipy, matplotlib (for plots only).
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

EPS = np.finfo(float).eps          # log-loss clipping, identical to scikit-learn
LOGIT_CLIP = 1e-6                  # clipping used only when taking logits for calibration fits
DEFAULT_THETAS = np.round(np.linspace(0.01, 0.99, 99), 4)

# ----------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------


def _as_binary(y):
    y = np.asarray(y)
    if not np.all(np.isin(y, (0, 1))):
        raise ValueError("binary task: y must contain only 0/1")
    return y.astype(float)


def _as_prob(p, name="prediction"):
    p = np.asarray(p, dtype=float)
    if np.any(~np.isfinite(p)):
        raise ValueError(f"{name}: predictions contain NaN/inf")
    if np.any((p < 0) | (p > 1)):
        raise ValueError(f"{name}: binary predictions must be probabilities in [0, 1] "
                         "(clip a linear-probability model first - see the regression section of the report)")
    return p


def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1.0), np.nan)


def _logit(p):
    p = np.clip(p, LOGIT_CLIP, 1 - LOGIT_CLIP)
    return np.log(p) - np.log1p(-p)


def holm(pvals):
    """Holm (1979) step-down adjustment; returns adjusted p-values in the original order."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return adj


# ----------------------------------------------------------------------------------------
# Per-observation losses (the "decomposable" metrics)
# ----------------------------------------------------------------------------------------


def brier_terms(y, p):
    return (p - y) ** 2


def logloss_terms(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def elementary_terms(y, p, theta):
    """Cost-weighted misclassification loss at threshold theta (Schervish 1989; Ehm et al. 2016):
    a false negative costs (1 - theta), a false positive costs theta, and we predict 1 iff p > theta."""
    return (1 - theta) * ((y == 1) & (p <= theta)) + theta * ((y == 0) & (p > theta))


# ----------------------------------------------------------------------------------------
# Sorted-score structure: lets every rank/threshold metric be computed in O(n) for any
# vector of (bootstrap) weights, without re-sorting.
# ----------------------------------------------------------------------------------------


class SortedScores:
    def __init__(self, p):
        self.u, self.g = np.unique(p, return_inverse=True)   # ascending unique scores, group id
        self.G = len(self.u)

    def class_weights(self, y, w):
        W1 = np.bincount(self.g, weights=w * y, minlength=self.G)
        W0 = np.bincount(self.g, weights=w * (1 - y), minlength=self.G)
        return W1, W0

    def auc(self, W1, W0):
        """Mann-Whitney form of the ROC AUC with ties counted 1/2 (Bamber 1975; Hanley & McNeil 1982)."""
        n1, n0 = W1.sum(), W0.sum()
        below0 = np.cumsum(W0) - W0                       # negative weight strictly below each score
        return float(np.sum(W1 * (below0 + 0.5 * W0)) / (n1 * n0))

    def average_precision(self, W1, W0):
        """AP = sum_k (R_k - R_{k-1}) P_k over descending distinct thresholds (scikit-learn definition)."""
        w1 = W1[::-1]
        tp = np.cumsum(w1)
        fp = np.cumsum(W0[::-1])
        m = w1 > 0                                      # zero-weight groups are not thresholds
        return float(np.sum(w1[m] / W1.sum() * tp[m] / (tp[m] + fp[m])))

    def counts_at(self, W1, W0, thetas):
        """FN(theta) and FP(theta) for the rule 'predict 1 iff p > theta'."""
        thetas = np.atleast_1d(thetas)
        idx = np.searchsorted(self.u, thetas, side="right") - 1
        c1 = np.concatenate([[0.0], np.cumsum(W1)])
        c0 = np.concatenate([[0.0], np.cumsum(W0)])
        fn = c1[idx + 1]                     # positives with p <= theta
        fp = W0.sum() - c0[idx + 1]          # negatives with p >  theta
        return fn, fp

    def murphy(self, W1, W0, thetas):
        fn, fp = self.counts_at(W1, W0, thetas)
        return ((1 - thetas) * fn + thetas * fp) / (W1.sum() + W0.sum())

    def max_youden(self, W1, W0):
        """max_t [TPR(t) - FPR(t)] = Kolmogorov-Smirnov distance between the class-conditional scores."""
        tpr = 1 - np.cumsum(W1) / W1.sum()
        fpr = 1 - np.cumsum(W0) / W0.sum()
        return float(max(0.0, np.max(np.concatenate([[0.0], tpr - fpr]))))

    def roc_curve(self, W1, W0):
        tpr = np.concatenate([[0.0], np.cumsum(W1[::-1]) / W1.sum()])
        fpr = np.concatenate([[0.0], np.cumsum(W0[::-1]) / W0.sum()])
        return fpr, tpr

    def pr_curve(self, W1, W0):
        tp = np.cumsum(W1[::-1])
        fp = np.cumsum(W0[::-1])
        return tp / W1.sum(), tp / (tp + fp)


# ----------------------------------------------------------------------------------------
# Threshold (confusion-matrix) metrics
# ----------------------------------------------------------------------------------------


def threshold_metrics(tp, fp, fn, tn, cost_fp=1.0, cost_fn=1.0):
    tp, fp, fn, tn = (np.asarray(v, dtype=float) for v in (tp, fp, fn, tn))
    n = tp + fp + fn + tn
    pos, neg = tp + fn, fp + tn
    tpr, tnr = _safe_div(tp, pos), _safe_div(tn, neg)
    ppv, npv = _safe_div(tp, tp + fp), _safe_div(tn, tn + fn)
    acc = (tp + tn) / n
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.where(mcc_den > 0, (tp * tn - fp * fn) / np.where(mcc_den > 0, mcc_den, 1.0), 0.0)
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    kappa = _safe_div(acc - pe, 1 - pe)
    cost = (cost_fp * fp + cost_fn * fn) / n
    trivial = np.minimum(cost_fn * pos, cost_fp * neg) / n     # best of 'flag none' / 'flag all'
    return {
        "accuracy": acc,
        "balanced_accuracy": (tpr + tnr) / 2,
        "tpr_recall": tpr,
        "tnr_specificity": tnr,
        "ppv_precision": ppv,
        "npv": npv,
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        "mcc": mcc,
        "cohen_kappa": kappa,
        "youden_j": tpr + tnr - 1,
        "expected_cost": cost,
        "cost_vs_trivial": _safe_div(cost, trivial),
    }


# ----------------------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------------------


def calibration_intercept_slope(y, p, w=None):
    """Cox (1958) recalibration: logit P(Y=1) = a + b * logit(p).
    Returns (citl, slope): citl is the intercept with the slope fixed at 1 (calibration-in-the-large,
    target 0) and slope is b from the two-parameter fit (target 1)."""
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y) if w is None else w
    a, beta = _cal_fit(y, _logit(np.asarray(p, dtype=float)), w, (0.0, np.array([0.0, 1.0])))
    return float(a), float(beta[1])


def ece(y, p, w=None, n_bins=10):
    """Expected calibration error with equal-width bins: sum_b (n_b/n) |ybar_b - pbar_b|."""
    w = np.ones_like(y) if w is None else w
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    resid = np.bincount(b, weights=w * (y - p), minlength=n_bins)
    return float(np.abs(resid).sum() / w.sum())


def spiegelhalter_z(y, p):
    """Spiegelhalter (1986) z-test of calibration; returns (z, two-sided p-value)."""
    num = np.sum((y - p) * (1 - 2 * p))
    den = np.sqrt(np.sum((1 - 2 * p) ** 2 * p * (1 - p)))
    z = num / den if den > 0 else np.nan
    return float(z), float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else float("nan")


def pav(x, y, w=None):
    """Isotonic (non-decreasing) regression of y on x by pool-adjacent-violators.
    Ties in x are pooled first, so equal forecasts receive equal recalibrated values."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    ux, inv = np.unique(x, return_inverse=True)
    sw = np.bincount(inv, weights=w)
    sy = np.bincount(inv, weights=w * y) / sw
    vals, wts, cnt = [], [], []
    for v, wt in zip(sy, sw):
        vals.append(v); wts.append(wt); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnt.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2)); wts.append(w1 + w2); cnt.append(c1 + c2)
    fitted_unique = np.repeat(vals, cnt)
    return fitted_unique[inv], ux, fitted_unique


def corp_decomposition(y, p, score="brier"):
    """CORP decomposition (Dimitriadis, Gneiting & Jordan 2021): mean score = MCB - DSC + UNC,
    using PAV-recalibrated forecasts. Exact, and MCB, DSC >= 0."""
    f = brier_terms if score == "brier" else logloss_terms
    p_cal, _, _ = pav(p, y)
    s_x = f(y, p).mean()
    s_c = f(y, p_cal).mean()
    s_r = f(y, np.full_like(p, y.mean())).mean()
    return {"score": s_x, "MCB": s_x - s_c, "DSC": s_r - s_c, "UNC": s_r}


# ----------------------------------------------------------------------------------------
# Paired tests
# ----------------------------------------------------------------------------------------


def delong(y, scores):
    """DeLong, DeLong & Clarke-Pearson (1988) covariance of correlated AUCs, using the
    midrank algorithm of Sun & Xu (2014). scores: array (k_models, n). Returns (aucs, cov)."""
    scores = np.atleast_2d(scores)
    pos = y == 1
    m, n = int(pos.sum()), int((~pos).sum())
    X, Y = scores[:, pos], scores[:, ~pos]
    k = scores.shape[0]
    tx = np.vstack([stats.rankdata(X[r]) for r in range(k)])
    ty = np.vstack([stats.rankdata(Y[r]) for r in range(k)])
    tz = np.vstack([stats.rankdata(np.concatenate([X[r], Y[r]])) for r in range(k)])
    aucs = (tz[:, :m].mean(axis=1) - (m + 1) / 2) / n
    v_pos = (tz[:, :m] - tx) / n            # structural components of the positives
    v_neg = 1.0 - (tz[:, m:] - ty) / m      # structural components of the negatives
    cov = np.atleast_2d(np.cov(v_pos)) / m + np.atleast_2d(np.cov(v_neg)) / n
    return aucs, cov


def mcnemar_midp(correct_a, correct_b):
    """McNemar (1947) test on discordant pairs; mid-p version (Fagerland, Lydersen & Laake 2013)."""
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    nd = b + c
    if nd == 0:
        return b, c, 1.0
    k = min(b, c)
    midp = 2 * stats.binom.cdf(k, nd, 0.5) - stats.binom.pmf(k, nd, 0.5)
    return b, c, float(min(1.0, midp))


def diebold_mariano(d, lag=0):
    """Diebold & Mariano (1995) test of equal expected loss from loss differences d_i.
    lag > 0 uses a Newey-West (Bartlett) long-run variance for serially dependent (time-ordered) d."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    dc = d - d.mean()
    lrv = np.dot(dc, dc) / n
    for l in range(1, lag + 1):
        lrv += 2 * (1 - l / (lag + 1)) * np.dot(dc[l:], dc[:-l]) / n
    se = np.sqrt(lrv / n)
    stat = d.mean() / se if se > 0 else np.nan
    return float(stat), float(2 * stats.norm.sf(abs(stat))) if np.isfinite(stat) else float("nan")


# ----------------------------------------------------------------------------------------
# Bootstrap weights
# ----------------------------------------------------------------------------------------


def _boot_weights(rng, n, cluster_codes=None):
    if cluster_codes is None:
        return np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
    k = cluster_codes.max() + 1
    draws = np.bincount(rng.integers(0, k, k), minlength=k).astype(float)
    return draws[cluster_codes]


# ----------------------------------------------------------------------------------------
# Binary evaluation
# ----------------------------------------------------------------------------------------

BINARY_BOOT_METRICS = ["brier", "log_loss", "brier_skill", "roc_auc", "average_precision",
                       "max_youden_ks", "cal_intercept", "cal_slope", "ece"]
LOWER_IS_BETTER = {"brier", "log_loss", "ece", "expected_cost", "cost_vs_trivial", "mse", "rmse", "mae",
                   "medae", "mape"}


class _ModelCache:
    """Per-model quantities that do not change across bootstrap replicates."""

    def __init__(self, y, p):
        self.p = p
        self.ss = SortedScores(p)
        self.bt = brier_terms(y, p)
        self.lt = logloss_terms(y, p)
        self.z = _logit(p)
        self.ece_bin = np.minimum((p * 10).astype(int), 9)
        self.cal_init = (0.0, np.array([0.0, 1.0]))


def _cal_fit(y, z, w, init, max_iter=100, tol=1e-10):
    """Intercept with slope fixed at 1 (calibration-in-the-large) and (intercept, slope) of the
    Cox (1958) recalibration model logit P(Y=1) = a + b z. Damped Newton-Raphson (step-halving on the
    log-likelihood) with a warm start, so over-confident forecasts cannot make the iteration diverge."""
    from scipy.special import expit

    def loglik(a0, b1, off):
        eta = a0 + b1 * z if off is None else a0 + off
        return np.sum(w * (y * eta - np.logaddexp(0.0, eta)))

    # calibration-in-the-large: slope fixed at 1 (z enters as an offset)
    a = float(init[0])
    for _ in range(max_iter):
        mu = expit(a + z)
        step = np.sum(w * (y - mu)) / max(np.sum(w * mu * (1 - mu)), 1e-300)
        t, ll0 = 1.0, loglik(a, 0.0, z)
        while loglik(a + t * step, 0.0, z) < ll0 - 1e-12 and t > 1e-6:
            t /= 2
        a += t * step
        if abs(t * step) < tol:
            break
    # two-parameter recalibration fit
    beta = np.array(init[1], dtype=float)
    for _ in range(max_iter):
        mu = expit(beta[0] + beta[1] * z)
        v = w * mu * (1 - mu)
        r = w * (y - mu)
        vz = v * z
        H = np.array([[v.sum(), vz.sum()], [vz.sum(), (vz * z).sum()]])
        g = np.array([r.sum(), (r * z).sum()])
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return a, np.array([np.nan, np.nan])
        t, ll0 = 1.0, loglik(beta[0], beta[1], None)
        while loglik(beta[0] + t * step[0], beta[1] + t * step[1], None) < ll0 - 1e-12 and t > 1e-6:
            t /= 2
        beta = beta + t * step
        if np.max(np.abs(t * step)) < tol:
            break
    return a, beta


def _binary_point_and_curves(y, c, w, threshold, cost_fp, cost_fn, ref_rate):
    ss = c.ss
    W1, W0 = ss.class_weights(y, w)
    sw = w.sum()
    ybar = W1.sum() / sw
    brier = float(np.dot(w, c.bt) / sw)
    rr = ybar if ref_rate is None else ref_rate
    brier_ref = ybar * (1 - ybar) + (ybar - rr) ** 2        # Brier score of the constant forecast rr
    a, beta = _cal_fit(y, c.z, w, c.cal_init)
    resid = np.bincount(c.ece_bin, weights=w * (y - c.p), minlength=10)
    fn, fp = ss.counts_at(W1, W0, threshold)
    tp, tn = W1.sum() - fn, W0.sum() - fp
    out = {
        "brier": brier,
        "log_loss": float(np.dot(w, c.lt) / sw),
        "brier_skill": 1 - brier / brier_ref,
        "roc_auc": ss.auc(W1, W0),
        "average_precision": ss.average_precision(W1, W0),
        "max_youden_ks": ss.max_youden(W1, W0),
        "cal_intercept": float(a),
        "cal_slope": float(beta[1]),
        "ece": float(np.abs(resid).sum() / sw),
    }
    out.update({k: float(v[0]) for k, v in threshold_metrics(tp, fp, fn, tn, cost_fp, cost_fn).items()})
    return out, (W1, W0), (a, beta)


@dataclass
class EvalResult:
    task: str
    metrics: pd.DataFrame                 # model, metric, estimate, ci_low, ci_high
    pairwise: pd.DataFrame                # metric, model_a, model_b, diff, ci_low, ci_high, test, p, p_holm
    extras: dict = field(default_factory=dict)

    def summary(self, metrics=None, digits=4):
        """Wide table 'estimate [ci_low, ci_high]' with one row per metric, one column per model."""
        m = self.metrics if metrics is None else self.metrics[self.metrics.metric.isin(metrics)]

        def fmt(r):
            if np.isnan(r.ci_low):
                return f"{r.estimate:.{digits}f}"
            return f"{r.estimate:.{digits}f} [{r.ci_low:.{digits}f}, {r.ci_high:.{digits}f}]"
        t = m.assign(cell=m.apply(fmt, axis=1)).pivot(index="metric", columns="model", values="cell")
        order = [k for k in m.metric.unique()]
        return t.loc[order]

    def to_markdown(self, path=None, digits=4):
        lines = ["## Model metrics (estimate [95% bootstrap CI])", "",
                 _df_to_md(self.summary(digits=digits).reset_index()), "",
                 "## Paired comparisons (difference = model_a - model_b)", "",
                 _df_to_md(self.pairwise.round(5)), ""]
        if "per_group" in self.extras:
            lines += ["## Per-group metrics", "", _df_to_md(self.extras["per_group"].round(4)), ""]
        text = "\n".join(lines)
        if path:
            with open(path, "w") as fh:
                fh.write(text)
        return text

    def plot(self, outdir, prefix=""):
        os.makedirs(outdir, exist_ok=True)
        if self.task == "binary":
            return _plot_binary(self, outdir, prefix)
        return []


def evaluate_binary(y, preds, threshold=None, cost_fp=1.0, cost_fn=1.0, n_boot=2000, seed=2026,
                    cluster=None, reference_rate=None, thetas=DEFAULT_THETAS, groups=None,
                    group_n_boot=500, alpha=0.05, hac_lag=0):
    """Score several probabilistic classifiers on one test set.

    y          : array of 0/1 outcomes (the frozen test labels)
    preds      : dict {model_name: array of P(y=1)}, all aligned with y
    threshold  : decision threshold theta for confusion-matrix metrics; default = the Bayes threshold
                 cost_fp / (cost_fp + cost_fn) (Elkan 2001), i.e. 0.5 for equal costs
    cluster    : optional array of cluster labels -> cluster bootstrap (e.g. group bookings, dates)
    reference_rate : constant forecast used in the Brier skill score (default: test-set base rate)
    groups     : optional array of group labels -> metrics recomputed within each group
    """
    y = _as_binary(y)
    names = list(preds)
    P = {k: _as_prob(v, k) for k, v in preds.items()}
    for k, v in P.items():
        if len(v) != len(y):
            raise ValueError(f"{k}: {len(v)} predictions but {len(y)} labels")
    if threshold is None:
        threshold = cost_fp / (cost_fp + cost_fn)
    n = len(y)
    cache = {k: _ModelCache(y, P[k]) for k in names}
    ss = {k: cache[k].ss for k in names}
    ones = np.ones(n)

    # ---- point estimates & curves
    point, curves = {}, {}
    for k in names:
        point[k], (W1, W0), init = _binary_point_and_curves(y, cache[k], ones, threshold, cost_fp, cost_fn,
                                                           reference_rate)
        cache[k].cal_init = init                     # warm start for the bootstrap fits
        corp_b = corp_decomposition(y, P[k], "brier")
        corp_l = corp_decomposition(y, P[k], "log")
        z, pz = spiegelhalter_z(y, P[k])
        point[k].update({"brier_MCB": corp_b["MCB"], "brier_DSC": corp_b["DSC"], "brier_UNC": corp_b["UNC"],
                         "logloss_MCB": corp_l["MCB"], "logloss_DSC": corp_l["DSC"],
                         "spiegelhalter_z": z, "spiegelhalter_p": pz,
                         "n_extreme_probs": float(np.sum((P[k] < EPS) | (P[k] > 1 - EPS))),
                         "n_confident_misses": float(np.sum(((P[k] < EPS) & (y == 1)) | ((P[k] > 1 - EPS) & (y == 0))))})
        p_cal, ux, fu = pav(P[k], y)
        curves[k] = {"roc": ss[k].roc_curve(W1, W0), "pr": ss[k].pr_curve(W1, W0),
                     "murphy": ss[k].murphy(W1, W0, thetas), "reliability": (ux, fu), "p": P[k]}
    aucs, cov = delong(y, np.vstack([P[k] for k in names]))

    # ---- bootstrap (same resamples for every model => paired)
    rng = np.random.default_rng(seed)
    codes = None if cluster is None else pd.factorize(np.asarray(cluster))[0]
    thr_names = list(threshold_metrics(1, 1, 1, 1).keys())
    boot_names = BINARY_BOOT_METRICS + thr_names
    B = np.full((n_boot, len(names), len(boot_names)), np.nan)
    BM = np.full((n_boot, len(names), len(thetas)), np.nan)
    for b in range(n_boot):
        w = _boot_weights(rng, n, codes)
        if np.sum(w * y) == 0 or np.sum(w * (1 - y)) == 0:
            continue
        for j, k in enumerate(names):
            pt, (W1, W0), _ = _binary_point_and_curves(y, cache[k], w, threshold, cost_fp, cost_fn,
                                                      reference_rate)
            B[b, j] = [pt[m] for m in boot_names]
            BM[b, j] = ss[k].murphy(W1, W0, thetas)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)

    rows = []
    for j, k in enumerate(names):
        for m, v in point[k].items():
            if m in boot_names:
                col = B[:, j, boot_names.index(m)]
                rows.append((k, m, v, np.nanpercentile(col, lo), np.nanpercentile(col, hi)))
            else:
                rows.append((k, m, v, np.nan, np.nan))
        rows.append((k, "roc_auc_delong_se", float(np.sqrt(cov[j, j])), np.nan, np.nan))
    metrics = pd.DataFrame(rows, columns=["model", "metric", "estimate", "ci_low", "ci_high"])

    # ---- pairwise comparisons
    prs = []
    for (ia, a), (ib, bname) in itertools.combinations(enumerate(names), 2):
        for m in ["brier", "log_loss", "roc_auc", "average_precision", "accuracy", "balanced_accuracy",
                  "f1", "mcc", "expected_cost", "cal_slope", "ece"]:
            jm = boot_names.index(m)
            dboot = B[:, ia, jm] - B[:, ib, jm]
            diff = point[a][m] - point[bname][m]
            test, pval = "paired bootstrap", 2 * min(np.nanmean(dboot <= 0), np.nanmean(dboot >= 0))
            if m == "roc_auc":
                var = cov[ia, ia] + cov[ib, ib] - 2 * cov[ia, ib]
                zst = (aucs[ia] - aucs[ib]) / np.sqrt(var) if var > 0 else np.nan
                test, pval = "DeLong", float(2 * stats.norm.sf(abs(zst)))
            elif m in ("brier", "log_loss"):
                f = brier_terms if m == "brier" else logloss_terms
                _, pval = diebold_mariano(f(y, P[a]) - f(y, P[bname]), hac_lag)
                test = "Diebold-Mariano" + (f" (NW lag {hac_lag})" if hac_lag else "")
            elif m == "accuracy":
                ca = (P[a] > threshold) == (y == 1)
                cb = (P[bname] > threshold) == (y == 1)
                _, _, pval = mcnemar_midp(ca, cb)
                test = "McNemar mid-p"
            prs.append((m, a, bname, diff, np.nanpercentile(dboot, lo), np.nanpercentile(dboot, hi),
                        test, min(1.0, float(pval))))
    pairwise = pd.DataFrame(prs, columns=["metric", "model_a", "model_b", "diff", "ci_low", "ci_high",
                                          "test", "p_value"])
    if len(pairwise):
        pairwise["p_holm"] = pairwise.groupby("metric")["p_value"].transform(lambda s: holm(s.values))

    # ---- Murphy-diagram differences against the model with the lowest Brier score
    best = min(names, key=lambda k: point[k]["brier"])
    jb = names.index(best)
    murphy = []
    for j, k in enumerate(names):
        d = BM[:, j] - BM[:, jb]
        for t_i, th in enumerate(thetas):
            murphy.append((k, th, curves[k]["murphy"][t_i], curves[k]["murphy"][t_i] - curves[best]["murphy"][t_i],
                           np.nanpercentile(d[:, t_i], lo), np.nanpercentile(d[:, t_i], hi)))
    murphy = pd.DataFrame(murphy, columns=["model", "theta", "mean_elementary_score", "diff_vs_best",
                                           "diff_ci_low", "diff_ci_high"])

    extras = {"curves": curves, "murphy": murphy, "best_by_brier": best, "threshold": threshold,
              "cost_fp": cost_fp, "cost_fn": cost_fn, "n": n, "prevalence": float(y.mean())}

    # ---- per-group metrics
    if groups is not None:
        groups = np.asarray(groups)
        tabs = []
        for gval in pd.unique(groups):
            msk = groups == gval
            sub = evaluate_binary(y[msk], {k: P[k][msk] for k in names}, threshold, cost_fp, cost_fn,
                                  n_boot=group_n_boot, seed=seed,
                                  cluster=None if cluster is None else np.asarray(cluster)[msk],
                                  reference_rate=reference_rate, thetas=thetas)
            t = sub.metrics[sub.metrics.metric.isin(["brier", "log_loss", "roc_auc", "cal_intercept",
                                                     "cal_slope", "accuracy", "mcc", "expected_cost"])]
            tabs.append(t.assign(group=gval, n=int(msk.sum()), prevalence=float(y[msk].mean())))
        extras["per_group"] = pd.concat(tabs, ignore_index=True)[
            ["group", "n", "prevalence", "model", "metric", "estimate", "ci_low", "ci_high"]]
    return EvalResult("binary", metrics, pairwise, extras)


# ----------------------------------------------------------------------------------------
# Regression evaluation (in case a member's model is a regression on a continuous target)
# ----------------------------------------------------------------------------------------


def _reg_point(y, yhat, w, ref_mean):
    e = yhat - y
    sw = w.sum()
    mse = np.sum(w * e ** 2) / sw
    ybar = np.sum(w * y) / sw if ref_mean is None else ref_mean
    out = {"mse": mse, "rmse": np.sqrt(mse), "mae": np.sum(w * np.abs(e)) / sw,
           "bias": np.sum(w * e) / sw, "r2_oos": 1 - np.sum(w * e ** 2) / np.sum(w * (y - ybar) ** 2)}
    if np.all(y != 0):
        out["mape"] = np.sum(w * np.abs(e / y)) / sw
    return out


def evaluate_regression(y, preds, train_mean=None, n_boot=2000, seed=2026, cluster=None, alpha=0.05,
                        hac_lag=0):
    """RMSE, MAE, bias, out-of-sample R^2 (relative to the TRAINING mean if given), MAPE (only if y != 0),
    median absolute error, with bootstrap CIs and Diebold-Mariano tests on squared / absolute errors."""
    y = np.asarray(y, dtype=float)
    names = list(preds)
    Y = {k: np.asarray(v, dtype=float) for k, v in preds.items()}
    n = len(y)
    ones = np.ones(n)
    point = {k: _reg_point(y, Y[k], ones, train_mean) for k in names}
    for k in names:
        point[k]["medae"] = float(np.median(np.abs(Y[k] - y)))
    mnames = list(point[names[0]].keys())
    boot_names = [m for m in mnames if m != "medae"]
    rng = np.random.default_rng(seed)
    codes = None if cluster is None else pd.factorize(np.asarray(cluster))[0]
    B = np.full((n_boot, len(names), len(boot_names)), np.nan)
    for b in range(n_boot):
        w = _boot_weights(rng, n, codes)
        for j, k in enumerate(names):
            pt = _reg_point(y, Y[k], w, train_mean)
            B[b, j] = [pt[m] for m in boot_names]
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    rows = []
    for j, k in enumerate(names):
        for m, v in point[k].items():
            if m in boot_names:
                col = B[:, j, boot_names.index(m)]
                rows.append((k, m, float(v), np.nanpercentile(col, lo), np.nanpercentile(col, hi)))
            else:
                rows.append((k, m, float(v), np.nan, np.nan))
    metrics = pd.DataFrame(rows, columns=["model", "metric", "estimate", "ci_low", "ci_high"])
    prs = []
    for (ia, a), (ib, bname) in itertools.combinations(enumerate(names), 2):
        for m in ["rmse", "mae", "r2_oos"]:
            jm = boot_names.index(m)
            dboot = B[:, ia, jm] - B[:, ib, jm]
            if m == "rmse":
                _, pval = diebold_mariano((Y[a] - y) ** 2 - (Y[bname] - y) ** 2, hac_lag)
                test = "Diebold-Mariano (squared error)"
            elif m == "mae":
                _, pval = diebold_mariano(np.abs(Y[a] - y) - np.abs(Y[bname] - y), hac_lag)
                test = "Diebold-Mariano (absolute error)"
            else:
                test, pval = "paired bootstrap", 2 * min(np.nanmean(dboot <= 0), np.nanmean(dboot >= 0))
            prs.append((m, a, bname, point[a][m] - point[bname][m], np.nanpercentile(dboot, lo),
                        np.nanpercentile(dboot, hi), test, min(1.0, float(pval))))
    pairwise = pd.DataFrame(prs, columns=["metric", "model_a", "model_b", "diff", "ci_low", "ci_high",
                                          "test", "p_value"])
    if len(pairwise):
        pairwise["p_holm"] = pairwise.groupby("metric")["p_value"].transform(lambda s: holm(s.values))
    return EvalResult("regression", metrics, pairwise, {"n": n})


# ----------------------------------------------------------------------------------------
# Plotting (static PNGs for the report)
# ----------------------------------------------------------------------------------------

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def style_axes(ax, xlabel=None, ylabel=None, title=None):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9, length=3, width=0.8)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", fontweight="bold")


def _legend(ax, **kw):
    leg = ax.legend(frameon=False, fontsize=9, labelcolor=INK2, **kw)
    return leg


def _plot_binary(res, outdir, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    curves = res.extras["curves"]
    names = list(curves)
    col = {k: SERIES[i % len(SERIES)] for i, k in enumerate(names)}
    files = []

    # Reliability (CORP) + sharpness
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(6.4, 6.6), gridspec_kw={"height_ratios": [3, 1.1]},
                                  sharex=True, facecolor=SURFACE)
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1, zorder=1)
    for k in names:
        ux, fu = curves[k]["reliability"]
        ax.step(ux, fu, where="post", color=col[k], lw=2, label=k, zorder=3)
    style_axes(ax, ylabel="Observed frequency (PAV-recalibrated)", title="Reliability diagram (CORP)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _legend(ax, loc="upper left")
    bins = np.linspace(0, 1, 41)
    for k in names:
        axh.hist(curves[k]["p"], bins=bins, histtype="step", color=col[k], lw=2, density=True)
    style_axes(axh, xlabel="Forecast probability p", ylabel="Density")
    fig.tight_layout()
    f = os.path.join(outdir, f"{prefix}reliability.png"); fig.savefig(f, dpi=160); plt.close(fig); files.append(f)

    # ROC and PR
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4), facecolor=SURFACE)
    a1.plot([0, 1], [0, 1], color=AXIS, lw=1)
    prev = res.extras["prevalence"]
    a2.axhline(prev, color=AXIS, lw=1)
    met = res.metrics.set_index(["model", "metric"]).estimate
    for k in names:
        fpr, tpr = curves[k]["roc"]
        a1.plot(fpr, tpr, color=col[k], lw=2, label=f"{k} (AUC {met[(k, 'roc_auc')]:.3f})")
        rec, prc = curves[k]["pr"]
        a2.plot(rec, prc, color=col[k], lw=2, label=f"{k} (AP {met[(k, 'average_precision')]:.3f})")
    style_axes(a1, "False positive rate", "True positive rate", "ROC curve")
    style_axes(a2, "Recall", "Precision", "Precision-recall curve")
    a2.text(0.99, prev + 0.015, f"no-skill = prevalence {prev:.3f}", ha="right", va="bottom", fontsize=8,
            color=MUTED)
    a1.set_xlim(0, 1); a1.set_ylim(0, 1); a2.set_xlim(0, 1); a2.set_ylim(0, 1.01)
    _legend(a1, loc="lower right"); _legend(a2, loc="lower left")
    fig.tight_layout()
    f = os.path.join(outdir, f"{prefix}roc_pr.png"); fig.savefig(f, dpi=160); plt.close(fig); files.append(f)

    # Murphy diagram
    md = res.extras["murphy"]
    best = res.extras["best_by_brier"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4), facecolor=SURFACE)
    for k in names:
        d = md[md.model == k]
        a1.plot(d.theta, d.mean_elementary_score, color=col[k], lw=2, label=k)
        if k != best:
            a2.fill_between(d.theta, d.diff_ci_low, d.diff_ci_high, color=col[k], alpha=0.12, lw=0)
            a2.plot(d.theta, d.diff_vs_best, color=col[k], lw=2, label=f"{k} minus {best}")
    a2.axhline(0, color=INK2, lw=1)
    th = res.extras["threshold"]
    for a in (a1, a2):
        a.axvline(th, color=AXIS, lw=1)
    style_axes(a1, "Cost ratio  theta = c_FP / (c_FP + c_FN)", "Mean elementary score (lower is better)",
               "Murphy diagram")
    style_axes(a2, "Cost ratio  theta", "Score difference (95% pointwise band)",
               f"Difference vs best-Brier model ({best})")
    _legend(a1, loc="upper right"); _legend(a2, loc="upper right")
    fig.tight_layout()
    f = os.path.join(outdir, f"{prefix}murphy.png"); fig.savefig(f, dpi=160); plt.close(fig); files.append(f)
    return files


def _df_to_md(df):
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(out)


# ----------------------------------------------------------------------------------------
# Command-line interface
# ----------------------------------------------------------------------------------------


def _load(truth_path, pred_specs, id_col="id", y_col="y", pred_col="prediction"):
    truth = pd.read_csv(truth_path)
    if truth[id_col].duplicated().any():
        raise ValueError("truth file has duplicated ids")
    preds = {}
    for spec in pred_specs:
        name, path = spec.split("=", 1)
        d = pd.read_csv(path)
        if d[id_col].duplicated().any():
            raise ValueError(f"{name}: duplicated ids")
        missing = set(truth[id_col]) - set(d[id_col])
        extra = set(d[id_col]) - set(truth[id_col])
        if missing or extra:
            raise ValueError(f"{name}: ids do not match the frozen test set "
                             f"({len(missing)} missing, {len(extra)} unexpected) - everyone must score "
                             "exactly the same rows")
        preds[name] = truth[[id_col]].merge(d[[id_col, pred_col]], on=id_col, how="left")[pred_col].values
    return truth, preds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", required=True, help="CSV with columns id,y (+ optional group/cluster columns)")
    ap.add_argument("--pred", action="append", required=True, metavar="NAME=PATH",
                    help="prediction CSV with columns id,prediction (repeat for each model)")
    ap.add_argument("--task", choices=["binary", "regression"], default="binary")
    ap.add_argument("--threshold", type=float, default=None,
                    help="decision threshold theta (default cost_fp/(cost_fp+cost_fn))")
    ap.add_argument("--cost-fp", type=float, default=1.0, help="cost of a false positive (default 1)")
    ap.add_argument("--cost-fn", type=float, default=1.0, help="cost of a false negative (default 1)")
    ap.add_argument("--group", default=None, help="truth column for per-group metrics (e.g. hotel)")
    ap.add_argument("--cluster", default=None, help="truth column for a cluster bootstrap")
    ap.add_argument("--train-mean", type=float, default=None, help="regression: training mean for R^2_OOS")
    ap.add_argument("--reference-rate", type=float, default=None,
                    help="binary: constant forecast for the Brier skill score (e.g. training prevalence)")
    ap.add_argument("--hac-lag", type=int, default=0, help="Newey-West lag for Diebold-Mariano tests")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--outdir", default="eval_results")
    args = ap.parse_args(argv)

    try:
        truth, preds = _load(args.truth, args.pred)
    except (ValueError, KeyError) as err:
        sys.exit(f"evaluate.py: {err}")
    os.makedirs(args.outdir, exist_ok=True)
    if args.task == "binary":
        res = evaluate_binary(truth["y"].values, preds, threshold=args.threshold, cost_fp=args.cost_fp,
                              cost_fn=args.cost_fn, n_boot=args.n_boot, seed=args.seed,
                              cluster=None if args.cluster is None else truth[args.cluster].values,
                              reference_rate=args.reference_rate,
                              groups=None if args.group is None else truth[args.group].values,
                              hac_lag=args.hac_lag)
        res.extras["murphy"].to_csv(os.path.join(args.outdir, "murphy.csv"), index=False)
        if "per_group" in res.extras:
            res.extras["per_group"].to_csv(os.path.join(args.outdir, "per_group.csv"), index=False)
        res.plot(args.outdir)
    else:
        res = evaluate_regression(truth["y"].values, preds, train_mean=args.train_mean, n_boot=args.n_boot,
                                  seed=args.seed,
                                  cluster=None if args.cluster is None else truth[args.cluster].values,
                                  hac_lag=args.hac_lag)
    res.metrics.to_csv(os.path.join(args.outdir, "metrics.csv"), index=False)
    res.pairwise.to_csv(os.path.join(args.outdir, "pairwise.csv"), index=False)
    text = res.to_markdown(os.path.join(args.outdir, "summary.md"))
    print(text)
    print(f"\nWrote metrics.csv, pairwise.csv, summary.md (+ figures) to {args.outdir}/")


if __name__ == "__main__":
    main()
