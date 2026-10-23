"""Checks evaluate.py against scikit-learn / scipy and against the identities derived in the report.
Run:  python test_evaluate.py      (or: pytest test_evaluate.py)"""
import numpy as np
from scipy import stats, integrate
from sklearn import metrics as skm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

import evaluate as ev

rng = np.random.default_rng(0)
N = 3000
Y = rng.binomial(1, 0.37, N).astype(float)
# rounded scores create many ties, which is where implementations usually disagree
P = np.clip(np.round(1 / (1 + np.exp(-(rng.normal(0, 1, N) + 1.6 * (Y - 0.5)))), 2), 0, 1)
P2 = np.clip(P + rng.normal(0, 0.08, N), 0, 1)


def test_proper_scores_match_sklearn():
    assert np.isclose(ev.brier_terms(Y, P).mean(), skm.brier_score_loss(Y, P))
    assert np.isclose(ev.logloss_terms(Y, P).mean(), skm.log_loss(Y, P, labels=[0, 1]))


def test_rank_metrics_match_sklearn():
    ss = ev.SortedScores(P)
    W1, W0 = ss.class_weights(Y, np.ones(N))
    assert np.isclose(ss.auc(W1, W0), skm.roc_auc_score(Y, P))
    assert np.isclose(ss.average_precision(W1, W0), skm.average_precision_score(Y, P))
    ks = stats.ks_2samp(P[Y == 0], P[Y == 1]).statistic
    assert np.isclose(ss.max_youden(W1, W0), ks)


def test_threshold_metrics_match_sklearn():
    for th in (0.3, 0.5, 2 / 3):
        yhat = (P > th).astype(int)
        tn, fp, fn, tp = skm.confusion_matrix(Y, yhat).ravel()
        m = ev.threshold_metrics(tp, fp, fn, tn)
        assert np.isclose(m["accuracy"], skm.accuracy_score(Y, yhat))
        assert np.isclose(m["balanced_accuracy"], skm.balanced_accuracy_score(Y, yhat))
        assert np.isclose(m["f1"], skm.f1_score(Y, yhat))
        assert np.isclose(m["mcc"], skm.matthews_corrcoef(Y, yhat))
        assert np.isclose(m["cohen_kappa"], skm.cohen_kappa_score(Y, yhat))
        assert np.isclose(m["ppv_precision"], skm.precision_score(Y, yhat))
        assert np.isclose(m["tpr_recall"], skm.recall_score(Y, yhat))
        ss = ev.SortedScores(P)
        fn2, fp2 = ss.counts_at(*ss.class_weights(Y, np.ones(N)), th)
        assert np.isclose(fn2[0], fn) and np.isclose(fp2[0], fp)


def test_weighted_equals_replicated():
    """Bootstrap weights must give exactly the metric of the physically resampled data."""
    w = np.bincount(rng.integers(0, N, N), minlength=N).astype(float)
    idx = np.repeat(np.arange(N), w.astype(int))
    c = ev._ModelCache(Y, P)
    pt_w, _, _ = ev._binary_point_and_curves(Y, c, w, 0.5, 1, 1, None)
    c2 = ev._ModelCache(Y[idx], P[idx])
    pt_r, _, _ = ev._binary_point_and_curves(Y[idx], c2, np.ones(len(idx)), 0.5, 1, 1, None)
    for k in pt_w:
        assert np.isclose(pt_w[k], pt_r[k], rtol=1e-7, atol=1e-9, equal_nan=True), k


def test_pav_matches_sklearn_isotonic():
    fit, _, _ = ev.pav(P, Y)
    iso = IsotonicRegression(y_min=0, y_max=1, increasing=True).fit(P, Y).predict(P)
    assert np.allclose(fit, iso)


def test_corp_decomposition_identity():
    for s in ("brier", "log"):
        d = ev.corp_decomposition(Y, P2, s)
        assert np.isclose(d["score"], d["MCB"] - d["DSC"] + d["UNC"])
        assert d["MCB"] >= -1e-12 and d["DSC"] >= -1e-12


def test_schervish_mixture_representation():
    """Brier = 2 * int_0^1 S_theta dtheta ; log loss = int_0^1 S_theta / (theta (1 - theta)) dtheta."""
    for p in (0.07, 0.37, 0.81):
        for y in (0, 1):
            f = lambda t: float(ev.elementary_terms(np.array([y]), np.array([p]), t)[0])
            b = 2 * (integrate.quad(f, 0, p)[0] + integrate.quad(f, p, 1)[0])
            l = integrate.quad(lambda t: f(t) / (t * (1 - t)), 0, p)[0] + \
                integrate.quad(lambda t: f(t) / (t * (1 - t)), p, 1)[0]
            assert np.isclose(b, (p - y) ** 2)
            assert np.isclose(l, -np.log(p if y == 1 else 1 - p))


def test_calibration_fit_matches_unpenalised_logistic_regression():
    z = ev._logit(P2)
    lr = LogisticRegression(C=1e12, tol=1e-10, max_iter=1000).fit(z.reshape(-1, 1), Y)
    a, b = ev.calibration_intercept_slope(Y, P2)
    assert np.isclose(b, lr.coef_[0, 0], atol=1e-5)


def test_ece_direct():
    bins = np.minimum((P * 10).astype(int), 9)
    direct = sum((bins == b).mean() * abs(Y[bins == b].mean() - P[bins == b].mean())
                 for b in range(10) if np.any(bins == b))
    assert np.isclose(ev.ece(Y, P), direct)


def test_delong_matches_brute_force():
    y = Y[:400]; s1 = P[:400]; s2 = P2[:400]
    pos, neg = y == 1, y == 0

    def psi(a, b):
        return (a > b) + 0.5 * (a == b)

    def components(s):
        v10 = np.array([psi(x, s[neg]).mean() for x in s[pos]])
        v01 = np.array([psi(s[pos], t).mean() for t in s[neg]])
        return v10, v01
    a10, a01 = components(s1)
    b10, b01 = components(s2)
    var_diff = (np.var(a10 - b10, ddof=1) / pos.sum()) + (np.var(a01 - b01, ddof=1) / neg.sum())
    aucs, cov = ev.delong(y, np.vstack([s1, s2]))
    assert np.allclose(aucs, [skm.roc_auc_score(y, s1), skm.roc_auc_score(y, s2)])
    assert np.isclose(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], var_diff)


def test_mcnemar_midp():
    ca = rng.random(500) < 0.8
    cb = rng.random(500) < 0.75
    b, c, p = ev.mcnemar_midp(ca, cb)
    k, n = min(b, c), b + c
    exact = min(1, 2 * stats.binom.cdf(k, n, 0.5))
    assert np.isclose(p, exact - stats.binom.pmf(k, n, 0.5))


def test_holm():
    assert np.allclose(ev.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_end_to_end_runs():
    res = ev.evaluate_binary(Y, {"A": P, "B": P2}, n_boot=50, groups=(np.arange(N) % 2))
    assert {"brier", "roc_auc", "mcc"} <= set(res.metrics.metric)
    assert len(res.pairwise) > 0 and "per_group" in res.extras
    reg = ev.evaluate_regression(Y + rng.normal(0, 1, N), {"A": P, "B": P2}, n_boot=50)
    assert {"rmse", "mae", "r2_oos"} <= set(reg.metrics.metric)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"all {len(tests)} checks passed")
