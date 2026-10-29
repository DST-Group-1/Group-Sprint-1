#!/usr/bin/env python3
"""
simulations.py - reproduces every number and figure in evaluation_metrics.md.

  python simulations.py --data path/to/hotel_bookings_clean.csv   (the group's cleaned file: columns
                                                               include is_canceled, arrival_date and split;
                                                               the data-based parts S5, S6 and the sizes and
                                                               rates used in S2-S4 are taken from it. Without
                                                               --data, S2-S4 use the cleaned file's values
                                                               stored in DEFAULTS and S5, S6 are skipped.)
Outputs: figures/*.png and sim_results.json

Sections
  S1  propriety: expected score as a function of the reported probability
  S2  prevalence (label) shift: which metrics move when only the cancellation rate changes
  S3  worked example: three synthetic submissions scored with evaluate.py
  S4  ranking noise: how often the truly-worse model wins, by metric and test-set size
  S5  regression: which functional each loss rewards, on the real ADR distribution
  S6  data diagnostics: duplicates, drift and a 'memoriser' under different splits

Provenance: drafted with the help of Claude (Anthropic) for Luke D's metrics research.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

import evaluate as ev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = "figures"
S1, S2, S3, S4 = ev.SERIES[:4]
OUT = {}

# Facts about the group's cleaned file (hotel_bookings_clean.csv: 118,558 rows; split column = train / validation / test,
# a random 70/15/15 split stratified by is_canceled that keeps identical rows together). configure(df) recomputes them
# from the file that is passed with --data.
DEFAULTS = dict(
    n_test=17903, pi_train=0.3731, pi_test=0.3675, city_share_test=0.6769,
    prev_city_test=0.4146, prev_resort_test=0.2690,
    rate_lo=0.2690, rate_hi=0.4202,                       # range of cancellation rates over split x hotel
    marks={"deduplicated data": 0.2792, "Resort hotel": 0.2801, "train": 0.3731, "test": 0.3675, "City hotel": 0.4191})
CFG = dict(DEFAULTS)


def configure(df):
    """Take test-set size, cancellation rates and the S2 marks from the cleaned file itself."""
    tr, te = df[df.split == "train"], df[df.split == "test"]
    raw = [c for c in df.columns if c != "split"]
    ded = df.drop_duplicates(subset=raw)
    rates = df.groupby(["split", "hotel"]).is_canceled.mean()
    CFG.update(n_test=int(len(te)), pi_train=float(tr.is_canceled.mean()), pi_test=float(te.is_canceled.mean()),
               city_share_test=float((te.hotel == "City Hotel").mean()),
               prev_city_test=float(te.loc[te.hotel == "City Hotel", "is_canceled"].mean()),
               prev_resort_test=float(te.loc[te.hotel == "Resort Hotel", "is_canceled"].mean()),
               rate_lo=float(rates.min()), rate_hi=float(rates.max()),
               marks={"deduplicated data": float(ded.is_canceled.mean()),
                      "Resort hotel": float(df.loc[df.hotel == "Resort Hotel", "is_canceled"].mean()),
                      "train": float(tr.is_canceled.mean()), "test": float(te.is_canceled.mean()),
                      "City hotel": float(df.loc[df.hotel == "City Hotel", "is_canceled"].mean())})
    OUT["config"] = CFG


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=160, facecolor=ev.SURFACE)
    plt.close(fig)
    print("saved", path)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    return np.log(p / (1 - p))


# ------------------------------------------------------------------------------------------
# S1  Propriety
# ------------------------------------------------------------------------------------------
def s1_propriety(q=0.37):
    p = np.linspace(0.001, 0.999, 999)
    curves = {
        "Brier score": (p - q) ** 2 + q * (1 - q),
        "Log loss": -(q * np.log(p) + (1 - q) * np.log(1 - p)),
        "Absolute error |p - y|": q * (1 - p) + (1 - q) * p,
        "Misclassification at 0.5": np.where(p > 0.5, 1 - q, q),
    }
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3), facecolor=ev.SURFACE)
    res = {}
    for ax, (name, e) in zip(axes, curves.items()):
        ax.axvline(q, color=ev.AXIS, lw=1)
        ax.plot(p, e, color=S1, lw=2)
        i = int(np.argmin(e))
        if name.startswith("Misclass"):
            ax.plot([0.001, 0.5], [q, q], color=S1, lw=5, solid_capstyle="round", alpha=0.3)
            note, where = "flat: every p \u2264 0.5\nis equally good", (0.03, 0.95, "left", "top")
        elif name.startswith("Absolute"):
            ax.plot(p[i], e[i], "o", color=S1, ms=8, mec=ev.SURFACE, mew=2, zorder=5, clip_on=False)
            note, where = "minimum at p = 0\n(report 0 or 1, never q)", (0.03, 0.95, "left", "top")
        else:
            ax.plot(p[i], e[i], "o", color=S1, ms=8, mec=ev.SURFACE, mew=2, zorder=5)
            note, where = f"minimum at p = {p[i]:.2f} = q", (0.5, 0.95, "center", "top")
        ax.text(where[0], where[1], note, transform=ax.transAxes, ha=where[2], va=where[3], fontsize=9,
                color=ev.INK2)
        ev.style_axes(ax, "Reported probability p", "Expected score" if ax is axes[0] else None, name)
        ax.set_xlim(0, 1)
        res[name] = float(p[i])
    axes[1].set_ylim(0.5, 2.0)
    fig.text(0.01, 0.985, f"Expected score of a forecast p when the true cancellation probability is q = {q} "
             "(vertical line)", ha="left", va="top", fontsize=10, color=ev.INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig1_propriety.png")
    OUT["S1_minimisers"] = res


# ------------------------------------------------------------------------------------------
# S2  Prevalence shift (binormal model, analytic + quadrature)
# ------------------------------------------------------------------------------------------
def s2_prevalence(auc=0.85, pi0=None, theta=0.5):
    pi0 = CFG["pi_train"] if pi0 is None else pi0
    delta = np.sqrt(2) * stats.norm.ppf(auc)        # AUC = Phi(delta / sqrt 2)
    c = -logit(pi0) / delta                          # p > theta  <=>  s > c   (theta = 0.5)
    c = (logit(theta) - logit(pi0)) / delta
    tpr = stats.norm.sf(c - delta / 2)
    fpr = stats.norm.sf(c + delta / 2)
    # quadrature for the proper scores
    x, wq = np.polynomial.hermite_e.hermegauss(120)
    wq = wq / wq.sum()

    def scores(pi):
        p1 = sigmoid(logit(pi0) + delta * (x + delta / 2))   # forecasts for cancelled bookings
        p0 = sigmoid(logit(pi0) + delta * (x - delta / 2))   # forecasts for non-cancelled bookings
        brier = pi * np.sum(wq * (1 - p1) ** 2) + (1 - pi) * np.sum(wq * p0 ** 2)
        ll = -(pi * np.sum(wq * np.log(p1)) + (1 - pi) * np.sum(wq * np.log(1 - p0)))
        return brier, ll

    pis = np.linspace(0.05, 0.95, 181)
    rows = []
    for pi in pis:
        q = pi * tpr + (1 - pi) * fpr
        ppv = pi * tpr / q
        acc = pi * tpr + (1 - pi) * (1 - fpr)
        f1 = 2 * ppv * tpr / (ppv + tpr)
        mcc = np.sqrt(pi * (1 - pi) / (q * (1 - q))) * (tpr - fpr)
        brier, ll = scores(pi)
        rows.append(dict(pi=pi, auc=auc, balanced_accuracy=(tpr + 1 - fpr) / 2, recall=tpr, precision=ppv,
                         f1=f1, mcc=mcc, accuracy=acc, brier=brier, log_loss=ll,
                         cal_intercept=logit(pi) - logit(pi0)))
    df = pd.DataFrame(rows)
    panels = [("auc", "ROC AUC"), ("balanced_accuracy", "Balanced accuracy"), ("recall", "Recall (TPR)"),
              ("precision", "Precision (PPV)"), ("f1", "F1"), ("mcc", "MCC"), ("accuracy", "Accuracy"),
              ("brier", "Brier score")]
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), facecolor=ev.SURFACE, sharex=True)
    for ax, (col, title) in zip(axes.ravel(), panels):
        ax.axvspan(CFG["rate_lo"], CFG["rate_hi"], color=ev.GRID, alpha=0.6, lw=0)
        ax.plot(df.pi, df[col], color=S1, lw=2)
        v0 = np.interp(pi0, df.pi, df[col])
        ax.plot(pi0, v0, "o", color=S1, ms=8, mec=ev.SURFACE, mew=2, zorder=5)
        ev.style_axes(ax, "Cancellation rate in test data" if ax in axes[1] else None, None, title)
        lo_, hi_ = df[col].min(), df[col].max()
        pad = max(0.02, 0.08 * (hi_ - lo_))
        ax.set_ylim(lo_ - pad, hi_ + pad)
        ax.set_xlim(0.05, 0.95)
    axes[0, 0].text((CFG["rate_lo"] + CFG["rate_hi"]) / 2, 0.03,
                    f"shaded: rates seen in\nour data ({CFG['rate_lo']:.2f}-{CFG['rate_hi']:.2f})", transform=axes[0, 0].get_xaxis_transform(),
                    fontsize=8, color=ev.MUTED, ha="center")
    fig.tight_layout()
    save(fig, "fig2_prevalence_shift.png")
    marks = CFG["marks"]
    tab = {k: {c: float(np.interp(v, df.pi, df[c])) for c in df.columns if c != "pi"} for k, v in marks.items()}
    OUT["S2_table"] = tab
    OUT["S2_rates"] = {"tpr": float(tpr), "fpr": float(fpr), "delta": float(delta)}


# ------------------------------------------------------------------------------------------
# S3  Worked example: three synthetic submissions
# ------------------------------------------------------------------------------------------
def make_submissions(n=None, seed=11):
    n = CFG["n_test"] if n is None else n
    rng = np.random.default_rng(seed)
    city = rng.random(n) < CFG["city_share_test"]
    y = (rng.random(n) < np.where(city, CFG["prev_city_test"], CFG["prev_resort_test"])).astype(float)
    lo, delta = logit(CFG["pi_test"]), 2.2
    s = rng.normal(0, 1, n) + delta * (y - 0.5)

    def informative(sig):          # calibrated log-odds given a noisy copy of the latent score
        return lo + delta * (s + rng.normal(0, sig, n)) / (1 + sig ** 2)
    eta_a, eta_b = informative(0.8), informative(0.45)
    p_a = sigmoid(eta_a)                                       # calibrated, less informative
    p_b = sigmoid(lo + 2.0 * (eta_b - lo))                     # informative, over-confident
    p_c = rng.binomial(100, sigmoid(lo + 0.8 * (eta_b - lo))) / 100   # 'vote fractions', under-confident
    hotel = np.where(city, "City", "Resort")
    return y, {"Model A": p_a, "Model B": p_b, "Model C": p_c}, hotel


def s3_worked_example():
    y, preds, hotel = make_submissions()
    res = ev.evaluate_binary(y, preds, threshold=0.5, n_boot=2000, groups=hotel, seed=2026)
    files = res.plot(FIG, prefix="fig3_worked_")
    print("saved", files)
    keep = ["brier", "brier_skill", "log_loss", "roc_auc", "average_precision", "cal_intercept", "cal_slope",
            "ece", "brier_MCB", "brier_DSC", "brier_UNC", "accuracy", "balanced_accuracy", "f1", "mcc",
            "cohen_kappa", "n_extreme_probs", "n_confident_misses"]
    OUT["S3_summary_md"] = ev._df_to_md(res.summary(keep, digits=4).reset_index())
    m = res.metrics
    OUT["S3_metrics"] = {f"{r.model}|{r.metric}": [r.estimate, r.ci_low, r.ci_high] for r in m.itertuples()}
    pw = res.pairwise[res.pairwise.metric.isin(["brier", "log_loss", "roc_auc", "accuracy", "mcc"])]
    OUT["S3_pairwise_md"] = ev._df_to_md(pw.assign(
        diff=pw["diff"].map(lambda v: f"{v:+.4f}"),
        ci=[f"[{a:+.4f}, {b:+.4f}]" for a, b in zip(pw.ci_low, pw.ci_high)],
        p_value=pw.p_value.map(lambda v: f"{v:.2g}"), p_holm=pw.p_holm.map(lambda v: f"{v:.2g}"))
        [["metric", "model_a", "model_b", "diff", "ci", "test", "p_value", "p_holm"]])
    md = res.extras["murphy"]
    a, b = md[md.model == "Model A"].set_index("theta"), md[md.model == "Model B"].set_index("theta")
    d = (b.mean_elementary_score - a.mean_elementary_score)
    sign = np.sign(d.values)
    OUT["S3_murphy_B_minus_A_crossings"] = [float(t) for t in d.index[1:][np.diff(sign) != 0]]
    OUT["S3_best_by_brier"] = res.extras["best_by_brier"]
    # cost ratios at which a model is significantly worse / better than the best-Brier model (95% pointwise bands)
    OUT["S3_murphy_significant_vs_best"] = {
        k: {"worse": [float(t) for t in g.theta[g.diff_ci_low > 0]],
            "better": [float(t) for t in g.theta[g.diff_ci_high < 0]]}
        for k, g in md.groupby("model") if k != res.extras["best_by_brier"]}
    pg = res.extras["per_group"]
    OUT["S3_per_group"] = {f"{r.group}|{r.model}|{r.metric}": [r.estimate, r.ci_low, r.ci_high, r.prevalence]
                           for r in pg.itertuples()}
    # dominance check of every pair over all theta
    dom = {}
    names = list(preds)
    for i in names:
        for j in names:
            if i < j:
                di = md[md.model == i].mean_elementary_score.values - md[md.model == j].mean_elementary_score.values
                dom[f"{i} vs {j}"] = {"i_better_share": float(np.mean(di < 0)), "j_better_share": float(np.mean(di > 0))}
    OUT["S3_dominance"] = dom
    # elementary scores at theta = 2/3 (walk cost = 2 x empty-room cost)
    th = 2 / 3
    OUT["S3_cost_theta_2_3"] = {k: float(np.mean(ev.elementary_terms(y, p, th))) for k, p in preds.items()}
    OUT["S3_cost_theta_0_85"] = {k: float(np.mean(ev.elementary_terms(y, p, 0.85))) for k, p in preds.items()}


# ------------------------------------------------------------------------------------------
# S4  Ranking noise and power
# ------------------------------------------------------------------------------------------
def s4_ranking_noise(R=1000, seed=4):
    rng = np.random.default_rng(seed)
    pi, delta, sa, sb = CFG["pi_test"], 2.2, 0.9, 0.95
    lo = logit(pi)

    def draw(n):
        y = (rng.random(n) < pi).astype(float)
        s = rng.normal(0, 1, n) + delta * (y - 0.5)
        pa = sigmoid(lo + delta * (s + rng.normal(0, sa, n)) / (1 + sa ** 2))
        pb = sigmoid(lo + delta * (s + rng.normal(0, sb, n)) / (1 + sb ** 2))
        return y, pa, pb

    def auc(y, p):
        r = stats.rankdata(p)
        n1 = y.sum()
        return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))

    y, pa, pb = draw(2_000_000)
    truth = {"brier": (ev.brier_terms(y, pa).mean(), ev.brier_terms(y, pb).mean()),
             "log_loss": (ev.logloss_terms(y, pa).mean(), ev.logloss_terms(y, pb).mean()),
             "auc": (auc(y, pa), auc(y, pb)),
             "accuracy": (np.mean((pa > 0.5) == y), np.mean((pb > 0.5) == y))}
    OUT["S4_population"] = {k: [float(v[0]), float(v[1])] for k, v in truth.items()}
    sizes = [500, 1000, 2000, 5000, 10000, CFG["n_test"]]
    rows = []
    for n in sizes:
        wrong = {k: 0 for k in truth}
        paired_sig = unpaired_sig = 0
        for _ in range(R):
            y, pa, pb = draw(n)
            la, lb = ev.brier_terms(y, pa), ev.brier_terms(y, pb)
            # A is truly better; count a wrong ordering as 1 and an exact tie as 1/2
            def bad(a, b, lower_better=True):
                return float(a > b if lower_better else a < b) + 0.5 * float(a == b)
            wrong["brier"] += bad(la.mean(), lb.mean())
            wrong["log_loss"] += bad(ev.logloss_terms(y, pa).mean(), ev.logloss_terms(y, pb).mean())
            wrong["auc"] += bad(auc(y, pa), auc(y, pb), lower_better=False)
            wrong["accuracy"] += bad(np.mean((pa > 0.5) == y), np.mean((pb > 0.5) == y), lower_better=False)
            d = la - lb
            paired_sig += abs(d.mean() / (d.std(ddof=1) / np.sqrt(n))) > 1.96
            se_u = np.sqrt(la.var(ddof=1) / n + lb.var(ddof=1) / n)
            unpaired_sig += abs((la.mean() - lb.mean()) / se_u) > 1.96
        rows.append(dict(n=n, **{f"wrong_{k}": v / R for k, v in wrong.items()},
                         power_paired=paired_sig / R, power_unpaired=unpaired_sig / R))
    df = pd.DataFrame(rows)
    OUT["S4_table"] = df.to_dict(orient="records")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=ev.SURFACE)
    lab = {"brier": "Brier score", "log_loss": "Log loss", "auc": "ROC AUC", "accuracy": "Accuracy at 0.5"}
    for col, (k, name) in zip(ev.SERIES, lab.items()):
        a1.plot(df.n, df[f"wrong_{k}"], color=col, lw=2, marker="o", ms=5, mec=ev.SURFACE, mew=1.5, label=name)
    a1.set_xscale("log")
    ev.style_axes(a1, "Test-set size n (log scale)", "P(truly worse model ranks first)",
                  "How often a single test split picks the wrong model")
    a1.legend(frameon=False, fontsize=9, labelcolor=ev.INK2, loc="upper right")
    a2.plot(df.n, df.power_paired, color=S1, lw=2, marker="o", ms=5, mec=ev.SURFACE, mew=1.5,
            label="paired test (Diebold-Mariano)")
    a2.plot(df.n, df.power_unpaired, color=S2, lw=2, marker="o", ms=5, mec=ev.SURFACE, mew=1.5,
            label="unpaired test (ignores pairing)")
    a2.set_xscale("log")
    ev.style_axes(a2, "Test-set size n (log scale)", "Power at the 5% level",
                  "Detecting the Brier difference")
    a2.set_ylim(0, 1.02)
    a2.legend(frameon=False, fontsize=9, labelcolor=ev.INK2, loc="upper left")
    ticks = [500, 1000, 2000, 5000, 10000, CFG["n_test"]]
    for a in (a1, a2):
        a.set_xticks(ticks)
        a.set_xticklabels(["500", "1k", "2k", "5k", "10k", f"{CFG['n_test'] / 1000:.0f}k"])
        a.minorticks_off()
    fig.tight_layout()
    save(fig, "fig4_ranking_noise.png")


# ------------------------------------------------------------------------------------------
# S5  Regression functionals on ADR
# ------------------------------------------------------------------------------------------
def s5_regression(df):
    adr = df.loc[df.adr > 0, "adr"].values
    c = np.linspace(30, 200, 1701)
    mse = np.array([np.mean((adr - v) ** 2) for v in c])
    mae = np.array([np.mean(np.abs(adr - v)) for v in c])
    ape = np.array([np.mean(np.abs(adr - v) / adr) for v in c])
    tau = 0.9
    pin = np.array([np.mean(np.maximum(tau * (adr - v), (tau - 1) * (adr - v))) for v in c])

    def wmedian(x, w):
        o = np.argsort(x)
        cw = np.cumsum(w[o])
        return float(x[o][np.searchsorted(cw, 0.5 * cw[-1])])
    exact = {"mean": float(adr.mean()), "median": float(np.median(adr)),
             "(-1)-median": wmedian(adr, 1 / adr), "0.9-quantile": float(np.quantile(adr, 0.9))}
    grid = {"MSE": float(c[np.argmin(mse)]), "MAE": float(c[np.argmin(mae)]), "MAPE": float(c[np.argmin(ape)]),
            "pinball 0.9": float(c[np.argmin(pin)])}
    OUT["S5_functionals"] = {"exact": exact, "grid_minimisers": grid, "n_positive_adr": int(len(adr))}
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3), facecolor=ev.SURFACE)
    for ax, (name, e, key) in zip(axes, [("Squared error", mse, "mean"), ("Absolute error", mae, "median"),
                                         ("Absolute percentage error", ape, "(-1)-median"),
                                         ("Pinball loss, tau = 0.9", pin, "0.9-quantile")]):
        rel = e / e.min()
        ax.plot(c, rel, color=S1, lw=2)
        i = int(np.argmin(e))
        ax.plot(c[i], rel[i], "o", color=S1, ms=8, mec=ev.SURFACE, mew=2, zorder=5)
        left = key == "0.9-quantile"
        ax.text(0.03 if left else 0.97, 0.05, f"optimal constant\n{c[i]:.1f} EUR = {key}", transform=ax.transAxes,
                ha="left" if left else "right", va="bottom", fontsize=9, color=ev.INK2)
        ev.style_axes(ax, "Constant prediction (EUR)", "Loss / minimum loss" if ax is axes[0] else None, name)
        ax.set_ylim(0.95, 1.6)
    fig.tight_layout()
    save(fig, "fig5_regression_functionals.png")
    # influence of the single extreme ADR on test RMSE
    big = df.adr.max()
    OUT["S5_outlier"] = {"max_adr": float(big), "min_adr": float(df.adr.min()),
                         "n_zero": int((df.adr == 0).sum()),
                         "share_of_total_variance": float((big - df.adr.mean()) ** 2 / (len(df) * df.adr.var(ddof=0)))}


# ------------------------------------------------------------------------------------------
# S6  Data diagnostics (no models): leakage, duplicates, memorisation under different splits
# ------------------------------------------------------------------------------------------
def s6_data(df):
    """Diagnostics on the cleaned file. The file carries the group's split (random 70/15/15, stratified by
    is_canceled, identical rows kept together) and has had the leaking columns removed. The chronological split
    by arrival date (train to 30 Sep 2016 / validation Oct-Dec 2016 / test 2017) is rebuilt from the dates for
    comparison."""
    df = df.copy()
    df["arrival_date"] = pd.to_datetime(df["arrival_date"])
    y = df.is_canceled
    raw = [c for c in df.columns if c not in ("split",)]                 # a row's identity: every column but the split tag
    date_cols = ["arrival_date", "arrival_date_year", "arrival_date_month", "arrival_date_week_number"]
    feats = [c for c in raw if c not in date_cols + ["is_canceled"]]
    spl = df.split.values
    chron = np.select([df.arrival_date <= "2016-09-30", df.arrival_date <= "2016-12-31"], ["train", "validation"],
                      "test")

    def split_stats(labels):
        return {k: {"n": int(len(g)), "prevalence": float(g.is_canceled.mean()),
                    "first": str(g.arrival_date.min().date()), "last": str(g.arrival_date.max().date()),
                    "city_share": float((g.hotel == "City Hotel").mean()),
                    "prev_city": float(g.loc[g.hotel == "City Hotel", "is_canceled"].mean()),
                    "prev_resort": float(g.loc[g.hotel == "Resort Hotel", "is_canceled"].mean())}
                for k, g in df.groupby(labels)}
    res = {"n": int(len(df)), "prevalence": float(y.mean()),
           "prevalence_by_hotel": df.groupby("hotel").is_canceled.mean().round(4).to_dict(),
           "split": split_stats(spl), "chronological_split": split_stats(chron),
           "nonrefund_cancel_rate": float(df.loc[df.deposit_type == "Non Refund", "is_canceled"].mean()),
           "nonrefund_share": float((df.deposit_type == "Non Refund").mean())}
    dup = df.duplicated(subset=raw)
    res["exact_duplicate_rows"] = int(dup.sum())
    res["duplicate_share"] = float(dup.mean())
    res["duplicate_share_by_split"] = {k: float(v) for k, v in dup.groupby(df.split).mean().items()}
    sizes = df.groupby(raw).size()
    res["duplicate_block_max"] = int(sizes.max())
    res["duplicate_blocks_gt1"] = int((sizes > 1).sum())
    dedup = df.drop_duplicates(subset=raw)
    res["dedup_n"], res["dedup_prevalence"] = int(len(dedup)), float(dedup.is_canceled.mean())
    nr = df.deposit_type == "Non Refund"
    res["nonrefund_duplicate_share"] = float(df.loc[nr].duplicated(subset=raw, keep=False).mean())
    for seg in ("Groups", "Offline TA/TO"):
        sel = df.market_segment == seg
        res[f"dup_share_in_{seg}"] = float(df.loc[sel & dup].shape[0] / max(dup.sum(), 1))
    # Lookup key: every booking feature except the arrival-date columns, so a temporal split does not rule out
    # matches by construction. Blocks of rows identical in ALL columns define the groups for the grouped split.
    key = pd.util.hash_pandas_object(df[feats].astype(str), index=False).values
    block = pd.util.hash_pandas_object(df[raw].astype(str), index=False).values
    # Booking event: same hotel, arrival date, lead time (hence booking date), stay, channel, agent, country, deposit
    # and customer type. Most plausibly several rooms of one booking that differ in room type, guests or price, so
    # they are not identical rows and the file's split does not keep them together.
    event_cols = ["hotel", "arrival_date", "lead_time", "stays_in_weekend_nights", "stays_in_week_nights",
                  "market_segment", "distribution_channel", "agent", "country", "deposit_type", "customer_type"]
    event = pd.util.hash_pandas_object(df[event_cols].astype(str), index=False).values
    ev_size = pd.Series(event).map(pd.Series(event).value_counts()).values
    ev_pure = (y.groupby(event).transform("min") == y.groupby(event).transform("max")).values
    res["event_rows_in_multi_row_events"] = float(np.mean(ev_size > 1))
    res["event_same_outcome_share"] = float(ev_pure[ev_size > 1].mean())
    rng = np.random.default_rng(2026)
    perm = rng.permutation(len(df))
    cut = int(0.8 * len(df))

    def memoriser(tr, te, k=key):
        ytr, yte = y.values[tr], y.values[te]
        table = pd.Series(ytr).groupby(k[tr]).mean()
        base = ytr.mean()
        hit = np.isin(k[te], table.index.values)
        p = np.where(hit, pd.Series(k[te]).map(table).fillna(base).values, base)
        ss = ev.SortedScores(p)
        W1, W0 = ss.class_weights(yte.astype(float), np.ones(len(te)))
        b = ev.brier_terms(yte, p).mean()
        return {"n_train": int(len(tr)), "n_test": int(len(te)), "test_prevalence": float(yte.mean()),
                "match_rate": float(hit.mean()), "auc": ss.auc(W1, W0), "brier": float(b),
                "bss": float(1 - b / (yte.mean() * (1 - yte.mean())))}
    out = {"random 80/20": memoriser(perm[:cut], perm[cut:])}
    ug = np.unique(block)
    test_groups = set(np.random.default_rng(7).choice(ug, size=int(0.2 * len(ug)), replace=False))
    g = np.array([k in test_groups for k in block])
    out["grouped random (identical rows kept together)"] = memoriser(np.where(~g)[0], np.where(g)[0])
    out["file split (train + validation -> test)"] = memoriser(np.where(spl != "test")[0], np.where(spl == "test")[0])
    out["file split (train -> validation)"] = memoriser(np.where(spl == "train")[0], np.where(spl == "validation")[0])
    out["file split, booking-event key (train + validation -> test)"] = memoriser(
        np.where(spl != "test")[0], np.where(spl == "test")[0], k=event)
    out["file split, booking-event key (train -> validation)"] = memoriser(
        np.where(spl == "train")[0], np.where(spl == "validation")[0], k=event)
    out["chronological (train + validation -> test)"] = memoriser(np.where(chron != "test")[0],
                                                                  np.where(chron == "test")[0])
    out["chronological (train -> validation)"] = memoriser(np.where(chron == "train")[0],
                                                           np.where(chron == "validation")[0])
    res["memoriser"] = out
    # the same memoriser, scored on the de-duplicated test rows only
    te_idx = np.where((spl == "test") & ~dup.values)[0]
    res["memoriser_test_dedup"] = memoriser(np.where(spl != "test")[0], te_idx)
    OUT["S6_data"] = res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="hotel_bookings_clean.csv (the group's cleaned file with a `split` column) - enables S5, S6 and takes sizes/rates from the data")
    ap.add_argument("--quick", action="store_true", help="fewer replications in S4")
    args = ap.parse_args()
    df = None
    if args.data:
        df = pd.read_csv(args.data)
        missing = {"split", "is_canceled", "arrival_date", "hotel"} - set(df.columns)
        if missing:
            raise SystemExit(f"{args.data} does not look like the cleaned file: missing columns {sorted(missing)}")
        configure(df)
    else:
        OUT["config"] = CFG
    s1_propriety()
    s2_prevalence()
    s3_worked_example()
    s4_ranking_noise(R=200 if args.quick else 1000)
    if df is not None:
        s5_regression(df)
        s6_data(df)
    with open("sim_results.json", "w") as fh:
        json.dump(OUT, fh, indent=1, default=float)
    print("wrote sim_results.json")


if __name__ == "__main__":
    main()
