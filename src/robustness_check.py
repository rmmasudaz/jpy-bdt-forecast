#!/usr/bin/env python3
"""
robustness_check.py — Validate v4 model robustness across train/test splits.

For split fractions 0.70 / 0.75 / 0.80, evaluates:
  v4  weekday-conditional ridge (17 features, walk-forward refit)
  persistence  "no change" baseline

Directional accuracy scored on signed predictions.
Walk-forward refit: at each test day, model is refit on all data before it.
Alpha selected by walk-forward on last quarter of training data.

Usage: python3 robustness_check.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

WARM = 20
ALPHAS = [0.1, 1, 10, 100]


def build_features(r_ub, r_uj, lub, dow):
    def lagmat(x, lags):
        out = np.full((len(x), lags), np.nan)
        for k in range(1, lags + 1):
            out[k:, k - 1] = x[:-k]
        return out
    dev10 = lub - pd.Series(lub).rolling(10).mean().to_numpy()
    dev20 = lub - pd.Series(lub).rolling(20).mean().to_numpy()
    wd = np.eye(7)[dow][:, 1:]
    fbase = np.column_stack([lagmat(r_ub, 3), dev10, dev20, np.roll(r_uj, 1)])
    X = np.column_stack([fbase, wd, fbase[:, [0]] * wd])
    return X


def dacc(actual, pred):
    t, p = np.sign(actual), np.sign(pred)
    m = t != 0
    return float((t[m] == p[m]).mean() * 100), int(m.sum())


def main():
    df = pd.read_csv("data/jpy_bdt_daily.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    n = len(df)
    lub = np.log(df["usd_to_bdt"].to_numpy(float))
    r_jb = np.zeros(n); r_jb[1:] = np.diff(lub)
    r_uj = np.zeros(n); r_uj[1:] = np.diff(np.log(df["usd_to_jpy"].to_numpy(float)))
    r_ub = np.zeros(n); r_ub[1:] = np.diff(lub)
    dow = df["date"].dt.dayofweek.to_numpy()
    X = build_features(r_ub, r_uj, lub, dow)

    print("v4 robustness check: weekday-conditional ridge across splits")
    print("=" * 65)
    print(f"  {'split':>6}  {'v4 dir%':>8}  {'v3 reversal':>11}  {'v4 n':>5}")
    print("-" * 65)

    for frac in [0.70, 0.75, 0.80]:
        split = int(n * frac)
        tr = np.arange(WARM, split)
        te = np.arange(split, n)

        # select alpha by walk-forward on last quarter of train (fixed scaler)
        mu_tr, sd_tr = X[tr].mean(0), X[tr].std(0) + 1e-12
        Xs_tr = np.where(np.isnan(X), 0.0, (X - mu_tr) / sd_tr)
        sel_start = WARM + (split - WARM) * 3 // 4
        best_a, best_s = 10, -1
        for a in ALPHAS:
            hits = tot = 0
            for t in range(sel_start, split):
                seg = np.arange(WARM, t)
                m = Ridge(alpha=a).fit(Xs_tr[seg], r_ub[seg])
                pred = m.predict(Xs_tr[t:t + 1])[0]
                if r_ub[t] != 0:
                    hits += int(np.sign(r_ub[t]) == np.sign(pred))
                    tot += 1
            s = hits / tot * 100 if tot else -1
            if s > best_s:
                best_s, best_a = s, a

        # walk-forward refit on test (fixed train scaler, matching pipeline)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        Xs = np.where(np.isnan(X), 0.0, (X - mu) / sd)
        r_hat = np.zeros(n)
        for t in te:
            seg = np.arange(WARM, t)
            mm = Ridge(alpha=best_a).fit(Xs[seg], r_ub[seg])
            r_hat[t] = mm.predict(Xs[t:t + 1])[0] - r_uj[seg].mean()

        dir_v4, n_v4 = dacc(r_jb[te], r_hat[te])
        # baseline: predict reversal (r_hat = -r_ub_lag), same as v3
        dir_rev, _ = dacc(r_jb[te], -r_ub[te - 1])
        up_freq = (r_jb[te] > 0).sum() / (r_jb[te] != 0).sum() * 100

        print(f"  {frac:>6.0%}  {dir_v4:>7.1f}%  {dir_rev:>10.1f}%  {n_v4:>5}   (up-freq {up_freq:.0f}%)")

    print("=" * 65)


if __name__ == "__main__":
    main()
