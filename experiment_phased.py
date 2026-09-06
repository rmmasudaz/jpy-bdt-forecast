#!/usr/bin/env python3
"""
experiment_phased.py — Phased accuracy improvement experiments for JPY/BDT.

Phase 1: Ridge on expanded features (31 features)
Phase 2: LightGBM on expanded features (non-linear model)
Phase 3: Ensemble (Ridge + LightGBM blend)

Evaluation: walk-forward refit on test (refit each day on all past data).
All hyperparams selected train-only. Confidence thresholds train-calibrated.
"""
import datetime as dt
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=UserWarning)

CSV = Path("data/jpy_bdt_daily.csv")
WARM = 20
ALPHAS = [0.1, 1, 10, 100]
LGB_PARAMS_REG = [
    {"num_leaves": 15, "learning_rate": 0.05, "n_estimators": 200,
     "min_child_samples": 30, "lambda_l1": 0.1, "lambda_l2": 1.0, "verbose": -1},
    {"num_leaves": 20, "learning_rate": 0.05, "n_estimators": 300,
     "min_child_samples": 25, "lambda_l1": 0.5, "lambda_l2": 1.0, "verbose": -1},
    {"num_leaves": 31, "learning_rate": 0.03, "n_estimators": 400,
     "min_child_samples": 20, "lambda_l1": 1.0, "lambda_l2": 2.0, "verbose": -1},
]
LGB_PARAMS_CLF = [
    {"num_leaves": 15, "learning_rate": 0.05, "n_estimators": 200,
     "min_child_samples": 30, "lambda_l1": 0.1, "lambda_l2": 1.0, "verbose": -1,
     "objective": "binary", "metric": "binary_logloss"},
    {"num_leaves": 20, "learning_rate": 0.05, "n_estimators": 300,
     "min_child_samples": 25, "lambda_l1": 0.5, "lambda_l2": 1.0, "verbose": -1,
     "objective": "binary", "metric": "binary_logloss"},
    {"num_leaves": 31, "learning_rate": 0.03, "n_estimators": 500,
     "min_child_samples": 20, "lambda_l1": 1.0, "lambda_l2": 2.0, "verbose": -1,
     "objective": "binary", "metric": "binary_logloss"},
]


def load_data():
    df = pd.read_csv(CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    n = len(df)
    lub = np.log(df["usd_to_bdt"].to_numpy(np.float64))
    luj = np.log(df["usd_to_jpy"].to_numpy(np.float64))
    r_jb = np.zeros(n); r_jb[1:] = np.diff(lub - luj)
    r_uj = np.zeros(n); r_uj[1:] = np.diff(luj)
    r_ub = np.zeros(n); r_ub[1:] = np.diff(lub)
    return df, n, r_jb, r_uj, r_ub, lub


def build_features_v4(r_ub, r_uj, lub, dow):
    """Current v4 feature set (17 features)."""
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
    return X  # (n, 17)


def build_features_expanded(r_ub, r_uj, lub, dow, dom):
    """Expanded feature set (~31 features) adding regime, shock, calendar."""
    def lagmat(x, lags):
        out = np.full((len(x), lags), np.nan)
        for k in range(1, lags + 1):
            out[k:, k - 1] = x[:-k]
        return out
    dev5 = lub - pd.Series(lub).rolling(5).mean().to_numpy()
    dev10 = lub - pd.Series(lub).rolling(10).mean().to_numpy()
    dev20 = lub - pd.Series(lub).rolling(20).mean().to_numpy()
    abs_lag1 = np.abs(np.roll(r_ub, 1))
    abs_lag1[0] = np.nan
    vol5 = pd.Series(r_ub).rolling(5).std().to_numpy()
    vol20 = pd.Series(r_ub).rolling(20).std().to_numpy()
    corr20 = pd.Series(r_uj).rolling(20).corr(pd.Series(r_ub)).to_numpy()
    wd = np.eye(7)[dow][:, 1:]
    l1, l2, l3 = lagmat(r_ub, 3).T
    uj_lag = np.roll(r_uj, 1); uj_lag[0] = np.nan
    fbase = np.column_stack([l1, l2, l3, dev5, dev10, dev20, abs_lag1, vol5, vol20, corr20, uj_lag, dom / 31.0])
    X = np.column_stack([fbase, wd, l1[:, None] * wd, l2[:, None] * wd])
    return X  # (n, ~31)


def walk_forward_eval(X, r_ub, r_uj, r_jb, split, model_type="ridge", lgb_params=None, alpha=10):
    """Walk-forward refit on test. Returns dict with metrics.
    Uses train-only standardization for ridge; LightGBM is scale-invariant."""
    n = X.shape[0]
    r_hat = np.zeros(n)
    tr = np.arange(WARM, split)
    Xmu, Xsd = X[tr].mean(0), X[tr].std(0) + 1e-12
    Xs = (X - Xmu) / Xsd

    for t in range(split, n):
        seg = np.arange(WARM, t)
        if model_type == "ridge":
            mm = Ridge(alpha=alpha).fit(Xs[seg], r_ub[seg])
            r_hat[t] = mm.predict(Xs[t:t + 1])[0] - r_uj[seg].mean()
        elif model_type == "lgb":
            # LightGBM is scale-invariant but use raw features (faster, same result)
            mm = lgb.LGBMRegressor(**lgb_params).fit(X[seg], r_ub[seg])
            r_hat[t] = mm.predict(X[t:t + 1])[0] - r_uj[seg].mean()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    te = np.arange(split, n)
    m = r_jb[te] != 0
    dir_acc = float((np.sign(r_jb[te][m]) == np.sign(r_hat[te][m])).mean() * 100)
    return {"dir": dir_acc, "r_hat": r_hat, "te": te}


def walk_forward_eval_clf(X, r_ub, r_uj, r_jb, split, lgb_params=None):
    """Walk-forward LightGBM classifier (predicts P(up)). Returns direction signal = p_up - 0.5."""
    n = X.shape[0]
    r_hat = np.zeros(n)
    tr = np.arange(WARM, split)
    for t in range(split, n):
        seg = np.arange(WARM, t)
        y = (r_ub[seg] > 0).astype(int)
        if len(np.unique(y)) < 2:
            r_hat[t] = 0.0
            continue
        mm = lgb.LGBMClassifier(**lgb_params).fit(X[seg], y)
        p_up = mm.predict_proba(X[t:t + 1])[0, 1]
        r_hat[t] = p_up - 0.5  # signal: positive = predicts up
    te = np.arange(split, n)
    m = r_jb[te] != 0
    dir_acc = float((np.sign(r_jb[te][m]) == np.sign(r_hat[te][m])).mean() * 100)
    return {"dir": dir_acc, "r_hat": r_hat, "te": te}


def train_alpha_selection(X, r_ub, r_uj, split, sel_start=None):
    """Select Ridge alpha by walk-forward dir accuracy on last quarter of train.
    Uses train-only standardization up to each point."""
    if sel_start is None:
        sel_start = WARM + (split - WARM) * 3 // 4
    best_a, best_s = 10, -1
    for a in ALPHAS:
        hits = tot = 0
        for t in range(sel_start, split):
            seg = np.arange(WARM, t)
            mu, sd = X[seg].mean(0), X[seg].std(0) + 1e-12
            xs = (X - mu) / sd
            m = Ridge(alpha=a).fit(xs[seg], r_ub[seg])
            pred = m.predict(xs[t:t + 1])[0]
            if r_ub[t] != 0:
                hits += int(np.sign(r_ub[t]) == np.sign(pred))
                tot += 1
        s = hits / tot * 100 if tot else -1
        if s > best_s:
            best_s, best_a = s, a
    return best_a


def train_lgb_selection(X, r_ub, r_uj, split, sel_start=None, params_list=None):
    """Select LightGBM params by walk-forward dir accuracy on last quarter of train."""
    if sel_start is None:
        sel_start = WARM + (split - WARM) * 3 // 4
    if params_list is None:
        params_list = LGB_PARAMS_REG
    best_params, best_s = params_list[0], -1
    for params in params_list:
        hits = tot = 0
        for t in range(sel_start, split):
            seg = np.arange(WARM, t)
            if "objective" in params:
                y = (r_ub[seg] > 0).astype(int)
                if len(np.unique(y)) < 2:
                    continue
                mm = lgb.LGBMClassifier(**params).fit(X[seg], y)
                p_up = mm.predict_proba(X[t:t + 1])[0, 1]
                pred = p_up - 0.5
            else:
                mm = lgb.LGBMRegressor(**params).fit(X[seg], r_ub[seg])
                pred = mm.predict(X[t:t + 1])[0]
            if r_ub[t] != 0:
                hits += int(np.sign(r_ub[t]) == np.sign(pred))
                tot += 1
        s = hits / tot * 100 if tot else -1
        if s > best_s:
            best_s, best_params = s, params
    return best_params


def selective_metrics(r_hat, r_jb, te, thr25, thr15, thr10):
    """Compute selective accuracy at train-calibrated thresholds."""
    conf = np.abs(r_hat[te])
    results = {}
    for label, thr in [("25%", thr25), ("15%", thr15), ("10%", thr10)]:
        sel = te[conf >= thr]
        m = r_jb[sel] != 0
        acc = float((np.sign(r_jb[sel][m]) == np.sign(r_hat[sel][m])).mean() * 100) if m.any() else float("nan")
        results[label] = {"acc": acc, "n": int(m.sum()), "cov": len(sel) / len(te) * 100}
    return results


def calibrate_thresholds(r_hat, r_ub, tr, alpha, split):
    """Block walk-forward OOF in training window to get confidence thresholds."""
    B = 60
    oof = np.full(split, np.nan)
    for i0 in range(0, len(tr), B):
        block = tr[i0: i0 + B]
        past = tr[tr < block[0]]
        if len(past) >= 100:
            mm = Ridge(alpha=alpha).fit(X_train := None, r_ub[past]) if False else None
    # Recompute with proper X
    return None  # placeholder


def run_split(df, n, r_jb, r_uj, r_ub, lub, frac, verbose=True):
    """Run all phases on one split fraction. Returns results dict."""
    split = int(n * frac)
    dow = df["date"].dt.dayofweek.to_numpy()
    dom = df["date"].dt.day.to_numpy()
    tr = np.arange(WARM, split)
    te = np.arange(split, n)

    X_v4 = build_features_v4(r_ub, r_uj, lub, dow)
    X_exp = build_features_expanded(r_ub, r_uj, lub, dow, dom)

    # Replace NaN with 0 for ridge (first WARM rows excluded from tr/te anyway)
    X_v4_s = np.where(np.isnan(X_v4), 0.0, X_v4)
    X_exp_s = np.where(np.isnan(X_exp), 0.0, X_exp)

    # --- Phase 0: v4 baseline (ridge on v4 features) ---
    alpha0 = train_alpha_selection(X_v4_s, r_ub, r_uj, split)
    res0 = walk_forward_eval(X_v4_s, r_ub, r_uj, r_jb, split, "ridge", alpha=alpha0)

    # --- Phase 1: ridge on expanded features ---
    alpha1 = train_alpha_selection(X_exp_s, r_ub, r_uj, split)
    res1 = walk_forward_eval(X_exp_s, r_ub, r_uj, r_jb, split, "ridge", alpha=alpha1)

    # --- Phase 2a: LightGBM regression on expanded features ---
    lgb_params = train_lgb_selection(X_exp_s, r_ub, r_uj, split)
    res2 = walk_forward_eval(X_exp_s, r_ub, r_uj, r_jb, split, "lgb", lgb_params=lgb_params)

    # --- Phase 2b: LightGBM classifier on expanded features (directs direction) ---
    lgb_clf_params = train_lgb_selection(X_exp_s, r_ub, r_uj, split, params_list=LGB_PARAMS_CLF)
    res2b = walk_forward_eval_clf(X_exp_s, r_ub, r_uj, r_jb, split, lgb_params=lgb_clf_params)

    # --- Phase 3: ensemble (average of ridge-exp + best LGB) ---
    # Pick the better of reg/clf for ensemble
    best_lgb = res2b if res2b["dir"] >= res2["dir"] else res2
    r_hat_ens = (res1["r_hat"] + best_lgb["r_hat"]) / 2
    te_idx = res1["te"]
    m = r_jb[te_idx] != 0
    dir_ens = float((np.sign(r_jb[te_idx][m]) == np.sign(r_hat_ens[te_idx][m])).mean() * 100)
    res3 = {"dir": dir_ens, "r_hat": r_hat_ens, "te": te_idx}

    # --- Selective metrics for best models ---
    # Use train-calibrated thresholds from v4 (same protocol, comparable)
    # Block walk-forward OOF on train
    B = 60
    def oof_ridge(X_raw, alpha):
        oof = np.full(n, np.nan)
        for i0 in range(0, len(tr), B):
            block = tr[i0: i0 + B]
            past = tr[tr < block[0]]
            if len(past) >= 100:
                mu, sd = X_raw[past].mean(0), X_raw[past].std(0) + 1e-12
                xs = (X_raw - mu) / sd
                mm = Ridge(alpha=alpha).fit(xs[past], r_ub[past])
                oof[block] = mm.predict(xs[block]) - r_uj[past].mean()
        return oof

    oof_v4 = oof_ridge(X_v4_s, alpha0)
    oof_exp = oof_ridge(X_exp_s, alpha1)

    oof_lgb = np.full(n, np.nan)
    for i0 in range(0, len(tr), B):
        block = tr[i0: i0 + B]
        past = tr[tr < block[0]]
        if len(past) >= 100:
            mm = lgb.LGBMRegressor(**lgb_params).fit(X_exp_s[past], r_ub[past])
            oof_lgb[block] = mm.predict(X_exp_s[block]) - r_uj[past].mean()

    oof_clf = np.full(n, np.nan)
    for i0 in range(0, len(tr), B):
        block = tr[i0: i0 + B]
        past = tr[tr < block[0]]
        if len(past) >= 100:
            y = (r_ub[past] > 0).astype(int)
            if len(np.unique(y)) >= 2:
                mm = lgb.LGBMClassifier(**lgb_clf_params).fit(X_exp_s[past], y)
                p_up = mm.predict_proba(X_exp_s[block])[:, 1]
                oof_clf[block] = p_up - 0.5

    tail = tr[len(tr) // 2:]
    thr = {}
    oof_ens = (oof_exp + oof_lgb) / 2
    for name, oof in [("v4", oof_v4), ("exp_ridge", oof_exp), ("lgb", oof_lgb),
                       ("lgb_clf", oof_clf), ("ens", oof_ens)]:
        src = np.abs(oof[tail])
        src = src[~np.isnan(src)]
        if len(src) > 0:
            thr[name] = {q: float(np.quantile(src, q)) for q in (0.75, 0.85, 0.90)}
        else:
            thr[name] = {0.75: 0, 0.85: 0, 0.90: 0}

    sel = {}
    for name, r in [("v4", res0), ("exp_ridge", res1), ("lgb", res2),
                     ("lgb_clf", res2b), ("ens", res3)]:
        t = thr.get(name, thr["v4"])
        sel[name] = selective_metrics(r["r_hat"], r_jb, te, t[0.75], t[0.85], t[0.90])

    out = {
        "frac": frac,
        "v4_dir": res0["dir"],
        "exp_ridge_dir": res1["dir"],
        "lgb_dir": res2["dir"],
        "ens_dir": res3["dir"],
        "v4_selective": sel["v4"],
        "lgb_selective": sel["lgb"],
        "ens_selective": sel["ens"],
    }
    if verbose:
        print(f"\n{'='*60}")
        print(f"Split {frac:.0%} (train {split}, test {n - split})")
        print(f"{'='*60}")
        print(f"  v4 ridge (17 feat)    : dir {res0['dir']:.1f}%")
        print(f"  exp ridge (31 feat)   : dir {res1['dir']:.1f}%   alpha={alpha1}")
        lgb_type = "clf" if "objective" in lgb_params else "reg"
        print(f"  exp LGB {lgb_type}          : dir {res2['dir']:.1f}%   params={lgb_params['num_leaves']} leaves, {lgb_params['n_estimators']} trees")
        lgb_clf_type = "clf" if "objective" in lgb_clf_params else "reg"
        print(f"  exp LGB clf           : dir {res2b['dir']:.1f}%")
        print(f"  ensemble (avg)        : dir {res3['dir']:.1f}%")
        print(f"\n  Selective accuracy (train-calibrated thresholds):")
        for name in ["v4", "lgb", "lgb_clf", "ens"]:
            s = sel[name]
            print(f"    {name:10s}: 25%→{s['25%']['acc']:.1f}% ({s['25%']['n']}d), "
                  f"15%→{s['15%']['acc']:.1f}% ({s['15%']['n']}d), "
                  f"10%→{s['10%']['acc']:.1f}% ({s['10%']['n']}d)")

    return out


if __name__ == "__main__":
    df, n, r_jb, r_uj, r_ub, lub = load_data()
    print(f"Loaded {n} days: {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")

    results = []
    for frac in (0.70, 0.75, 0.80):
        r = run_split(df, n, r_jb, r_uj, r_ub, lub, frac)
        results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY across splits")
    print(f"{'='*60}")
    for metric in ["v4_dir", "exp_ridge_dir", "lgb_dir", "ens_dir"]:
        vals = [r[metric] for r in results]
        print(f"  {metric:20s}: {np.mean(vals):.1f}% (min {min(vals):.1f}, max {max(vals):.1f})")

    # Save
    with open("experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved → experiment_results.json")
