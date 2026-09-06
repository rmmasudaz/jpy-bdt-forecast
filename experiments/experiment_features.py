#!/usr/bin/env python3
"""
experiment_features.py — Does adding features actually improve the LSTM?

Compares feature sets on the SAME chronological 80/20 split:

    A  baseline : [r, |r|]                          (the original model)
    B  +USDlegs : A + [USD/JPY r, USD/BDT r]        (cross-rate decomposition)
    C  +vol/mom : B + [rv5, mom5, mom20]            (realised vol & momentum)
    D  +dow     : C + [dow_sin, dow_cos]            (day-of-week seasonality)

The JPY/BDT rate factors EXACTLY as  USD/BDT / USD/JPY, so the two USD legs are
economically meaningful inputs, not just extra noise:

    ln(JPY/BDT)_t - ln(JPY/BDT)_{t-1} = ln(USD/BDT)_t - ln(USD/JPY)_t

Bangladesh manages the Taka against the USD (gradual adjustments -> persistent
returns), while JPY/USD is the volatile market leg. If those components carry
any predictable autocorrelation, the LSTM should exploit it.

Every config gets identical treatment: same split, same architecture, same
validation magnitude-calibration. Only the columns differ. Optional ensemble
over `--seeds` models averages the standardised-return predictions.

Usage:
    python3 experiment_features.py [--csv data/jpy_bdt_daily.csv] [--seq-len 30] [--seeds 3]
"""

from __future__ import annotations

import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Input, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, LSTM


def build_model(seq_len: int, n_features: int, seed: int) -> Sequential:
    tf.keras.utils.set_random_seed(seed)
    model = Sequential(
        [
            Input(shape=(seq_len, n_features)),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="huber")
    return model


def make_windows(feat: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(seq_len, len(feat)):
        X.append(feat[i - seq_len : i])
        y.append(feat[i, 0])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def fit_model(model, X_tr, y_tr, X_val, y_val, epochs=100):
    cbs = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
    ]
    return model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=epochs,
                     batch_size=32, callbacks=cbs, verbose=0)


def metrics_levels(yt, yp):
    yt, yp = np.asarray(yt, float), np.asarray(yp, float)
    return dict(
        mae=float(np.mean(np.abs(yt - yp))),
        rmse=float(np.sqrt(np.mean((yt - yp) ** 2))),
        mape=float(np.mean(np.abs((yt - yp) / yt)) * 100.0),
    )


def directional_accuracy(yt, yp):
    t, p = np.sign(np.asarray(yt)), np.sign(np.asarray(yp))
    m = t != 0
    return float((t[m] == p[m]).mean() * 100.0) if m.any() else float("nan")


def train_ensemble(feat, rets, rate, seq_len, n_seeds):
    """Train `n_seeds` models on the same split; return list of models + scaler + indices."""
    n = len(rate)
    split = int(n * 0.8)

    scaler = StandardScaler().fit(feat[:split])
    feat_s = scaler.transform(feat).astype(np.float32)

    X, y = make_windows(feat_s, seq_len)
    y_idx = np.arange(seq_len, n)
    train_mask = y_idx < split
    test_mask = ~train_mask
    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask], y[test_mask]

    val_cut = int(len(X_tr) * 0.88)
    X_fit, y_fit = X_tr[:val_cut], y_tr[:val_cut]
    X_val, y_val = X_tr[val_cut:], y_tr[val_cut:]

    models = []
    for s in range(n_seeds):
        m = build_model(seq_len, feat.shape[1], seed=1000 + s)
        fit_model(m, X_fit, y_fit, X_val, y_val)
        models.append(m)

    return {
        "models": models, "scaler": scaler, "feat_s": feat_s,
        "y_idx": y_idx, "train_mask": train_mask, "test_mask": test_mask,
        "val_cut": val_cut, "split": split,
    }


def ensemble_predict(models, X) -> np.ndarray:
    preds = np.stack([m.predict(X, verbose=0).ravel() for m in models])
    return preds.mean(axis=0)


def evaluate(feat, rets, rate, seq_len, n_seeds):
    n = len(rate)
    split = int(n * 0.8)
    env = train_ensemble(feat, rets, rate, seq_len, n_seeds)

    # validation rows indices
    vi = env["y_idx"][env["train_mask"]][env["val_cut"]:]
    # build windows for validation targets (features strictly before target day)
    X_val_rows = np.stack([env["feat_s"][i - seq_len : i] for i in vi])
    rv = ensemble_predict(env["models"], X_val_rows)

    rmean, rstd = float(rets.mean()), float(rets.std(ddof=1))
    best_lam, best_mae = 1.0, float("inf")
    for lam in np.arange(0.0, 3.01, 0.1):
        rhat = rv * lam * rstd + rmean
        lvl = rate[vi - 1] * np.exp(rhat)
        m = float(np.mean(np.abs(rate[vi] - lvl)))
        if m < best_mae:
            best_mae, best_lam = m, lam

    # 1-step walk-forward on the test set
    X_te = np.stack([env["feat_s"][i - seq_len : i] for i in env["y_idx"][env["test_mask"]]])
    r_hat_std = ensemble_predict(env["models"], X_te)
    r_hat = r_hat_std * best_lam * rstd + rmean
    te_idx = env["y_idx"][env["test_mask"]]
    actual_levels = rate[te_idx]
    prev_levels = rate[te_idx - 1]
    pred_levels = prev_levels * np.exp(r_hat)

    m_lstm = metrics_levels(actual_levels, pred_levels)
    m_base = metrics_levels(actual_levels, prev_levels)
    dir_acc = directional_accuracy(rets[te_idx], r_hat_std)

    # 30-day recursive backtest: each model recurses its own state, averaged per step
    bt_steps = min(30, n - split)
    windows = [env["feat_s"][split - seq_len : split].copy() for _ in env["models"]]
    bt_std = np.empty(bt_steps)
    for i in range(bt_steps):
        r_step = np.mean([
            float(m.predict(np.ascontiguousarray(w[np.newaxis, ...], dtype=np.float32), verbose=0)[0, 0])
            for m, w in zip(env["models"], windows)
        ])
        bt_std[i] = r_step
        for j, w in enumerate(windows):
            w = np.roll(w, -1, axis=0)
            w[-1, 0] = r_step
            w[-1, 1] = abs(r_step)
            windows[j] = w
    bt_rets = bt_std * best_lam * rstd + rmean
    bt_levels = rate[split - 1] * np.exp(np.cumsum(bt_rets))
    bt_actual = rate[split : split + bt_steps]
    m_bt = metrics_levels(bt_actual, bt_levels)

    return {
        "mae": m_lstm["mae"], "rmse": m_lstm["rmse"], "mape": m_lstm["mape"],
        "dir_acc": dir_acc, "lam": best_lam,
        "baseline_mae": m_base["mae"], "bt_mae": m_bt["mae"],
    }


def build_feature_sets(df: pd.DataFrame) -> dict[str, np.ndarray]:
    rate = df["jpy_to_bdt"].to_numpy(float)
    logr = np.log(rate)
    ret = np.zeros(len(rate)); ret[1:] = logr[1:] - logr[:-1]

    ruj = np.zeros(len(rate)); ruj[1:] = np.diff(np.log(df["usd_to_jpy"].to_numpy(float)))
    rub = np.zeros(len(rate)); rub[1:] = np.diff(np.log(df["usd_to_bdt"].to_numpy(float)))

    s = pd.Series(ret)
    rv5 = s.rolling(5, min_periods=1).std().fillna(0.0).to_numpy()
    mom5 = s.rolling(5, min_periods=1).sum().to_numpy()
    mom20 = s.rolling(20, min_periods=1).sum().to_numpy()

    dow = df["date"].dt.dayofweek.to_numpy(float)
    dow_sin = np.sin(2 * np.pi * dow / 7.0)
    dow_cos = np.cos(2 * np.pi * dow / 7.0)

    absr = np.abs(ret)
    return {
        "A baseline": np.column_stack([ret, absr]),
        "B +USDlegs": np.column_stack([ret, absr, ruj, rub]),
        "C +vol/mom": np.column_stack([ret, absr, ruj, rub, rv5, mom5, mom20]),
        "D +dow": np.column_stack([ret, absr, ruj, rub, rv5, mom5, mom20, dow_sin, dow_cos]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/jpy_bdt_daily.csv")
    ap.add_argument("--seq-len", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--only", default=None, help="e.g. B to run one config")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    rate = df["jpy_to_bdt"].to_numpy(float)
    logr = np.log(rate)
    ret = np.zeros(len(rate)); ret[1:] = logr[1:] - logr[:-1]
    print(f"Loaded {len(rate)} rows  {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}  "
          f"(ensemble seeds = {args.seeds})")

    sets = build_feature_sets(df)
    if args.only:
        key = next(k for k in sets if k.startswith(args.only))
        sets = {key: sets[key]}

    results = {}
    for name, feat in sets.items():
        r = evaluate(feat, ret, rate, args.seq_len, args.seeds)
        results[name] = r
        print(f"\n[{name}]  n_features={feat.shape[1]}")
        print(f"  1-step MAE {r['mae']:.5f}  RMSE {r['rmse']:.5f}  MAPE {r['mape']:.3f}%  "
              f"dir_acc {r['dir_acc']:.1f}%  lambda {r['lam']:.1f}")
        print(f"  baseline MAE {r['baseline_mae']:.5f} | 30d backtest MAE {r['bt_mae']:.5f}")

    print("\n=== Summary (sorted by directional accuracy) ===")
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["dir_acc"], reverse=True):
        print(f"  {name:<12} dir {r['dir_acc']:5.1f}%  MAE {r['mae']:.5f}  "
              f"vs base {100*(r['baseline_mae']-r['mae'])/r['baseline_mae']:+5.2f}%  backtest {r['bt_mae']:.5f}")


if __name__ == "__main__":
    main()
