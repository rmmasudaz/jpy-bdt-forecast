#!/usr/bin/env python3
"""
experiment_hybrid.py — Exploit the cross-rate decomposition for higher accuracy.

Key facts established in analysis:
  * JPY/BDT return  ==  USD/BDT return  -  USD/JPY return   (EXACT, to 1e-6)
  * USD/BDT is a managed rate and MEAN-REVERTS  (lag-1 autocorr ≈ -0.26)
      =>  sign(-r_ub_lag) predicts next-day JPY/BDT direction ≈ 56% (train) / 62% (test)
  * USD/JPY is a free-floating rate ≈ random walk

Models compared on the same 80/20 split (3-seed ensembles unless trivial):

  M1  LSTM direct on JPY/BDT         (current v2: features r,|r|,r_uj,r_ub)
  M2  mean-reversion signal          r_hat = -c * r_ub_lag
  M3  LSTM + explicit mean-rev feat  (adds -r_ub_lag as an input column)
  M4  LSTM on the taka leg           target = r_ub, combine r_hat_jb = r_hat_ub - mu_uj

Lambda (return-magnitude scale) is always chosen on the validation slice of the
training set — never on the test set.

Usage:  python3 experiment_hybrid.py [--seeds 3]
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

SEED = 42


def build_model(seq_len: int, n_features: int, seed: int) -> Sequential:
    tf.keras.utils.set_random_seed(seed)
    model = Sequential([
        Input(shape=(seq_len, n_features)),
        LSTM(64, return_sequences=True), Dropout(0.2),
        LSTM(32), Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="huber")
    return model


def make_windows(feat, seq_len, target):
    X, y = [], []
    for i in range(seq_len, len(feat)):
        X.append(feat[i - seq_len:i])
        y.append(target[i])
    return np.asarray(X, np.float32), np.asarray(y, np.float32)


def fit_model(model, X_tr, y_tr, X_val, y_val, epochs=100):
    cbs = [EarlyStopping("val_loss", patience=8, restore_best_weights=True),
           ReduceLROnPlateau("val_loss", factor=0.5, patience=4, min_lr=1e-5)]
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=epochs,
              batch_size=32, callbacks=cbs, verbose=0)


def ens_predict(models, X):
    return np.stack([m.predict(X, verbose=0).ravel() for m in models]).mean(0)


def dacc(actual, pred):
    t, p = np.sign(actual), np.sign(pred)
    m = t != 0
    return float((t[m] == p[m]).mean() * 100), int(m.sum())


def calibrate_on_val(pred_val, out_mean, out_std, vi, rate):
    """lambda in standardized-prediction space, minimizing val 1-step level MAE."""
    best_lam, best_mae = 1.0, float("inf")
    for lam in np.arange(0.0, 3.01, 0.1):
        rhat = pred_val * lam * out_std + out_mean
        m = np.mean(np.abs(rate[vi] - rate[vi - 1] * np.exp(rhat)))
        if m < best_mae:
            best_mae, best_lam = m, lam
    return best_lam, best_mae


def run_lstm(feat, target, split, seq_len, n_seeds, out_mean, out_std, rate):
    """Train ensemble; calibrate lambda on validation; return standardized and
    lambda-scaled raw test predictions."""
    scaler = StandardScaler().fit(feat[:split])
    fs = scaler.transform(feat).astype(np.float32)
    X, y = make_windows(fs, seq_len, target)
    y_idx = np.arange(seq_len, len(target))
    tr = y_idx < split
    X_tr, y_tr = X[tr], y[tr]
    val_cut = int(len(X_tr) * 0.88)

    models = []
    for s in range(n_seeds):
        m = build_model(seq_len, feat.shape[1], SEED + s)
        fit_model(m, X_tr[:val_cut], y_tr[:val_cut], X_tr[val_cut:], y_tr[val_cut:])
        models.append(m)

    # validation windows for lambda calibration
    vi = y_idx[tr][val_cut:]
    X_val = np.stack([fs[i - seq_len:i] for i in vi])
    lam, _ = calibrate_on_val(ens_predict(models, X_val), out_mean, out_std, vi, rate)

    # test predictions
    te_idx = y_idx[~tr]
    X_te = np.stack([fs[i - seq_len:i] for i in te_idx])
    r_hat_std = ens_predict(models, X_te)
    r_hat = r_hat_std * lam * out_std + out_mean
    return r_hat_std, r_hat, te_idx


def report(name, r_hat_std, r_hat_raw, r_jb, te_idx, rate):
    """Direction is scored on the model's SIGNED output (r_hat_std); MAE on the
    lambda-scaled level forecast (r_hat_raw). Scaling by a positive lambda does
    not change sign, but it can shrink forecasts to ~0 (constant sign), which
    would make directional accuracy meaningless."""
    a, cnt = dacc(r_jb[te_idx], r_hat_std)
    mae = float(np.mean(np.abs(rate[te_idx] - rate[te_idx - 1] * np.exp(r_hat_raw))))
    print(f"{name:<36} dir {a:5.1f}% (n={cnt})  1-step MAE {mae:.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seq-len", type=int, default=30)
    args = ap.parse_args()

    df = pd.read_csv("data/jpy_bdt_daily.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    n = len(df); split = int(n * 0.8)
    rate = df["jpy_to_bdt"].to_numpy(float)
    r_jb = np.zeros(n); r_jb[1:] = np.diff(np.log(rate))
    r_uj = np.zeros(n); r_uj[1:] = np.diff(np.log(df["usd_to_jpy"].to_numpy(float)))
    r_ub = np.zeros(n); r_ub[1:] = np.diff(np.log(df["usd_to_bdt"].to_numpy(float)))
    absr = np.abs(r_jb)
    mu_uj = float(r_uj[:split].mean())
    jb_mean, jb_std = float(r_jb.mean()), float(r_jb.std(ddof=1))
    ub_mean, ub_std = float(r_ub.mean()), float(r_ub.std(ddof=1))
    te_idx = np.arange(split, n)
    print(f"{n} rows, split {split}, seeds {args.seeds}\n")

    # ---- M2: mean-reversion signal (deterministic) ---------------------------
    r_hat_mr = -0.25 * r_ub[:-1]            # raw-return pred for day t from lag t-1
    report("M2 mean-rev -0.25*r_ub_lag",
           r_hat_mr[split - 1:], r_hat_mr[split - 1:], r_jb, te_idx, rate)

    # ---- LSTM models ---------------------------------------------------------
    feat_base = np.column_stack([r_jb, absr, r_uj, r_ub])                       # M1
    mr = np.zeros(n); mr[1:] = -r_ub[:-1]                                       # M3 feat
    feat_m3 = np.column_stack([r_jb, absr, r_uj, r_ub, mr])
    feat_m4 = np.column_stack([r_ub, np.abs(r_ub), r_uj, absr, r_jb])           # target r_ub

    s1, r1, te1 = run_lstm(feat_base, r_jb, split, args.seq_len, args.seeds, jb_mean, jb_std, rate)
    report("M1 LSTM direct (v2)", s1, r1, r_jb, te1, rate)

    s3, r3, te3 = run_lstm(feat_m3, r_jb, split, args.seq_len, args.seeds, jb_mean, jb_std, rate)
    report("M3 LSTM + mean-rev feature", s3, r3, r_jb, te3, rate)

    s4, r_hat_ub, te4 = run_lstm(feat_m4, r_ub, split, args.seq_len, args.seeds, ub_mean, ub_std, rate)
    report("M4 LSTM on taka leg", s4, r_hat_ub - mu_uj, r_jb, te4, rate)


if __name__ == "__main__":
    main()
