# train.py
#
# HMM по остаточной волатильности z_resid + режимный VAR(1) **в факторном пространстве**
# + ансамбль GRU, предсказывающих остаток в X-пространстве.
#
# Дополнительно:
#   - term-structure волатильности по step_in_seq
#   - AR(1) по log z_resid^2
#   - SABR/CEV-нормализация диффов
#   - realized vol по окнам (5, 20, 60)
#   - vol-of-vol по z_resid (по тем же окнам)
#   - leverage-фича ||dX|| * sign(d z_resid)
#   - PCA-факторы по X и факторная волатильность (factor_rms)
#   - динамические матрицы переходов HMM (low/high vol)
#   - EWMA/IGARCH-вола по z_resid: sigma_ewma_t, sigma_ewma_next_t, shock_norm_t, shock_energy_t
#   - режимный VAR(1) по PCA-факторам (baseline в факторном пространстве)
#
# Запуск (пример):
#   python train.py --train_path train.parquet --out_dir . --n_ensemble 3 --lr 0.001 --n_factors 8

import os
import argparse
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


# ---------- utils ----------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_fourier_features(t_norm: np.ndarray, ks=(1, 2, 3, 4)):
    feats = []
    for k in ks:
        w = 2 * np.pi * k
        feats.append(np.sin(w * t_norm))
        feats.append(np.cos(w * t_norm))
    return np.stack(feats, axis=-1)  # [T, 2*len(ks)]


def compute_x_stats_and_z(df: pd.DataFrame, feature_cols):
    X = df[feature_cols].to_numpy(np.float64)  # [T, N]
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0) + 1e-8
    Xn = (X - x_mean) / x_std
    z = np.sqrt((Xn ** 2).mean(axis=1))        # [T]
    return X, x_mean, x_std, z


def fit_hmm_on_z(z: np.ndarray, n_regimes: int, em_iter: int, search_reps: int, maxiter: int):
    print(f"[HMM] Fitting MarkovRegression on z (len={len(z)}, k={n_regimes})")
    mod = MarkovRegression(
        z,
        k_regimes=n_regimes,
        trend="c",
        switching_variance=True,
    )
    res = mod.fit(
        em_iter=em_iter,
        search_reps=search_reps,
        maxiter=maxiter,
        disp=True,
    )
    smoothed = res.smoothed_marginal_probabilities
    if hasattr(smoothed, "to_numpy"):
        gammas = smoothed.to_numpy()  # [T, K]
    else:
        gammas = np.asarray(smoothed)
    states_hat = gammas.argmax(axis=1).astype(np.int64)
    return gammas, states_hat


def estimate_transition_matrix(states: np.ndarray, seq_ix: np.ndarray, n_regimes: int):
    print("[HMM] Estimating transition matrix (unconditional)...")
    K = n_regimes
    trans_counts = np.ones((K, K), dtype=np.float64) * 1e-3  # сглаживание

    same_seq = (seq_ix[1:] == seq_ix[:-1])
    idx = np.where(same_seq)[0]
    for t in idx:
        i = states[t]
        j = states[t + 1]
        trans_counts[i, j] += 1.0

    trans_mat = trans_counts / trans_counts.sum(axis=1, keepdims=True)
    return trans_mat


def estimate_transition_matrices_by_z(states: np.ndarray,
                                      seq_ix: np.ndarray,
                                      z_resid: np.ndarray,
                                      n_regimes: int,
                                      z_thresh: float):
    """
    Две матрицы переходов:
      - trans_low: при z_resid < z_thresh
      - trans_high: при z_resid >= z_thresh
    """
    print("[HMM] Estimating transition matrices conditioned on z_resid...")
    K = n_regimes
    low_counts = np.ones((K, K), dtype=np.float64) * 1e-3
    high_counts = np.ones((K, K), dtype=np.float64) * 1e-3

    same_seq = (seq_ix[1:] == seq_ix[:-1])
    idx = np.where(same_seq)[0]
    for t in idx:
        i = states[t]
        j = states[t + 1]
        if z_resid[t] < z_thresh:
            low_counts[i, j] += 1.0
        else:
            high_counts[i, j] += 1.0

    trans_low = low_counts / low_counts.sum(axis=1, keepdims=True)
    trans_high = high_counts / high_counts.sum(axis=1, keepdims=True)
    return trans_low, trans_high


def estimate_z_emission_params(z: np.ndarray, states: np.ndarray, n_regimes: int):
    print("[HMM] Estimating emission params for z...")
    K = n_regimes
    mean_z = np.zeros(K, dtype=np.float64)
    std_z = np.zeros(K, dtype=np.float64)
    for r in range(K):
        mask = (states == r)
        if mask.sum() == 0:
            mean_z[r] = z.mean()
            std_z[r] = z.std() + 1e-6
        else:
            zr = z[mask]
            mean_z[r] = zr.mean()
            std_z[r] = zr.std() + 1e-6
    return mean_z, std_z


def compute_sabr_betas_and_dstd(X: np.ndarray, seq_ix: np.ndarray):
    """
    Оценка beta_j в модели типа SABR/CEV для каждой фичи и std нормализованных диффов.
    dX_j ~ |X_j|^{beta_j}.
    """
    print("[SABR] Estimating betas and d_std...")
    T, N = X.shape
    if T < 2:
        betas = np.zeros(N, dtype=np.float64)
        d_std = np.ones(N, dtype=np.float64)
        return betas, d_std

    same_seq = (seq_ix[1:] == seq_ix[:-1])
    if not same_seq.any():
        betas = np.zeros(N, dtype=np.float64)
        d_std = np.ones(N, dtype=np.float64)
        return betas, d_std

    X_prev = X[:-1][same_seq]   # [M,N]
    X_next = X[1:][same_seq]    # [M,N]
    dx = X_next - X_prev        # [M,N]

    betas = np.zeros(N, dtype=np.float64)
    for j in range(N):
        d_col = dx[:, j]
        x_col = X_prev[:, j]
        mask = (np.abs(d_col) > 1e-8) & (np.abs(x_col) > 1e-8)
        if mask.sum() < 100:
            betas[j] = 0.0
            continue
        y = np.log(np.abs(d_col[mask]))
        x = np.log(np.abs(x_col[mask]))
        if np.std(x) < 1e-8:
            betas[j] = 0.0
            continue
        b, a = np.polyfit(x, y, 1)  # y ≈ a + b x
        betas[j] = float(np.clip(b, -1.0, 1.0))

    sabr_scale = np.power(np.abs(X_prev) + 1e-8, betas[None, :])
    d_sabr = dx / sabr_scale
    d_std = d_sabr.std(axis=0) + 1e-8

    return betas, d_std


def fit_var1_weighted(X_t: np.ndarray,
                      Y: np.ndarray,
                      weights: np.ndarray,
                      lam: float):
    """
    Взвешенный VAR(1) в общем виде:
      y = A x + c

    X_t, Y: [M, D] — здесь D может быть либо N (фичи), либо F (факторы).
    """
    M, D = X_t.shape
    if M == 0 or weights.sum() <= 1e-8:
        return np.zeros((D, D), dtype=np.float64), np.zeros(D, dtype=np.float64)

    w = weights.clip(min=1e-8)
    sqrt_w = np.sqrt(w).reshape(-1, 1)        # [M,1]
    Xw = X_t * sqrt_w                         # [M,D]
    Yw = Y * sqrt_w                           # [M,D]
    onesw = sqrt_w                            # [M,1]

    Phi = np.concatenate([Xw, onesw], axis=1) # [M, D+1]

    XtWX = Phi.T @ Phi                        # (D+1, D+1)
    XtWY = Phi.T @ Yw                         # (D+1, D)

    reg = np.eye(D + 1, dtype=np.float64)
    reg[-1, -1] = 0.0   # не штрафуем сдвиг
    XtWX_reg = XtWX + lam * reg

    Theta = np.linalg.solve(XtWX_reg, XtWY)   # (D+1, D)

    Theta_x = Theta[:D, :]                    # (D, D)
    c = Theta[D, :]                           # (D,)
    A = Theta_x.T                             # (D, D) так, чтобы y = A @ x + c
    return A, c


def fit_var1_per_regime(X: np.ndarray,
                        seq_ix: np.ndarray,
                        gammas: np.ndarray,
                        n_regimes: int,
                        lam: float = 1e-2,
                        min_weight: float = 100.0):
    """
    VAR(1) по каждому режиму (soft assignment по gamma_t[s]) в D-мерном пространстве X.
    Здесь X может быть:
      - либо исходное X_t (N-мерное),
      - либо PCA-факторы f_t (F-мерные).
    """
    print("[VAR] Fitting VAR(1) per regime (ridge)...")
    T, D = X.shape
    K = n_regimes

    same_seq = (seq_ix[1:] == seq_ix[:-1])
    idx = np.where(same_seq)[0]

    X_t = X[:-1][idx]        # [M,D]
    Y = X[1:][idx]           # [M,D]
    G = gammas[:-1][idx]     # [M,K]
    M = X_t.shape[0]
    print(f"[VAR] transitions M = {M}")

    # глобальный VAR (для fallback)
    w_all = np.ones(M, dtype=np.float64)
    A_global, c_global = fit_var1_weighted(X_t, Y, w_all, lam)

    var_A = np.zeros((K, D, D), dtype=np.float64)
    var_c = np.zeros((K, D), dtype=np.float64)

    for r in range(K):
        w_r = G[:, r]
        total_w = float(w_r.sum())
        print(f"  regime {r}: total weight = {total_w:.1f}")
        if total_w < min_weight:
            var_A[r] = A_global
            var_c[r] = c_global
            continue
        A_r, c_r = fit_var1_weighted(X_t, Y, w_r, lam)
        var_A[r] = A_r
        var_c[r] = c_r

    return var_A, var_c


def compute_ewma_features(z_resid: np.ndarray,
                          seq_ix: np.ndarray,
                          lam: float):
    """
    EWMA / IGARCH-подобная условная волатильность по z_resid.
    Для каждой последовательности:
      h_0 = global_mean(z_resid^2)
      h_{t+1} = lam * h_t + (1 - lam) * z_resid_t^2

    Для шага t считаем фичи:
      sigma_ewma_t   = sqrt(h_t)
      sigma_ewma_next_t = sqrt(h_{t+1})
      shock_norm_t   = z_resid_t / sigma_ewma_t
      shock_energy_t = z_resid_t^2 / h_t
    """
    print("[EWMA] Computing conditional volatility features (IGARCH-style)...")
    T = len(z_resid)
    assert T == len(seq_ix)
    r = z_resid.astype(np.float64)
    r2 = r * r

    # глобальная дисперсия для инициализации
    init_var = float(r2.mean()) + 1e-12

    sigma_ewma = np.zeros(T, dtype=np.float64)
    sigma_ewma_next = np.zeros(T, dtype=np.float64)
    shock_norm = np.zeros(T, dtype=np.float64)
    shock_energy = np.zeros(T, dtype=np.float64)

    eps = 1e-12
    uniq = np.unique(seq_ix)
    for s in uniq:
        idx = np.where(seq_ix == s)[0]
        # последовательность уже отсортирована по step_in_seq
        h = init_var
        for t in idx:
            sigma_t = math.sqrt(max(h, eps))
            sigma_ewma[t] = sigma_t
            shock_norm[t] = r[t] / sigma_t
            shock_energy[t] = r2[t] / max(h, eps)

            h_next = lam * h + (1.0 - lam) * r2[t]
            sigma_ewma_next[t] = math.sqrt(max(h_next, eps))

            h = h_next

    return sigma_ewma, sigma_ewma_next, shock_norm, shock_energy, init_var


# ---------- GRU модель ----------

class GRUResidual(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, x, h0=None):
        # x: [B, L, D]
        out, h_last = self.gru(x, h0)   # [B,L,H]
        y = self.out(out)               # [B,L,N]
        return y, h_last


# ---------- Dataset для GRU ----------

class GRUDataset(Dataset):
    """
    По одной последовательности:
      inputs:  [L, D]
      targets: [L, N]
      mask:    [L]

    Структура фич на шаге t:
      [x_norm (N),
       d_norm_sabr (N),
       gamma (K),
       y_base_norm (N),
       time/Fourier (10),
       extra]

    extra включает:
      - z_raw, z_resid, sigma_next              (3)
      - realized vol по z_resid (5,20,60)       (3)
      - vol-of-vol по z_resid (5,20,60)         (3)
      - leverage lev                            (1)
      - factor_rms (RMS по PCA-факторам)        (1)
      - factor_norm (F)                         (F)
      - sigma_ewma_t, sigma_ewma_next_t         (2)
      - shock_norm_t, shock_energy_t            (2)

    Итого extra: F + 15
    => D = 3N + K + F + 25

    Здесь baseline y_base строится через режимный VAR в факторном пространстве:
      - сначала считаем факторы f_t из x_norm_t
      - VAR(1) по f_t по режимам -> f_base_{t+1}
      - X_base_{t+1} ≈ x_mean + x_std * (f_base @ pca_components^T)
      - y_base_norm = Xn_base = (X_base - x_mean) / x_std
    """
    def __init__(self, df: pd.DataFrame, feature_cols, K: int,
                 x_mean, x_std, d_std,
                 var_A, var_c,         # VAR по факторам: (K,F,F), (K,F)
                 betas,
                 mu_h: float,
                 phi_h: float,
                 n_factors: int,
                 factor_mean: np.ndarray,
                 factor_std: np.ndarray,
                 pca_components: np.ndarray,   # (N,F)
                 fourier_ks=(1, 2, 3, 4)):
        self.df = df
        self.feature_cols = feature_cols
        self.K = K
        self.x_mean = x_mean.astype(np.float32)
        self.x_std = x_std.astype(np.float32)
        self.d_std = d_std.astype(np.float32)
        self.var_A = var_A.astype(np.float32)   # (K,F,F)
        self.var_c = var_c.astype(np.float32)   # (K,F)
        self.betas = betas.astype(np.float32)   # (N,)
        self.mu_h = float(mu_h)
        self.phi_h = float(phi_h)
        self.fourier_ks = fourier_ks

        self.n_factors = int(n_factors)
        self.factor_mean = factor_mean.astype(np.float32)
        self.factor_std = factor_std.astype(np.float32)
        self.pca_components = pca_components.astype(np.float32)  # (N,F)
        self.factor_cols = [f"factor_{k}" for k in range(self.n_factors)]

        self.groups = list(df.groupby("seq_ix", sort=False))
        self.N = len(feature_cols)
        self.D = 3 * self.N + self.K + self.n_factors + 25

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        _, g = self.groups[idx]
        g = g.sort_values("step_in_seq")

        X = g[self.feature_cols].to_numpy(np.float32)          # [T,N]
        steps = g["step_in_seq"].to_numpy(np.int32)            # [T]
        need = g["need_prediction"].to_numpy(bool)             # [T]
        z_raw_seq = g["z_raw"].to_numpy(np.float32)            # [T]
        z_resid_seq = g["z_resid"].to_numpy(np.float32)        # [T]
        lev_seq = g["lev"].to_numpy(np.float32)                # [T]
        sigma_ewma_seq = g["sigma_ewma"].to_numpy(np.float32)          # [T]
        sigma_ewma_next_seq = g["sigma_ewma_next"].to_numpy(np.float32)  # [T]
        shock_norm_seq = g["shock_norm"].to_numpy(np.float32)          # [T]
        shock_energy_seq = g["shock_energy"].to_numpy(np.float32)      # [T]

        if self.n_factors > 0:
            F_seq = g[self.factor_cols].to_numpy(np.float32)   # [T,F]
            F_norm_seq = (F_seq - self.factor_mean) / self.factor_std
            factor_rms_seq = np.sqrt((F_norm_seq ** 2).mean(axis=1))  # [T]
        else:
            F_norm_seq = np.zeros((len(g), 0), dtype=np.float32)
            factor_rms_seq = np.zeros((len(g),), dtype=np.float32)

        T, N = X.shape

        gammas = []
        for k in range(self.K):
            gammas.append(g[f"gamma_{k}"].to_numpy(np.float32))
        gamma = np.stack(gammas, axis=1)                       # [T,K]

        Xn = (X - self.x_mean) / self.x_std                    # [T,N]

        # SABR-нормированные диффы
        d = np.zeros_like(X, dtype=np.float32)
        if T > 1:
            dx = X[1:] - X[:-1]
            sabr_scale = np.power(np.abs(X[:-1]) + 1e-8,
                                  self.betas[None, :])
            d[1:] = (dx / sabr_scale).astype(np.float32)
        dn = d / self.d_std                                    # [T,N]

        # time/Fourier
        t_norm = steps.astype(np.float32) / 999.0
        t_feats = np.stack([t_norm, t_norm * t_norm], axis=1)  # [T,2]
        f_feats = make_fourier_features(t_norm, ks=self.fourier_ks).astype(np.float32)
        tf = np.concatenate([t_feats, f_feats], axis=1)        # [T,10]

        # AR(1) по log-vol residual: sigma_{t+1} (прогноз)
        h_seq = np.log(z_resid_seq.astype(np.float64) ** 2 + 1e-8)  # [T]
        sigma_next = np.sqrt(np.exp(self.mu_h + self.phi_h * h_seq)).astype(np.float32)  # [T]

        # realized vol и vol-of-vol по z_resid
        windows = (5, 20, 60)
        max_w = max(windows)

        # realized vol (z_resid)
        rv = np.zeros((T, len(windows)), dtype=np.float32)
        hist = []
        for t in range(T):
            v = float(z_resid_seq[t])
            hist.append(v * v)
            if len(hist) > max_w:
                hist.pop(0)
            arr = np.array(hist, dtype=np.float32)
            for i, w in enumerate(windows):
                Lw = min(len(arr), w)
                rv[t, i] = 0.0 if Lw == 0 else float(np.sqrt(arr[-Lw:].mean()))

        # vol-of-vol (differences z_resid)
        vov = np.zeros((T, len(windows)), dtype=np.float32)
        hist_d = []
        prev_z = None
        for t in range(T):
            if prev_z is None:
                dv2 = 0.0
            else:
                dv = float(z_resid_seq[t] - prev_z)
                dv2 = dv * dv
            prev_z = float(z_resid_seq[t])
            hist_d.append(dv2)
            if len(hist_d) > max_w:
                hist_d.pop(0)
            arr_d = np.array(hist_d, dtype=np.float32)
            for i, w in enumerate(windows):
                Lw = min(len(arr_d), w)
                vov[t, i] = 0.0 if Lw == 0 else float(np.sqrt(arr_d[-Lw:].mean()))

        L = T - 1
        inputs = np.zeros((L, self.D), dtype=np.float32)
        targets = np.zeros((L, N), dtype=np.float32)
        mask = np.zeros((L,), dtype=np.float32)

        for t in range(L):
            x_t = X[t]
            x_norm_t = Xn[t]
            d_norm_t = dn[t]
            gamma_t = gamma[t]
            tf_t = tf[t]
            z_raw_t = z_raw_seq[t]
            z_resid_t = z_resid_seq[t]
            sigma_next_t = sigma_next[t]
            rv_t = rv[t]
            vov_t = vov[t]
            lev_t = lev_seq[t]
            factor_rms_t = factor_rms_seq[t]
            factor_norm_t = F_norm_seq[t]  # [F]
            sigma_ewma_t = sigma_ewma_seq[t]
            sigma_ewma_next_t = sigma_ewma_next_seq[t]
            shock_norm_t = shock_norm_seq[t]
            shock_energy_t = shock_energy_seq[t]

            # baseline через факторный VAR:
            # 1) факторы f_t из x_norm_t
            if self.n_factors > 0:
                f_t = x_norm_t @ self.pca_components  # [F]
                # 2) режимный VAR по факторам
                y_states_f = np.einsum("kij,j->ki", self.var_A, f_t) + self.var_c  # [K,F]
                f_base = (gamma_t[:, None] * y_states_f).sum(axis=0)               # [F]
                # 3) обратно в Xn-пространство
                x_base_norm = f_base @ self.pca_components.T                       # [N]
            else:
                x_base_norm = np.zeros((N,), dtype=np.float32)

            y_base = x_base_norm * self.x_std + self.x_mean                        # [N]
            y_base_norm = x_base_norm                                             # [N]

            y_true = X[t + 1]                                                     # [N]
            r_true = (y_true - y_base) / self.x_std                               # [N]

            # extra:
            extra = np.concatenate(
                [
                    np.array([z_raw_t, z_resid_t, sigma_next_t], dtype=np.float32),
                    rv_t.astype(np.float32),
                    vov_t.astype(np.float32),
                    np.array([lev_t, factor_rms_t], dtype=np.float32),
                    factor_norm_t.astype(np.float32),
                    np.array([sigma_ewma_t, sigma_ewma_next_t,
                              shock_norm_t, shock_energy_t], dtype=np.float32),
                ],
                axis=0,
            )  # [F + 15]

            inp = np.concatenate(
                [x_norm_t, d_norm_t, gamma_t, y_base_norm, tf_t, extra],
                axis=0,
            )

            inputs[t] = inp
            targets[t] = r_true
            mask[t] = 1.0 if need[t] else 0.0

        return (
            torch.from_numpy(inputs),   # [L,D]
            torch.from_numpy(targets),  # [L,N]
            torch.from_numpy(mask),     # [L]
        )


# ---------- R^2 ----------

def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    diff = y_true - y_pred
    sse = (diff ** 2).sum(axis=0)
    mean_y = y_true.mean(axis=0)
    sst = ((y_true - mean_y) ** 2).sum(axis=0)
    sst = np.where(sst <= 1e-12, 1e-12, sst)
    r2_vec = 1.0 - sse / sst
    return float(r2_vec.mean())


# ---------- main training ----------

def train(args):
    set_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device("cpu")

    print(f"[data] loading {args.train_path}")
    df = pd.read_parquet(args.train_path)
    df = df.sort_values(["seq_ix", "step_in_seq"]).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in ("seq_ix", "step_in_seq", "need_prediction")]
    print(f"[data] N features = {len(feature_cols)}")

    X, x_mean, x_std, z_raw = compute_x_stats_and_z(df, feature_cols)
    seq_ix = df["seq_ix"].to_numpy(np.int64)
    steps = df["step_in_seq"].to_numpy(np.int32)
    T, N = X.shape

    # ----- PCA факторы -----
    Xn = (X - x_mean) / x_std
    if args.n_factors > 0 and N > 0:
        F = min(args.n_factors, N)
        print(f"[PCA] computing {F} factors via SVD...")
        U, S, Vt = np.linalg.svd(Xn, full_matrices=False)  # Xn = U S Vt
        pca_components = Vt[:F].T  # (N, F)
        factors = Xn @ pca_components  # (T, F)
        factor_mean = factors.mean(axis=0)
        factor_std = factors.std(axis=0) + 1e-8
        factors_f32 = factors.astype(np.float32)
        for k in range(F):
            df[f"factor_{k}"] = factors_f32[:, k]
    else:
        F = 0
        pca_components = np.zeros((N, 0), dtype=np.float64)
        factors = np.zeros((T, 0), dtype=np.float64)
        factor_mean = np.zeros((0,), dtype=np.float64)
        factor_std = np.ones((0,), dtype=np.float64)

    # сырой z
    df["z_raw"] = z_raw

    # term-structure волы по шагу
    print("[term] Estimating volatility term-structure over step_in_seq...")
    vol_term = df.groupby("step_in_seq")["z_raw"].mean()
    vol_term = vol_term.rolling(15, min_periods=1, center=True).mean()
    max_step = int(df["step_in_seq"].max())

    z_term_full = np.zeros(max_step + 1, dtype=np.float64)
    z_term_full[:] = vol_term.iloc[-1]
    z_term_full[vol_term.index.to_numpy()] = vol_term.to_numpy()

    # остаточная вола
    z_resid = df["z_raw"].to_numpy() / (z_term_full[steps] + 1e-8)
    df["z_resid"] = z_resid

    # EWMA / IGARCH-подобная волатильность по z_resid
    sigma_ewma, sigma_ewma_next, shock_norm, shock_energy, ewma_init_var = compute_ewma_features(
        z_resid=z_resid,
        seq_ix=seq_ix,
        lam=args.ewma_lambda,
    )
    df["sigma_ewma"] = sigma_ewma.astype(np.float32)
    df["sigma_ewma_next"] = sigma_ewma_next.astype(np.float32)
    df["shock_norm"] = shock_norm.astype(np.float32)
    df["shock_energy"] = shock_energy.astype(np.float32)

    # leverage feature: ||dX|| * sign(d z_resid)
    print("[SV] Computing leverage feature...")
    same_seq = (seq_ix[1:] == seq_ix[:-1])

    dX = np.zeros_like(X)          # [T, N]
    if same_seq.any():
        dX[1:][same_seq] = X[1:][same_seq] - X[:-1][same_seq]

    norm_dX = np.sqrt((dX ** 2).sum(axis=1))   # [T]

    dz = np.zeros_like(z_resid)
    dz[1:] = z_resid[1:] - z_resid[:-1]

    lev = norm_dX * np.sign(dz)               # [T]
    df["lev"] = lev.astype(np.float32)

    # AR(1) по log-vol residual
    print("[SV] Fitting AR(1) on log-vol residual...")
    h = np.log(z_resid ** 2 + 1e-8)
    if len(h) > 1:
        h_t = h[:-1]
        h_next = h[1:]
        A = np.vstack([np.ones_like(h_t), h_t]).T  # [T-1,2]
        theta, *_ = np.linalg.lstsq(A, h_next, rcond=None)
        mu_h, phi_h = theta.astype(np.float64)
    else:
        mu_h, phi_h = 0.0, 0.0

    # HMM по z_resid
    gammas, states_hat = fit_hmm_on_z(
        z=z_resid,
        n_regimes=args.n_regimes,
        em_iter=args.em_iter,
        search_reps=args.search_reps,
        maxiter=args.maxiter,
    )

    K = args.n_regimes
    for k in range(K):
        df[f"gamma_{k}"] = gammas[:, k].astype(np.float32)

    trans_mat = estimate_transition_matrix(states_hat, seq_ix, K)
    mean_z, std_z = estimate_z_emission_params(z_resid, states_hat, K)

    # условные матрицы переходов по низкой/высокой воле
    z_thresh = float(np.median(z_resid))
    trans_mat_low, trans_mat_high = estimate_transition_matrices_by_z(
        states_hat, seq_ix, z_resid, K, z_thresh
    )
    trans_alpha = 4.0  # крутизна логистики для смешивания матриц

    # режимный VAR(1) в факторном пространстве
    if F > 0:
        print("[VAR] Using PCA-factor VAR baseline...")
        var_A_f, var_c_f = fit_var1_per_regime(
            X=factors,
            seq_ix=seq_ix,
            gammas=gammas,
            n_regimes=K,
            lam=args.var_ridge,
            min_weight=args.var_min_weight,
        )
    else:
        print("[VAR] No factors, using zero VAR baseline")
        var_A_f = np.zeros((K, 0, 0), dtype=np.float64)
        var_c_f = np.zeros((K, 0), dtype=np.float64)

    # SABR-беты и std дифференциалов по X
    betas, d_std = compute_sabr_betas_and_dstd(X, seq_ix)

    # начальное распределение режимов
    counts = np.bincount(states_hat, minlength=K).astype(np.float64)
    pi0 = counts / counts.sum()

    # датасет для GRU
    full_ds = GRUDataset(
        df=df,
        feature_cols=feature_cols,
        K=K,
        x_mean=x_mean,
        x_std=x_std,
        d_std=d_std,
        var_A=var_A_f,
        var_c=var_c_f,
        betas=betas,
        mu_h=mu_h,
        phi_h=phi_h,
        n_factors=F,
        factor_mean=factor_mean,
        factor_std=factor_std,
        pca_components=pca_components,
        fourier_ks=(1, 2, 3, 4),
    )

    val_size = max(1, int(len(full_ds) * 0.2))
    train_size = len(full_ds) - val_size
    tr_ds, val_ds = random_split(full_ds, [train_size, val_size],
                                 generator=torch.Generator().manual_seed(args.seed))

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    in_dim = full_ds.D
    N_out = full_ds.N

    loss_fn = nn.MSELoss(reduction="sum")

    all_states = []
    all_scores = []

    for ens_id in range(args.n_ensemble):
        print(f"\n=== Ensemble member {ens_id + 1}/{args.n_ensemble} ===")

        set_seed(args.seed + ens_id * 113)

        model = GRUResidual(
            in_dim=in_dim,
            hidden=args.hidden,
            out_dim=N_out,
            num_layers=args.gru_layers,
            dropout=args.gru_dropout,
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

        best_val_r2 = -1e9
        best_state = None
        bad = 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            tot_loss = 0.0
            tot_count = 0.0

            for xb, yb, mb in tr_loader:
                xb = xb.to(device)  # [B,L,D]
                yb = yb.to(device)  # [B,L,N]
                mb = mb.to(device)  # [B,L]

                opt.zero_grad()
                y_pred, _ = model(xb)  # [B,L,N]
                mask = mb.unsqueeze(-1)  # [B,L,1]

                diff = (y_pred - yb) * mask
                loss = loss_fn(diff, torch.zeros_like(diff))
                count = mask.sum().item()
                if count > 0:
                    (loss / count).backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    tot_loss += loss.item()
                    tot_count += count

            train_loss = tot_loss / max(1.0, tot_count)
            sched.step()

            # validation R^2 (на полных y)
            model.eval()
            y_true_all, y_pred_all = [], []
            with torch.no_grad():
                for xb, yb, mb in val_loader:
                    xb = xb.to(device)  # [1,L,D]
                    yb = yb.to(device)
                    mb = mb.to(device)
                    y_pred_norm, _ = model(xb)  # [1,L,N]

                    xb_np = xb.cpu().numpy()[0]         # [L,D]
                    yb_np = yb.cpu().numpy()[0]         # [L,N]
                    y_pred_norm_np = y_pred_norm.cpu().numpy()[0]
                    mb_np = mb.cpu().numpy()[0]         # [L]

                    Lseq = xb_np.shape[0]
                    Nloc = N_out
                    Kloc = K

                    # y_base_norm лежит сразу после (x_norm, d_norm, gamma)
                    y_base_norm_seq = xb_np[:, 2 * Nloc + Kloc:3 * Nloc + Kloc]

                    for t in range(Lseq):
                        if mb_np[t] < 0.5:
                            continue
                        y_base_norm = y_base_norm_seq[t]
                        y_base = y_base_norm * x_std + x_mean

                        r_true_norm = yb_np[t]
                        r_pred_norm = y_pred_norm_np[t]
                        r_true = r_true_norm * x_std
                        r_pred = r_pred_norm * x_std

                        y_true = y_base + r_true
                        y_pred = y_base + r_pred

                        y_true_all.append(y_true)
                        y_pred_all.append(y_pred)

            if len(y_true_all) > 0:
                y_true_arr = np.stack(y_true_all, axis=0)
                y_pred_arr = np.stack(y_pred_all, axis=0)
                val_r2 = compute_r2(y_true_arr, y_pred_arr)
            else:
                val_r2 = -1e9

            print(f"[ens {ens_id} | {epoch:03d}] train_loss={train_loss:.6f} | val_R2={val_r2:.4f}")

            if val_r2 > best_val_r2 + 1e-5:
                best_val_r2 = val_r2
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= args.patience:
                    print("Early stopping.")
                    break

        all_states.append(best_state if best_state is not None else model.state_dict())
        all_scores.append(best_val_r2)
        print(f"[ens {ens_id}] best val R² = {best_val_r2:.4f}")

    # --- save artifacts ---
    os.makedirs(args.out_dir, exist_ok=True)

    np.savez(
        os.path.join(args.out_dir, "regime_hmm_params.npz"),
        K=np.int64(K),
        N=np.int64(len(feature_cols)),
        var_A=var_A_f,
        var_c=var_c_f,
        trans_mat=trans_mat,
        trans_mat_low=trans_mat_low,
        trans_mat_high=trans_mat_high,
        z_thresh=np.float64(z_thresh),
        trans_alpha=np.float64(trans_alpha),
        mean_z=mean_z,
        std_z=std_z,
        # скейлы X
        x_mean=x_mean,
        x_std=x_std,
        d_std=d_std,
        betas=betas,
        z_term=z_term_full,
        max_step=np.int64(max_step),
        mu_h=mu_h,
        phi_h=phi_h,
        pi0=pi0,
        n_factors=np.int64(F),
        pca_components=pca_components,
        factor_mean=factor_mean,
        factor_std=factor_std,
        ewma_lambda=np.float64(args.ewma_lambda),
        ewma_init_var=np.float64(ewma_init_var),
    )

    torch.save(
        {
            "state_dict": all_states[0],
            "state_dicts": all_states,
            "in_dim": in_dim,
            "hidden": args.hidden,
            "N": N_out,
            "num_layers": args.gru_layers,
            "dropout": args.gru_dropout,
        },
        os.path.join(args.out_dir, "hmm_gru.pt"),
    )

    print(f"[save] regime_hmm_params.npz & hmm_gru.pt -> {args.out_dir}")
    print(f"[best] ensemble val R² (mean) = {float(np.mean(all_scores)):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", type=str, default="train.parquet")
    ap.add_argument("--out_dir", type=str, default=".")
    ap.add_argument("--n_regimes", type=int, default=2)
    ap.add_argument("--em_iter", type=int, default=20)
    ap.add_argument("--search_reps", type=int, default=5)
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--var_ridge", type=float, default=1e-2)
    ap.add_argument("--var_min_weight", type=float, default=100.0)
    ap.add_argument("--hidden", type=int, default=160)
    ap.add_argument("--gru_layers", type=int, default=2)
    ap.add_argument("--gru_dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_ensemble", type=int, default=1)
    ap.add_argument("--n_factors", type=int, default=8)
    ap.add_argument("--ewma_lambda", type=float, default=0.97)
    args = ap.parse_args()
    train(args)

