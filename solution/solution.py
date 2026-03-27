import os
import math
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn

from utils import DataPoint


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
        out, h_last = self.gru(x, h0)   # [B,L,H]
        y = self.out(out)               # [B,L,N]
        return y, h_last


class PredictionModel:
    def __init__(self):
        torch.set_num_threads(1)
        base_dir = os.path.dirname(__file__)

        # ---- HMM + VAR + фичи ----
        params_path = os.path.join(base_dir, "regime_hmm_params.npz")
        if not os.path.isfile(params_path):
            # fallback: чистый persistence
            self.N = None
            self.K = 0
            self.var_A = None
            self.var_c = None
            self.trans_mat = None
            self.trans_mat_low = None
            self.trans_mat_high = None
            self.z_thresh = 0.0
            self.trans_alpha = 0.0
            self.mean_z = None
            self.std_z = None
            self.x_mean = None
            self.x_std = None
            self.d_std = None
            self.betas = None
            self.z_term = None
            self.max_step = 999
            self.mu_h = 0.0
            self.phi_h = 0.0
            self.pi0 = None
            self.n_factors = 0
            self.pca_components = None
            self.factor_mean = None
            self.factor_std = None
            self.ewma_lambda = 0.97
            self.ewma_init_var = 1.0
        else:
            data = np.load(params_path)

            self.K = int(data["K"])
            self.N = int(data["N"])

            # факторный VAR(1): (K,F,F), (K,F)
            self.var_A = data["var_A"].astype(np.float64)
            self.var_c = data["var_c"].astype(np.float64)

            # HMM
            self.trans_mat = data["trans_mat"].astype(np.float64)  # (K,K)
            self.trans_mat_low = data["trans_mat_low"].astype(np.float64)
            self.trans_mat_high = data["trans_mat_high"].astype(np.float64)
            self.z_thresh = float(data["z_thresh"])
            self.trans_alpha = float(data["trans_alpha"])

            self.mean_z = data["mean_z"].astype(np.float64)        # (K,)
            self.std_z = data["std_z"].astype(np.float64)          # (K,)

            # скейлы X
            self.x_mean = data["x_mean"].astype(np.float64)        # (N,)
            self.x_std = data["x_std"].astype(np.float64)          # (N,)
            self.d_std = data["d_std"].astype(np.float64)          # (N,)
            self.betas = data["betas"].astype(np.float64)          # (N,)

            # term-structure
            self.z_term = data["z_term"].astype(np.float64)        # (max_step+1,)
            self.max_step = int(data["max_step"])

            # AR(1) по лог-воле
            self.mu_h = float(data["mu_h"])
            self.phi_h = float(data["phi_h"])

            # начальное распред. режимов
            self.pi0 = data["pi0"].astype(np.float64)              # (K,)

            # PCA-факторы
            self.n_factors = int(data.get("n_factors", np.int64(0)))
            self.pca_components = data["pca_components"].astype(np.float64)  # (N,F)
            self.factor_mean = data["factor_mean"].astype(np.float64)        # (F,)
            self.factor_std = data["factor_std"].astype(np.float64)          # (F,)

            # EWMA / IGARCH
            self.ewma_lambda = float(data.get("ewma_lambda", np.float64(0.97)))
            self.ewma_init_var = float(data.get("ewma_init_var", np.float64(1.0)))

        # ---- GRU ансамбль ----
        self.models: List[GRUResidual] = []
        self.h_states: List[Optional[torch.Tensor]] = []
        self.in_dim: Optional[int] = None
        self.hidden_dim: Optional[int] = None
        self.num_layers: int = 1

        gru_path = os.path.join(base_dir, "hmm_gru.pt")
        if os.path.isfile(gru_path):
            ckpt = torch.load(gru_path, map_location="cpu")
            self.in_dim = int(ckpt["in_dim"])
            self.hidden_dim = int(ckpt["hidden"])
            N_gru = int(ckpt["N"])
            self.num_layers = int(ckpt.get("num_layers", 1))
            dropout = float(ckpt.get("dropout", 0.0))

            state_dicts = ckpt.get("state_dicts", None)
            if state_dicts is not None and isinstance(state_dicts, list) and len(state_dicts) > 0:
                for sd in state_dicts:
                    m = GRUResidual(
                        in_dim=self.in_dim,
                        hidden=self.hidden_dim,
                        out_dim=N_gru,
                        num_layers=self.num_layers,
                        dropout=dropout,
                    )
                    m.load_state_dict(sd)
                    m.eval()
                    self.models.append(m)
            else:
                m = GRUResidual(
                    in_dim=self.in_dim,
                    hidden=self.hidden_dim,
                    out_dim=N_gru,
                    num_layers=self.num_layers,
                    dropout=dropout,
                )
                m.load_state_dict(ckpt["state_dict"])
                m.eval()
                self.models.append(m)

            for _ in self.models:
                self.h_states.append(
                    torch.zeros(self.num_layers, 1, self.hidden_dim, dtype=torch.float32)
                )

        # ---- состояние по последовательности ----
        self.curr_seq_ix: Optional[int] = None
        self.prev_x: Optional[np.ndarray] = None
        self.prev_z_resid: Optional[float] = None

        # HMM-состояние
        self.pi: Optional[np.ndarray] = None  # (K,)

        # EWMA / IGARCH
        self.h_ewma: Optional[float] = None

        # realized vol / vol-of-vol
        self.rv_windows = (5, 20, 60)
        self.max_window = max(self.rv_windows)
        self.hist_z2: List[float] = []
        self.hist_dz2: List[float] = []
        self.prev_z_for_vov: Optional[float] = None

    # ---------- helpers ----------

    def _reset_seq(self, seq_ix: int, state_dim: int):
        self.curr_seq_ix = seq_ix
        self.prev_x = None
        self.prev_z_resid = None
        self.hist_z2 = []
        self.hist_dz2 = []
        self.prev_z_for_vov = None

        # если размерность не совпала — fallback на простую persistence
        if (self.N is None) or (self.N != state_dim):
            self.N = state_dim
            self.K = 0
            self.var_A = None
            self.var_c = None
            self.trans_mat = None
            self.trans_mat_low = None
            self.trans_mat_high = None
            self.mean_z = None
            self.std_z = None
            self.x_mean = None
            self.x_std = None
            self.d_std = None
            self.betas = None
            self.z_term = None
            self.max_step = 999
            self.mu_h = 0.0
            self.phi_h = 0.0
            self.pi0 = None
            self.n_factors = 0
            self.pca_components = None
            self.factor_mean = None
            self.factor_std = None

        # HMM init
        if self.pi0 is not None and self.K > 0:
            self.pi = self.pi0.copy()
        elif self.K > 0:
            self.pi = np.ones(self.K, dtype=np.float64) / float(self.K)
        else:
            self.pi = None

        # EWMA init
        self.h_ewma = float(self.ewma_init_var)

        # GRU hidden
        if self.models and self.hidden_dim is not None:
            self.h_states = []
            for _ in self.models:
                self.h_states.append(
                    torch.zeros(self.num_layers, 1, self.hidden_dim, dtype=torch.float32)
                )

    def _compute_z_raw_and_resid(self, x: np.ndarray, step_in_seq: int):
        if self.x_mean is None or self.x_std is None:
            # простая RMS
            z_raw = float(np.sqrt((x.astype(np.float64) ** 2).mean()))
            z_resid = z_raw
            return z_raw, z_resid

        xn = (x.astype(np.float64) - self.x_mean) / self.x_std
        z_raw = float(np.sqrt((xn ** 2).mean()))

        if self.z_term is not None:
            step_clamped = int(min(max(step_in_seq, 0), self.max_step))
            z_term_t = float(self.z_term[step_clamped])
        else:
            z_term_t = 1.0

        z_resid = z_raw / (z_term_t + 1e-8)
        return z_raw, z_resid

    def _hmm_filter_step(self, z_resid: float):
        if (
            self.pi is None
            or self.K <= 0
            or self.trans_mat is None
            or self.mean_z is None
            or self.std_z is None
        ):
            return

        # динамическая матрица переходов (low/high mix по z_resid)
        trans = self.trans_mat
        if (self.trans_mat_low is not None) and (self.trans_mat_high is not None):
            x = self.trans_alpha * (z_resid - self.z_thresh)
            # logistic weight для high-vol матрицы
            try:
                w_high = 1.0 / (1.0 + math.exp(-x))
            except OverflowError:
                w_high = 1.0 if x > 0 else 0.0
            w_low = 1.0 - w_high
            trans = w_low * self.trans_mat_low + w_high * self.trans_mat_high  # (K,K)

        pi_pred = self.pi @ trans  # (K,)

        # эмиссия: гауссиан по z_resid
        diff = (z_resid - self.mean_z) / (self.std_z + 1e-12)
        log_L = -0.5 * diff * diff - np.log(self.std_z + 1e-12)

        log_pi_pred = np.log(pi_pred + 1e-15)
        log_post = log_pi_pred + log_L

        m = log_post.max()
        post = np.exp(log_post - m)
        s = post.sum()
        if (not np.isfinite(s)) or s <= 0:
            if self.pi0 is not None:
                self.pi = self.pi0.copy()
            else:
                self.pi = np.ones_like(pi_pred) / float(len(pi_pred))
        else:
            self.pi = post / s

    # ---------- main API ----------

    def predict(self, data_point: DataPoint) -> Optional[np.ndarray]:
        x = data_point.state.astype(np.float64)
        N_state = x.shape[0]

        # новая последовательность?
        if (self.curr_seq_ix is None) or (data_point.seq_ix != self.curr_seq_ix):
            self._reset_seq(data_point.seq_ix, N_state)

        # fallback: если нет обученных параметров — persistence baseline
        if (
            self.var_A is None
            or self.var_c is None
            or self.x_mean is None
            or self.x_std is None
            or self.d_std is None
            or not self.models
            or self.in_dim is None
        ):
            self.prev_x = x.copy()
            if not data_point.need_prediction:
                return None
            return x.astype(np.float32)

        # --- HMM + baseline VAR в факторном пространстве ---
        z_raw, z_resid = self._compute_z_raw_and_resid(
            x, data_point.step_in_seq
        )

        # HMM фильтр
        self._hmm_filter_step(z_resid)

        x64 = x.astype(np.float64)
        # нормировка X
        x_norm = (x64 - self.x_mean) / self.x_std

        # режимные вероятности
        if self.pi is not None and self.K > 0:
            pi = self.pi.astype(np.float64)
        elif self.K > 0:
            pi = np.ones(self.K, dtype=np.float64) / float(self.K)
        else:
            pi = np.zeros(0, dtype=np.float64)

        # факторный baseline
        if self.n_factors > 0 and self.pca_components is not None and self.pca_components.shape[1] == self.n_factors:
            # факторы из x_norm
            f_t = x_norm @ self.pca_components  # [F]
            # режимный VAR в факторном пространстве
            y_states_f = np.einsum("kij,j->ki", self.var_A, f_t) + self.var_c  # [K,F]
            if pi.size == self.K and self.K > 0:
                f_base = (pi[:, None] * y_states_f).sum(axis=0)  # [F]
            else:
                f_base = y_states_f.mean(axis=0)
            # обратно в Xn
            x_base_norm = f_base @ self.pca_components.T        # [N]
        else:
            x_base_norm = np.zeros_like(x_norm)

        y_base = x_base_norm * self.x_std + self.x_mean         # [N]
        y_base_norm = x_base_norm.astype(np.float32)            # [N]

        # SABR-нормированные диффы
        if self.prev_x is not None and self.betas is not None and self.d_std is not None:
            dx = x64 - self.prev_x
            sabr_scale = np.power(np.abs(self.prev_x) + 1e-8, self.betas)
            d_sabr = dx / sabr_scale
            d_norm = (d_sabr / self.d_std).astype(np.float32)
        else:
            d_norm = np.zeros_like(x_norm, dtype=np.float32)

        # time / Fourier
        t_norm = float(data_point.step_in_seq) / 999.0
        t2 = t_norm * t_norm
        fourier = []
        for k in (1, 2, 3, 4):
            w = 2.0 * math.pi * k
            fourier.append(math.sin(w * t_norm))
            fourier.append(math.cos(w * t_norm))
        tf = np.array([t_norm, t2] + fourier, dtype=np.float32)  # [10]

        # AR(1) по log-vol residual -> sigma_next
        h_t = math.log(z_resid * z_resid + 1e-8)
        sigma_next = math.sqrt(max(1e-12, math.exp(self.mu_h + self.phi_h * h_t)))

        # realized vol / vol-of-vol по z_resid
        v2 = z_resid * z_resid
        self.hist_z2.append(v2)
        if len(self.hist_z2) > self.max_window:
            self.hist_z2.pop(0)
        arr_z2 = np.array(self.hist_z2, dtype=np.float64)

        rv_vals = []
        for w in self.rv_windows:
            Lw = min(len(arr_z2), w)
            if Lw == 0:
                rv_vals.append(0.0)
            else:
                rv_vals.append(float(math.sqrt(arr_z2[-Lw:].mean())))
        rv = np.array(rv_vals, dtype=np.float32)  # [3]

        # vol-of-vol
        if self.prev_z_for_vov is None:
            dv2 = 0.0
        else:
            dz_vov = z_resid - self.prev_z_for_vov
            dv2 = dz_vov * dz_vov
        self.prev_z_for_vov = z_resid

        self.hist_dz2.append(dv2)
        if len(self.hist_dz2) > self.max_window:
            self.hist_dz2.pop(0)
        arr_dz2 = np.array(self.hist_dz2, dtype=np.float64)

        vov_vals = []
        for w in self.rv_windows:
            Lw = min(len(arr_dz2), w)
            if Lw == 0:
                vov_vals.append(0.0)
            else:
                vov_vals.append(float(math.sqrt(arr_dz2[-Lw:].mean())))
        vov = np.array(vov_vals, dtype=np.float32)  # [3]

        # leverage
        if self.prev_x is None or self.prev_z_resid is None:
            lev = 0.0
        else:
            dX = x64 - self.prev_x
            norm_dX = float(math.sqrt(float(np.sum(dX * dX))))
            dz_sign = z_resid - self.prev_z_resid
            if abs(dz_sign) < 1e-12:
                s = 0.0
            else:
                s = 1.0 if dz_sign > 0 else -1.0
            lev = norm_dX * s

        # PCA-факторы для extra-фич
        if self.n_factors > 0 and self.pca_components is not None and self.pca_components.shape[1] == self.n_factors:
            factors = x_norm @ self.pca_components  # [F]
            factor_norm = (factors - self.factor_mean) / self.factor_std
            factor_rms = float(math.sqrt(float(np.mean(factor_norm ** 2)))) if self.n_factors > 0 else 0.0
        else:
            factor_norm = np.zeros((0,), dtype=np.float32)
            factor_rms = 0.0

        # EWMA / IGARCH фичи
        if self.h_ewma is None:
            self.h_ewma = float(self.ewma_init_var)
        h_curr = self.h_ewma
        sigma_ewma_t = float(math.sqrt(max(h_curr, 1e-12)))
        shock_norm = float(z_resid / sigma_ewma_t) if sigma_ewma_t > 0 else 0.0
        shock_energy = float((z_resid * z_resid) / max(h_curr, 1e-12))

        h_next = self.ewma_lambda * h_curr + (1.0 - self.ewma_lambda) * (z_resid * z_resid)
        sigma_ewma_next = float(math.sqrt(max(h_next, 1e-12)))
        self.h_ewma = h_next

        # gamma / режимные вероятности
        if self.pi is not None and self.K > 0:
            gamma_vec = self.pi.astype(np.float32)
        elif self.K > 0:
            gamma_vec = (np.ones(self.K, dtype=np.float32) / float(self.K))
        else:
            gamma_vec = np.zeros((0,), dtype=np.float32)

        # extra-фичи
        extra = np.concatenate(
            [
                np.array([z_raw, z_resid, sigma_next], dtype=np.float32),
                rv,                            # [3]
                vov,                           # [3]
                np.array([lev, factor_rms], dtype=np.float32),
                factor_norm.astype(np.float32),
                np.array(
                    [sigma_ewma_t, sigma_ewma_next, shock_norm, shock_energy],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        )

        # полный вектор признаков
        feats = np.concatenate(
            [
                x_norm.astype(np.float32),
                d_norm.astype(np.float32),
                gamma_vec,
                y_base_norm,
                tf,
                extra,
            ],
            axis=0,
        )

        # проверим размерность
        if self.in_dim is not None and feats.shape[0] != self.in_dim:
            # если что-то пошло не так по размеру — fallback на persistence
            self.prev_x = x64.copy()
            self.prev_z_resid = z_resid
            if not data_point.need_prediction:
                return None
            return x.astype(np.float32)

        inp = torch.from_numpy(feats.reshape(1, 1, -1))  # [1,1,D]

        # --- GRU ансамбль ---
        r_pred_norm_total = np.zeros((self.N,), dtype=np.float64)
        cnt = 0
        for i, model in enumerate(self.models):
            h_i = self.h_states[i]
            with torch.no_grad():
                y_norm_seq, h_new = model(inp, h_i)
            self.h_states[i] = h_new
            r_pred_norm_i = y_norm_seq.cpu().numpy().reshape(-1).astype(np.float64)  # [N]
            r_pred_norm_total += r_pred_norm_i
            cnt += 1

        if cnt > 0:
            r_pred_norm = r_pred_norm_total / float(cnt)
        else:
            r_pred_norm = np.zeros((self.N,), dtype=np.float64)

        # обновляем prev
        self.prev_x = x64.copy()
        self.prev_z_resid = z_resid

        if not data_point.need_prediction:
            return None

        # денормализация остатка и финальный прогноз
        r_pred = r_pred_norm * self.x_std
        y_hat = y_base + r_pred

        return y_hat.astype(np.float32)

