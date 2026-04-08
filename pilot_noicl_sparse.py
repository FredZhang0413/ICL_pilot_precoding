"""
Pilot-Based In-Context Learning for Multi-User MISO Precoding
with Curriculum Self-Evolution Training

COMPLETE RUNNABLE VERSION with on-the-fly testing and baseline comparisons.

Copy this file to your local environment with PyTorch+CUDA and run directly:
    python pilot_icl_precoding_v2.py

Features:
  - 4 baselines computed at startup (MMSE/OptPLam × Perfect/Imperfect CSI)
  - On-the-fly evaluation on FIXED test set every epoch
  - 3-phase curriculum training with self-bootstrapping
  - Clean terminal output with rate comparisons

Requirements: PyTorch >= 2.0, tqdm
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
import numpy as np
from typing import Optional, Tuple, List, Dict
from tqdm import tqdm
import warnings
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###############################################################################
# 1. CONFIGURATION
###############################################################################
class Config:
    def __init__(self, **kwargs):
        # System
        self.K = kwargs.get('K', 16)
        self.N = kwargs.get('N', 16)
        self.L_p = kwargs.get('L_p', 32)
        self.P_max = kwargs.get('P_max', 1.0)
        self.SNR_dB = kwargs.get('SNR_dB', 15)
        self.sigma2 = self.P_max / (10 ** (self.SNR_dB / 10))

        # Channel model (geometric sparse mmWave, cluster-based)
        self.ch_n_clusters = kwargs.get('ch_n_clusters', 3)
        self.ch_n_rays_per_cluster = kwargs.get('ch_n_rays_per_cluster', 5)
        self.ch_angle_spread_deg = kwargs.get('ch_angle_spread_deg', 7.0)

        # No-ICL model I/O dimensions
        # input token feature: 2*L_p (real/imag), sequence length: K
        # output compressed representation: 2*K -> (p, lambda)
        self.in_token_dim = 2 * self.L_p
        self.seq_len = self.K
        self.out_dim = 2 * self.K

        # ICL
        self.n_demos = kwargs.get('n_demos', 4)

        # Pilot Encoder
        self.encoder_hidden = kwargs.get('encoder_hidden', 128)

        # Transformer (no PE, with causal mask)
        self.d_model = kwargs.get('d_model', 128)
        self.n_heads = kwargs.get('n_heads', 4)
        self.n_layers = kwargs.get('n_layers', 3)
        self.d_ff = kwargs.get('d_ff', 256)
        self.dropout = kwargs.get('dropout', 0.0)

        # Training
        self.batch_size = kwargs.get('batch_size', 64)
        self.lr = kwargs.get('lr', 3e-4)
        self.lr_min = kwargs.get('lr_min', 5e-5)
        self.weight_decay = kwargs.get('weight_decay', 1e-4)

        # Dataset
        self.initial_dataset_size = kwargs.get('initial_dataset_size', 3000)
        self.opt_iters = kwargs.get('opt_iters', 500)   # Adam iters for (p,lam) labels
        self.opt_lr = kwargs.get('opt_lr', 0.03)

        # Curriculum
        self.phase1_epochs = kwargs.get('phase1_epochs', 30)
        self.phase2_epochs = kwargs.get('phase2_epochs', 70)
        self.total_epochs = self.phase1_epochs + self.phase2_epochs
        self.r_max = kwargs.get('r_max', 0.85)
        self.steps_per_epoch = kwargs.get('steps_per_epoch', 80)

        # Self-bootstrapping
        self.tau_start = kwargs.get('tau_start', 30)
        self.tau_end = kwargs.get('tau_end', 65)
        self.max_dataset_size = kwargs.get('max_dataset_size', 30000)

        # Loss balancing: scale unsup rate loss to match supervised MSE magnitude
        # Diagnostic showed |rate|/|MSE| ~ 100x at Phase 1/2 boundary → scale = 0.005
        self.unsup_scale = kwargs.get('unsup_scale', 0.005)

        # Eval
        self.n_test = kwargs.get('n_test', 300)


###############################################################################
# 2. SIGNAL PROCESSING UTILITIES
###############################################################################
def generate_channel(B: int, K: int, N: int,
                     n_clusters: int = 3,
                     n_rays_per_cluster: int = 5,
                     angle_spread_deg: float = 7.0) -> torch.Tensor:
    """
    Geometric sparse mmWave channel (cluster-based), shape (B, K, N) complex.

    User-k channel:
      h_k = sqrt(N/L) * sum_{l=1..L} alpha_{k,l} * a_t(phi_{k,l}),
      a_t(phi)[n] = exp(j*pi*n*sin(phi)) / sqrt(N),  n=0,...,N-1

    where L = N_c * N_r (clusters × rays per cluster).
    """
    L = n_clusters * n_rays_per_cluster
    angle_spread = math.radians(angle_spread_deg)

    # Cluster center AoDs (uniform over [-pi/2, pi/2]) and per-ray offsets
    cluster_means = (torch.rand(B, K, n_clusters, 1, device=device) - 0.5) * math.pi
    ray_offsets = torch.randn(B, K, n_clusters, n_rays_per_cluster, device=device) * angle_spread
    angles = (cluster_means + ray_offsets).clamp(-math.pi / 2, math.pi / 2).reshape(B, K, L)

    # Complex path gains alpha ~ CN(0,1)
    alpha = (
        torch.randn(B, K, L, device=device)
        + 1j * torch.randn(B, K, L, device=device)
    ) / math.sqrt(2)

    # ULA steering vectors a_t(phi), half-wavelength spacing
    ant_idx = torch.arange(N, device=device, dtype=torch.float32).view(1, 1, 1, N)
    phase = math.pi * ant_idx * torch.sin(angles).unsqueeze(-1)  # (B,K,L,N)
    steering = torch.polar(torch.ones_like(phase), phase) / math.sqrt(N)

    # Aggregate sparse paths
    H = math.sqrt(N / L) * torch.sum(alpha.unsqueeze(-1) * steering, dim=2)
    return H


def generate_pilot_matrix(K: int, L_p: int) -> torch.Tensor:
    """Random Gaussian pilot: Phi ~ CN(0, 1/L_p), shape (K, L_p) complex."""
    P_r = torch.randn(K, L_p, device=device) / math.sqrt(2 * L_p)
    P_i = torch.randn(K, L_p, device=device) / math.sqrt(2 * L_p)
    return torch.complex(P_r, P_i)


def pilot_observe(H: torch.Tensor, Phi: torch.Tensor, sigma2: float) -> torch.Tensor:
    """Y = H^T Phi + N, returns (B, N, L_p) complex."""
    B, K, N = H.shape
    L_p = Phi.size(1)
    Y = H.transpose(-1, -2) @ Phi.unsqueeze(0).expand(B, -1, -1)
    noise_r = torch.randn(B, N, L_p, device=device) * math.sqrt(sigma2 / 2)
    noise_i = torch.randn(B, N, L_p, device=device) * math.sqrt(sigma2 / 2)
    return Y + torch.complex(noise_r, noise_i)


def pilot_to_real(Y: torch.Tensor) -> torch.Tensor:
    """(B, N, L_p) complex -> (B, 2*N*L_p) real."""
    return torch.cat([Y.real, Y.imag], dim=1).reshape(Y.size(0), -1)


def pilot_observe_full(H: torch.Tensor, Psi: torch.Tensor, sigma2: float) -> torch.Tensor:
    """
    Full-size pilot signal for no-ICL input tokens.
    S = H @ Psi + N, returns (B, K, L_p) complex.
    """
    B, K, _ = H.shape
    L_p = Psi.size(1)
    S = H @ Psi.unsqueeze(0).expand(B, -1, -1)
    noise_r = torch.randn(B, K, L_p, device=device) * math.sqrt(sigma2 / 2)
    noise_i = torch.randn(B, K, L_p, device=device) * math.sqrt(sigma2 / 2)
    return S + torch.complex(noise_r, noise_i)


def pilot_signal_to_tokens(S: torch.Tensor) -> torch.Tensor:
    """(B, K, L_p) complex -> (B, K, 2*L_p) real tokens."""
    return torch.cat([S.real, S.imag], dim=-1)


def tokens_to_state(tokens: torch.Tensor) -> torch.Tensor:
    """(B, K, 2*L_p) -> (B, 2*K*L_p)."""
    return tokens.reshape(tokens.size(0), -1)


def state_to_tokens(state: torch.Tensor, K: int, L_p: int) -> torch.Tensor:
    """(B, 2*K*L_p) -> (B, K, 2*L_p)."""
    return state.view(state.size(0), K, 2 * L_p)


def hhat_to_real(H_hat: torch.Tensor) -> torch.Tensor:
    """(B, K, N) complex -> (B, 2*K*N) real."""
    B, K, N = H_hat.shape
    return torch.cat([H_hat.real, H_hat.imag], dim=-1).reshape(B, 2 * K * N)


def mmse_channel_est(Y: torch.Tensor, Phi: torch.Tensor, sigma2: float) -> torch.Tensor:
    """MMSE estimate: H_hat = Phi^* (Phi^T Phi^* + sigma2 I)^{-1} Y^T, returns (B, K, N)."""
    K, L_p = Phi.shape
    A = Phi.T @ Phi.conj() + sigma2 * torch.eye(L_p, device=device, dtype=Phi.dtype)
    PA = Phi.conj() @ torch.linalg.inv(A)  # (K, L_p)
    return torch.matmul(PA.unsqueeze(0), Y.transpose(-1, -2))  # (B, K, N)


def compute_sum_rate(H: torch.Tensor, W: torch.Tensor, sigma2: float) -> torch.Tensor:
    """Sum rate for MU-MISO. H:(B,K,N), W:(B,N,K), returns (B,)."""
    HW = H @ W  # (B, K, K)
    sig = torch.abs(torch.diagonal(HW, dim1=-2, dim2=-1)) ** 2
    tot = torch.sum(torch.abs(HW) ** 2, dim=-1)
    interf = tot - sig
    SINR = sig / (interf + sigma2)
    return torch.log2(1 + SINR).sum(dim=-1)


def mmse_beamformer(H: torch.Tensor, P_max: float, sigma2: float) -> torch.Tensor:
    """MMSE (regularized ZF) beamformer. Returns W: (B, N, K)."""
    B, K, N = H.shape
    H_H = H.conj().transpose(-1, -2)
    A = H_H @ H + sigma2 * torch.eye(N, device=device, dtype=H.dtype).unsqueeze(0)
    W = torch.linalg.solve(A, H_H)
    pw = torch.sum(torch.abs(W) ** 2, dim=(1, 2), keepdim=False).real
    W = W * torch.sqrt(P_max / (pw + 1e-8)).view(B, 1, 1)
    return W


def reconstruct_precoder(H: torch.Tensor, p: torch.Tensor, lam: torch.Tensor,
                         sigma2: float) -> torch.Tensor:
    """
    Optimal structure: w_k = sqrt(p_k) * v_k
    v_k = A^{-1} h_k / ||A^{-1} h_k||, A = I + (1/sigma2) H^H diag(lam) H
    H:(B,K,N), p:(B,K), lam:(B,K) -> W:(B,N,K)
    """
    B, K, N = H.shape
    h = H.conj().transpose(-1, -2)  # (B, N, K): conjugate transpose, columns are h_1*,...,h_K*
    lam_diag = torch.diag_embed(lam / sigma2).to(torch.cfloat)  # (B, K, K)
    eye = torch.eye(N, device=device, dtype=torch.cfloat).unsqueeze(0).expand(B, -1, -1)
    A = eye + h @ lam_diag @ h.conj().transpose(-1, -2)  # (B, N, N)
    A_inv_h = torch.linalg.solve(A, h)  # (B, N, K)
    norms = torch.norm(A_inv_h, dim=1, keepdim=True).real + 1e-8  # (B, 1, K)
    V = A_inv_h / norms
    W = V * torch.sqrt(p).unsqueeze(1).to(torch.cfloat)  # (B, N, K)
    return W


###############################################################################
# 3. GROUND TRUTH LABEL GENERATION: Optimize (p, lambda) via Adam
###############################################################################
@torch.no_grad()
def _eval_rate(H, p_logits, lam_logits, P_max, sigma2):
    p = F.softmax(p_logits, dim=-1) * P_max
    lam = F.softmax(lam_logits, dim=-1) * P_max
    W = reconstruct_precoder(H, p, lam, sigma2)
    return compute_sum_rate(H, W, sigma2), p, lam


def generate_optimal_params(H: torch.Tensor, P_max: float, sigma2: float,
                            n_iters: int = 500, lr: float = 0.03,
                            n_restarts: int = 3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Optimize (p, lambda) to maximize sum-rate via Adam with multiple restarts.
    Returns: (best_p, best_lam, best_rate)
    """
    B, K, N = H.shape
    H_d = H.detach()
    best_rate = torch.full((B,), -float('inf'), device=device)
    best_p = torch.zeros(B, K, device=device)
    best_lam = torch.zeros(B, K, device=device)

    for restart in range(n_restarts):
        p_log = (torch.randn(B, K, device=device) * 0.1).detach().requires_grad_(True)
        lam_log = (torch.randn(B, K, device=device) * 0.1).detach().requires_grad_(True)
        opt = Adam([p_log, lam_log], lr=lr)

        for i in range(n_iters):
            p = F.softmax(p_log, dim=-1) * P_max
            lam = F.softplus(lam_log)          # λ only needs to be positive, no sum constraint
            W = reconstruct_precoder(H_d, p, lam, sigma2)
            rate = compute_sum_rate(H_d, W, sigma2)
            loss = -rate.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Snapshot best
        with torch.no_grad():
            p_snap = F.softmax(p_log, dim=-1) * P_max
            lam_snap = F.softplus(lam_log)
            W_snap = reconstruct_precoder(H_d, p_snap, lam_snap, sigma2)
            rate_snap = compute_sum_rate(H_d, W_snap, sigma2)
            improved = rate_snap > best_rate
            if improved.any():
                best_rate[improved] = rate_snap[improved]
                best_p[improved] = p_snap[improved]
                best_lam[improved] = lam_snap[improved]

    return best_p.detach(), best_lam.detach(), best_rate.detach()


###############################################################################
# 4. COMPUTE ALL 4 BASELINES ON A FIXED TEST SET
###############################################################################
def compute_baselines(H_test: torch.Tensor, Phi: torch.Tensor,
                      cfg: Config) -> Dict[str, float]:
    """
    Compute 4 baselines on a fixed test set.
    Returns dict with keys: mmse_perfect, mmse_imperfect, opt_perfect, opt_imperfect
    """
    B = H_test.size(0)
    sigma2 = cfg.sigma2
    P_max = cfg.P_max
    bs = min(64, B)
    results = {k: [] for k in ['mmse_perfect', 'mmse_imperfect', 'opt_perfect', 'opt_imperfect']}

    for start in range(0, B, bs):
        end = min(start + bs, B)
        H = H_test[start:end]

        # Generate pilot and estimate
        Y = pilot_observe(H, Phi, sigma2)
        H_hat = mmse_channel_est(Y, Phi, sigma2)

        with torch.no_grad():
            # 1. MMSE BF + Perfect CSI
            W1 = mmse_beamformer(H, P_max, sigma2)
            results['mmse_perfect'].append(compute_sum_rate(H, W1, sigma2))

            # 2. MMSE BF + Imperfect CSI
            W2 = mmse_beamformer(H_hat, P_max, sigma2)
            results['mmse_imperfect'].append(compute_sum_rate(H, W2, sigma2))

        # 3. Opt (p,lam) + Perfect CSI
        p3, lam3, _ = generate_optimal_params(H, P_max, sigma2,
                                               n_iters=cfg.opt_iters, lr=cfg.opt_lr, n_restarts=2)
        with torch.no_grad():
            W3 = reconstruct_precoder(H, p3, lam3, sigma2)
            results['opt_perfect'].append(compute_sum_rate(H, W3, sigma2))

        # 4. Opt (p,lam) + Imperfect CSI (design with H_hat, evaluate with H)
        p4, lam4, _ = generate_optimal_params(H_hat, P_max, sigma2,
                                               n_iters=cfg.opt_iters, lr=cfg.opt_lr, n_restarts=2)
        with torch.no_grad():
            W4 = reconstruct_precoder(H_hat, p4, lam4, sigma2)
            results['opt_imperfect'].append(compute_sum_rate(H, W4, sigma2))

        print(f"  Baselines [{end}/{B}] "
              f"MMSE-P: {torch.cat(results['mmse_perfect']).mean():.2f}  "
              f"MMSE-E: {torch.cat(results['mmse_imperfect']).mean():.2f}  "
              f"OPT-P: {torch.cat(results['opt_perfect']).mean():.2f}  "
              f"OPT-E: {torch.cat(results['opt_imperfect']).mean():.2f}")

    return {k: torch.cat(v).mean().item() for k, v in results.items()}


###############################################################################
# 5. NO-ICL TRANSFORMER (direct pilot-token sequence -> compressed solution)
###############################################################################
class CausalBlock(nn.Module):
    def __init__(self, d: int, heads: int, d_ff: int, drop: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(),
                                nn.Linear(d_ff, d), nn.Dropout(drop))

    def forward(self, x, mask):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask)[0]
        return x + self.ff(self.ln2(x))


class PilotNoICLModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.K, self.L_p, self.P_max = cfg.K, cfg.L_p, cfg.P_max
        self.proj_in = nn.Linear(cfg.in_token_dim, cfg.d_model)
        self.ln_in = nn.LayerNorm(cfg.d_model)
        self.blocks = nn.ModuleList([
            CausalBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.ln_out = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.out_dim)
        self.apply(self._init_w)
        print(f"Model params: {sum(p.numel() for p in self.parameters()):,}")

    def _init_w(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _extract(self, raw):
        p = torch.sigmoid(raw[:, :self.K])
        p = p / (p.sum(-1, keepdim=True) + 1e-8) * self.P_max
        lam = F.softplus(raw[:, self.K:2 * self.K])
        return p, lam

    def forward(self, pilot_tokens: torch.Tensor):
        """pilot_tokens: (B, K, 2*L_p) -> (p, lam), each (B, K)."""
        B, L, _ = pilot_tokens.shape
        mask = torch.triu(torch.ones(L, L, device=pilot_tokens.device, dtype=torch.bool), diagonal=1)
        x = self.ln_in(self.proj_in(pilot_tokens))
        for blk in self.blocks:
            x = blk(x, mask)
        z = self.ln_out(x).mean(dim=1)   # compressed representation
        raw = self.head(z)               # (B, 2K)
        return self._extract(raw)


###############################################################################
# 6. FIXED SUPERVISED DATASET (no expansion)
###############################################################################
class SupDataset:
    def __init__(self, state: torch.Tensor, H: torch.Tensor,
                 p: torch.Tensor, lam: torch.Tensor, rates: torch.Tensor):
        self.state = state
        self.H = H
        self.p = p
        self.lam = lam
        self.rates = rates
        self._n = state.size(0)

    @property
    def size(self):
        return self._n


###############################################################################
# 7. ON-THE-FLY EVALUATION
###############################################################################
@torch.no_grad()
def evaluate_model(model: PilotNoICLModel,
                   H_test: torch.Tensor,
                   Psi: torch.Tensor,
                   cfg: Config) -> float:
    model.eval()
    B_total = H_test.size(0)
    bs = min(cfg.batch_size, B_total)
    all_rates = []

    for start in range(0, B_total, bs):
        end = min(start + bs, B_total)
        H = H_test[start:end]
        S = pilot_observe_full(H, Psi, cfg.sigma2)
        tokens = pilot_signal_to_tokens(S)
        p_pred, lam_pred = model(tokens)
        W = reconstruct_precoder(H, p_pred, lam_pred, cfg.sigma2)
        all_rates.append(compute_sum_rate(H, W, cfg.sigma2))

    model.train()
    return torch.cat(all_rates).mean().item()


###############################################################################
# 8. MAIN TRAINING LOOP (no-ICL, no dataset growth)
###############################################################################
def train(cfg: Config):
    print("=" * 75)
    print("  PILOT-TO-SOLUTION NO-ICL BASELINE (Sparse Channel)")
    print("=" * 75)
    print(f"  Device: {device}")
    print(f"  System: K={cfg.K}, N={cfg.N}, L_p={cfg.L_p}, SNR={cfg.SNR_dB}dB, sigma2={cfg.sigma2:.6f}")
    print(f"  Channel: mmWave sparse (Nc={cfg.ch_n_clusters}, Nr={cfg.ch_n_rays_per_cluster}, "
          f"spread={cfg.ch_angle_spread_deg} deg)")
    print(f"  Input tokens: feature_dim={cfg.in_token_dim}, seq_len={cfg.seq_len}")
    print(f"  Output repr: dim={cfg.out_dim} -> (p, lambda)")
    print(f"  Model: d={cfg.d_model}, heads={cfg.n_heads}, layers={cfg.n_layers}, ff={cfg.d_ff}")
    print(f"  Train: Phase1={cfg.phase1_epochs}ep, Phase2={cfg.phase2_epochs}ep, "
          f"BS={cfg.batch_size}, steps/ep={cfg.steps_per_epoch}, "
          f"lr={cfg.lr:.0e}->{cfg.lr_min:.0e} (cosine)")
    print()

    # Baseline pilot matrix (for imperfect-CSI baselines)
    Phi = generate_pilot_matrix(cfg.K, cfg.L_p)
    # No-ICL input pilot matrix (full-size pilot signal, shape N x L_p)
    Psi = generate_pilot_matrix(cfg.N, cfg.L_p)

    print("Generating fixed test set...")
    H_test = generate_channel(cfg.n_test, cfg.K, cfg.N,
                              n_clusters=cfg.ch_n_clusters,
                              n_rays_per_cluster=cfg.ch_n_rays_per_cluster,
                              angle_spread_deg=cfg.ch_angle_spread_deg)

    print("\nComputing baselines (this may take several minutes)...")
    baselines = compute_baselines(H_test, Phi, cfg)
    print("\n" + "-" * 75)
    print("  BASELINES on fixed test set:")
    print(f"    1. MMSE BF + Perfect CSI:      {baselines['mmse_perfect']:.4f} bps/Hz")
    print(f"    2. MMSE BF + Imperfect CSI:    {baselines['mmse_imperfect']:.4f} bps/Hz")
    print(f"    3. Opt(p,lam) + Perfect CSI:   {baselines['opt_perfect']:.4f} bps/Hz")
    print(f"    4. Opt(p,lam) + Imperfect CSI: {baselines['opt_imperfect']:.4f} bps/Hz")
    print("-" * 75)

    # Fixed small supervised dataset: (state, label) with true-CSI WMMSE labels
    print(f"\nGenerating fixed supervised dataset (M0={cfg.initial_dataset_size}, true-CSI labels)...")
    states, H_all, p_all, lam_all, rate_all = [], [], [], [], []
    gen_bs = min(64, cfg.initial_dataset_size)
    for s in range(0, cfg.initial_dataset_size, gen_bs):
        e = min(s + gen_bs, cfg.initial_dataset_size)
        H_b = generate_channel(e - s, cfg.K, cfg.N,
                               n_clusters=cfg.ch_n_clusters,
                               n_rays_per_cluster=cfg.ch_n_rays_per_cluster,
                               angle_spread_deg=cfg.ch_angle_spread_deg)
        S_b = pilot_observe_full(H_b, Psi, cfg.sigma2)
        state_b = tokens_to_state(pilot_signal_to_tokens(S_b))
        p_b, lam_b, _ = generate_optimal_params(H_b, cfg.P_max, cfg.sigma2,
                                                n_iters=cfg.opt_iters, lr=cfg.opt_lr)
        with torch.no_grad():
            W_b = reconstruct_precoder(H_b, p_b, lam_b, cfg.sigma2)
            r_b = compute_sum_rate(H_b, W_b, cfg.sigma2)
        states.append(state_b)
        H_all.append(H_b)
        p_all.append(p_b)
        lam_all.append(lam_b)
        rate_all.append(r_b)
        print(f"  [{e}/{cfg.initial_dataset_size}] avg rate: {r_b.mean():.4f}", flush=True)

    sup_data = SupDataset(
        state=torch.cat(states, dim=0),
        H=torch.cat(H_all, dim=0),
        p=torch.cat(p_all, dim=0),
        lam=torch.cat(lam_all, dim=0),
        rates=torch.cat(rate_all, dim=0),
    )
    print(f"  Dataset ready: {sup_data.size} samples, avg rate: {sup_data.rates.mean():.4f}")

    model = PilotNoICLModel(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.total_epochs,
                                                      eta_min=cfg.lr_min)

    print("\n" + "=" * 75)
    header = (f"{'Ep':>3s} {'Ph':>2s} {'r':>4s} | {'MSE':>8s} {'UnsRate':>8s} | "
              f"{'TestRate':>8s} | {'DS':>5s} {'Add':>4s} | "
              f"{'B1':>6s} {'B2':>6s} {'B3':>6s} {'B4':>6s}")
    print(header)
    print(f"{'':>3s} {'':>2s} {'':>4s} | {'':>8s} {'':>8s} | "
          f"{'':>8s} | {'':>5s} {'':>4s} | "
          f"{baselines['mmse_perfect']:>6.2f} {baselines['mmse_imperfect']:>6.2f} "
          f"{baselines['opt_perfect']:>6.2f} {baselines['opt_imperfect']:>6.2f}  <- baselines")
    print("-" * 75)

    best_test_rate = 0.0
    history: Dict[str, List] = {'test_rate': [], 'train_rate': [], 'mse': [],
                                'ds_size': [], 'ep_added': [], 'phase': []}

    for epoch in range(cfg.total_epochs):
        model.train()
        t0 = time.time()

        if epoch < cfg.phase1_epochs:
            phase, r = 1, 0.0
        else:
            phase = 2
            prog = (epoch - cfg.phase1_epochs) / max(1, cfg.phase2_epochs - 1)
            r = min(prog * cfg.r_max, cfg.r_max)

        ep_mse, ep_rate, ep_steps = 0.0, 0.0, 0
        ep_added = 0  # no dataset expansion in no-ICL baseline

        for _ in range(cfg.steps_per_epoch):
            B = cfg.batch_size
            is_unsup = torch.rand(B, device=device) < r

            # supervised part from fixed dataset
            s_idx = torch.randint(0, sup_data.size, (B,), device=device)
            q_state_sup = sup_data.state[s_idx]
            q_H_sup = sup_data.H[s_idx]
            q_p_gt = sup_data.p[s_idx]
            q_lam_gt = sup_data.lam[s_idx]

            # unsupervised part from random channels
            q_H_unsup = generate_channel(B, cfg.K, cfg.N,
                                         n_clusters=cfg.ch_n_clusters,
                                         n_rays_per_cluster=cfg.ch_n_rays_per_cluster,
                                         angle_spread_deg=cfg.ch_angle_spread_deg)
            S_unsup = pilot_observe_full(q_H_unsup, Psi, cfg.sigma2)
            q_state_unsup = tokens_to_state(pilot_signal_to_tokens(S_unsup))

            q_H = torch.where(is_unsup.view(B, 1, 1).expand_as(q_H_sup), q_H_unsup, q_H_sup)
            q_state = torch.where(is_unsup.view(B, 1).expand_as(q_state_sup), q_state_unsup, q_state_sup)
            q_tokens = state_to_tokens(q_state, cfg.K, cfg.L_p)

            p_pred, lam_pred = model(q_tokens)

            mse_per = (F.mse_loss(p_pred, q_p_gt, reduction='none').sum(-1) +
                       F.mse_loss(lam_pred, q_lam_gt, reduction='none').sum(-1))

            W_pred = reconstruct_precoder(q_H, p_pred, lam_pred, cfg.sigma2)
            rate_pred = compute_sum_rate(q_H, W_pred, cfg.sigma2)

            loss_per = torch.where(is_unsup, -rate_pred * cfg.unsup_scale, mse_per)
            loss = loss_per.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            n_sup = (~is_unsup).sum().item()
            n_unsup = is_unsup.sum().item()
            if n_sup > 0:
                ep_mse += mse_per[~is_unsup].mean().item()
            if n_unsup > 0:
                ep_rate += rate_pred[is_unsup].mean().item()
            ep_steps += 1

        scheduler.step()

        test_rate = evaluate_model(model, H_test, Psi, cfg)
        best_test_rate = max(best_test_rate, test_rate)

        avg_mse = ep_mse / max(1, ep_steps)
        avg_rate = ep_rate / max(1, ep_steps)
        dt = time.time() - t0

        history['test_rate'].append(test_rate)
        history['train_rate'].append(avg_rate)
        history['mse'].append(avg_mse)
        history['ds_size'].append(sup_data.size)
        history['ep_added'].append(ep_added)
        history['phase'].append(phase)

        if epoch == cfg.phase1_epochs:
            ds_rate_avg = sup_data.rates.mean().item()
            print(f"\n  [Loss balance] Phase 1 final MSE={avg_mse:.5f}, "
                  f"dataset avg rate={ds_rate_avg:.2f} bps/Hz")
            print(f"  [Loss balance] |rate|/|MSE| ~ {ds_rate_avg/max(avg_mse,1e-8):.0f}x  "
                  f"-> unsup_scale={cfg.unsup_scale:.4f}  "
                  f"(scaled rate loss ~ {ds_rate_avg*cfg.unsup_scale:.4f})\n", flush=True)

        print(f"{epoch+1:3d} {phase:>2d} {r:.2f} | "
              f"{avg_mse:8.5f} {avg_rate:8.4f} | "
              f"{test_rate:8.4f} | "
              f"{sup_data.size:5d} {ep_added:4d} | "
              f"{baselines['mmse_perfect']:6.2f} {baselines['mmse_imperfect']:6.2f} "
              f"{baselines['opt_perfect']:6.2f} {baselines['opt_imperfect']:6.2f}  "
              f"({dt:.1f}s)", flush=True)

    b4 = baselines['opt_imperfect']
    rows = [
        ("MMSE BF + Perfect CSI",        baselines['mmse_perfect']),
        ("MMSE BF + Imperfect CSI",      baselines['mmse_imperfect']),
        ("Opt(p,lam) + Perfect CSI",     baselines['opt_perfect']),
        ("Opt(p,lam) + Imperfect CSI",   b4),
        ("No-ICL Model (best epoch)",    best_test_rate),
        ("No-ICL Model (final epoch)",   test_rate),
    ]
    print("\n" + "=" * 68)
    print("  TRAINING COMPLETE")
    print("=" * 68)
    print(f"  {'Method':<38} {'bps/Hz':>8}  {'vs B4':>7}")
    print("-" * 68)
    for name, val in rows:
        pct = f"{100*val/b4:.1f}%" if b4 > 0 else "-"
        print(f"  {name:<38} {val:>8.4f}  {pct:>7}")
    print(f"\n  Fixed supervised dataset: {sup_data.size} samples (no expansion)")
    print("=" * 68)

    _plot_curves(history, baselines, cfg, save_path='training_curves_noicl_sparse.png')
    return model, sup_data, baselines


###############################################################################
# PLOTTING
###############################################################################
def _plot_curves(history: Dict, baselines: Dict, cfg: 'Config',
                 save_path: str = 'training_curves_noicl_sparse.png') -> None:
    ep = list(range(1, len(history['test_rate']) + 1))
    p1 = cfg.phase1_epochs

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"No-ICL Pilot2Solution  K={cfg.K} N={cfg.N} L_p={cfg.L_p} "
                 f"SNR={cfg.SNR_dB}dB", fontsize=12)

    ax = axes[0, 0]
    ax.plot(ep, history['test_rate'], 'b-o', ms=2, lw=1.4, label='No-ICL model (test)')
    colors = ['#333333', '#2ca02c', '#d62728', '#ff7f0e']
    labels = ['MMSE BF Perfect', 'MMSE BF Imperfect', 'Opt(p,lam) Perfect', 'Opt(p,lam) Imperfect']
    keys = ['mmse_perfect', 'mmse_imperfect', 'opt_perfect', 'opt_imperfect']
    for c, lbl, k in zip(colors, labels, keys):
        ax.axhline(baselines[k], color=c, linestyle='--', lw=1.2, label=lbl)
    ax.axvline(p1 + 0.5, color='gray', linestyle=':', lw=1.0, label='Ph1→Ph2')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Sum Rate (bps/Hz)')
    ax.set_title('Test Sum Rate vs Epoch')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    p2_ep = [e for e, ph in zip(ep, history['phase']) if ph == 2]
    p2_rates = [r for r, ph in zip(history['train_rate'], history['phase']) if ph == 2]
    ax.plot(p2_ep, p2_rates, 'r-o', ms=2, lw=1.4, label='Train rate (unsup)')
    ax.axhline(baselines['opt_imperfect'], color='#ff7f0e', linestyle='--', lw=1.2,
               label='Opt(p,lam) Imperfect')
    ax.axhline(baselines['mmse_imperfect'], color='#2ca02c', linestyle='--', lw=1.2,
               label='MMSE BF Imperfect')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Sum Rate (bps/Hz)')
    ax.set_title('Unsupervised Training Rate (Phase 2)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    p1_ep = [e for e, ph in zip(ep, history['phase']) if ph == 1]
    p1_mse = [m for m, ph in zip(history['mse'], history['phase']) if ph == 1]
    ax.semilogy(p1_ep, p1_mse, 'g-o', ms=2, lw=1.4, label='Supervised MSE')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE (log)')
    ax.set_title('Supervised MSE Loss (Phase 1)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(ep, history['ds_size'], 'purple', lw=1.4, label='Dataset size (fixed)')
    ax.axvline(p1 + 0.5, color='gray', linestyle=':', lw=1.0, label='Ph1→Ph2')
    ax2 = ax.twinx()
    ax2.bar(ep, history['ep_added'], color='mediumpurple', alpha=0.4, label='Added/epoch')
    ax2.set_ylabel('Samples added per epoch', color='mediumpurple')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Total dataset size')
    ax.set_title('Dataset (No Expansion)')
    ax.legend(loc='upper left', fontsize=8); ax2.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Training curves saved → {save_path}")


###############################################################################
# 9. ENTRY POINT
###############################################################################
if __name__ == "__main__":
    cfg = Config(
        K=32, N=32, L_p=20,
        P_max=1.0, SNR_dB=20,
        n_demos=32,
        encoder_hidden=256,
        d_model=512, n_heads=8, n_layers=6, d_ff=1024,
        dropout=0.0,
        batch_size=64,
        lr=2e-4, lr_min=5e-5,
        weight_decay=1e-4,
        initial_dataset_size=1024,
        opt_iters=500, opt_lr=0.03,
        phase1_epochs=16,
        phase2_epochs=500,
        steps_per_epoch=80,
        r_max=0.85,
        unsup_scale=0.005,
        tau_start=30, tau_end=65,
        max_dataset_size=50000,
        n_test=300,
    )
    train(cfg)
