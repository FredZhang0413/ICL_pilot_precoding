"""
Pilot-Based In-Context Learning for MU-MISO Precoding — v5
Direct Pilot-to-Beamformer via Compressed ICL (No Channel Estimation)

ARCHITECTURE OVERVIEW:
  Phase 0 (Pretraining):
    0a) PilotEncoder + ChannelDecoder (EDN): Y -> z_pilot -> H_hat, loss=||H-H_hat||^2
    0b) BFEncoder + BFDecoder (EDN): W* -> c_bf -> W_hat, loss=||W*-W_hat||^2_F
  Phase 1 (Supervised warm-start):
    Freeze encoders. Train ICLTransformer + BFDecoder on MSE in beamformer space.
  Phase 2 (Curriculum self-evolution):
    Train ICLTransformer + BFDecoder on sum-rate loss. Self-bootstrapping with
    dual-threshold admission.

INFERENCE (no channel estimation needed):
  Y_query -> PilotEncoder -> z_q
  {(Y_i, W_i*)} -> PilotEncoder, BFEncoder -> {(z_i, c_i)}  (from dataset)
  ICL Transformer({z_1, c_1, ..., z_l, c_l, z_q}) -> c_hat
  BFDecoder(c_hat) -> W_hat  (full-size beamformer, power-normalized)

KEY FEATURES:
  - FiLM conditioning on BOTH encoders (SNR-adaptive)
  - Pretrained encoder-decoder networks (Phase 0)
  - Dual-threshold self-bootstrapping (per-instance + cross-instance)
  - Periodic bottom-pruning of unsupervised samples
  - Dataset stores [H, Y, W_label] in full size
  - Truncated DFT pilot matrix for sparse channels
  - All rates evaluated on TRUE channel H

Requirements: PyTorch >= 2.0, matplotlib
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
import warnings
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_global_seed(seed: int, deterministic: bool = False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


###############################################################################
# 1. CONFIGURATION
###############################################################################
class Config:
    def __init__(self, **kwargs):
        # Reproducibility
        self.seed = kwargs.get('seed', 2026)
        self.deterministic = kwargs.get('deterministic', False)

        # System
        self.K = kwargs.get('K', 32)
        self.N = kwargs.get('N', 28)
        self.L_p = kwargs.get('L_p', 20)     # pilot length < min(K, N)
        self.P_max = kwargs.get('P_max', 1.0)
        self.SNR_dB = kwargs.get('SNR_dB', 20)
        self.sigma2 = self.P_max / (10 ** (self.SNR_dB / 10))

        # Sparse channel
        self.ch_n_clusters = kwargs.get('ch_n_clusters', 3)
        self.ch_n_rays = kwargs.get('ch_n_rays', 5)
        self.ch_spread_deg = kwargs.get('ch_spread_deg', 10.0)

        # Token / compressed dimensions
        self.D_tok = kwargs.get('D_tok', 256)  # unified token dim for ICL sequence

        # Encoder-Decoder pretraining (Phase 0)
        self.edn_hidden = kwargs.get('edn_hidden', 512)
        self.edn_epochs = kwargs.get('edn_epochs', 2000)
        self.edn_lr = kwargs.get('edn_lr', 1e-3)
        self.edn_batch = kwargs.get('edn_batch', 128)
        self.edn_n_samples = kwargs.get('edn_n_samples', 5000)

        # ICL Transformer
        self.n_demos = kwargs.get('n_demos', 5)
        self.d_model = kwargs.get('d_model', 512)
        self.n_heads = kwargs.get('n_heads', 8)
        self.n_layers = kwargs.get('n_layers', 6)
        self.d_ff = kwargs.get('d_ff', 1024)
        self.dropout = kwargs.get('dropout', 0.0)

        # Context selection
        self.context_k_cand = kwargs.get('context_k_cand', 20)
        self.context_pool_size = kwargs.get('context_pool_size', 512)
        self.mmr_alpha = kwargs.get('mmr_alpha', 0.65)

        # Training
        self.batch_size = kwargs.get('batch_size', 64)
        self.lr = kwargs.get('lr', 2e-4)
        self.lr_min = kwargs.get('lr_min', 5e-5)
        self.weight_decay = kwargs.get('weight_decay', 1e-4)
        self.initial_ds_size = kwargs.get('initial_ds_size', 1024)
        self.wmmse_iters = kwargs.get('wmmse_iters', 500)
        self.wmmse_lr = kwargs.get('wmmse_lr', 0.03)
        self.wmmse_label_restarts = kwargs.get('wmmse_label_restarts', 2)
        self.unsup_scale = kwargs.get('unsup_scale', 0.01)

        # Hybrid loss (soft switch: MSE -> rate)
        self.hybrid_calib_steps = kwargs.get('hybrid_calib_steps', 4)
        self.hybrid_calib_batch = kwargs.get('hybrid_calib_batch', 16)
        self.hybrid_rate_gain = kwargs.get('hybrid_rate_gain', 1.0)
        self.hybrid_rate_scale_min = kwargs.get('hybrid_rate_scale_min', 0.01)
        self.hybrid_rate_scale_max = kwargs.get('hybrid_rate_scale_max', 1000.0)
        self.hybrid_switch_power = kwargs.get('hybrid_switch_power', 1.0)
        self.unsup_mix_transition_size = kwargs.get('unsup_mix_transition_size', 1024)

        # Residual anti-lazy regularization
        # Encourage model output to deviate from reference MMSE-imperfect beamformer.
        self.lazy_residual_min_ratio = kwargs.get('lazy_residual_min_ratio', 0.10)
        self.lazy_residual_weight = kwargs.get('lazy_residual_weight', 0.05)
        # Supervised branch: also learn target residual direction, not only final W.
        self.sup_residual_target_weight = kwargs.get('sup_residual_target_weight', 0.25)

        # Curriculum
        self.phase1_epochs = kwargs.get('phase1_epochs', 30)
        self.phase2_epochs = kwargs.get('phase2_epochs', 500)
        self.total_epochs = self.phase1_epochs + self.phase2_epochs
        self.r_max = kwargs.get('r_max', 0.85)
        self.steps_per_epoch = kwargs.get('steps_per_epoch', 80)

        # Dual-threshold bootstrapping
        self.boot_alpha_start = kwargs.get('boot_alpha_start', 0.90)
        self.boot_alpha_end = kwargs.get('boot_alpha_end', 1.05)
        self.boot_beta_start = kwargs.get('boot_beta_start', 0.50)
        self.boot_beta_end = kwargs.get('boot_beta_end', 0.80)
        self.max_ds_size = kwargs.get('max_ds_size', 50000)

        # Bottom pruning
        self.prune_every = kwargs.get('prune_every', 5)
        self.prune_drop_start = kwargs.get('prune_drop_start', 0.0)
        self.prune_drop_end = kwargs.get('prune_drop_end', 0.10)
        self.prune_min_unsup = kwargs.get('prune_min_unsup', 4096)

        # Plateau detection
        self.plateau_window = kwargs.get('plateau_window', 10)
        self.plateau_thresh = kwargs.get('plateau_thresh', 0.3)
        self.plateau_boost = kwargs.get('plateau_boost', 0.10)

        # Eval
        self.n_test = kwargs.get('n_test', 200)
        self.rate_save_every = kwargs.get('rate_save_every', 10)
        self.train_rate_pt = kwargs.get('train_rate_pt', 'training_rate.pt')
        self.test_rate_pt = kwargs.get('test_rate_pt', 'testing_rate.pt')
        self.rate_curve_png = kwargs.get('rate_curve_png', 'train_test_rate_curve.png')


###############################################################################
# 2. SIGNAL PROCESSING
###############################################################################
def generate_channel(B, K, N, n_cl=3, n_ray=5, spread=10.0):
    """Cluster-based sparse mmWave channel. Returns (B,K,N) complex."""
    L = n_cl * n_ray; asp = math.radians(spread)
    cm = (torch.rand(B,K,n_cl,1,device=device)-0.5)*math.pi
    ro = torch.randn(B,K,n_cl,n_ray,device=device)*asp
    ang = (cm+ro).clamp(-math.pi/2,math.pi/2).reshape(B,K,L)
    alp = (torch.randn(B,K,L,device=device)+1j*torch.randn(B,K,L,device=device))/math.sqrt(2)
    idx = torch.arange(N,device=device,dtype=torch.float32).view(1,1,1,N)
    ph = math.pi*idx*torch.sin(ang).unsqueeze(-1)
    st = torch.polar(torch.ones_like(ph), ph)/math.sqrt(N)
    return math.sqrt(N/L)*torch.sum(alp.unsqueeze(-1)*st, dim=2)


def generate_pilot_dft(K, L_p):
    """Truncated DFT pilot matrix — better coherence for sparse channels than Gaussian."""
    F_full = torch.fft.fft(torch.eye(K, device=device)) / math.sqrt(K)
    return F_full[:, :L_p].contiguous()  # (K, L_p) complex


def generate_pilot_gaussian(K, L_p):
    """Random Gaussian pilot as fallback."""
    r = torch.randn(K, L_p, device=device)/math.sqrt(2*L_p)
    i = torch.randn(K, L_p, device=device)/math.sqrt(2*L_p)
    return torch.complex(r, i)


def pilot_observe(H, Phi, sigma2):
    """Y = H^T Phi + N, returns (B,N,L_p) complex."""
    B,K,N = H.shape; Lp = Phi.size(1)
    Y = H.transpose(-1,-2) @ Phi.unsqueeze(0).expand(B,-1,-1)
    nr = torch.randn(B,N,Lp,device=device)*math.sqrt(sigma2/2)
    ni = torch.randn(B,N,Lp,device=device)*math.sqrt(sigma2/2)
    return Y + torch.complex(nr, ni)


def pilot_to_real(Y):
    """(B,N,L_p) complex -> (B, 2*N*L_p) real."""
    return torch.cat([Y.real, Y.imag], dim=1).reshape(Y.size(0), -1)


def real_to_pilot(x, N, L_p):
    """(B, 2*N*L_p) real -> (B,N,L_p) complex."""
    B = x.size(0)
    x = x.view(B, 2*N, L_p)
    return torch.complex(x[:, :N, :], x[:, N:, :])


def bf_to_real(W):
    """(B,N,K) complex -> (B, 2*N*K) real."""
    return torch.cat([W.real, W.imag], dim=1).reshape(W.size(0), -1)


def real_to_bf(x, N, K):
    """(B, 2*N*K) real -> (B,N,K) complex."""
    B = x.size(0)
    x = x.view(B, 2*N, K)
    return torch.complex(x[:, :N, :], x[:, N:, :])


def channel_to_real(H):
    """(B,K,N) complex -> (B, 2*K*N) real."""
    return torch.cat([H.real, H.imag], dim=-1).reshape(H.size(0), -1)


def real_to_channel(x, K, N):
    """(B, 2*K*N) real -> (B,K,N) complex."""
    B = x.size(0); x = x.view(B, K, 2*N)
    return torch.complex(x[:,:,:N], x[:,:,N:])


def mmse_channel_est(Y, Phi, sigma2):
    K,Lp = Phi.shape
    A = Phi.T @ Phi.conj() + sigma2*torch.eye(Lp, device=device, dtype=Phi.dtype)
    PA = Phi.conj() @ torch.linalg.inv(A)
    return torch.matmul(PA.unsqueeze(0), Y.transpose(-1,-2))


def ls_channel_est(Y, Phi):
    """Least-squares channel estimate using pseudo-inverse. Y:(B,N,Lp) -> H_hat:(B,K,N)."""
    Phi_pinv = torch.linalg.pinv(Phi)  # (Lp, K)
    Ht_hat = Y @ Phi_pinv.unsqueeze(0)  # (B, N, K)
    return Ht_hat.transpose(-1, -2).contiguous()  # (B, K, N)


def compute_sum_rate(H, W, sigma2):
    """Always on TRUE H. H:(B,K,N), W:(B,N,K), returns (B,)."""
    HW = H @ W
    sig = torch.abs(torch.diagonal(HW, dim1=-2, dim2=-1))**2
    tot = torch.sum(torch.abs(HW)**2, dim=-1)
    SINR = sig / (tot - sig + sigma2)
    return torch.log2(1+SINR).sum(-1)


def mmse_beamformer(H, P_max, sigma2):
    B,K,N = H.shape; HH = H.conj().transpose(-1,-2)
    A = HH@H + sigma2*torch.eye(N,device=device,dtype=H.dtype).unsqueeze(0)
    W = torch.linalg.solve(A, HH)
    pw = torch.sum(torch.abs(W)**2, dim=(1,2)).real
    return W*torch.sqrt(P_max/(pw+1e-8)).view(B,1,1)


def power_normalize(W, P_max):
    """Normalize beamformer to satisfy ||W||_F^2 = P_max."""
    pw = torch.sum(torch.abs(W)**2, dim=(1,2), keepdim=True).real
    return W * torch.sqrt(P_max / (pw + 1e-8))


def generate_wmmse_labels(H, P_max, sigma2, n_iters=500, lr=0.03, n_restarts=3):
    """Generate WMMSE-proxy beamformer labels via gradient-based optimization on TRUE H."""
    B,K,N = H.shape; Hd = H.detach()
    best_rate = torch.full((B,), -float('inf'), device=device)
    best_W = torch.zeros(B, N, K, device=device, dtype=torch.cfloat)
    for _ in range(n_restarts):
        Wr = (torch.randn(B,N,K,device=device)*0.05).requires_grad_(True)
        Wi = (torch.randn(B,N,K,device=device)*0.05).requires_grad_(True)
        opt = Adam([Wr, Wi], lr=lr)
        for _ in range(n_iters):
            W = power_normalize(torch.complex(Wr, Wi), P_max)
            rate = compute_sum_rate(Hd, W, sigma2)
            (-rate.sum()).backward(); opt.step(); opt.zero_grad()
        with torch.no_grad():
            W_snap = power_normalize(torch.complex(Wr, Wi), P_max)
            r_snap = compute_sum_rate(Hd, W_snap, sigma2)
            imp = r_snap > best_rate
            if imp.any():
                best_rate[imp] = r_snap[imp]; best_W[imp] = W_snap[imp]
    return best_W.detach(), best_rate.detach()


###############################################################################
# 3. FiLM MODULE (shared by both encoders)
###############################################################################
class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation conditioned on log(sigma2)."""
    def __init__(self, n_channels):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, n_channels), nn.GELU(),
                                 nn.Linear(n_channels, 2*n_channels))
    def forward(self, x, sigma2):
        """x: (B, C, *), sigma2: float -> modulated x."""
        B = x.size(0); C = x.size(1)
        log_s = torch.full((B,1), math.log(sigma2+1e-10), device=x.device)
        params = self.net(log_s)  # (B, 2C)
        gamma = params[:, :C]; beta = params[:, C:]
        # Reshape for broadcasting: works for (B,C,L) or (B,C)
        if x.dim() == 3:
            gamma = gamma.unsqueeze(-1); beta = beta.unsqueeze(-1)
        return gamma * x + beta


###############################################################################
# 4. PILOT ENCODER + CHANNEL DECODER (pretrained EDN)
###############################################################################
class PilotEncoder(nn.Module):
    """CNN + attention pooling + FiLM. Maps Y_real to z ∈ R^{D_tok}."""
    def __init__(self, N, L_p, D_tok, hidden=512):
        super().__init__()
        self.N, self.L_p, self.D_tok = N, L_p, D_tok
        self.conv1 = nn.Conv1d(2*N, hidden, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.ln = nn.LayerNorm(hidden)
        self.attn_q = nn.Parameter(torch.randn(1,1,hidden)*0.02)
        self.attn_k = nn.Linear(hidden, hidden)
        self.attn_v = nn.Linear(hidden, hidden)
        self.proj = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                  nn.Linear(hidden, D_tok))

    def forward(self, x, sigma2=None):
        """x: (B, 2*N*L_p) -> (B, D_tok)."""
        B = x.size(0)
        x = x.view(B, 2*self.N, self.L_p)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = self.ln(x.transpose(1,2))  # (B, Lp, H)
        q = self.attn_q.expand(B,-1,-1)
        k,v = self.attn_k(x), self.attn_v(x)
        w = F.softmax(torch.bmm(q, k.transpose(1,2))/math.sqrt(k.size(-1)), -1)
        return self.proj(torch.bmm(w, v).squeeze(1))


class ChannelDecoder(nn.Module):
    """Decodes z_pilot back to H_hat. Used only during Phase 0a pretraining."""
    def __init__(self, K, N, D_tok, hidden=512):
        super().__init__()
        self.K, self.N = K, N
        self.net = nn.Sequential(nn.Linear(D_tok, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 2*K*N))
    def forward(self, z):
        return self.net(z)  # (B, 2KN) real


###############################################################################
# 5. BF ENCODER + BF DECODER (pretrained EDN)
###############################################################################
class BFEncoder(nn.Module):
    """Compresses W ∈ C^{N×K} to c ∈ R^{D_tok}. FiLM-conditioned on SNR."""
    def __init__(self, N, K, D_tok, hidden=512):
        super().__init__()
        self.N, self.K = N, K
        self.net = nn.Sequential(nn.Linear(2*N*K, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU())
        self.proj = nn.Linear(hidden, D_tok)

    def forward(self, W_real, sigma2=None):
        """W_real: (B, 2NK) -> (B, D_tok)."""
        h = self.net(W_real)  # (B, hidden)
        return self.proj(h)


class BFDecoder(nn.Module):
    """Decompresses c_hat ∈ R^{D_tok} to W_hat ∈ C^{N×K}, power-normalized."""
    def __init__(self, N, K, D_tok, P_max, hidden=512):
        super().__init__()
        self.N, self.K, self.P_max = N, K, P_max
        self.net = nn.Sequential(nn.Linear(D_tok, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 2*N*K))

    def forward(self, c, normalize=True):
        """c: (B, D_tok) -> W_hat: (B, N, K) complex."""
        x = self.net(c)  # (B, 2NK)
        W = real_to_bf(x, self.N, self.K)
        return power_normalize(W, self.P_max) if normalize else W


###############################################################################
# 6. PHASE 0: PRETRAIN ENCODER-DECODER NETWORKS
###############################################################################
def build_static_labeled_dataset(cfg, Phi, n_samples):
    """
    Build ONE fixed labeled dataset for:
      - Phase 0a pilot EDN pretrain
      - Phase 0b BF EDN pretrain
      - Phase 1 initial supervised dataset

    Each datapoint includes at least:
            H (true channel), Y_real (pilot observation), W_real (MMSE-perfect label),
            label_rate, mmse_rate
    """
    print("\n[Static labels] Building one shared labeled dataset...")
    print(f"  total samples={n_samples} (shared by Phase0 + initial dataset)")

    H_cache, Y_cache, H_real_cache = [], [], []
    W_cache, lr_cache, mr_cache = [], [], []

    for s in range(0, n_samples, cfg.edn_batch):
        e = min(s + cfg.edn_batch, n_samples)
        b = e - s
        H = generate_channel(b, cfg.K, cfg.N,
                             cfg.ch_n_clusters, cfg.ch_n_rays, cfg.ch_spread_deg)
        Y = pilot_observe(H, Phi, cfg.sigma2)

        W_star = mmse_beamformer(H, cfg.P_max, cfg.sigma2)
        label_r = compute_sum_rate(H, W_star, cfg.sigma2)
        mmse_r = label_r

        H_cache.append(H.detach())
        Y_cache.append(pilot_to_real(Y))
        H_real_cache.append(channel_to_real(H))
        W_cache.append(bf_to_real(W_star))
        lr_cache.append(label_r.detach())
        mr_cache.append(mmse_r.detach())

        if (e % max(cfg.edn_batch * 10, 1) == 0) or (e == n_samples):
            print(f"    cache [{e}/{n_samples}]", flush=True)

    out = {
        'H': torch.cat(H_cache, dim=0),
        'Y_real': torch.cat(Y_cache, dim=0),
        'H_real': torch.cat(H_real_cache, dim=0),
        'W_real': torch.cat(W_cache, dim=0),
        'label_rate': torch.cat(lr_cache, dim=0),
        'wmmse_rate': torch.cat(lr_cache, dim=0),
        'mmse_rate': torch.cat(mr_cache, dim=0),
    }
    print(f"  Shared labeled cache ready: {out['H'].size(0)} samples")
    return out


def pretrain_pilot_edn(cfg, labeled_cache):
    """Phase 0a: Train PilotEncoder + ChannelDecoder as autoencoder."""
    print("\n[Phase 0a] Pretraining PilotEncoder + ChannelDecoder...")
    enc = PilotEncoder(cfg.N, cfg.L_p, cfg.D_tok, cfg.edn_hidden).to(device)
    dec = ChannelDecoder(cfg.K, cfg.N, cfg.D_tok, cfg.edn_hidden).to(device)
    opt = Adam(list(enc.parameters()) + list(dec.parameters()), lr=cfg.edn_lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, cfg.edn_epochs, eta_min=0)

    n_total = min(cfg.edn_n_samples, labeled_cache['Y_real'].size(0))
    Y_cache = labeled_cache['Y_real'][:n_total]
    H_cache = labeled_cache['H_real'][:n_total]
    print(f"  Using shared labeled cache: {n_total} samples")

    for ep in range(cfg.edn_epochs):
        ep_loss = 0.0; n_batch = 0
        perm = torch.randperm(n_total, device=device)
        for s in range(0, n_total, cfg.edn_batch):
            e = min(s + cfg.edn_batch, n_total)
            idx = perm[s:e]
            Y_real = Y_cache[idx]
            H_real = H_cache[idx]
            z = enc(Y_real, sigma2=cfg.sigma2)
            H_hat_real = dec(z)
            loss = F.mse_loss(H_hat_real, H_real)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters())+list(dec.parameters()), 5.0)
            opt.step(); ep_loss += loss.item(); n_batch += 1
        sched.step()
        if (ep+1) % 20 == 0 or ep == 0:
            print(f"  Epoch {ep+1}/{cfg.edn_epochs}  MSE={ep_loss/n_batch:.6f}")
    print(f"  PilotEncoder pretrained. Params: {sum(p.numel() for p in enc.parameters()):,}")
    return enc


def pretrain_bf_edn(cfg, labeled_cache):
    """Phase 0b: Train BFEncoder + BFDecoder as autoencoder."""
    print("\n[Phase 0b] Pretraining BFEncoder + BFDecoder...")
    enc = BFEncoder(cfg.N, cfg.K, cfg.D_tok, cfg.edn_hidden).to(device)
    dec = BFDecoder(cfg.N, cfg.K, cfg.D_tok, cfg.P_max, cfg.edn_hidden).to(device)
    opt = Adam(list(enc.parameters()) + list(dec.parameters()), lr=cfg.edn_lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, cfg.edn_epochs, eta_min=0)

    n_total = min(cfg.edn_n_samples, labeled_cache['W_real'].size(0))
    W_cache = labeled_cache['W_real'][:n_total]
    print(f"  Using shared labeled cache: {n_total} samples")

    for ep in range(cfg.edn_epochs):
        ep_loss = 0.0; n_batch = 0
        perm = torch.randperm(n_total, device=device)
        for s in range(0, n_total, cfg.edn_batch):
            e = min(s + cfg.edn_batch, n_total)
            idx = perm[s:e]
            W_real = W_cache[idx]
            c = enc(W_real, sigma2=cfg.sigma2)
            W_hat = dec(c)
            W_hat_real = bf_to_real(W_hat)
            loss = F.mse_loss(W_hat_real, W_real)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters())+list(dec.parameters()), 5.0)
            opt.step(); ep_loss += loss.item(); n_batch += 1
        sched.step()
        if (ep+1) % 20 == 0 or ep == 0:
            print(f"  Epoch {ep+1}/{cfg.edn_epochs}  MSE={ep_loss/n_batch:.6f}")
    print(f"  BFEncoder pretrained. Params: {sum(p.numel() for p in enc.parameters()):,}")
    return enc, dec


###############################################################################
# 7. ICL TRANSFORMER (unchanged structure)
###############################################################################
class CausalBlock(nn.Module):
    def __init__(self, d, heads, d_ff, drop=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d,d_ff), nn.GELU(), nn.Linear(d_ff,d), nn.Dropout(drop))
    def forward(self, x, mask):
        h = self.ln1(x); x = x+self.attn(h,h,h,attn_mask=mask)[0]
        return x+self.ff(self.ln2(x))

class ICLTransformer(nn.Module):
    def __init__(self, D_tok, d, heads, layers, d_ff, drop=0.0):
        super().__init__()
        self.proj_in = nn.Linear(D_tok, d); self.ln_in = nn.LayerNorm(d)
        self.blocks = nn.ModuleList([CausalBlock(d, heads, d_ff, drop) for _ in range(layers)])
        self.ln_out = nn.LayerNorm(d); self.proj_out = nn.Linear(d, D_tok)
    def forward(self, seq):
        L = seq.size(1)
        mask = torch.triu(torch.ones(L,L,device=seq.device,dtype=torch.bool), diagonal=1)
        x = self.ln_in(self.proj_in(seq))
        for blk in self.blocks: x = blk(x, mask)
        return self.proj_out(self.ln_out(x))


###############################################################################
# 8. FULL MODEL
###############################################################################
class PilotICLModelV5(nn.Module):
    """
    Complete model: PilotEncoder (frozen) + BFEncoder (frozen) +
                    ICLTransformer (trainable) + BFDecoder (trainable)
    """
    def __init__(self, pilot_enc, bf_enc, bf_dec, cfg):
        super().__init__()
        self.pilot_enc = pilot_enc  # frozen after Phase 0
        self.bf_enc = bf_enc        # frozen after Phase 0
        self.transformer = ICLTransformer(cfg.D_tok, cfg.d_model, cfg.n_heads,
                                          cfg.n_layers, cfg.d_ff, cfg.dropout)
        self.bf_dec = bf_dec        # trainable
        self.D_tok = cfg.D_tok
        n_train = sum(p.numel() for p in self.transformer.parameters()) + \
                  sum(p.numel() for p in self.bf_dec.parameters())
        n_frozen = sum(p.numel() for p in self.pilot_enc.parameters()) + \
                   sum(p.numel() for p in self.bf_enc.parameters())
        print(f"Model: trainable={n_train:,} frozen={n_frozen:,} total={n_train+n_frozen:,}")

    def forward(self, demo_pil_real, demo_W_real, query_pil_real, sigma2=None):
        """
        demo_pil_real:  (B, l, 2*N*L_p)  — demo pilot observations
        demo_W_real:    (B, l, 2*N*K)    — demo BF solutions
        query_pil_real: (B, 2*N*L_p)     — query pilot observation
        sigma2:         float            — for FiLM conditioning

        Returns: Delta-W (B, N, K) complex (residual beamformer)
        """
        B, l, _ = demo_pil_real.shape

        # Encode all pilots (demos + query) through frozen PilotEncoder
        all_pil = torch.cat([demo_pil_real.reshape(B*l, -1), query_pil_real], 0)
        with torch.no_grad():
            all_z = self.pilot_enc(all_pil, sigma2=sigma2)
        demo_z = all_z[:B*l].reshape(B, l, self.D_tok)
        query_z = all_z[B*l:]  # (B, D_tok)

        # Encode demo BF solutions through frozen BFEncoder
        with torch.no_grad():
            demo_c = self.bf_enc(demo_W_real.reshape(B*l, -1), sigma2=sigma2)
        demo_c = demo_c.reshape(B, l, self.D_tok)

        # Build ICL sequence: {z_1, c_1, z_2, c_2, ..., z_l, c_l, z_query}
        tokens = []
        for i in range(l):
            tokens.append(demo_z[:, i])  # state token
            tokens.append(demo_c[:, i])  # label token
        tokens.append(query_z)           # query state token
        seq = torch.stack(tokens, dim=1)  # (B, 2l+1, D_tok)

        # ICL Transformer (trainable)
        out = self.transformer(seq)
        c_hat = out[:, -1]  # (B, D_tok) — predicted compressed BF

        # BF Decoder (trainable) — predicts residual beamformer (unnormalized)
        dW = self.bf_dec(c_hat, normalize=False)  # (B, N, K) complex
        return dW


###############################################################################
# 9. DYNAMIC DATASET — stores [H, Y_real, W_real, rate, mmse_rate, is_sup]
###############################################################################
class DynDataset:
    def __init__(self, max_sz=50000):
        self.max_sz = max_sz
        self.H = self.Y_real = self.W_real = self.rates = self.mmse_rates = self.is_sup = None
        self._n = 0; self.n_sup = 0; self.n_unsup = 0

    @property
    def size(self): return self._n

    def add(self, H, Y_real, W_real, rates, mmse_rates=None, supervised=True):
        H, Y_real, W_real, rates = [x.detach() for x in [H, Y_real, W_real, rates]]
        flag = torch.full((H.size(0),), bool(supervised), device=device, dtype=torch.bool)
        mr = mmse_rates.detach() if mmse_rates is not None else torch.zeros_like(rates)
        if self.H is None:
            self.H, self.Y_real, self.W_real = H, Y_real, W_real
            self.rates, self.mmse_rates, self.is_sup = rates, mr, flag
        else:
            assert self.H is not None and self.Y_real is not None and self.W_real is not None
            assert self.rates is not None and self.mmse_rates is not None and self.is_sup is not None
            self.H = torch.cat([self.H, H])
            self.Y_real = torch.cat([self.Y_real, Y_real])
            self.W_real = torch.cat([self.W_real, W_real])
            self.rates = torch.cat([self.rates, rates])
            self.mmse_rates = torch.cat([self.mmse_rates, mr])
            self.is_sup = torch.cat([self.is_sup, flag])
        m = H.size(0)
        if supervised: self.n_sup += m
        else: self.n_unsup += m
        # Overflow pruning: keep all supervised samples, fill remaining slots
        # with highest-rate unsupervised samples.
        if self.H.size(0) > self.max_sz:
            assert self.rates is not None and self.is_sup is not None
            sup_idx = torch.where(self.is_sup)[0]
            uns_idx = torch.where(~self.is_sup)[0]

            if sup_idx.numel() >= self.max_sz:
                # Extremely rare: keep best supervised samples by rate.
                keep_sup = sup_idx[torch.topk(self.rates[sup_idx], self.max_sz).indices]
                keep_idx = keep_sup
            else:
                n_uns_keep = self.max_sz - sup_idx.numel()
                if uns_idx.numel() > n_uns_keep:
                    keep_uns = uns_idx[torch.topk(self.rates[uns_idx], n_uns_keep).indices]
                else:
                    keep_uns = uns_idx
                keep_idx = torch.cat([sup_idx, keep_uns], dim=0)

            for a in ['H','Y_real','W_real','rates','mmse_rates','is_sup']:
                setattr(self, a, getattr(self, a)[keep_idx])
        assert self.H is not None and self.is_sup is not None
        self._n = self.H.size(0)
        self.n_sup = int(self.is_sup.sum().item())
        self.n_unsup = int((~self.is_sup).sum().item())

    def prune_unsup_bottom(self, drop_ratio, min_keep=0):
        if self._n == 0 or drop_ratio <= 0: return 0
        assert self.H is not None and self.rates is not None and self.is_sup is not None
        ui = torch.where(~self.is_sup)[0]
        nu = ui.numel()
        if nu == 0: return 0
        nd = min(int(nu*drop_ratio), max(0, nu-min_keep))
        if nd <= 0: return 0
        worst = ui[torch.topk(self.rates[ui], k=nd, largest=False).indices]
        keep = torch.ones(self._n, device=device, dtype=torch.bool); keep[worst] = False
        for a in ['H','Y_real','W_real','rates','mmse_rates','is_sup']:
            setattr(self, a, getattr(self, a)[keep])
        assert self.H is not None and self.is_sup is not None
        self._n = self.H.size(0)
        self.n_sup = int(self.is_sup.sum().item())
        self.n_unsup = int((~self.is_sup).sum().item())
        return nd


###############################################################################
# 10. CONTEXT SELECTION (Random sampling from dataset)
###############################################################################
@torch.no_grad()
def select_demos(model, dataset, query_pil, cfg):
    B = query_pil.size(0)
    l = cfg.n_demos
    return torch.randint(0, dataset.size, (B, l), device=device)


###############################################################################
# 11. BASELINES & EVALUATION
###############################################################################
def compute_baselines(H_test, Phi, cfg):
    B = H_test.size(0); s2 = cfg.sigma2; Pm = cfg.P_max; bs = min(64, B)
    res = {k: [] for k in ['mmse_perf','mmse_imp','wmmse_perf','wmmse_imp']}
    for s in range(0, B, bs):
        e = min(s+bs, B); H = H_test[s:e]
        Y = pilot_observe(H, Phi, s2); Hh = mmse_channel_est(Y, Phi, s2)
        with torch.no_grad():
            res['mmse_perf'].append(compute_sum_rate(H, mmse_beamformer(H,Pm,s2), s2))
            res['mmse_imp'].append(compute_sum_rate(H, mmse_beamformer(Hh,Pm,s2), s2))
        W3,_ = generate_wmmse_labels(H, Pm, s2, n_iters=cfg.wmmse_iters, lr=cfg.wmmse_lr, n_restarts=2)
        with torch.no_grad(): res['wmmse_perf'].append(compute_sum_rate(H, W3, s2))
        W4,_ = generate_wmmse_labels(Hh, Pm, s2, n_iters=cfg.wmmse_iters, lr=cfg.wmmse_lr, n_restarts=2)
        with torch.no_grad(): res['wmmse_imp'].append(compute_sum_rate(H, W4, s2))
        print(f"  [{e}/{B}] B1={torch.cat(res['mmse_perf']).mean():.2f} "
              f"B2={torch.cat(res['mmse_imp']).mean():.2f} "
              f"B3={torch.cat(res['wmmse_perf']).mean():.2f} "
              f"B4={torch.cat(res['wmmse_imp']).mean():.2f}")
    return {k: torch.cat(v).mean().item() for k, v in res.items()}


@torch.no_grad()
def evaluate(model, dataset, H_test, Phi, cfg):
    model.eval()
    B = H_test.size(0); bs = min(cfg.batch_size, B); all_r = []
    for s in range(0, B, bs):
        e = min(s+bs, B); H = H_test[s:e]; b = H.size(0)
        Y = pilot_observe(H, Phi, cfg.sigma2); pil = pilot_to_real(Y)
        idx = select_demos(model, dataset, pil, cfg)
        dp = dataset.Y_real[idx]   # (b, l, 2NLp)
        dw = dataset.W_real[idx]   # (b, l, 2NK)
        H_ls = ls_channel_est(Y, Phi)
        W_base = mmse_beamformer(H_ls, cfg.P_max, cfg.sigma2)
        dW = model(dp, dw, pil, sigma2=cfg.sigma2)
        W_hat = power_normalize(W_base + dW, cfg.P_max)
        all_r.append(compute_sum_rate(H, W_hat, cfg.sigma2))
    model.train()
    return torch.cat(all_r).mean().item()


@torch.no_grad()
def calibrate_hybrid_rate_scale(model, dataset, Phi, cfg):
    """Quick small-range probing for balancing MSE and rate magnitudes."""
    model.eval()
    n_steps = max(1, cfg.hybrid_calib_steps)
    B = max(1, min(cfg.hybrid_calib_batch, cfg.batch_size))
    assert dataset.is_sup is not None
    sup_pool = torch.where(dataset.is_sup)[0]
    if sup_pool.numel() == 0:
        sup_pool = torch.arange(dataset.size, device=device)

    mse_vals, rate_vals = [], []
    for _ in range(n_steps):
        si = sup_pool[torch.randint(0, sup_pool.numel(), (B,), device=device)]
        q_H = dataset.H[si]
        q_W_gt_real = dataset.W_real[si]

        q_Y = pilot_observe(q_H, Phi, cfg.sigma2)
        q_pil = pilot_to_real(q_Y)

        di = select_demos(model, dataset, q_pil, cfg)
        dp = dataset.Y_real[di]
        dw = dataset.W_real[di]

        H_ls = ls_channel_est(q_Y, Phi)
        W_base = mmse_beamformer(H_ls, cfg.P_max, cfg.sigma2)
        dW = model(dp, dw, q_pil, sigma2=cfg.sigma2)
        W_hat = power_normalize(W_base + dW, cfg.P_max)
        W_hat_real = bf_to_real(W_hat)

        mse_vals.append(F.mse_loss(W_hat_real, q_W_gt_real, reduction='none').sum(-1).mean())
        rate_vals.append(compute_sum_rate(q_H, W_hat, cfg.sigma2).mean())

    mse_ref = torch.stack(mse_vals).mean().item()
    rate_ref = abs(torch.stack(rate_vals).mean().item())

    raw = (mse_ref / (rate_ref + 1e-8)) * cfg.hybrid_rate_gain
    scale = float(np.clip(raw, cfg.hybrid_rate_scale_min, cfg.hybrid_rate_scale_max))
    model.train()
    return scale, mse_ref, rate_ref


###############################################################################
# 12. MAIN TRAINING LOOP
###############################################################################
def train(cfg):
    set_global_seed(cfg.seed, cfg.deterministic)
    print("="*75)
    print("  PILOT-to-BEAMFORMER ICL v5 (Compressed BF, No Channel Est.)")
    print("="*75)
    print(f"  K={cfg.K} N={cfg.N} L_p={cfg.L_p} SNR={cfg.SNR_dB}dB D_tok={cfg.D_tok}")
    print(f"  Compression: pilot {2*cfg.N*cfg.L_p}→{cfg.D_tok}, BF {2*cfg.N*cfg.K}→{cfg.D_tok}")
    print(f"  seed={cfg.seed} deterministic={cfg.deterministic}")

    # ---- Shared static labeled dataset (generated ONCE) ----
    Phi = generate_pilot_dft(cfg.K, cfg.L_p)
    n_static = max(cfg.edn_n_samples, cfg.initial_ds_size)
    labeled_cache = build_static_labeled_dataset(cfg, Phi, n_static)

    # ---- Phase 0: Pretrain encoders on shared static cache ----
    pilot_enc = pretrain_pilot_edn(cfg, labeled_cache)
    bf_enc, bf_dec = pretrain_bf_edn(cfg, labeled_cache)

    # Freeze encoders
    for p in pilot_enc.parameters(): p.requires_grad_(False)
    for p in bf_enc.parameters(): p.requires_grad_(False)

    # ---- Test set & baselines ----
    H_test = generate_channel(cfg.n_test, cfg.K, cfg.N, cfg.ch_n_clusters,
                              cfg.ch_n_rays, cfg.ch_spread_deg)
    print("\nComputing baselines...")
    bl = compute_baselines(H_test, Phi, cfg)
    for k,v in bl.items(): print(f"  {k}: {v:.4f}")

    # ---- Initial labeled dataset from the same shared static cache ----
    print(f"\nBuilding initial dataset from shared cache (M0={cfg.initial_ds_size})...")
    ds = DynDataset(max_sz=cfg.max_ds_size)
    init_n = min(cfg.initial_ds_size, labeled_cache['H'].size(0))
    gbs = min(64, init_n)
    for s in range(0, init_n, gbs):
        e = min(s+gbs, init_n)
        ds.add(
            labeled_cache['H'][s:e],
            labeled_cache['Y_real'][s:e],
            labeled_cache['W_real'][s:e],
            labeled_cache['label_rate'][s:e],
            mmse_rates=labeled_cache['mmse_rate'][s:e],
            supervised=True,
        )
        print(f"  [{e}/{init_n}] rate={labeled_cache['label_rate'][s:e].mean():.2f}")

    # ---- Build full model ----
    model = PilotICLModelV5(pilot_enc, bf_enc, bf_dec, cfg).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.total_epochs, eta_min=cfg.lr_min)

    # ---- Training ----
    print("\n"+"="*75)
    best_test = 0.0
    hist = {'test':[], 'train':[], 'mse':[], 'ds':[], 'add':[], 'ph':[]}
    hybrid_rate_scale = None

    for epoch in range(cfg.total_epochs):
        model.train(); t0 = time.time()
        assert ds.H is not None and ds.W_real is not None and ds.Y_real is not None
        if epoch < cfg.phase1_epochs:
            phase, r = 1, 0.0
        else:
            phase = 2
            # Staircase schedule as requested:
            # 0.25, 0.5, 0.75 each for 50 epochs, then 1.0 afterwards.
            p2e = epoch - cfg.phase1_epochs
            if p2e < 50:
                r = 0.25
            elif p2e < 100:
                r = 0.50
            elif p2e < 150:
                r = 0.75
            else:
                r = 1.0

        ep_prog = epoch/max(1, cfg.total_epochs-1)
        alpha_t = cfg.boot_alpha_start + (cfg.boot_alpha_end-cfg.boot_alpha_start)*ep_prog
        beta_t = cfg.boot_beta_start + (cfg.boot_beta_end-cfg.boot_beta_start)*ep_prog

        ep_mse, ep_rate, ep_add, ep_n = 0., 0., 0, 0

        # Supervised pool for MSE-query branch: ONLY initial/supervised datapoints.
        assert ds.is_sup is not None
        sup_pool = torch.where(ds.is_sup)[0]
        if sup_pool.numel() == 0:
            sup_pool = torch.arange(ds.size, device=device)
        uns_pool = torch.where(~ds.is_sup)[0]

        for step in range(cfg.steps_per_epoch):
            B = cfg.batch_size; l = cfg.n_demos
            # Sample unsupervised query tokens by ratio r.
            # Even when uns_pool is empty, unsup queries are allowed and will use fresh samples,
            # enabling bootstrap growth of the unsupervised pool.
            if r >= 1.0:
                is_unsup = torch.ones(B, device=device, dtype=torch.bool)
            else:
                is_unsup = torch.rand(B, device=device) < r

            # Supervised queries for MSE branch from supervised-only pool.
            sidx = sup_pool[torch.randint(0, sup_pool.numel(), (B,), device=device)]
            q_H_sup = ds.H[sidx]
            q_W_gt_real = ds.W_real[sidx]

            # Unsupervised rate-loss queries: mix fresh and existing unsup (about half-half).
            q_H_unsup = q_H_sup.clone()
            q_W_unsup_real = q_W_gt_real.clone()
            uns_idx = torch.where(is_unsup)[0]
            n_uns = int(uns_idx.numel())
            if n_uns > 0:
                order = uns_idx[torch.randperm(n_uns, device=device)]
                if uns_pool.numel() > 0:
                    # Gradually increase old-unsup proportion to 50% as unsup pool grows.
                    grow = min(1.0, float(uns_pool.numel()) / max(1, cfg.unsup_mix_transition_size))
                    old_ratio = 0.5 * grow
                    n_old = int(round(n_uns * old_ratio))
                    n_fresh = n_uns - n_old
                else:
                    n_old = 0
                    n_fresh = n_uns

                if n_old > 0:
                    old_slots = order[:n_old]
                    uidx = uns_pool[torch.randint(0, uns_pool.numel(), (n_old,), device=device)]
                    q_H_unsup[old_slots] = ds.H[uidx]
                    q_W_unsup_real[old_slots] = ds.W_real[uidx]

                if n_fresh > 0:
                    fresh_slots = order[n_old:n_old+n_fresh]
                    fresh_H = generate_channel(n_fresh, cfg.K, cfg.N,
                                               cfg.ch_n_clusters, cfg.ch_n_rays, cfg.ch_spread_deg)
                    q_H_unsup[fresh_slots] = fresh_H

            q_H = torch.where(is_unsup.view(B,1,1).expand_as(q_H_sup), q_H_unsup, q_H_sup)
            q_W_gt_real = torch.where(is_unsup.view(B,1).expand_as(q_W_gt_real), q_W_unsup_real, q_W_gt_real)

            # All inputs are noisy pilots (training = inference)
            q_Y = pilot_observe(q_H, Phi, cfg.sigma2)
            q_pil = pilot_to_real(q_Y)

            # Demo pairs from dataset
            di = select_demos(model, ds, q_pil, cfg)
            dp = ds.Y_real[di]   # (B, l, 2NLp)
            dw = ds.W_real[di]   # (B, l, 2NK)

            # Residual beamforming: W = W_base(LS+MMSE-imp) + DeltaW
            H_ls = ls_channel_est(q_Y, Phi)
            W_base = mmse_beamformer(H_ls, cfg.P_max, cfg.sigma2)
            dW = model(dp, dw, q_pil, sigma2=cfg.sigma2)
            W_hat = power_normalize(W_base + dW, cfg.P_max)

            # Losses — all rates on TRUE H
            rate_pred = compute_sum_rate(q_H, W_hat, cfg.sigma2)
            W_hat_real = bf_to_real(W_hat)
            mse_per = F.mse_loss(W_hat_real, q_W_gt_real, reduction='none').sum(-1)
            sup_mask = ~is_unsup
            if sup_mask.any():
                mse_sup = mse_per[sup_mask].mean()
            else:
                mse_sup = torch.zeros((), device=device, dtype=mse_per.dtype)

            # Anti-lazy residual regularization:
            # keep output from collapsing too close to MMSE-imperfect reference W_base.
            res_norm = torch.linalg.vector_norm((W_hat - W_base).reshape(B, -1), dim=1)
            base_norm = torch.linalg.vector_norm(W_base.reshape(B, -1), dim=1).clamp_min(1e-8)
            res_ratio = res_norm / base_norm
            lazy_pen_per = F.relu(cfg.lazy_residual_min_ratio - res_ratio).pow(2)
            if is_unsup.any():
                lazy_pen = lazy_pen_per[is_unsup].mean()
            else:
                lazy_pen = lazy_pen_per.mean()

            # Residual-targeted supervised objective:
            # DeltaW* = W* - W_base, directly supervising residual direction/magnitude.
            q_W_gt = real_to_bf(q_W_gt_real, cfg.N, cfg.K)
            dW_star = q_W_gt - W_base
            dW_real = bf_to_real(dW)
            dW_star_real = bf_to_real(dW_star)
            res_tgt_per = F.mse_loss(dW_real, dW_star_real, reduction='none').sum(-1)
            if sup_mask.any():
                res_tgt_sup = res_tgt_per[sup_mask].mean()
            else:
                res_tgt_sup = torch.zeros((), device=device, dtype=mse_per.dtype)

            if phase == 1:
                loss = (
                    mse_sup
                    + cfg.sup_residual_target_weight * res_tgt_sup
                    + cfg.lazy_residual_weight * lazy_pen
                )
            else:
                # Soft switch in Phase 2: weighted hybrid from MSE to rate
                if hybrid_rate_scale is None:
                    hybrid_rate_scale, mse_ref, rate_ref = calibrate_hybrid_rate_scale(model, ds, Phi, cfg)
                    print(f"  [Hybrid calibration] mse_ref={mse_ref:.4f} rate_ref={rate_ref:.4f} "
                          f"scale={hybrid_rate_scale:.4f}", flush=True)

                p2_prog = (epoch - cfg.phase1_epochs + 1) / max(1, cfg.phase2_epochs)
                p2_prog = float(np.clip(p2_prog, 0.0, 1.0))
                w_rate = p2_prog ** cfg.hybrid_switch_power
                w_mse = 1.0 - w_rate

                rate_loss = -rate_pred.mean()
                # Final stage (r>=1): pure sum-rate objective as requested.
                if r >= 1.0:
                    loss = (
                        hybrid_rate_scale * rate_loss
                        + cfg.sup_residual_target_weight * res_tgt_sup
                        + cfg.lazy_residual_weight * lazy_pen
                    )
                else:
                    loss = (
                        w_mse * mse_sup
                        + w_rate * (hybrid_rate_scale * rate_loss)
                        + cfg.sup_residual_target_weight * res_tgt_sup
                        + cfg.lazy_residual_weight * lazy_pen
                    )

            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()

            # Dual-threshold self-bootstrapping
            with torch.no_grad():
                if phase == 2:
                    n_uns_batch = int(is_unsup.sum().item())
                    n_cand = n_uns_batch // 2
                    if n_cand <= 0:
                        pass
                    else:
                    # New unsupervised candidates are generated fresh and admitted by threshold.
                        cand_H = generate_channel(n_cand, cfg.K, cfg.N, cfg.ch_n_clusters,
                                                  cfg.ch_n_rays, cfg.ch_spread_deg)
                        cand_Y = pilot_observe(cand_H, Phi, cfg.sigma2)
                        cand_pil = pilot_to_real(cand_Y)
                        cdi = select_demos(model, ds, cand_pil, cfg)
                        cdp = ds.Y_real[cdi]
                        cdw = ds.W_real[cdi]

                        cand_H_ls = ls_channel_est(cand_Y, Phi)
                        cand_W_base = mmse_beamformer(cand_H_ls, cfg.P_max, cfg.sigma2)
                        cand_dW = model(cdp, cdw, cand_pil, sigma2=cfg.sigma2)
                        cand_W_hat = power_normalize(cand_W_base + cand_dW, cfg.P_max)
                        cand_rate = compute_sum_rate(cand_H, cand_W_hat, cfg.sigma2)

                        cand_H_hat = mmse_channel_est(cand_Y, Phi, cfg.sigma2)
                        cand_W_mmse_imp = mmse_beamformer(cand_H_hat, cfg.P_max, cfg.sigma2)
                        cand_mmse_imp_r = compute_sum_rate(cand_H, cand_W_mmse_imp, cfg.sigma2)

                        per_ok = cand_rate > alpha_t * cand_mmse_imp_r
                        cross_ok = cand_rate >= torch.quantile(cand_rate, beta_t) if cand_rate.numel() > 1 \
                            else torch.ones_like(cand_rate, dtype=torch.bool)
                        good = per_ok & cross_ok
                        if good.any():
                            ds.add(cand_H[good], cand_pil[good], bf_to_real(cand_W_hat[good]),
                                   cand_rate[good], mmse_rates=cand_mmse_imp_r[good], supervised=False)
                            ep_add += good.sum().item()

                            # Refresh unsupervised pool after admissions.
                            assert ds.is_sup is not None
                            uns_pool = torch.where(~ds.is_sup)[0]

            ns = (~is_unsup).sum().item(); nu = is_unsup.sum().item()
            if ns > 0:
                ep_mse += mse_per[~is_unsup].mean().item()
            if nu > 0:
                ep_rate += rate_pred[is_unsup].mean().item()
            ep_n += 1

        scheduler.step()

        # Bottom pruning
        nd = 0
        if phase == 2 and (epoch+1) % cfg.prune_every == 0:
            fp = (epoch-cfg.phase1_epochs)/max(1, cfg.phase2_epochs-1)
            dr = cfg.prune_drop_start + (cfg.prune_drop_end-cfg.prune_drop_start)*fp
            nd = ds.prune_unsup_bottom(dr, cfg.prune_min_unsup)

        # Evaluate
        tr = evaluate(model, ds, H_test, Phi, cfg)
        best_test = max(best_test, tr)

        am = ep_mse/max(1,ep_n); ar = ep_rate/max(1,ep_n); dt = time.time()-t0
        hist['test'].append(tr); hist['train'].append(ar); hist['mse'].append(am)
        hist['ds'].append(ds.size); hist['add'].append(ep_add); hist['ph'].append(phase)

        # Periodically overwrite metric snapshots (.pt)
        if (epoch + 1) % cfg.rate_save_every == 0:
            ep_tensor = torch.arange(1, len(hist['train']) + 1)
            torch.save({
                'epochs': ep_tensor,
                'train_rate': torch.tensor(hist['train'], dtype=torch.float32),
            }, cfg.train_rate_pt)
            torch.save({
                'epochs': ep_tensor,
                'test_rate': torch.tensor(hist['test'], dtype=torch.float32),
            }, cfg.test_rate_pt)

        # Plateau detection retained for monitoring only (r follows fixed schedule).

        print(f"{epoch+1:3d} ph{phase} r={r:.2f} | "
              f"mse={am:.5f} rate={ar:.2f} | test={tr:.3f} best={best_test:.3f} | "
              f"DS={ds.size} +{ep_add} -{nd} | "
              f"α={alpha_t:.2f} β={beta_t:.0%} ({dt:.1f}s)", flush=True)

    # Final
    b4 = bl['wmmse_imp']
    print("\n"+"="*68)
    print("  COMPLETE")
    for k,v in [("B1 MMSE+Perf", bl['mmse_perf']), ("B2 MMSE+Imp", bl['mmse_imp']),
                ("B3 WMMSE+Perf", bl['wmmse_perf']), ("B4 WMMSE+Imp", b4),
                ("ICL best", best_test), ("ICL final", tr)]:
        print(f"  {k:<20} {v:8.3f}  ({100*v/b4:.1f}% of B4)" if b4>0 else f"  {k} {v}")
    print(f"  Dataset: {ds.size} ({ds.n_sup} sup + {ds.n_unsup} unsup)")

    # Ensure final overwrite save
    ep_tensor = torch.arange(1, len(hist['train']) + 1)
    torch.save({
        'epochs': ep_tensor,
        'train_rate': torch.tensor(hist['train'], dtype=torch.float32),
    }, cfg.train_rate_pt)
    torch.save({
        'epochs': ep_tensor,
        'test_rate': torch.tensor(hist['test'], dtype=torch.float32),
    }, cfg.test_rate_pt)

    # Plot train/test rate curve vs epochs
    plt.figure(figsize=(8, 5))
    epochs = np.arange(1, len(hist['train']) + 1)
    plt.plot(epochs, hist['train'], label='Train rate', linewidth=1.8)
    plt.plot(epochs, hist['test'], label='Test rate', linewidth=1.8)
    plt.xlabel('Epoch')
    plt.ylabel('Sum rate')
    plt.title('Train/Test Sum-Rate vs Epochs')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.rate_curve_png, dpi=180)
    plt.close()

    print(f"  Saved: {cfg.train_rate_pt}, {cfg.test_rate_pt}, {cfg.rate_curve_png}")
    return model, ds, bl


###############################################################################
# 13. ENTRY POINT
###############################################################################
if __name__ == "__main__":
    cfg = Config(
        K=32, N=32, L_p=20,
        P_max=1.0, SNR_dB=20,
        ch_n_clusters=3, ch_n_rays=5, ch_spread_deg=10.0,
        # Token / compression
        D_tok=256,
        # Encoder-Decoder pretraining
        edn_hidden=128, edn_epochs=1000, edn_lr=1e-3, edn_batch=128, edn_n_samples=5000,
        # ICL Transformer
        n_demos=5,
        d_model=512, n_heads=8, n_layers=6, d_ff=1024, dropout=0.0,
        # Training
        batch_size=64, lr=1e-4, lr_min=5e-5, weight_decay=1e-4,
        initial_ds_size=1024, wmmse_iters=100, wmmse_lr=0.03, wmmse_label_restarts=2,
        unsup_scale=0.01,
        # Anti-lazy residual regularization
        lazy_residual_min_ratio=0.10,
        lazy_residual_weight=0.05,
        sup_residual_target_weight=0.25,
        # Curriculum
        phase1_epochs=50, phase2_epochs=1000, r_max=0.85, steps_per_epoch=80,
        # Dual-threshold
        boot_alpha_start=0.60, boot_alpha_end=0.90,
        boot_beta_start=0.60, boot_beta_end=0.90,
        max_ds_size=100000,
        # Pruning
        prune_every=10, prune_drop_start=0.0, prune_drop_end=0.10, prune_min_unsup=4096,
        # Plateau
        plateau_window=10, plateau_thresh=0.3, plateau_boost=0.10,
        # Eval
        n_test=200,
    )
    train(cfg)