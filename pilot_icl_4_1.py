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


###############################################################################
# 1. CONFIGURATION
###############################################################################
class Config:
    def __init__(self, **kwargs):
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
        self.edn_epochs = kwargs.get('edn_epochs', 100)
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
        self.unsup_scale = kwargs.get('unsup_scale', 0.01)

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
        self.prune_drop_end = kwargs.get('prune_drop_end', 0.20)
        self.prune_min_unsup = kwargs.get('prune_min_unsup', 2048)

        # Plateau detection
        self.plateau_window = kwargs.get('plateau_window', 10)
        self.plateau_thresh = kwargs.get('plateau_thresh', 0.3)
        self.plateau_boost = kwargs.get('plateau_boost', 0.10)

        # Eval
        self.n_test = kwargs.get('n_test', 200)


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
        self.film = FiLMLayer(hidden)
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
        if sigma2 is not None: x = self.film(x, sigma2)
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
        self.film = FiLMLayer(hidden)
        self.proj = nn.Linear(hidden, D_tok)

    def forward(self, W_real, sigma2=None):
        """W_real: (B, 2NK) -> (B, D_tok)."""
        h = self.net(W_real)  # (B, hidden)
        if sigma2 is not None:
            h = self.film(h, sigma2)
        return self.proj(h)


class BFDecoder(nn.Module):
    """Decompresses c_hat ∈ R^{D_tok} to W_hat ∈ C^{N×K}, power-normalized."""
    def __init__(self, N, K, D_tok, P_max, hidden=512):
        super().__init__()
        self.N, self.K, self.P_max = N, K, P_max
        self.net = nn.Sequential(nn.Linear(D_tok, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 2*N*K))

    def forward(self, c):
        """c: (B, D_tok) -> W_hat: (B, N, K) complex, power-normalized."""
        x = self.net(c)  # (B, 2NK)
        W = real_to_bf(x, self.N, self.K)
        return power_normalize(W, self.P_max)


###############################################################################
# 6. PHASE 0: PRETRAIN ENCODER-DECODER NETWORKS
###############################################################################
def pretrain_pilot_edn(cfg):
    """Phase 0a: Train PilotEncoder + ChannelDecoder as autoencoder."""
    print("\n[Phase 0a] Pretraining PilotEncoder + ChannelDecoder...")
    enc = PilotEncoder(cfg.N, cfg.L_p, cfg.D_tok, cfg.edn_hidden).to(device)
    dec = ChannelDecoder(cfg.K, cfg.N, cfg.D_tok, cfg.edn_hidden).to(device)
    Phi = generate_pilot_dft(cfg.K, cfg.L_p)
    opt = Adam(list(enc.parameters()) + list(dec.parameters()), lr=cfg.edn_lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, cfg.edn_epochs, eta_min=1e-5)

    for ep in range(cfg.edn_epochs):
        ep_loss = 0.0; n_batch = 0
        for _ in range(cfg.edn_n_samples // cfg.edn_batch):
            H = generate_channel(cfg.edn_batch, cfg.K, cfg.N,
                                 cfg.ch_n_clusters, cfg.ch_n_rays, cfg.ch_spread_deg)
            Y = pilot_observe(H, Phi, cfg.sigma2)
            Y_real = pilot_to_real(Y)
            H_real = channel_to_real(H)
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
    return enc, Phi


def pretrain_bf_edn(cfg):
    """Phase 0b: Train BFEncoder + BFDecoder as autoencoder."""
    print("\n[Phase 0b] Pretraining BFEncoder + BFDecoder...")
    enc = BFEncoder(cfg.N, cfg.K, cfg.D_tok, cfg.edn_hidden).to(device)
    dec = BFDecoder(cfg.N, cfg.K, cfg.D_tok, cfg.P_max, cfg.edn_hidden).to(device)
    opt = Adam(list(enc.parameters()) + list(dec.parameters()), lr=cfg.edn_lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, cfg.edn_epochs, eta_min=1e-5)

    for ep in range(cfg.edn_epochs):
        ep_loss = 0.0; n_batch = 0
        for _ in range(cfg.edn_n_samples // cfg.edn_batch):
            H = generate_channel(cfg.edn_batch, cfg.K, cfg.N,
                                 cfg.ch_n_clusters, cfg.ch_n_rays, cfg.ch_spread_deg)
            W_star, _ = generate_wmmse_labels(H, cfg.P_max, cfg.sigma2,
                                              n_iters=200, lr=cfg.wmmse_lr, n_restarts=1)
            W_real = bf_to_real(W_star)
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

        Returns: W_hat (B, N, K) complex, power-normalized
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

        # BF Decoder (trainable) — recovers full-size beamformer
        W_hat = self.bf_dec(c_hat)  # (B, N, K) complex, power-normalized
        return W_hat


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
            self.H = torch.cat([self.H, H])
            self.Y_real = torch.cat([self.Y_real, Y_real])
            self.W_real = torch.cat([self.W_real, W_real])
            self.rates = torch.cat([self.rates, rates])
            self.mmse_rates = torch.cat([self.mmse_rates, mr])
            self.is_sup = torch.cat([self.is_sup, flag])
        m = H.size(0)
        if supervised: self.n_sup += m
        else: self.n_unsup += m
        # Overflow pruning: keep top-rate samples
        if self.H.size(0) > self.max_sz:
            _, idx = torch.topk(self.rates, self.max_sz)
            for a in ['H','Y_real','W_real','rates','mmse_rates','is_sup']:
                setattr(self, a, getattr(self, a)[idx])
        self._n = self.H.size(0)
        self.n_sup = int(self.is_sup.sum().item())
        self.n_unsup = int((~self.is_sup).sum().item())

    def prune_unsup_bottom(self, drop_ratio, min_keep=0):
        if self._n == 0 or drop_ratio <= 0: return 0
        ui = torch.where(~self.is_sup)[0]
        nu = ui.numel()
        if nu == 0: return 0
        nd = min(int(nu*drop_ratio), max(0, nu-min_keep))
        if nd <= 0: return 0
        worst = ui[torch.topk(self.rates[ui], k=nd, largest=False).indices]
        keep = torch.ones(self._n, device=device, dtype=torch.bool); keep[worst] = False
        for a in ['H','Y_real','W_real','rates','mmse_rates','is_sup']:
            setattr(self, a, getattr(self, a)[keep])
        self._n = self.H.size(0)
        self.n_sup = int(self.is_sup.sum().item())
        self.n_unsup = int((~self.is_sup).sum().item())
        return nd


###############################################################################
# 10. CONTEXT SELECTION (Similarity + MMR on pilot embeddings)
###############################################################################
@torch.no_grad()
def select_demos(model, dataset, query_pil, cfg):
    B = query_pil.size(0); l = cfg.n_demos
    if dataset.size <= l:
        return torch.randint(0, dataset.size, (B, l), device=device)
    ps = min(cfg.context_pool_size, dataset.size)
    pi = torch.randperm(dataset.size, device=device)[:ps] if ps < dataset.size \
        else torch.arange(dataset.size, device=device)
    z_q = F.normalize(model.pilot_enc(query_pil, sigma2=cfg.sigma2), dim=-1)
    z_p = F.normalize(model.pilot_enc(dataset.Y_real[pi], sigma2=cfg.sigma2), dim=-1)
    sim = z_q @ z_p.T
    kc = min(max(l, cfg.context_k_cand), ps)
    cand = torch.topk(sim, k=kc, dim=1).indices
    chosen = torch.zeros(B, l, dtype=torch.long, device=device)
    alpha = cfg.mmr_alpha
    for b in range(B):
        c = cand[b]; ce = z_p[c]; qs = sim[b,c]; cs = ce@ce.T
        sel = [int(torch.argmax(qs).item())]
        mask = torch.zeros(kc, dtype=torch.bool, device=device); mask[sel[0]] = True
        for _ in range(1, l):
            ms = cs[:, sel].max(1).values
            sc = alpha*qs-(1-alpha)*ms; sc[mask] = -1e9
            nxt = int(torch.argmax(sc).item()); sel.append(nxt); mask[nxt] = True
        chosen[b] = c[torch.tensor(sel, device=device)]
    return pi[chosen]


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
        W_hat = model(dp, dw, pil, sigma2=cfg.sigma2)
        all_r.append(compute_sum_rate(H, W_hat, cfg.sigma2))
    model.train()
    return torch.cat(all_r).mean().item()


###############################################################################
# 12. MAIN TRAINING LOOP
###############################################################################
def train(cfg):
    print("="*75)
    print("  PILOT-to-BEAMFORMER ICL v5 (Compressed BF, No Channel Est.)")
    print("="*75)
    print(f"  K={cfg.K} N={cfg.N} L_p={cfg.L_p} SNR={cfg.SNR_dB}dB D_tok={cfg.D_tok}")
    print(f"  Compression: pilot {2*cfg.N*cfg.L_p}→{cfg.D_tok}, BF {2*cfg.N*cfg.K}→{cfg.D_tok}")

    # ---- Phase 0: Pretrain encoders ----
    pilot_enc, Phi = pretrain_pilot_edn(cfg)
    bf_enc, bf_dec = pretrain_bf_edn(cfg)

    # Freeze encoders
    for p in pilot_enc.parameters(): p.requires_grad_(False)
    for p in bf_enc.parameters(): p.requires_grad_(False)

    # ---- Test set & baselines ----
    H_test = generate_channel(cfg.n_test, cfg.K, cfg.N, cfg.ch_n_clusters,
                              cfg.ch_n_rays, cfg.ch_spread_deg)
    print("\nComputing baselines...")
    bl = compute_baselines(H_test, Phi, cfg)
    for k,v in bl.items(): print(f"  {k}: {v:.4f}")

    # ---- Initial labeled dataset: WMMSE on perfect CSI ----
    print(f"\nGenerating initial dataset (M0={cfg.initial_ds_size})...")
    ds = DynDataset(max_sz=cfg.max_ds_size)
    gbs = min(64, cfg.initial_ds_size)
    for s in range(0, cfg.initial_ds_size, gbs):
        e = min(s+gbs, cfg.initial_ds_size)
        H = generate_channel(e-s, cfg.K, cfg.N, cfg.ch_n_clusters, cfg.ch_n_rays, cfg.ch_spread_deg)
        W_star, _ = generate_wmmse_labels(H, cfg.P_max, cfg.sigma2,
                                          n_iters=cfg.wmmse_iters, lr=cfg.wmmse_lr)
        Y = pilot_observe(H, Phi, cfg.sigma2)
        with torch.no_grad():
            rate = compute_sum_rate(H, W_star, cfg.sigma2)
            mmse_r = compute_sum_rate(H, mmse_beamformer(H, cfg.P_max, cfg.sigma2), cfg.sigma2)
        ds.add(H, pilot_to_real(Y), bf_to_real(W_star), rate, mmse_rates=mmse_r, supervised=True)
        print(f"  [{e}/{cfg.initial_ds_size}] rate={rate.mean():.2f}")

    # ---- Build full model ----
    model = PilotICLModelV5(pilot_enc, bf_enc, bf_dec, cfg).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.total_epochs, eta_min=cfg.lr_min)

    # ---- Training ----
    print("\n"+"="*75)
    best_test = 0.0
    hist = {'test':[], 'train':[], 'mse':[], 'ds':[], 'add':[], 'ph':[]}
    r_boost = 0.0

    for epoch in range(cfg.total_epochs):
        model.train(); t0 = time.time()
        if epoch < cfg.phase1_epochs: phase, r = 1, 0.0
        else:
            phase = 2
            prog = (epoch-cfg.phase1_epochs)/max(1, cfg.phase2_epochs-1)
            r = min(prog*cfg.r_max + r_boost, cfg.r_max)

        ep_prog = epoch/max(1, cfg.total_epochs-1)
        alpha_t = cfg.boot_alpha_start + (cfg.boot_alpha_end-cfg.boot_alpha_start)*ep_prog
        beta_t = cfg.boot_beta_start + (cfg.boot_beta_end-cfg.boot_beta_start)*ep_prog

        ep_mse, ep_rate, ep_add, ep_n = 0., 0., 0, 0

        for step in range(cfg.steps_per_epoch):
            B = cfg.batch_size; l = cfg.n_demos
            is_unsup = torch.rand(B, device=device) < r

            # Supervised queries from dataset
            si = torch.randint(0, ds.size, (B,), device=device)
            q_H_sup = ds.H[si]; q_W_gt_real = ds.W_real[si]

            # Unsupervised queries: fresh channels
            q_H_unsup = generate_channel(B, cfg.K, cfg.N, cfg.ch_n_clusters,
                                         cfg.ch_n_rays, cfg.ch_spread_deg)
            q_H = torch.where(is_unsup.view(B,1,1).expand_as(q_H_sup), q_H_unsup, q_H_sup)

            # All inputs are noisy pilots (training = inference)
            q_Y = pilot_observe(q_H, Phi, cfg.sigma2)
            q_pil = pilot_to_real(q_Y)

            # Demo pairs from dataset
            di = select_demos(model, ds, q_pil, cfg)
            dp = ds.Y_real[di]   # (B, l, 2NLp)
            dw = ds.W_real[di]   # (B, l, 2NK)

            # Forward
            W_hat = model(dp, dw, q_pil, sigma2=cfg.sigma2)  # (B,N,K) complex

            # Losses — all rates on TRUE H
            rate_pred = compute_sum_rate(q_H, W_hat, cfg.sigma2)
            W_hat_real = bf_to_real(W_hat)
            mse_per = F.mse_loss(W_hat_real, q_W_gt_real, reduction='none').sum(-1)

            loss_per = torch.where(is_unsup, -rate_pred*cfg.unsup_scale, mse_per)
            loss = loss_per.mean()

            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()

            # Dual-threshold self-bootstrapping
            with torch.no_grad():
                if is_unsup.any():
                    ur = rate_pred[is_unsup]
                    # Per-instance: must beat alpha * MMSE_rate (perfect CSI baseline)
                    mmse_r_u = compute_sum_rate(
                        q_H[is_unsup],
                        mmse_beamformer(q_H[is_unsup], cfg.P_max, cfg.sigma2),
                        cfg.sigma2)
                    per_ok = ur > alpha_t * mmse_r_u
                    # Cross-instance: top beta percentile in batch
                    cross_ok = ur >= torch.quantile(ur, beta_t) if ur.numel() > 1 \
                        else torch.ones_like(ur, dtype=torch.bool)
                    good = per_ok & cross_ok
                    if good.any():
                        gi = torch.where(is_unsup)[0][good]
                        Y_gi = pilot_to_real(pilot_observe(q_H[gi], Phi, cfg.sigma2))
                        ds.add(q_H[gi], Y_gi, bf_to_real(W_hat[gi]),
                               rate_pred[gi], mmse_rates=mmse_r_u[good], supervised=False)
                        ep_add += good.sum().item()

            ns = (~is_unsup).sum().item(); nu = is_unsup.sum().item()
            if ns > 0: ep_mse += mse_per[~is_unsup].mean().item()
            if nu > 0: ep_rate += rate_pred[is_unsup].mean().item()
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

        # Plateau detection
        W = cfg.plateau_window
        if phase == 2 and len(hist['test']) > W:
            rec = hist['test'][-W:]
            if max(rec)-min(rec) < cfg.plateau_thresh:
                r_boost = min(r_boost + cfg.plateau_boost, cfg.r_max)

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
    return model, ds, bl


###############################################################################
# 13. ENTRY POINT
###############################################################################
if __name__ == "__main__":
    cfg = Config(
        K=32, N=28, L_p=20,
        P_max=1.0, SNR_dB=20,
        ch_n_clusters=3, ch_n_rays=5, ch_spread_deg=10.0,
        # Token / compression
        D_tok=256,
        # Encoder-Decoder pretraining
        edn_hidden=512, edn_epochs=100, edn_lr=1e-3, edn_batch=128, edn_n_samples=5000,
        # ICL Transformer
        n_demos=5,
        d_model=512, n_heads=8, n_layers=6, d_ff=1024, dropout=0.0,
        # Training
        batch_size=64, lr=2e-4, lr_min=5e-5, weight_decay=1e-4,
        initial_ds_size=1024, wmmse_iters=500, wmmse_lr=0.03,
        unsup_scale=0.01,
        # Curriculum
        phase1_epochs=30, phase2_epochs=500, r_max=0.85, steps_per_epoch=80,
        # Dual-threshold
        boot_alpha_start=0.90, boot_alpha_end=1.05,
        boot_beta_start=0.50, boot_beta_end=0.80,
        max_ds_size=50000,
        # Pruning
        prune_every=5, prune_drop_start=0.0, prune_drop_end=0.20, prune_min_unsup=2048,
        # Plateau
        plateau_window=10, plateau_thresh=0.3, plateau_boost=0.10,
        # Eval
        n_test=200,
    )
    train(cfg)
