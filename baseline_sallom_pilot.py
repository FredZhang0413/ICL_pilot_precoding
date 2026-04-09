"""
BASELINE: SALLO-M Transformer for Pilot-Based Beamforming (No CSI)

Adapted from the SALLO-M Transformer (Zhang et al.) for the noisy-pilot
scenario. The original SALLO-M assumes perfect CSI; this adaptation operates
under limited-length noisy pilots only.

KEY DIFFERENCES FROM ORIGINAL SALLO-M:
  1. Auxiliary variable C^(0) = zero-padded pilot Y (not true channel H)
  2. Initial beamformer W^(0) = random (not LMMSE — no H available)
  3. NO PGD refinement at any layer (requires true H for gradient)
  4. NO sample/attention masking (single fixed K,N configuration)
  5. Channel model: cluster-based sparse (not Gaussian)
  6. Training loss: -sum_rate(H_true, W^(T)) — H_true known during training

ARCHITECTURE (per layer, same as original):
  Dual tokenization: user-level (B, 4K, N) + antenna-level (B, 4N, K)
  Multi-head self-attention on each stream
  MLP refinement + residual connections
  Average-merge of user/antenna streams → updated (C, W)
  Normalize W to satisfy ||W||^2 = P_max

TRAINING: Sliding-window (window=3, 10 layers, 8 phases of 100 epochs each)
"""

import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Optional
import warnings
import time

warnings.filterwarnings("ignore")
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")


###############################################################################
# 1. SIGNAL PROCESSING UTILITIES
###############################################################################
def sum_rate(H, W, sigma2=1.0):
    """Sum rate for MU-MISO. H:(B,K,N), W:(B,N,K), returns scalar."""
    prod = th.bmm(H, W)
    signal_power = th.abs(th.diagonal(prod, dim1=-2, dim2=-1))**2
    interference = th.sum(th.abs(prod)**2, dim=-1) - signal_power
    SINR = signal_power / (interference + sigma2)
    return th.log2(1 + SINR).sum(dim=-1).mean()


def sum_rate_per_sample(H, W, sigma2=1.0):
    """Sum rate per sample. H:(B,K,N), W:(B,N,K), returns (B,)."""
    prod = th.bmm(H, W)
    signal_power = th.abs(th.diagonal(prod, dim1=-2, dim2=-1))**2
    interference = th.sum(th.abs(prod)**2, dim=-1) - signal_power
    SINR = signal_power / (interference + sigma2)
    return th.log2(1 + SINR).sum(dim=-1)


def generate_sparse_channel(B, K, N, n_cl=3, n_ray=5, spread_deg=10.0):
    """Cluster-based sparse mmWave channel. Returns (B,K,N) complex."""
    L = n_cl * n_ray; asp = math.radians(spread_deg)
    cm = (th.rand(B,K,n_cl,1,device=device)-0.5)*math.pi
    ro = th.randn(B,K,n_cl,n_ray,device=device)*asp
    ang = (cm+ro).clamp(-math.pi/2,math.pi/2).reshape(B,K,L)
    alp = (th.randn(B,K,L,device=device)+1j*th.randn(B,K,L,device=device))/math.sqrt(2)
    idx = th.arange(N,device=device,dtype=th.float32).view(1,1,1,N)
    ph = math.pi*idx*th.sin(ang).unsqueeze(-1)
    st = th.polar(th.ones_like(ph),ph)/math.sqrt(N)
    return math.sqrt(N/L)*th.sum(alp.unsqueeze(-1)*st, dim=2)


def generate_pilot_dft(K, L_p):
    """Truncated DFT pilot matrix. Returns (K, L_p) complex."""
    return (th.fft.fft(th.eye(K, device=device)) / math.sqrt(K))[:, :L_p].contiguous()


def pilot_observe(H, Phi, sigma2):
    """Y = H^T Phi + N, returns (B,N,L_p) complex."""
    B, K, N = H.shape; Lp = Phi.size(1)
    Y = H.transpose(-1,-2) @ Phi.unsqueeze(0).expand(B,-1,-1)
    nr = th.randn(B,N,Lp,device=device)*math.sqrt(sigma2/2)
    ni = th.randn(B,N,Lp,device=device)*math.sqrt(sigma2/2)
    return Y + th.complex(nr, ni)


def mmse_beamformer(H, P_max, sigma2):
    """MMSE BF on true H for baseline comparison."""
    B,K,N = H.shape; HH = H.conj().transpose(-1,-2)
    A = HH@H + sigma2*th.eye(N,device=device,dtype=H.dtype).unsqueeze(0)
    W = th.linalg.solve(A, HH)
    pw = th.sum(th.abs(W)**2, dim=(1,2)).real
    return W * th.sqrt(P_max/(pw+1e-8)).view(B,1,1)


def power_normalize(W, P_max):
    pw = th.sum(th.abs(W)**2, dim=(1,2), keepdim=True).real
    return W * th.sqrt(P_max / (pw + 1e-8))


def ls_channel_est(Y, Phi):
    return (Y @ th.linalg.pinv(Phi).unsqueeze(0)).transpose(-1,-2).contiguous()


def mmse_channel_est(Y, Phi, sigma2):
    K,Lp = Phi.shape
    A = Phi.T @ Phi.conj() + sigma2*th.eye(Lp, device=device, dtype=Phi.dtype)
    return th.matmul((Phi.conj()@th.linalg.inv(A)).unsqueeze(0), Y.transpose(-1,-2))


###############################################################################
# 2. MULTI-HEAD SELF-ATTENTION — USER-LEVEL (from SALLO-M)
###############################################################################
class MultiHeadAttentionUser(nn.Module):
    def __init__(self, token_dim, seq_len, d_model, num_heads, embed_dim, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.token_dim = token_dim  # N
        self.seq_len = seq_len      # 4K
        self.num_tokens_h = seq_len // 2  # 2K
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.x_embedding = nn.Linear(token_dim, embed_dim)
        self.x_ln = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, d_model)
        self.k_proj = nn.Linear(embed_dim, d_model)
        self.v_proj = nn.Linear(embed_dim, d_model)
        self.mlp_head_collect = nn.Sequential(
            nn.Linear(d_model, 4*token_dim), nn.GELU(),
            nn.Linear(4*token_dim, 2*token_dim), nn.GELU(),
            nn.Linear(2*token_dim, token_dim),
        )
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x_hu_wu):
        B, L, D = x_hu_wu.shape
        h, d = self.num_heads, self.head_dim
        x_ln = self.x_ln(self.x_embedding(x_hu_wu))
        Q = self.q_proj(x_ln).view(B, L, h, d).transpose(1, 2)
        K = self.k_proj(x_ln).view(B, L, h, d).transpose(1, 2)
        V = self.v_proj(x_ln).view(B, L, h, d).transpose(1, 2)
        scores = (Q @ K.transpose(-2,-1)) / math.sqrt(d)
        attn = self.attn_drop(F.softmax(scores, dim=-1))
        context = (attn @ V).transpose(1,2).contiguous().view(B, L, h*d)
        context = self.mlp_head_collect(context)
        y_h = context[:, :self.num_tokens_h, :] + x_hu_wu[:, :self.num_tokens_h, :]
        y_w = context[:, self.num_tokens_h:, :] + x_hu_wu[:, self.num_tokens_h:, :]
        return y_h, y_w


###############################################################################
# 3. MULTI-HEAD SELF-ATTENTION — ANTENNA-LEVEL (from SALLO-M)
###############################################################################
class MultiHeadAttentionAntenna(nn.Module):
    def __init__(self, token_dim, seq_len, d_model, num_heads, embed_dim, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.token_dim = token_dim  # K
        self.seq_len = seq_len      # 4N
        self.num_tokens_h = seq_len // 2
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.x_embedding = nn.Linear(token_dim, embed_dim)
        self.x_ln = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, d_model)
        self.k_proj = nn.Linear(embed_dim, d_model)
        self.v_proj = nn.Linear(embed_dim, d_model)
        self.mlp_head_collect = nn.Sequential(
            nn.Linear(d_model, 4*token_dim), nn.GELU(),
            nn.Linear(4*token_dim, 2*token_dim), nn.GELU(),
            nn.Linear(2*token_dim, token_dim),
        )
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x_ha_wa):
        B, L, D = x_ha_wa.shape
        h, d = self.num_heads, self.head_dim
        x_ln = self.x_ln(self.x_embedding(x_ha_wa))
        Q = self.q_proj(x_ln).view(B, L, h, d).transpose(1, 2)
        K = self.k_proj(x_ln).view(B, L, h, d).transpose(1, 2)
        V = self.v_proj(x_ln).view(B, L, h, d).transpose(1, 2)
        scores = (Q @ K.transpose(-2,-1)) / math.sqrt(d)
        attn = self.attn_drop(F.softmax(scores, dim=-1))
        context = (attn @ V).transpose(1,2).contiguous().view(B, L, h*d)
        context = self.mlp_head_collect(context)
        y_h = context[:, :self.num_tokens_h, :] + x_ha_wa[:, :self.num_tokens_h, :]
        y_w = context[:, self.num_tokens_h:, :] + x_ha_wa[:, self.num_tokens_h:, :]
        return y_h, y_w


###############################################################################
# 4. TRANSFORMER BLOCK — DUAL-STREAM (from SALLO-M, no masking)
###############################################################################
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.K = config.K; self.N = config.N
        d_model = config.d_model; n_head = config.n_head; embed_dim = config.embed_dim

        self.self_attn_user = MultiHeadAttentionUser(
            token_dim=self.N, seq_len=4*self.K, d_model=d_model,
            num_heads=n_head, embed_dim=embed_dim)
        self.self_attn_antenna = MultiHeadAttentionAntenna(
            token_dim=self.K, seq_len=4*self.N, d_model=d_model,
            num_heads=n_head, embed_dim=embed_dim)

        self.y_h_u_ln = nn.LayerNorm(self.N); self.y_w_u_ln = nn.LayerNorm(self.N)
        self.y_h_a_ln = nn.LayerNorm(self.K); self.y_w_a_ln = nn.LayerNorm(self.K)

        self.mlp_h_u = nn.Sequential(
            nn.Linear(self.N,4*self.N),nn.GELU(),nn.Linear(4*self.N,4*self.N),nn.GELU(),
            nn.Linear(4*self.N,self.N),nn.LayerNorm(self.N))
        self.mlp_w_u = nn.Sequential(
            nn.Linear(self.N,4*self.N),nn.GELU(),nn.Linear(4*self.N,4*self.N),nn.GELU(),
            nn.Linear(4*self.N,self.N))
        self.mlp_h_a = nn.Sequential(
            nn.Linear(self.K,4*self.K),nn.GELU(),nn.Linear(4*self.K,4*self.K),nn.GELU(),
            nn.Linear(4*self.K,self.K),nn.LayerNorm(self.K))
        self.mlp_w_a = nn.Sequential(
            nn.Linear(self.K,4*self.K),nn.GELU(),nn.Linear(4*self.K,4*self.K),nn.GELU(),
            nn.Linear(4*self.K,self.K))

    def forward(self, X_hu_wu, X_ha_wa):
        # User-level attention (no mask)
        y_h_u, y_w_u = self.self_attn_user(X_hu_wu)
        # Antenna-level attention (no mask)
        y_h_a, y_w_a = self.self_attn_antenna(X_ha_wa)

        # User-level MLP
        y_h_u_next = y_h_u + self.mlp_h_u(self.y_h_u_ln(y_h_u))
        y_w_u_out = y_w_u + self.mlp_w_u(self.y_w_u_ln(y_w_u))

        # Antenna-level MLP
        y_h_a_next = y_h_a + self.mlp_h_a(self.y_h_a_ln(y_h_a))
        y_w_a_out = y_w_a + self.mlp_w_a(self.y_w_a_ln(y_w_a))

        # Merge user and antenna streams for auxiliary variable C
        y_h_a_real = y_h_a_next[:, :self.N, :].transpose(-1,-2)  # (B,K,N)
        y_h_a_imag = y_h_a_next[:, self.N:, :].transpose(-1,-2)
        y_h_a_uf = th.cat([y_h_a_real, y_h_a_imag], dim=1)       # (B,2K,N)
        y_h_combined = (y_h_u_next + y_h_a_uf) / 2.0

        # Merge for beamformer W + normalize
        y_w_a_real = y_w_a_out[:, :self.N, :].transpose(-1,-2)
        y_w_a_imag = y_w_a_out[:, self.N:, :].transpose(-1,-2)
        y_w_a_uf = th.cat([y_w_a_real, y_w_a_imag], dim=1)
        y_w_combined = (y_w_u_out + y_w_a_uf) / 2.0
        # Power normalize the BF output
        y_w_flat = y_w_combined.reshape(y_w_combined.size(0), -1)
        y_w_flat = y_w_flat / y_w_flat.norm(dim=1, keepdim=True)
        y_w_next = y_w_flat.reshape(y_w_combined.shape)

        return y_h_combined, y_w_next


###############################################################################
# 5. FULL L2O TRANSFORMER MODEL (no PGD)
###############################################################################
class SALLOMPilotModel(nn.Module):
    """
    SALLO-M Transformer adapted for noisy pilot input.
    No PGD refinement, no masking, no LMMSE initialization.
    """
    def __init__(self, config):
        super().__init__()
        self.K = config.K; self.N = config.N
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[SALLO-M Pilot] Model parameters: {n_params:,}")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward_layers(self, X_hu_wu, X_ha_wa, layer_range):
        """Forward through specified layer range. Returns list of per-layer W solutions."""
        C_user, W_user = X_hu_wu, X_ha_wa
        W_traj = []
        for idx in layer_range:
            blk = self.blocks[idx]
            y_h, y_w = blk(C_user, W_user)

            # Reconstruct complex C and W from real representation
            C_complex = y_h[:, :self.K, :] + 1j * y_h[:, self.K:, :]  # (B,K,N)
            W_complex = y_w[:, :self.K, :] + 1j * y_w[:, self.K:, :]  # (B,K,N)
            W_traj.append(W_complex)

            # Prepare inputs for next layer
            C_user = th.cat([C_complex.real, C_complex.imag], dim=1)  # (B,2K,N)
            W_update = th.cat([W_complex.real, W_complex.imag], dim=1)
            C_user = th.cat([C_user, W_update], dim=1)  # (B,4K,N)

            C_ant = th.cat([C_complex.real.transpose(-1,-2), C_complex.imag.transpose(-1,-2)], dim=1)
            W_ant = th.cat([W_complex.real.transpose(-1,-2), W_complex.imag.transpose(-1,-2)], dim=1)
            W_user = th.cat([C_ant, W_ant], dim=1)  # (B,4N,K)

        return W_traj


###############################################################################
# 6. CONFIGURATION
###############################################################################
class Config:
    def __init__(self, **kw):
        self.seed = kw.get('seed', 2026)
        self.K = kw.get('K', 32)
        self.N = kw.get('N', 32)
        self.L_p = kw.get('L_p', 20)
        self.P_max = kw.get('P_max', 1.0)
        self.SNR_dB = kw.get('SNR_dB', 20)
        self.sigma2 = self.P_max / (10 ** (self.SNR_dB / 10))
        self.ch_n_clusters = kw.get('ch_n_clusters', 3)
        self.ch_n_rays = kw.get('ch_n_rays', 5)
        self.ch_spread_deg = kw.get('ch_spread_deg', 10.0)
        # Transformer
        self.d_model = kw.get('d_model', 768)
        self.n_head = kw.get('n_head', 12)
        self.embed_dim = kw.get('embed_dim', 128)
        self.n_layers = kw.get('n_layers', 10)
        # Training
        self.batch_size = kw.get('batch_size', 64)
        self.lr = kw.get('lr', 8e-5)
        self.weight_decay = kw.get('weight_decay', 0.0)
        self.epochs_per_phase = kw.get('epochs_per_phase', 100)
        self.window_size = kw.get('window_size', 3)
        self.steps_per_epoch = kw.get('steps_per_epoch', 200)
        self.n_test = kw.get('n_test', 200)

    @property
    def n_phases(self):
        return self.n_layers - self.window_size + 1

    @property
    def total_epochs(self):
        return self.n_phases * self.epochs_per_phase


###############################################################################
# 7. TRAINING LOOP (Sliding-Window, No PGD)
###############################################################################
def train(cfg):
    th.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if th.cuda.is_available(): th.cuda.manual_seed_all(cfg.seed)

    print("="*70)
    print("  SALLO-M PILOT BASELINE (No CSI, No PGD)")
    print("="*70)
    print(f"  K={cfg.K} N={cfg.N} L_p={cfg.L_p} SNR={cfg.SNR_dB}dB")
    print(f"  Layers={cfg.n_layers} Window={cfg.window_size} Phases={cfg.n_phases}")
    print(f"  d_model={cfg.d_model} n_head={cfg.n_head} embed={cfg.embed_dim}")

    Phi = generate_pilot_dft(cfg.K, cfg.L_p)

    # Test set
    H_test = generate_sparse_channel(cfg.n_test, cfg.K, cfg.N,
                                      cfg.ch_n_clusters, cfg.ch_n_rays, cfg.ch_spread_deg)
    # Baselines
    with th.no_grad():
        W_mmse_p = mmse_beamformer(H_test, cfg.P_max, cfg.sigma2)
        bl_mmse_p = sum_rate_per_sample(H_test, W_mmse_p, cfg.sigma2).mean().item()
        Y_test = pilot_observe(H_test, Phi, cfg.sigma2)
        Hh = mmse_channel_est(Y_test, Phi, cfg.sigma2)
        W_mmse_e = mmse_beamformer(Hh, cfg.P_max, cfg.sigma2)
        bl_mmse_e = sum_rate_per_sample(H_test, W_mmse_e, cfg.sigma2).mean().item()
    print(f"\n  Baselines: MMSE-Perfect={bl_mmse_p:.2f}  MMSE-Imperfect={bl_mmse_e:.2f}")

    model = SALLOMPilotModel(cfg).to(device)

    print(f"\n  Training: {cfg.total_epochs} epochs ({cfg.n_phases} phases × {cfg.epochs_per_phase} ep)")
    print("="*70)

    best_test = 0.0
    hist = {'test': [], 'train': []}

    for epoch in range(cfg.total_epochs):
        t0 = time.time()

        # Determine sliding-window position
        phase = epoch // cfg.epochs_per_phase
        window_start = min(phase, cfg.n_layers - cfg.window_size)
        window_end = window_start + cfg.window_size - 1  # inclusive

        # Set trainable layers
        for idx, blk in enumerate(model.blocks):
            if window_start <= idx <= window_end:
                blk.train()
                for p in blk.parameters(): p.requires_grad_(True)
            else:
                blk.eval()
                for p in blk.parameters(): p.requires_grad_(False)

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = th.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)

        ep_rate = 0.; ep_n = 0

        for step in range(cfg.steps_per_epoch):
            B = cfg.batch_size
            # Generate fresh sparse channels
            H = generate_sparse_channel(B, cfg.K, cfg.N,
                                        cfg.ch_n_clusters, cfg.ch_n_rays, cfg.ch_spread_deg)
            # Noisy pilot observation
            Y = pilot_observe(H, Phi, cfg.sigma2)  # (B, N, L_p)

            # === Construct initial auxiliary variable C^(0) from pilot ===
            # Y^T: (B, L_p, N) → zero-pad user dim from L_p to K → (B, K, N)
            Y_t = Y.transpose(-1, -2)  # (B, L_p, N)
            C0 = th.zeros(B, cfg.K, cfg.N, device=device, dtype=th.cfloat)
            C0[:, :cfg.L_p, :] = Y_t  # first L_p "users" filled with pilot data

            # === Initial beamformer W^(0): MMSE BF from imperfect CSI ===
            # Imperfect CSI is obtained via LS channel estimate using MP-inverse on pilots.
            H_ls = ls_channel_est(Y, Phi)                             # (B, K, N)
            W0_nk = mmse_beamformer(H_ls, cfg.P_max, cfg.sigma2)      # (B, N, K)
            W0 = W0_nk.transpose(-1, -2).contiguous()                 # (B, K, N)

            # === Build dual-tokenization inputs ===
            # User-level: (B, 4K, N)
            C_user = th.cat([C0.real, C0.imag], dim=1)  # (B, 2K, N)
            W_user = th.cat([W0.real, W0.imag], dim=1)  # (B, 2K, N)
            X_hu_wu = th.cat([C_user, W_user], dim=1)   # (B, 4K, N)

            # Antenna-level: (B, 4N, K)
            C_ant = th.cat([C0.real.transpose(-1,-2), C0.imag.transpose(-1,-2)], dim=1)
            W_ant = th.cat([W0.real.transpose(-1,-2), W0.imag.transpose(-1,-2)], dim=1)
            X_ha_wa = th.cat([C_ant, W_ant], dim=1)  # (B, 4N, K)

            # Forward through layers 0..window_end
            W_traj = model.forward_layers(X_hu_wu, X_ha_wa, range(window_end + 1))

            # Loss: -sum_rate of FINAL layer's beamformer on TRUE H
            W_final = W_traj[-1]  # (B, K, N) complex
            W_final_nk = W_final.transpose(-1, -2)  # (B, N, K)
            W_final_nk = power_normalize(W_final_nk, cfg.P_max)
            rate = sum_rate(H, W_final_nk, cfg.sigma2)
            loss = -rate

            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(trainable, max_norm=200.0)
            optimizer.step()

            ep_rate += rate.item(); ep_n += 1

        # Evaluate on test set
        model.eval()
        test_rates = []
        with th.no_grad():
            for s in range(0, cfg.n_test, cfg.batch_size):
                e = min(s+cfg.batch_size, cfg.n_test)
                Hb = H_test[s:e]; b = Hb.size(0)
                Yb = pilot_observe(Hb, Phi, cfg.sigma2)

                # Same initialization as training
                Yt = Yb.transpose(-1,-2)
                C0 = th.zeros(b, cfg.K, cfg.N, device=device, dtype=th.cfloat)
                C0[:, :cfg.L_p, :] = Yt
                H_ls_b = ls_channel_est(Yb, Phi)                       # (b, K, N)
                W0 = mmse_beamformer(H_ls_b, cfg.P_max, cfg.sigma2).transpose(-1, -2).contiguous()

                Cu = th.cat([C0.real, C0.imag], dim=1)
                Wu = th.cat([W0.real, W0.imag], dim=1)
                Xhu = th.cat([Cu, Wu], dim=1)
                Ca = th.cat([C0.real.transpose(-1,-2), C0.imag.transpose(-1,-2)], dim=1)
                Wa = th.cat([W0.real.transpose(-1,-2), W0.imag.transpose(-1,-2)], dim=1)
                Xha = th.cat([Ca, Wa], dim=1)

                W_traj = model.forward_layers(Xhu, Xha, range(cfg.n_layers))
                Wf = power_normalize(W_traj[-1].transpose(-1,-2), cfg.P_max)
                test_rates.append(sum_rate_per_sample(Hb, Wf, cfg.sigma2))
        tr = th.cat(test_rates).mean().item()
        best_test = max(best_test, tr)

        ar = ep_rate / max(1, ep_n); dt = time.time() - t0
        hist['test'].append(tr); hist['train'].append(ar)

        # Save training-rate snapshot every 10 epochs
        if (epoch + 1) % 10 == 0:
            ep_tensor = th.arange(1, len(hist['train']) + 1)
            th.save({
                'epochs': ep_tensor,
                'train_rate': th.tensor(hist['train'], dtype=th.float32),
            }, 'train_rate_sallom.pt')

        print(f"{epoch+1:3d} ph{phase} [{window_start}-{window_end}] | "
              f"train={ar:.2f} test={tr:.3f} best={best_test:.3f} | "
              f"B1={bl_mmse_p:.2f} B2={bl_mmse_e:.2f} ({dt:.1f}s)", flush=True)

    # Final summary
    print("\n"+"="*68)
    print("  SALLO-M PILOT BASELINE COMPLETE")
    for k,v in [("MMSE+Perfect", bl_mmse_p), ("MMSE+Imperfect", bl_mmse_e),
                ("SALLO-M best", best_test), ("SALLO-M final", tr)]:
        print(f"  {k:<25} {v:8.3f}")

    # Final overwrite save
    ep_tensor = th.arange(1, len(hist['train']) + 1)
    th.save({
        'epochs': ep_tensor,
        'train_rate': th.tensor(hist['train'], dtype=th.float32),
    }, 'train_rate_sallom.pt')

    return model, hist


###############################################################################
# 8. ENTRY POINT
###############################################################################
if __name__ == "__main__":
    cfg = Config(
        K=32, N=32, L_p=20,
        P_max=1.0, SNR_dB=20,
        ch_n_clusters=3, ch_n_rays=5, ch_spread_deg=10.0,
        d_model=768, n_head=12, embed_dim=128,
        n_layers=10, window_size=3,
        batch_size=64, lr=1e-4, weight_decay=0.0,
        epochs_per_phase=50, steps_per_epoch=200,
        n_test=200,
    )
    train(cfg)
