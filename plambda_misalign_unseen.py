"""
Multi-Scenario ICL Precoding with (p, lambda) — Context Misalignment Study

This script studies the performance degradation when the ICL context is
drawn from a DIFFERENT scenario than the actual query channel, simulating
a scenario misidentification event during deployment.

MISALIGNMENT SETUP:
  During inference, for each test sample:
    - The TRUE query pilot is generated from scenario B (the actual environment)
    - But we INCORRECTLY believe it is from scenario A (misidentified)
    - So we draw context demo pairs from scenario A's dataset
    - The model sees a mismatched (context_A, query_B) pair
  This tests the robustness of ICL: can it still produce reasonable precoders
  when the demonstration context does not match the query distribution?

CONTROL FLOW:
  - cfg.load_pretrained = False (default): train from scratch, save checkpoint
  - cfg.load_pretrained = True: load saved checkpoint, skip training

TRAINING: identical to pilot_icl_plambda_multi_scenario.py (unchanged)
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
from dataclasses import dataclass
import warnings
import time
import os
import os.path as osp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_global_seed(seed, det=False):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if det: torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False


###############################################################################
# 1. SCENARIO DEFINITIONS
###############################################################################
@dataclass
class ScenarioDef:
    """Channel scenario definition. Each scenario has distinct (clusters, rays, spread)."""
    name: str
    sid: int
    ch_type: str        # 'sparse' or 'rayleigh'
    n_clusters: int = 3
    n_rays: int = 5
    spread_deg: float = 10.0

SCENARIOS = [
    ScenarioDef("Ultra Dense Scatter", 0, 'sparse', 8, 8, 25.0),
    ScenarioDef("Near-Field LoS",      1, 'sparse', 1, 20, 1.0),
    ScenarioDef("Far-Field LoS",       2, 'sparse', 1, 6, 4.0),
    ScenarioDef("Rayleigh iid",        3, 'rayleigh'),
]

# Deliberately far from the four seen tasks (high clusters + very low rays + very wide spread).
UNSEEN_SCENARIO = ScenarioDef("Extreme Blockage WideSpread", 10, 'sparse', 12, 1, 40.0)


###############################################################################
# 2. CONFIGURATION
###############################################################################
class Config:
    def __init__(self, **kw):
        self.seed = kw.get('seed', 2026)
        self.K = kw.get('K', 32); self.N = kw.get('N', 32)
        self.L_p = kw.get('L_p', 20); self.P_max = kw.get('P_max', 1.0)
        self.SNR_dB = kw.get('SNR_dB', 20)
        self.sigma2 = self.P_max / (10 ** (self.SNR_dB / 10))

        self.scenarios = kw.get('scenarios', SCENARIOS)
        self.n_scenarios = len(self.scenarios)
        self.scenario_ids = [sc.sid for sc in self.scenarios]
        self.sid_to_scenario = {sc.sid: sc for sc in self.scenarios}

        # Token dimension: 4K (state = 4K from pilot encoder, label = [p,lam,0,0] = 4K)
        self.token_dim = 4 * self.K
        self.label_dim = 2 * self.K

        # Pilot encoder (Phase 0 pretraining)
        self.encoder_hidden = kw.get('encoder_hidden', 256)
        self.edn_epochs = kw.get('edn_epochs', 500)
        self.edn_lr = kw.get('edn_lr', 1e-3)
        self.edn_batch = kw.get('edn_batch', 128)
        self.edn_n_samples = kw.get('edn_n_samples', 8000)  # total across all scenarios

        # ICL Transformer
        self.n_demos = kw.get('n_demos', 5)
        self.d_model = kw.get('d_model', 512); self.n_heads = kw.get('n_heads', 8)
        self.n_layers = kw.get('n_layers', 6); self.d_ff = kw.get('d_ff', 1024)
        self.dropout = kw.get('dropout', 0.0)

        # Training
        self.batch_size = kw.get('batch_size', 64)
        self.lr = kw.get('lr', 1e-4); self.lr_min = kw.get('lr_min', 5e-5)
        self.weight_decay = kw.get('weight_decay', 1e-4)
        self.init_ds_per_scenario = kw.get('init_ds_per_scenario', 256)

        # (p,lambda) label generation via Adam on H_hat
        self.opt_iters = kw.get('opt_iters', 300)
        self.opt_lr = kw.get('opt_lr', 0.03)
        self.opt_restarts = kw.get('opt_restarts', 2)
        self.use_robust_labels = kw.get('use_robust_labels', True)  # optimize on H_hat

        # Loss
        self.unsup_scale = kw.get('unsup_scale', 0.005)

        # Curriculum schedule
        self.phase1_epochs = kw.get('phase1_epochs', 100)
        self.phase2_epochs = kw.get('phase2_epochs', 500)
        self.total_epochs = self.phase1_epochs + self.phase2_epochs
        self.steps_per_epoch = kw.get('steps_per_epoch', 80)

        # Dual-threshold self-bootstrapping
        self.boot_alpha_start = kw.get('boot_alpha_start', 0.60)
        self.boot_alpha_end = kw.get('boot_alpha_end', 0.90)
        self.boot_beta_start = kw.get('boot_beta_start', 0.60)
        self.boot_beta_end = kw.get('boot_beta_end', 0.90)
        self.max_ds_per_scenario = kw.get('max_ds_per_scenario', 20000)

        # Bottom pruning
        self.prune_every = kw.get('prune_every', 10)
        self.prune_drop_start = kw.get('prune_drop_start', 0.0)
        self.prune_drop_end = kw.get('prune_drop_end', 0.10)
        self.prune_min_unsup = kw.get('prune_min_unsup', 1024)
        # Growth-friendly pruning controls (to avoid aggressive dataset shrinkage)
        self.prune_enable = kw.get('prune_enable', True)
        self.prune_warmup_phase2_epochs = kw.get('prune_warmup_phase2_epochs', 200)
        self.prune_drop_ratio_cap = kw.get('prune_drop_ratio_cap', 0.01)
        self.prune_drop_vs_add_ratio = kw.get('prune_drop_vs_add_ratio', 0.10)
        self.prune_max_drop_per_scenario = kw.get('prune_max_drop_per_scenario', 32)

        # Evaluation
        self.n_test_per_scenario = kw.get('n_test_per_scenario', 100)

        # Checkpoint save/load control
        self.load_pretrained = kw.get('load_pretrained', False)
        self.ckpt_path = kw.get('ckpt_path', 'plambda_multi_scenario.pt')

        # Inference test: context/query rate matrix
        self.n_test_context_query_matrix = kw.get('n_test_context_query_matrix', 200)
        self.ctx_query_matrix_plot_path = kw.get('ctx_query_matrix_plot_path', 'context_query_rate_matrix_4x4.png')

        # Training-curve logging
        self.train_curve_save_every = kw.get('train_curve_save_every', 10)
        self.train_curve_dir = kw.get('train_curve_dir', 'training_curves_4x4')
        self.train_curve_plot_path = kw.get('train_curve_plot_path', 'training_curves_4x4.png')

        # Unseen adaptation study
        self.unseen_scenario = kw.get('unseen_scenario', UNSEEN_SCENARIO)
        self.unseen_context_size = kw.get('unseen_context_size', 1024)
        self.unseen_n_test = kw.get('unseen_n_test', 200)


###############################################################################
# 3. CHANNEL GENERATION
###############################################################################
def generate_sparse_channel(B, K, N, n_cl, n_ray, spread_deg):
    """Cluster-based geometric sparse mmWave channel. Returns (B,K,N) complex."""
    L = n_cl * n_ray; asp = math.radians(spread_deg)
    cm = (torch.rand(B,K,n_cl,1,device=device)-0.5)*math.pi
    ro = torch.randn(B,K,n_cl,n_ray,device=device)*asp
    ang = (cm+ro).clamp(-math.pi/2,math.pi/2).reshape(B,K,L)
    alp = (torch.randn(B,K,L,device=device)+1j*torch.randn(B,K,L,device=device))/math.sqrt(2)
    idx = torch.arange(N,device=device,dtype=torch.float32).view(1,1,1,N)
    ph = math.pi*idx*torch.sin(ang).unsqueeze(-1)
    st = torch.polar(torch.ones_like(ph),ph)/math.sqrt(N)
    return math.sqrt(N/L)*torch.sum(alp.unsqueeze(-1)*st, dim=2)

def generate_rayleigh_channel(B, K, N):
    """iid Rayleigh fading channel. Returns (B,K,N) complex."""
    return (torch.randn(B,K,N,device=device)+1j*torch.randn(B,K,N,device=device))/math.sqrt(2)

def generate_channel_scenario(B, K, N, sc: ScenarioDef):
    """Dispatch channel generation by scenario type."""
    if sc.ch_type == 'rayleigh': return generate_rayleigh_channel(B, K, N)
    return generate_sparse_channel(B, K, N, sc.n_clusters, sc.n_rays, sc.spread_deg)


###############################################################################
# 4. SIGNAL PROCESSING
###############################################################################
def generate_pilot_dft(K, L_p):
    """Truncated DFT pilot matrix (K, L_p) complex."""
    return (torch.fft.fft(torch.eye(K,device=device))/math.sqrt(K))[:,:L_p].contiguous()

def pilot_observe(H, Phi, sigma2):
    """Y = H^T Phi + N. H:(B,K,N), Phi:(K,Lp) -> Y:(B,N,Lp) complex."""
    B,K,N = H.shape; Lp = Phi.size(1)
    Y = H.transpose(-1,-2) @ Phi.unsqueeze(0).expand(B,-1,-1)
    nr = torch.randn(B,N,Lp,device=device)*math.sqrt(sigma2/2)
    ni = torch.randn(B,N,Lp,device=device)*math.sqrt(sigma2/2)
    return Y + torch.complex(nr, ni)

def pilot_to_real(Y):
    """(B,N,Lp) complex -> (B, 2*N*Lp) real."""
    return torch.cat([Y.real,Y.imag],dim=1).reshape(Y.size(0),-1)

def channel_to_real(H):
    """(B,K,N) complex -> (B, 2*K*N) real."""
    return torch.cat([H.real,H.imag],dim=-1).reshape(H.size(0),-1)

def compute_sum_rate(H, W, sigma2):
    """Sum rate on TRUE channel H. H:(B,K,N), W:(B,N,K) -> (B,)."""
    HW = H @ W; sig = torch.abs(torch.diagonal(HW,dim1=-2,dim2=-1))**2
    tot = torch.sum(torch.abs(HW)**2,dim=-1); SINR = sig/(tot-sig+sigma2)
    return torch.log2(1+SINR).sum(-1)

def mmse_beamformer(H, P_max, sigma2):
    """MMSE (regularized ZF) beamformer. H:(B,K,N) -> W:(B,N,K)."""
    B,K,N = H.shape; HH = H.conj().transpose(-1,-2)
    A = HH@H + sigma2*torch.eye(N,device=device,dtype=H.dtype).unsqueeze(0)
    W = torch.linalg.solve(A, HH)
    pw = torch.sum(torch.abs(W)**2,dim=(1,2)).real
    return W*torch.sqrt(P_max/(pw+1e-8)).view(B,1,1)

def power_normalize(W, P_max):
    pw = torch.sum(torch.abs(W)**2,dim=(1,2),keepdim=True).real
    return W*torch.sqrt(P_max/(pw+1e-8))

def mmse_channel_est(Y, Phi, sigma2):
    """MMSE channel estimate from pilots. Returns (B,K,N) complex."""
    K,Lp = Phi.shape
    A = Phi.T@Phi.conj()+sigma2*torch.eye(Lp,device=device,dtype=Phi.dtype)
    return torch.matmul((Phi.conj()@torch.linalg.inv(A)).unsqueeze(0),Y.transpose(-1,-2))

def reconstruct_precoder(H, p, lam, sigma2):
    """
    Reconstruct BF from optimal structure:
      w_k = sqrt(p_k) * A^{-1} h_k / ||A^{-1} h_k||
      A = I_N + (1/sigma2) * H^H diag(lam) H
    H:(B,K,N), p:(B,K), lam:(B,K) -> W:(B,N,K).
    """
    B,K,N = H.shape
    h = H.conj().transpose(-1,-2)  # (B,N,K)
    lam_diag = torch.diag_embed(lam/sigma2).to(torch.cfloat)
    eye = torch.eye(N,device=device,dtype=torch.cfloat).unsqueeze(0)
    A = eye + h @ lam_diag @ h.conj().transpose(-1,-2)
    A_inv_h = torch.linalg.solve(A, h)
    norms = torch.norm(A_inv_h, dim=1, keepdim=True).real + 1e-8
    V = A_inv_h / norms
    return V * torch.sqrt(p).unsqueeze(1).to(torch.cfloat)


###############################################################################
# 5. (p, lambda) LABEL GENERATION
###############################################################################
def generate_optimal_params(H, P_max, sigma2, n_iters=300, lr=0.03, n_restarts=2):
    """
    Optimize (p, lambda) via Adam to maximize sum-rate.
    H can be either true channel or MMSE estimate (robust mode).
    Returns (best_p, best_lam, best_rate).
    """
    B,K,N = H.shape; Hd = H.detach()
    best_rate = torch.full((B,),-float('inf'),device=device)
    best_p = torch.zeros(B,K,device=device)
    best_lam = torch.zeros(B,K,device=device)

    for _ in range(n_restarts):
        p_log = (torch.randn(B,K,device=device)*0.1).requires_grad_(True)
        lam_log = (torch.randn(B,K,device=device)*0.1).requires_grad_(True)
        opt = Adam([p_log, lam_log], lr=lr)
        for _ in range(n_iters):
            p = F.softmax(p_log,dim=-1)*P_max
            lam = F.softplus(lam_log)
            W = reconstruct_precoder(Hd, p, lam, sigma2)
            (-compute_sum_rate(Hd,W,sigma2).sum()).backward()
            opt.step(); opt.zero_grad()
        with torch.no_grad():
            p_s = F.softmax(p_log,dim=-1)*P_max; lam_s = F.softplus(lam_log)
            W_s = reconstruct_precoder(Hd, p_s, lam_s, sigma2)
            r_s = compute_sum_rate(Hd, W_s, sigma2)
            imp = r_s > best_rate
            if imp.any():
                best_rate[imp]=r_s[imp]; best_p[imp]=p_s[imp]; best_lam[imp]=lam_s[imp]
    return best_p.detach(), best_lam.detach(), best_rate.detach()


###############################################################################
# 6. NETWORK MODULES
###############################################################################
class PilotEncoder(nn.Module):
    """CNN + attention pooling. Maps Y_real -> state token z ∈ R^{4K}."""
    def __init__(self, N, L_p, K, hidden=256):
        super().__init__()
        self.N, self.L_p, self.K = N, L_p, K
        self.conv1 = nn.Conv1d(2*N, hidden, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.ln = nn.LayerNorm(hidden)
        self.attn_q = nn.Parameter(torch.randn(1,1,hidden)*0.02)
        self.attn_k = nn.Linear(hidden, hidden)
        self.attn_v = nn.Linear(hidden, hidden)
        self.proj = nn.Sequential(nn.Linear(hidden,hidden), nn.GELU(), nn.Linear(hidden,4*K))

    def forward(self, x, sigma2=None):
        """x: (B, 2*N*L_p) -> (B, 4K)."""
        B = x.size(0); x = x.view(B, 2*self.N, self.L_p)
        x = F.gelu(self.conv1(x)); x = F.gelu(self.conv2(x))
        x = self.ln(x.transpose(1,2))
        q = self.attn_q.expand(B,-1,-1); k,v = self.attn_k(x), self.attn_v(x)
        w = F.softmax(torch.bmm(q,k.transpose(1,2))/math.sqrt(k.size(-1)),-1)
        return self.proj(torch.bmm(w,v).squeeze(1))


class ChannelDecoder(nn.Module):
    """Decodes z -> H_hat_real. Used only during Phase 0 pretraining."""
    def __init__(self, K, N, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4*K,hidden),nn.GELU(),
                                 nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,2*K*N))
    def forward(self, z): return self.net(z)


class CausalBlock(nn.Module):
    def __init__(self, d, heads, d_ff, drop=0.0):
        super().__init__()
        self.ln1=nn.LayerNorm(d)
        self.attn=nn.MultiheadAttention(d,heads,dropout=drop,batch_first=True)
        self.ln2=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,d_ff),nn.GELU(),nn.Linear(d_ff,d),nn.Dropout(drop))
    def forward(self, x, mask):
        h=self.ln1(x); x=x+self.attn(h,h,h,attn_mask=mask)[0]
        return x+self.ff(self.ln2(x))


class ICLTransformer(nn.Module):
    def __init__(self, tok, d, heads, layers, d_ff, drop=0.0):
        super().__init__()
        self.proj_in=nn.Linear(tok,d); self.ln_in=nn.LayerNorm(d)
        self.blocks=nn.ModuleList([CausalBlock(d,heads,d_ff,drop) for _ in range(layers)])
        self.ln_out=nn.LayerNorm(d); self.proj_out=nn.Linear(d,tok)
    def forward(self, seq):
        L=seq.size(1); mask=torch.triu(torch.ones(L,L,device=seq.device,dtype=torch.bool),diagonal=1)
        x=self.ln_in(self.proj_in(seq))
        for blk in self.blocks: x=blk(x,mask)
        return self.proj_out(self.ln_out(x))


class PilotICLModel_PLambda(nn.Module):
    """
    ICL model with (p,lambda) parameterization.
    No BFEncoder needed — labels are directly [p, lambda, 0, 0] ∈ R^{4K}.
    """
    def __init__(self, pilot_enc, cfg):
        super().__init__()
        self.pilot_enc = pilot_enc
        self.K = cfg.K; self.P_max = cfg.P_max
        self.token_dim = cfg.token_dim; self.label_dim = cfg.label_dim
        self.transformer = ICLTransformer(
            cfg.token_dim, cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.d_ff, cfg.dropout)

    def _label_token(self, p, lam):
        """Construct zero-padded label token: [p, lam, 0, 0] ∈ R^{4K}."""
        B = p.size(0)
        tok = torch.zeros(B, self.token_dim, device=p.device)
        tok[:, :self.K] = p
        tok[:, self.K:2*self.K] = lam
        return tok

    def _extract(self, raw):
        """Extract (p, lam) from the 4K-dim Transformer output.
        p: sigmoid + L1-normalize to sum to P_max.
        lam: softplus for positivity (no sum constraint)."""
        p = torch.sigmoid(raw[:, :self.K])
        p = p / (p.sum(-1, keepdim=True) + 1e-8) * self.P_max
        lam = F.softplus(raw[:, self.K:2*self.K])
        return p, lam

    def forward(self, demo_pilots, demo_p, demo_lam, query_pilot):
        """
        demo_pilots: (B, l, 2*N*Lp) — demo pilot observations
        demo_p:      (B, l, K)       — demo power allocations
        demo_lam:    (B, l, K)       — demo dual variables
        query_pilot: (B, 2*N*Lp)    — query pilot observation
        Returns: (p_pred, lam_pred) each (B, K).
        """
        B, l, pd = demo_pilots.shape

        # Encode all pilots (demos + query) through frozen PilotEncoder
        all_pil = torch.cat([demo_pilots.reshape(B*l, pd), query_pilot], dim=0)
        with torch.no_grad():
            all_z = self.pilot_enc(all_pil)  # ((B*l+B), 4K)
        demo_z = all_z[:B*l].reshape(B, l, self.token_dim)
        query_z = all_z[B*l:]  # (B, 4K)

        # Build ICL token sequence: [z_1, y_1, z_2, y_2, ..., z_l, y_l, z_q]
        tokens = []
        for i in range(l):
            tokens.append(demo_z[:, i])                           # state token
            tokens.append(self._label_token(demo_p[:, i], demo_lam[:, i]))  # label token
        tokens.append(query_z)                                    # query state
        seq = torch.stack(tokens, dim=1)  # (B, 2l+1, 4K)

        out = self.transformer(seq)
        return self._extract(out[:, -1])  # predict (p, lam) at query position


###############################################################################
# 7. PER-SCENARIO DATASET (stores H, p, lam, rates)
###############################################################################
class DynDataset:
    """Per-scenario dynamic dataset for (p,lambda) parameterization."""
    def __init__(self, max_sz=20000):
        self.max_sz = max_sz
        # Keep tensor buffers non-optional to avoid None-propagation in static analysis.
        self.H = torch.empty((0, 0, 0), dtype=torch.cfloat, device=device)
        self.p = torch.empty((0, 0), dtype=torch.float32, device=device)
        self.lam = torch.empty((0, 0), dtype=torch.float32, device=device)
        self.rates = torch.empty((0,), dtype=torch.float32, device=device)
        self.mmse_rates = torch.empty((0,), dtype=torch.float32, device=device)
        self.is_sup = torch.empty((0,), dtype=torch.bool, device=device)
        self._n = 0; self.n_sup = 0; self.n_unsup = 0

    @property
    def size(self): return self._n

    def add(self, H, p, lam, rates, mmse_rates=None, supervised=True):
        H,p,lam,rates = [x.detach() for x in [H,p,lam,rates]]
        flag = torch.full((H.size(0),),bool(supervised),device=device,dtype=torch.bool)
        mr = mmse_rates.detach() if mmse_rates is not None else torch.zeros_like(rates)
        if self._n == 0:
            self.H,self.p,self.lam,self.rates,self.mmse_rates,self.is_sup = H,p,lam,rates,mr,flag
        else:
            self.H=torch.cat([self.H,H]); self.p=torch.cat([self.p,p])
            self.lam=torch.cat([self.lam,lam]); self.rates=torch.cat([self.rates,rates])
            self.mmse_rates=torch.cat([self.mmse_rates,mr]); self.is_sup=torch.cat([self.is_sup,flag])
        if supervised: self.n_sup+=H.size(0)
        else: self.n_unsup+=H.size(0)
        # Capacity management: keep top-rate samples
        if self.H.size(0) > self.max_sz:
            sup_idx=torch.where(self.is_sup)[0]; uns_idx=torch.where(~self.is_sup)[0]
            if sup_idx.numel()>=self.max_sz:
                keep=sup_idx[torch.topk(self.rates[sup_idx],self.max_sz).indices]
            else:
                nk=self.max_sz-sup_idx.numel()
                ku=uns_idx[torch.topk(self.rates[uns_idx],min(nk,uns_idx.numel())).indices] if uns_idx.numel()>nk else uns_idx
                keep=torch.cat([sup_idx,ku])
            for a in ['H','p','lam','rates','mmse_rates','is_sup']:
                setattr(self,a,getattr(self,a)[keep])
        self._n=self.H.size(0)
        self.n_sup=int(self.is_sup.sum().item()); self.n_unsup=int((~self.is_sup).sum().item())

    def prune_unsup_bottom(self, drop_ratio, min_keep=0, max_drop=None):
        if self._n==0 or drop_ratio<=0: return 0
        ui=torch.where(~self.is_sup)[0]; nu=ui.numel()
        if nu==0: return 0
        nd=min(int(nu*drop_ratio),max(0,nu-min_keep))
        if max_drop is not None:
            nd=min(nd,int(max_drop))
        if nd<=0: return 0
        worst=ui[torch.topk(self.rates[ui],k=nd,largest=False).indices]
        keep=torch.ones(self._n,device=device,dtype=torch.bool); keep[worst]=False
        for a in ['H','p','lam','rates','mmse_rates','is_sup']:
            setattr(self,a,getattr(self,a)[keep])
        self._n=self.H.size(0)
        self.n_sup=int(self.is_sup.sum().item()); self.n_unsup=int((~self.is_sup).sum().item())
        return nd


class MultiScenarioDataset:
    """Wraps per-scenario DynDataset instances."""
    def __init__(self, scenarios, max_per_scenario=20000):
        self.scenarios = scenarios
        self.datasets = {sc.sid: DynDataset(max_sz=max_per_scenario) for sc in scenarios}
    def get(self, sid): return self.datasets[sid]
    @property
    def total_size(self): return sum(ds.size for ds in self.datasets.values())
    def summary(self):
        lines = []
        for sc in self.scenarios:
            ds = self.datasets[sc.sid]
            lines.append(f"  S{sc.sid} {sc.name:<18} size={ds.size:5d} (sup={ds.n_sup}, unsup={ds.n_unsup})")
        return "\n".join(lines)


###############################################################################
# 8. PHASE 0: PRETRAIN PILOT ENCODER (on mixed-scenario data)
###############################################################################
def build_mixed_labeled_cache(cfg, Phi):
    """Generate labeled data from ALL scenarios for Phase 0 + initial datasets."""
    print("\n[Mixed Cache] Building multi-scenario labeled data...")
    per_sc = cfg.edn_n_samples // cfg.n_scenarios
    cache = {k: [] for k in ['H','Y_real','H_real','p','lam','label_rate','mmse_rate','scenario_id']}

    for sc in cfg.scenarios:
        print(f"  S{sc.sid} {sc.name}: {per_sc} samples...", end=" ", flush=True)
        for s in range(0, per_sc, cfg.edn_batch):
            e = min(s+cfg.edn_batch, per_sc); b = e-s
            H = generate_channel_scenario(b, cfg.K, cfg.N, sc)
            Y = pilot_observe(H, Phi, cfg.sigma2)
            H_hat = mmse_channel_est(Y, Phi, cfg.sigma2)

            # Generate (p,lam) labels — robust mode uses H_hat, legacy uses H
            H_design = H_hat if cfg.use_robust_labels else H
            p_opt, lam_opt, _ = generate_optimal_params(
                H_design, cfg.P_max, cfg.sigma2, cfg.opt_iters, cfg.opt_lr, cfg.opt_restarts)

            # Evaluate rate on TRUE channel with precoder from design channel
            with torch.no_grad():
                W_opt = reconstruct_precoder(H_design, p_opt, lam_opt, cfg.sigma2)
                r_opt = compute_sum_rate(H, W_opt, cfg.sigma2)
                mmse_r = compute_sum_rate(H, mmse_beamformer(H, cfg.P_max, cfg.sigma2), cfg.sigma2)

            cache['H'].append(H); cache['Y_real'].append(pilot_to_real(Y))
            cache['H_real'].append(channel_to_real(H))
            cache['p'].append(p_opt); cache['lam'].append(lam_opt)
            cache['label_rate'].append(r_opt); cache['mmse_rate'].append(mmse_r)
            cache['scenario_id'].append(torch.full((b,),sc.sid,device=device,dtype=torch.long))
        print("done", flush=True)

    out = {k: torch.cat(v,dim=0) for k,v in cache.items()}
    print(f"  Total: {out['H'].size(0)} samples across {cfg.n_scenarios} scenarios")
    return out


def pretrain_pilot_edn(cfg, cache):
    """Phase 0: PilotEncoder + ChannelDecoder autoencoder on mixed data."""
    print("\n[Phase 0] Pretraining PilotEncoder (multi-scenario)...")
    enc = PilotEncoder(cfg.N, cfg.L_p, cfg.K, cfg.encoder_hidden).to(device)
    dec = ChannelDecoder(cfg.K, cfg.N, cfg.encoder_hidden).to(device)
    opt = Adam(list(enc.parameters())+list(dec.parameters()), lr=cfg.edn_lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, cfg.edn_epochs, eta_min=0)
    n = cache['Y_real'].size(0)
    for ep in range(cfg.edn_epochs):
        el=0.; nb=0; perm=torch.randperm(n,device=device)
        for s in range(0,n,cfg.edn_batch):
            e=min(s+cfg.edn_batch,n); idx=perm[s:e]
            loss=F.mse_loss(dec(enc(cache['Y_real'][idx])), cache['H_real'][idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters())+list(dec.parameters()),5.0)
            opt.step(); el+=loss.item(); nb+=1
        sched.step()
        if (ep+1)%100==0 or ep==0: print(f"  Ep {ep+1}/{cfg.edn_epochs} MSE={el/nb:.6f}")
    return enc


###############################################################################
# 9. BASELINES + EVALUATION (per-scenario)
###############################################################################
def compute_per_scenario_baselines(H_tests, Phi, cfg):
    all_bl = {}
    for sc in cfg.scenarios:
        H = H_tests[sc.sid]; s2=cfg.sigma2; Pm=cfg.P_max
        Y = pilot_observe(H, Phi, s2); Hh = mmse_channel_est(Y, Phi, s2)
        with torch.no_grad():
            b_mmse_p = compute_sum_rate(H, mmse_beamformer(H,Pm,s2), s2).mean().item()
            b_mmse_e = compute_sum_rate(H, mmse_beamformer(Hh,Pm,s2), s2).mean().item()
        # Opt (p,lam) on H_hat
        p_opt,lam_opt,_ = generate_optimal_params(Hh,Pm,s2,cfg.opt_iters,cfg.opt_lr,cfg.opt_restarts)
        with torch.no_grad():
            W_opt = reconstruct_precoder(Hh, p_opt, lam_opt, s2)
            b_opt_e = compute_sum_rate(H, W_opt, s2).mean().item()
        all_bl[sc.sid] = {'mmse_perf':b_mmse_p,'mmse_imp':b_mmse_e,'opt_imp':b_opt_e,'name':sc.name}
        print(f"  S{sc.sid} {sc.name:<18} MMSE-P={b_mmse_p:.2f} MMSE-E={b_mmse_e:.2f} Opt-E={b_opt_e:.2f}")
    return all_bl


@torch.no_grad()
def evaluate_per_scenario(model, mds, H_tests, Phi, cfg):
    """Evaluate model on each scenario test set. Context from same scenario."""
    model.eval(); results = {}
    for sc in cfg.scenarios:
        ds = mds.get(sc.sid)
        if ds.size < cfg.n_demos: continue
        H = H_tests[sc.sid]; B=H.size(0); bs=min(cfg.batch_size,B); rates=[]
        for s in range(0,B,bs):
            e=min(s+bs,B); Hb=H[s:e]; b=Hb.size(0)
            Y=pilot_observe(Hb,Phi,cfg.sigma2); pil=pilot_to_real(Y)
            H_hat=mmse_channel_est(Y,Phi,cfg.sigma2)

            # Random context from per-scenario dataset
            d_idx=torch.randint(0,ds.size,(b,cfg.n_demos),device=device)
            # Build demo pilots on-the-fly from stored H
            d_H=ds.H[d_idx].reshape(b*cfg.n_demos,cfg.K,cfg.N)
            d_Y=pilot_observe(d_H,Phi,cfg.sigma2)
            d_pil=pilot_to_real(d_Y).reshape(b,cfg.n_demos,-1)
            d_p=ds.p[d_idx]; d_lam=ds.lam[d_idx]

            p_pred,lam_pred = model(d_pil, d_p, d_lam, pil)
            W_hat = reconstruct_precoder(H_hat, p_pred, lam_pred, cfg.sigma2)
            rates.append(compute_sum_rate(Hb, W_hat, cfg.sigma2))
        results[sc.sid] = torch.cat(rates).mean().item()
    model.train(); return results


###############################################################################
# 10. MAIN TRAINING LOOP
###############################################################################
def train(cfg):
    set_global_seed(cfg.seed)
    print("="*75)
    print("  MULTI-SCENARIO ICL PRECODING with (p,lambda) Parameterization")
    print("="*75)
    print(f"  K={cfg.K} N={cfg.N} L_p={cfg.L_p} SNR={cfg.SNR_dB}dB")
    print(f"  Token dim: {cfg.token_dim} (= 4K), label dim: {cfg.label_dim} (= 2K)")
    for sc in cfg.scenarios:
        tag = f"({sc.n_clusters}cl,{sc.n_rays}r,{sc.spread_deg}°)" if sc.ch_type=='sparse' else "(iid)"
        print(f"    S{sc.sid}: {sc.name:<18} {sc.ch_type:8s} {tag}")

    Phi = generate_pilot_dft(cfg.K, cfg.L_p)

    # Build mixed labeled cache (generates (p,lam) labels for all scenarios)
    cache = build_mixed_labeled_cache(cfg, Phi)

    # Phase 0: Pretrain PilotEncoder only (no BF EDN needed for (p,lam) approach)
    pilot_enc = pretrain_pilot_edn(cfg, cache)
    for p in pilot_enc.parameters(): p.requires_grad_(False)

    # Per-scenario test sets + baselines
    H_tests = {}
    for sc in cfg.scenarios:
        H_tests[sc.sid] = generate_channel_scenario(cfg.n_test_per_scenario, cfg.K, cfg.N, sc)
    print("\nPer-scenario baselines:")
    bl = compute_per_scenario_baselines(H_tests, Phi, cfg)

    # Initialize per-scenario datasets from cache
    mds = MultiScenarioDataset(cfg.scenarios, max_per_scenario=cfg.max_ds_per_scenario)
    sids = cache['scenario_id']
    for sc in cfg.scenarios:
        mask = sids==sc.sid; idx = torch.where(mask)[0][:cfg.init_ds_per_scenario]
        if idx.numel() > 0:
            ds = mds.get(sc.sid)
            ds.add(cache['H'][idx], cache['p'][idx], cache['lam'][idx],
                   cache['label_rate'][idx], mmse_rates=cache['mmse_rate'][idx], supervised=True)

    # Ensure every scenario has enough demos
    for sc in cfg.scenarios:
        ds = mds.get(sc.sid)
        if ds.size >= cfg.n_demos: continue
        need = cfg.n_demos - ds.size
        H_ex = generate_channel_scenario(need, cfg.K, cfg.N, sc)
        Y_ex = pilot_observe(H_ex, Phi, cfg.sigma2); H_hat_ex = mmse_channel_est(Y_ex, Phi, cfg.sigma2)
        H_design = H_hat_ex if cfg.use_robust_labels else H_ex
        p_ex,lam_ex,_ = generate_optimal_params(H_design,cfg.P_max,cfg.sigma2,cfg.opt_iters,cfg.opt_lr)
        with torch.no_grad():
            W_ex = reconstruct_precoder(H_design,p_ex,lam_ex,cfg.sigma2)
            r_ex = compute_sum_rate(H_ex, W_ex, cfg.sigma2)
        ds.add(H_ex, p_ex, lam_ex, r_ex, supervised=True)
    print(f"\nInitial datasets:\n{mds.summary()}")

    # Model (no BF encoder/decoder — just PilotEncoder + ICL Transformer)
    model = PilotICLModel_PLambda(pilot_enc, cfg).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.total_epochs, eta_min=cfg.lr_min)

    print(f"\nModel params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable")
    print("\n"+"="*75)
    best_avg = 0.0
    os.makedirs(cfg.train_curve_dir, exist_ok=True)
    curve_hist = {sc.sid: {'name': sc.name, 'epochs': [], 'train_rate': []} for sc in cfg.scenarios}

    for epoch in range(cfg.total_epochs):
        model.train(); t0 = time.time()
        if epoch < cfg.phase1_epochs: phase, r = 1, 0.0
        else:
            phase = 2; p2e = epoch - cfg.phase1_epochs
            if p2e < 50: r = 0.25
            elif p2e < 100: r = 0.50
            elif p2e < 150: r = 0.75
            else: r = 1.0

        ep_prog = epoch / max(1, cfg.total_epochs-1)
        alpha_t = cfg.boot_alpha_start + (cfg.boot_alpha_end-cfg.boot_alpha_start)*ep_prog
        beta_t = cfg.boot_beta_start + (cfg.boot_beta_end-cfg.boot_beta_start)*ep_prog
        ep_loss, ep_rate, ep_add, ep_n = 0., 0., 0, 0
        per_sid_rate_sum = {sid: 0.0 for sid in cfg.scenario_ids}
        per_sid_rate_cnt = {sid: 0 for sid in cfg.scenario_ids}

        eligible_sids = [sid for sid in cfg.scenario_ids if mds.get(sid).size >= cfg.n_demos]
        if not eligible_sids: raise RuntimeError("No scenario has enough demos.")

        for step in range(cfg.steps_per_epoch):
            # Batch-level round-robin: one scenario per batch
            batch_sid = eligible_sids[(epoch*cfg.steps_per_epoch+step) % len(eligible_sids)]
            ds = mds.get(batch_sid); sc = cfg.sid_to_scenario[batch_sid]
            B = cfg.batch_size; is_unsup = torch.rand(B,device=device) < r

            # Supervised queries from dataset
            sup_idx = torch.randint(0, ds.size, (B,), device=device)
            q_H_sup = ds.H[sup_idx]; q_p_gt = ds.p[sup_idx]; q_lam_gt = ds.lam[sup_idx]

            # Unsupervised queries: fresh channel samples
            q_H_unsup = generate_channel_scenario(B, cfg.K, cfg.N, sc)
            q_H = torch.where(is_unsup.view(B,1,1).expand_as(q_H_sup), q_H_unsup, q_H_sup)

            # Pilot observation + MMSE estimate
            q_Y = pilot_observe(q_H, Phi, cfg.sigma2)
            q_pil = pilot_to_real(q_Y)
            q_H_hat = mmse_channel_est(q_Y, Phi, cfg.sigma2)

            # Random context demos from per-scenario dataset
            d_idx = torch.randint(0, ds.size, (B, cfg.n_demos), device=device)
            d_H = ds.H[d_idx].reshape(B*cfg.n_demos, cfg.K, cfg.N)
            d_Y = pilot_observe(d_H, Phi, cfg.sigma2)
            d_pil = pilot_to_real(d_Y).reshape(B, cfg.n_demos, -1)
            d_p = ds.p[d_idx]; d_lam = ds.lam[d_idx]

            # Forward: predict (p, lam)
            p_pred, lam_pred = model(d_pil, d_p, d_lam, q_pil)

            # Reconstruct W using H_hat (no true channel at inference)
            W_pred = reconstruct_precoder(q_H_hat, p_pred, lam_pred, cfg.sigma2)
            rate_pred = compute_sum_rate(q_H, W_pred, cfg.sigma2)

            # Loss: supervised MSE on (p,lam) + unsupervised negative rate
            mse_per = F.mse_loss(p_pred,q_p_gt,reduction='none').sum(-1) + \
                      F.mse_loss(lam_pred,q_lam_gt,reduction='none').sum(-1)
            loss_per = torch.where(is_unsup, -rate_pred*cfg.unsup_scale, mse_per)
            loss = loss_per.mean()

            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0); optimizer.step()
            batch_rate = rate_pred.mean().item()
            ep_loss += loss.item(); ep_rate += batch_rate; ep_n += 1
            per_sid_rate_sum[batch_sid] += batch_rate
            per_sid_rate_cnt[batch_sid] += 1

            # Self-bootstrapping: admit high-quality unsupervised solutions
            if phase == 2 and is_unsup.any():
                with torch.no_grad():
                    ur = rate_pred[is_unsup]
                    uH = q_H[is_unsup]; up = p_pred[is_unsup]; ul = lam_pred[is_unsup]
                    mmse_r = compute_sum_rate(uH, mmse_beamformer(uH,cfg.P_max,cfg.sigma2), cfg.sigma2)
                    per_ok = ur > alpha_t * mmse_r
                    cross_ok = ur >= torch.quantile(ur,beta_t) if ur.numel()>1 else torch.ones_like(ur,dtype=torch.bool)
                    good = per_ok & cross_ok
                    if good.any():
                        mds.get(batch_sid).add(uH[good],up[good],ul[good],ur[good],
                                               mmse_rates=mmse_r[good],supervised=False)
                        ep_add += good.sum().item()

        scheduler.step()

        # Bottom pruning (growth-friendly and heavily suppressed)
        nd = 0
        if cfg.prune_enable and phase==2 and (epoch+1)%cfg.prune_every==0:
            fp=(epoch-cfg.phase1_epochs)/max(1,cfg.phase2_epochs-1)
            p2e = epoch - cfg.phase1_epochs
            if p2e >= cfg.prune_warmup_phase2_epochs:
                dr_raw=cfg.prune_drop_start+(cfg.prune_drop_end-cfg.prune_drop_start)*fp
                dr=min(dr_raw, cfg.prune_drop_ratio_cap)
                global_budget=min(
                    int(ep_add * cfg.prune_drop_vs_add_ratio),
                    cfg.prune_max_drop_per_scenario * len(cfg.scenarios),
                )
                if global_budget > 0:
                    for sc in cfg.scenarios:
                        if global_budget <= 0:
                            break
                        ds_sc = mds.get(sc.sid)
                        if ds_sc.n_unsup <= cfg.prune_min_unsup:
                            continue
                        local_cap = min(global_budget, cfg.prune_max_drop_per_scenario)
                        ddrop = ds_sc.prune_unsup_bottom(dr, cfg.prune_min_unsup, max_drop=local_cap)
                        nd += ddrop
                        global_budget -= ddrop

        # Evaluation
        sc_rates = evaluate_per_scenario(model, mds, H_tests, Phi, cfg)
        avg_rate = np.mean([v for v in sc_rates.values()]) if sc_rates else 0.0
        best_avg = max(best_avg, avg_rate)

        dt = time.time()-t0
        if (epoch+1)%10==0 or epoch==0:
            rate_str = " ".join([f"S{sc.sid}={sc_rates.get(sc.sid,0):.1f}" for sc in cfg.scenarios])
            print(f"{epoch+1:3d} ph{phase} r={r:.2f} | loss={ep_loss/max(1,ep_n):.4f} rate={ep_rate/max(1,ep_n):.2f} | "
                  f"avg={avg_rate:.2f} best={best_avg:.2f} | DS={mds.total_size} +{ep_add} -{nd} | "
                  f"{rate_str} ({dt:.1f}s)", flush=True)

        if (epoch + 1) % cfg.train_curve_save_every == 0:
            for sc in cfg.scenarios:
                sid = sc.sid
                if per_sid_rate_cnt[sid] == 0:
                    continue
                sid_rate = per_sid_rate_sum[sid] / per_sid_rate_cnt[sid]
                curve_hist[sid]['epochs'].append(epoch + 1)
                curve_hist[sid]['train_rate'].append(sid_rate)
                out_path = os.path.join(cfg.train_curve_dir, f"train_rate_s{sid}.pt")
                torch.save({
                    'scenario_id': sid,
                    'scenario_name': curve_hist[sid]['name'],
                    'epochs': curve_hist[sid]['epochs'],
                    'train_rate': curve_hist[sid]['train_rate'],
                }, out_path)

    # Final summary
    print("\n"+"="*75)
    print("  MULTI-SCENARIO (p,lambda) TRAINING COMPLETE")
    print("="*75)
    for sc in cfg.scenarios:
        ds=mds.get(sc.sid); tr=sc_rates.get(sc.sid,0.0)
        print(f"  S{sc.sid} {sc.name:<18} test={tr:.2f} "
              f"MMSE-P={bl[sc.sid]['mmse_perf']:.2f} MMSE-E={bl[sc.sid]['mmse_imp']:.2f} "
              f"Opt-E={bl[sc.sid]['opt_imp']:.2f} DS={ds.size}")
    print(f"\n  Average: {avg_rate:.3f} (best: {best_avg:.3f})")

    # Save checkpoint: model weights + per-scenario datasets
    ckpt = {
        'model_state_dict': model.state_dict(),
        'pilot_enc_state_dict': pilot_enc.state_dict(),
        'cfg_dict': {
            'K': cfg.K, 'N': cfg.N, 'L_p': cfg.L_p, 'P_max': cfg.P_max,
            'SNR_dB': cfg.SNR_dB, 'D_tok': cfg.token_dim,
        },
        'datasets': {sid: {
            'H': mds.get(sid).H, 'p': mds.get(sid).p, 'lam': mds.get(sid).lam,
            'rates': mds.get(sid).rates, 'mmse_rates': mds.get(sid).mmse_rates,
            'is_sup': mds.get(sid).is_sup,
        } for sid in cfg.scenario_ids if mds.get(sid).size > 0},
    }
    torch.save(ckpt, cfg.ckpt_path)
    print(f"\n  Checkpoint saved -> {cfg.ckpt_path}")

    plt.figure(figsize=(9, 6))
    for sc in cfg.scenarios:
        sid = sc.sid
        if len(curve_hist[sid]['epochs']) == 0:
            continue
        plt.plot(curve_hist[sid]['epochs'], curve_hist[sid]['train_rate'], marker='o', label=f"S{sid} {sc.name}")
    plt.xlabel('Epoch')
    plt.ylabel('Train rate (scenario-wise epoch average)')
    plt.title('Per-Scenario Training Curves')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig(cfg.train_curve_plot_path, dpi=180)
    plt.close()
    print(f"  Training curve figure saved -> {cfg.train_curve_plot_path}")

    return model, mds, bl, Phi


###############################################################################
# 11. CHECKPOINT LOADING
###############################################################################
def load_from_checkpoint(cfg):
    """
    Load a trained model + per-scenario datasets from a saved checkpoint.
    Rebuilds the PilotEncoder (frozen) and PilotICLModel_PLambda, then loads
    saved state dicts. Also restores per-scenario DynDataset contents.
    Returns (model, mds, Phi).
    """
    print(f"\n[Load] Loading checkpoint from {cfg.ckpt_path}...")
    ckpt = torch.load(cfg.ckpt_path, map_location=device)

    Phi = generate_pilot_dft(cfg.K, cfg.L_p)

    # Rebuild PilotEncoder and load weights
    pilot_enc = PilotEncoder(cfg.N, cfg.L_p, cfg.K, cfg.encoder_hidden).to(device)
    pilot_enc.load_state_dict(ckpt['pilot_enc_state_dict'])
    for p in pilot_enc.parameters(): p.requires_grad_(False)

    # Rebuild full model and load weights
    model = PilotICLModel_PLambda(pilot_enc, cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Restore per-scenario datasets
    mds = MultiScenarioDataset(cfg.scenarios, max_per_scenario=cfg.max_ds_per_scenario)
    ds_data = ckpt.get('datasets', {})
    for sid, data in ds_data.items():
        sid = int(sid)
        ds = mds.get(sid)
        ds.H = data['H']; ds.p = data['p']; ds.lam = data['lam']
        ds.rates = data['rates']; ds.mmse_rates = data['mmse_rates']
        ds.is_sup = data['is_sup']
        ds._n = ds.H.size(0)
        ds.n_sup = int(ds.is_sup.sum().item())
        ds.n_unsup = int((~ds.is_sup).sum().item())
    print(f"  Datasets loaded:\n{mds.summary()}")

    return model, mds, Phi


@torch.no_grad()
def evaluate_context_query_rate_matrix(model, mds, Phi, cfg):
    """
    Build a 4x4 table:
      - X axis: context scenario used to build in-context demos
      - Y axis: query scenario used to generate query token/pilot
      - Cell(y, x): average test sum-rate of produced BF solution
    """
    model.eval()
    scenario_list = list(cfg.scenarios)
    S = len(scenario_list)
    if S != 4:
        raise RuntimeError(f"Expected exactly 4 scenarios for 4x4 table, got {S}.")
    n_test = cfg.n_test_context_query_matrix
    rate_matrix = np.full((S, S), np.nan, dtype=np.float64)

    print("\n" + "=" * 75)
    print("  CONTEXT-QUERY 4x4 RATE TABLE")
    print("=" * 75)
    print("  X-axis: context scenario, Y-axis: query scenario")
    print(f"  n_test per cell: {n_test}")

    # Pre-generate query test sets for each query scenario.
    q_data = {}
    for sc in scenario_list:
        H_q = generate_channel_scenario(n_test, cfg.K, cfg.N, sc)
        Y_q = pilot_observe(H_q, Phi, cfg.sigma2)
        q_data[sc.sid] = {
            'H': H_q,
            'pil': pilot_to_real(Y_q),
            'H_hat': mmse_channel_est(Y_q, Phi, cfg.sigma2),
        }

    for x, sc_ctx in enumerate(scenario_list):
        ds_ctx = mds.get(sc_ctx.sid)
        if ds_ctx.size < cfg.n_demos:
            print(f"  WARNING: context S{sc_ctx.sid} has insufficient demos ({ds_ctx.size}), skipping column")
            continue

        for y, sc_q in enumerate(scenario_list):
            td = q_data[sc_q.sid]
            H_q = td['H']; pil_q = td['pil']; H_hat_q = td['H_hat']
            bs = min(cfg.batch_size, n_test)
            batch_rates = []

            for s in range(0, n_test, bs):
                e = min(s + bs, n_test)
                b = e - s

                # Context demos from context scenario x
                d_idx = torch.randint(0, ds_ctx.size, (b, cfg.n_demos), device=device)
                d_H = ds_ctx.H[d_idx].reshape(b * cfg.n_demos, cfg.K, cfg.N)
                d_Y = pilot_observe(d_H, Phi, cfg.sigma2)
                d_pil = pilot_to_real(d_Y).reshape(b, cfg.n_demos, -1)
                d_p = ds_ctx.p[d_idx]
                d_lam = ds_ctx.lam[d_idx]

                # Query token from scenario y
                p_pred, lam_pred = model(d_pil, d_p, d_lam, pil_q[s:e])
                W_hat = reconstruct_precoder(H_hat_q[s:e], p_pred, lam_pred, cfg.sigma2)
                batch_rates.append(compute_sum_rate(H_q[s:e], W_hat, cfg.sigma2))

            rate_matrix[y, x] = torch.cat(batch_rates).mean().item()

    print("\n  4x4 Average Rate Matrix (rows=query, cols=context)")
    hdr = "  " + " " * 12
    for sc in scenario_list:
        hdr += f" {('S'+str(sc.sid)):>8}"
    print(hdr)
    for y, sc_q in enumerate(scenario_list):
        row = f"  {('S'+str(sc_q.sid)+' '+sc_q.name[:8]):>12}"
        for x in range(S):
            row += f" {rate_matrix[y, x]:8.2f}"
        print(row)

    # Plot matrix-like table with values.
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(rate_matrix, cmap='YlGnBu', aspect='auto')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Average Test Rate')

    ax.set_xticks(np.arange(S))
    ax.set_yticks(np.arange(S))
    ax.set_xticklabels([f"S{sc.sid}\n{sc.name}" for sc in scenario_list], rotation=0)
    ax.set_yticklabels([f"S{sc.sid} {sc.name}" for sc in scenario_list])
    ax.set_xlabel('Context Scenario (context pairs source)')
    ax.set_ylabel('Query Scenario (query token source)')
    ax.set_title('4x4 Context-Query Average Test Rate Table')

    # Annotate each cell with numeric value.
    for y in range(S):
        for x in range(S):
            ax.text(x, y, f"{rate_matrix[y, x]:.2f}", ha='center', va='center', color='black', fontsize=9)

    fig.tight_layout()
    fig.savefig(cfg.ctx_query_matrix_plot_path, dpi=180)
    plt.close(fig)
    print(f"\n  Saved 4x4 table figure -> {cfg.ctx_query_matrix_plot_path}")

    return {
        'scenario_list': scenario_list,
        'rate_matrix': rate_matrix,
    }


def train_or_load(cfg):
    """Utility wrapper for a consistent train/load flow."""
    if cfg.load_pretrained:
        return load_from_checkpoint(cfg)
    model, mds, _, Phi = train(cfg)
    return model, mds, Phi


def build_opt_context_dataset_for_scenario(sc, n_samples, Phi, cfg):
    """Build context dataset for a scenario via robust Opt(p,lambda) on H_hat."""
    ds = DynDataset(max_sz=n_samples)
    bs = min(cfg.batch_size, n_samples)
    for s in range(0, n_samples, bs):
        e = min(s + bs, n_samples)
        b = e - s
        H = generate_channel_scenario(b, cfg.K, cfg.N, sc)
        Y = pilot_observe(H, Phi, cfg.sigma2)
        H_hat = mmse_channel_est(Y, Phi, cfg.sigma2)
        # This path solves p/lambda with gradient-based optimization, so grad must be enabled.
        with torch.enable_grad():
            p_opt, lam_opt, _ = generate_optimal_params(
                H_hat, cfg.P_max, cfg.sigma2, cfg.opt_iters, cfg.opt_lr, cfg.opt_restarts
            )
        W_opt = reconstruct_precoder(H_hat, p_opt, lam_opt, cfg.sigma2)
        rates = compute_sum_rate(H, W_opt, cfg.sigma2)
        mmse_rates = compute_sum_rate(H, mmse_beamformer(H, cfg.P_max, cfg.sigma2), cfg.sigma2)
        ds.add(H, p_opt, lam_opt, rates, mmse_rates=mmse_rates, supervised=True)
    return ds


@torch.no_grad()
def evaluate_query_with_context_dataset(model, ds_ctx, query_sc, Phi, cfg, n_test):
    """Evaluate ICL rate for fixed context dataset and fixed query scenario."""
    if ds_ctx.size < cfg.n_demos:
        raise RuntimeError(f"Context dataset too small ({ds_ctx.size}) for n_demos={cfg.n_demos}.")

    H_q = generate_channel_scenario(n_test, cfg.K, cfg.N, query_sc)
    Y_q = pilot_observe(H_q, Phi, cfg.sigma2)
    pil_q = pilot_to_real(Y_q)
    H_hat_q = mmse_channel_est(Y_q, Phi, cfg.sigma2)

    mmse_imp = compute_sum_rate(H_q, mmse_beamformer(H_hat_q, cfg.P_max, cfg.sigma2), cfg.sigma2).mean().item()

    bs = min(cfg.batch_size, n_test)
    rates = []
    for s in range(0, n_test, bs):
        e = min(s + bs, n_test)
        b = e - s
        d_idx = torch.randint(0, ds_ctx.size, (b, cfg.n_demos), device=device)
        d_H = ds_ctx.H[d_idx].reshape(b * cfg.n_demos, cfg.K, cfg.N)
        d_Y = pilot_observe(d_H, Phi, cfg.sigma2)
        d_pil = pilot_to_real(d_Y).reshape(b, cfg.n_demos, -1)
        d_p = ds_ctx.p[d_idx]
        d_lam = ds_ctx.lam[d_idx]

        p_pred, lam_pred = model(d_pil, d_p, d_lam, pil_q[s:e])
        W_hat = reconstruct_precoder(H_hat_q[s:e], p_pred, lam_pred, cfg.sigma2)
        rates.append(compute_sum_rate(H_q[s:e], W_hat, cfg.sigma2))

    icl_rate = torch.cat(rates).mean().item()
    return {'icl_rate': icl_rate, 'mmse_imp': mmse_imp}


def run_unseen_task_adaptation_study(base_cfg: Config):
    """
    Four-case task-adaptation study:
      1) train on 4 seen, test unseen aligned (ctx=unseen, query=unseen)
      2) train on 4 seen, test unseen misaligned (ctx=seen_i, query=unseen)
      3) train on 4+unseen, test new seen aligned (ctx=unseen, query=unseen)
      4) train on 4+unseen, test new seen misaligned (ctx=seen_i, query=unseen)
    """
    print("\n" + "=" * 85)
    print("  FOUR-CASE UNSEEN TASK ADAPTATION STUDY")
    print("=" * 85)

    seen4 = list(base_cfg.scenarios)
    unseen_sc = base_cfg.unseen_scenario
    n_test = base_cfg.unseen_n_test

    # A) Model trained on original 4 seen tasks
    cfg_seen4 = Config(**{
        **base_cfg.__dict__,
        'scenarios': seen4,
        'ckpt_path': 'plambda_seen4.pt',
        'load_pretrained': base_cfg.load_pretrained,
    })
    model4, mds4, Phi4 = train_or_load(cfg_seen4)

    # Build unseen context (not in training set) for case 1
    ds_unseen_ctx_for_4 = build_opt_context_dataset_for_scenario(
        unseen_sc, cfg_seen4.unseen_context_size, Phi4, cfg_seen4
    )

    case1 = evaluate_query_with_context_dataset(
        model4, ds_unseen_ctx_for_4, unseen_sc, Phi4, cfg_seen4, n_test
    )

    case2_each = []
    for sc in seen4:
        res = evaluate_query_with_context_dataset(
            model4, mds4.get(sc.sid), unseen_sc, Phi4, cfg_seen4, n_test
        )
        case2_each.append((sc, res))

    # B) Model trained on original 4 + the new unseen task (now seen)
    seen5 = seen4 + [unseen_sc]
    cfg_seen5 = Config(**{
        **base_cfg.__dict__,
        'scenarios': seen5,
        'ckpt_path': 'plambda_seen4_plus_unseen.pt',
        'load_pretrained': base_cfg.load_pretrained,
    })
    model5, mds5, Phi5 = train_or_load(cfg_seen5)

    case3 = evaluate_query_with_context_dataset(
        model5, mds5.get(unseen_sc.sid), unseen_sc, Phi5, cfg_seen5, n_test
    )

    case4_each = []
    for sc in seen4:
        res = evaluate_query_with_context_dataset(
            model5, mds5.get(sc.sid), unseen_sc, Phi5, cfg_seen5, n_test
        )
        case4_each.append((sc, res))

    # Print summary
    print("\n" + "-" * 85)
    print(f"Unseen scenario: S{unseen_sc.sid} {unseen_sc.name}"
          f" ({unseen_sc.ch_type}, cl={unseen_sc.n_clusters}, rays={unseen_sc.n_rays}, spread={unseen_sc.spread_deg})")
    print("-" * 85)

    print("\n[Case 1] Train on seen4; Test aligned on unseen (ctx=unseen, query=unseen)")
    print(f"  ICL={case1['icl_rate']:.3f}  MMSE-E={case1['mmse_imp']:.3f}"
          f"  Ratio={case1['icl_rate']/max(case1['mmse_imp'],1e-8):.3f}")

    print("\n[Case 2] Train on seen4; Test misaligned (ctx=seen_i, query=unseen)")
    vals2 = []
    for sc, res in case2_each:
        vals2.append(res['icl_rate'])
        ratio = res['icl_rate'] / max(res['mmse_imp'], 1e-8)
        print(f"  ctx=S{sc.sid} {sc.name:<20} ICL={res['icl_rate']:.3f} MMSE-E={res['mmse_imp']:.3f} Ratio={ratio:.3f}")
    print(f"  Avg over seen contexts: {np.mean(vals2):.3f}")

    print("\n[Case 3] Train on seen4+unseen; Test aligned on new seen task (ctx=unseen, query=unseen)")
    print(f"  ICL={case3['icl_rate']:.3f}  MMSE-E={case3['mmse_imp']:.3f}"
          f"  Ratio={case3['icl_rate']/max(case3['mmse_imp'],1e-8):.3f}")

    print("\n[Case 4] Train on seen4+unseen; Test misaligned (ctx=old_seen_i, query=new_seen_unseen)")
    vals4 = []
    for sc, res in case4_each:
        vals4.append(res['icl_rate'])
        ratio = res['icl_rate'] / max(res['mmse_imp'], 1e-8)
        print(f"  ctx=S{sc.sid} {sc.name:<20} ICL={res['icl_rate']:.3f} MMSE-E={res['mmse_imp']:.3f} Ratio={ratio:.3f}")
    print(f"  Avg over old-seen contexts: {np.mean(vals4):.3f}")

    return {
        'case1': case1,
        'case2_each': case2_each,
        'case3': case3,
        'case4_each': case4_each,
    }


###############################################################################
# 13. ENTRY POINT
###############################################################################
if __name__ == "__main__":
    cfg = Config(
        K=32, N=32, L_p=20, P_max=1.0, SNR_dB=20,
        encoder_hidden=128,
        edn_epochs=500, edn_lr=1e-3, edn_batch=128, edn_n_samples=8000,
        n_demos=5, d_model=512, n_heads=8, n_layers=6, d_ff=1024,
        batch_size=64, lr=1e-4, lr_min=5e-5, weight_decay=1e-4,
        init_ds_per_scenario=256,
        opt_iters=300, opt_lr=0.03, opt_restarts=2,
        use_robust_labels=True,
        unsup_scale=0.005,
        phase1_epochs=100, phase2_epochs=500, steps_per_epoch=80,
        boot_alpha_start=0.60, boot_alpha_end=0.90,
        boot_beta_start=0.60, boot_beta_end=0.90,
        max_ds_per_scenario=50000,
        prune_enable=True,
        prune_every=10, prune_drop_start=0.0, prune_drop_end=0.02, prune_min_unsup=1024,
        prune_warmup_phase2_epochs=200,
        prune_drop_ratio_cap=0.01,
        prune_drop_vs_add_ratio=0.10,
        prune_max_drop_per_scenario=32,
        n_test_per_scenario=100,
        # Checkpoint control
        load_pretrained=False,       # Set True to skip training and load checkpoint
        ckpt_path='plambda_multi_scenario.pt',
        # Inference context-query matrix
        n_test_context_query_matrix=200,
        ctx_query_matrix_plot_path='context_query_rate_matrix_4x4.png',
        train_curve_save_every=10,
        train_curve_dir='training_curves_4x4',
        train_curve_plot_path='training_curves_4x4.png',
        unseen_scenario=UNSEEN_SCENARIO,
        unseen_context_size=1024,
        unseen_n_test=200,
    )

    # Modes:
    #   - train_seen5_only: train 4 seen + unseen and save plambda_seen4_plus_unseen.pt
    #   - study: run four-case study (train or load depending on cfg.load_pretrained)
    #   - study_load_only: force loading two checkpoints for inference-only study
    run_mode = 'study_load_only'

    if run_mode == 'train_seen5_only':
        cfg_seen5 = Config(**{
            **cfg.__dict__,
            'scenarios': list(SCENARIOS) + [UNSEEN_SCENARIO],
            'load_pretrained': False,
            'ckpt_path': 'plambda_seen4_plus_unseen.pt',
        })
        print("\n[Run Mode] train_seen5_only -> training and saving plambda_seen4_plus_unseen.pt")
        model5, mds5, bl5, Phi5 = train(cfg_seen5)
    elif run_mode == 'study':
        # Requested four-case unseen adaptation study (results are printed).
        study_out = run_unseen_task_adaptation_study(cfg)
    elif run_mode == 'study_load_only':
        # Force loading both models from checkpoint and run only inference study.
        ckpt_seen4 = 'plambda_seen4.pt'
        ckpt_seen5 = 'plambda_seen4_plus_unseen.pt'
        missing = [p for p in [ckpt_seen4, ckpt_seen5] if not osp.exists(p)]
        if missing:
            raise FileNotFoundError(
                "Missing checkpoints for inference-only study: " + ", ".join(missing)
            )
        cfg.load_pretrained = True
        print("\n[Run Mode] study_load_only -> loading plambda_seen4.pt + plambda_seen4_plus_unseen.pt")
        study_out = run_unseen_task_adaptation_study(cfg)
    else:
        raise ValueError(f"Unknown run_mode: {run_mode}")
