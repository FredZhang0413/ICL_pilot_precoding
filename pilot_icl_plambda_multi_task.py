"""
Multi-Scenario Pilot-Based ICL Precoding with (p, lambda) Parameterization

Merges two approaches:
  1. Multi-scenario batch-level round-robin training (from pilot_icl_multi_task_4_5.py)
  2. Optimal (p,lambda) beamformer structure (from pilot_icl_plambda_4_11.py)

KEY DIFFERENCES FROM THE COMPRESSED-BF VERSION:
  - NO BFEncoder or BFDecoder. The BF solution is represented as (p, lambda) ∈ R^{2K},
    which is zero-padded to match the 4K-dimensional pilot state token.
  - The ICL label token is [p, lambda, 0, 0] ∈ R^{4K} (no learned compression).
  - W is reconstructed via the optimal structure: w_k = sqrt(p_k) * A^{-1} h_k / ||A^{-1} h_k||,
    using MMSE channel estimate H_hat (since true H is unavailable at inference).
  - Phase 0 only pretrains the PilotEncoder (no BF EDN needed).
  - Supervised labels are (p*, lambda*) obtained by Adam optimization on H_hat (robust mode).

ARCHITECTURE:
  PilotEncoder: Y_real -> z ∈ R^{4K}  (CNN + attention pooling)
  ICL Transformer: {z_1, y_1, ..., z_l, y_l, z_q} -> y_hat ∈ R^{4K}
    where y_i = [p_i, lambda_i, 0, 0] is the zero-padded label token
  Output head: y_hat[:K] -> p (sigmoid + L1-normalize), y_hat[K:2K] -> lambda (softplus)
  Precoder: W = reconstruct_precoder(H_hat, p, lambda, sigma2)

MULTI-SCENARIO DESIGN:
  - 8 channel scenarios (S0-S7), shared ICL Transformer, per-scenario datasets
  - NO scenario tag in model input — ICL identifies scenario from context
  - Batch-level round-robin: each mini-batch uses a single scenario, cycling through all
  - Per-scenario self-bootstrapping and bottom-pruning
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
    ScenarioDef("Dense Urban",    0, 'sparse', 3, 5, 10.0),
    ScenarioDef("LoS-Dominant",   1, 'sparse', 1, 10, 3.0),
    ScenarioDef("Rich Scatter",   2, 'sparse', 6, 3, 5.0),
    ScenarioDef("Suburban",       3, 'sparse', 2, 8, 15.0),
    ScenarioDef("Indoor Office",  4, 'sparse', 5, 2, 20.0),
    ScenarioDef("Near-LoS",       5, 'sparse', 1, 15, 2.0),
    ScenarioDef("Moderate Urban", 6, 'sparse', 4, 4, 8.0),
    ScenarioDef("Rayleigh iid",   7, 'rayleigh'),
]


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

        # Evaluation
        self.n_test_per_scenario = kw.get('n_test_per_scenario', 100)


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
        self.H = self.p = self.lam = self.rates = self.mmse_rates = self.is_sup = None
        self._n = 0; self.n_sup = 0; self.n_unsup = 0

    @property
    def size(self): return self._n

    def add(self, H, p, lam, rates, mmse_rates=None, supervised=True):
        H,p,lam,rates = [x.detach() for x in [H,p,lam,rates]]
        flag = torch.full((H.size(0),),bool(supervised),device=device,dtype=torch.bool)
        mr = mmse_rates.detach() if mmse_rates is not None else torch.zeros_like(rates)
        if self.H is None:
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

    def prune_unsup_bottom(self, drop_ratio, min_keep=0):
        if self._n==0 or drop_ratio<=0: return 0
        ui=torch.where(~self.is_sup)[0]; nu=ui.numel()
        if nu==0: return 0
        nd=min(int(nu*drop_ratio),max(0,nu-min_keep))
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
        # Opt (p,lam) on true H (perfect) and H_hat (imperfect)
        p_opt_p,lam_opt_p,_ = generate_optimal_params(H,Pm,s2,cfg.opt_iters,cfg.opt_lr,cfg.opt_restarts)
        with torch.no_grad():
            W_opt_p = reconstruct_precoder(H, p_opt_p, lam_opt_p, s2)
            b_opt_p = compute_sum_rate(H, W_opt_p, s2).mean().item()
        # Opt (p,lam) on H_hat
        p_opt,lam_opt,_ = generate_optimal_params(Hh,Pm,s2,cfg.opt_iters,cfg.opt_lr,cfg.opt_restarts)
        with torch.no_grad():
            W_opt = reconstruct_precoder(Hh, p_opt, lam_opt, s2)
            b_opt_e = compute_sum_rate(H, W_opt, s2).mean().item()
        all_bl[sc.sid] = {
            'mmse_perf': b_mmse_p,
            'mmse_imp': b_mmse_e,
            'opt_perf': b_opt_p,
            'opt_imp': b_opt_e,
            'name': sc.name,
        }
        print(
            f"  S{sc.sid} {sc.name:<18} MMSE-P={b_mmse_p:.2f} MMSE-E={b_mmse_e:.2f} "
            f"Opt-P={b_opt_p:.2f} Opt-E={b_opt_e:.2f}"
        )
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
    # For stable display: if one scenario has no batch this epoch, reuse its last train-rate.
    last_sc_train_rate = {sc.sid: bl[sc.sid]['mmse_imp'] for sc in cfg.scenarios}

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
        ep_loss, ep_add, ep_n = 0., 0, 0
        ep_rate_sum, ep_rate_cnt = 0.0, 0
        sc_rate_sum = {sid: 0.0 for sid in cfg.scenario_ids}
        sc_rate_cnt = {sid: 0 for sid in cfg.scenario_ids}

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
            ep_loss += loss.item(); ep_n += 1

            # Train-rate logging policy (never show zero):
            #   Phase 1 -> supervised sample rate.
            #   Phase 2 -> switch to unsup sample rate only when this scenario has
            #              enough unsup samples (> one mini-batch), else keep sup rate.
            n_sup = int((~is_unsup).sum().item())
            n_unsup = int(is_unsup.sum().item())
            use_unsup_for_log = (phase == 2 and ds.n_unsup > cfg.batch_size)
            if use_unsup_for_log and n_unsup > 0:
                rs = rate_pred[is_unsup].sum().item(); rc = n_unsup
            elif n_sup > 0:
                rs = rate_pred[~is_unsup].sum().item(); rc = n_sup
            elif n_unsup > 0:
                # Rare fallback when a batch is fully unsup before switch condition.
                rs = rate_pred[is_unsup].sum().item(); rc = n_unsup
            else:
                rs = 0.0; rc = 0

            if rc > 0:
                ep_rate_sum += rs; ep_rate_cnt += rc
                sc_rate_sum[batch_sid] += rs; sc_rate_cnt[batch_sid] += rc

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

        # Bottom pruning
        nd = 0
        if phase==2 and (epoch+1)%cfg.prune_every==0:
            fp=(epoch-cfg.phase1_epochs)/max(1,cfg.phase2_epochs-1)
            dr=cfg.prune_drop_start+(cfg.prune_drop_end-cfg.prune_drop_start)*fp
            for sc in cfg.scenarios: nd+=mds.get(sc.sid).prune_unsup_bottom(dr,cfg.prune_min_unsup)

        # Evaluation
        sc_rates = evaluate_per_scenario(model, mds, H_tests, Phi, cfg)
        avg_rate = np.mean([v for v in sc_rates.values()]) if sc_rates else 0.0
        best_avg = max(best_avg, avg_rate)

        # Per-scenario train-rate for display; fallback to last value if no samples this epoch.
        sc_train_rates = {}
        for sid in cfg.scenario_ids:
            if sc_rate_cnt[sid] > 0:
                sc_train_rates[sid] = sc_rate_sum[sid] / max(1, sc_rate_cnt[sid])
            else:
                sc_train_rates[sid] = last_sc_train_rate[sid]
            last_sc_train_rate[sid] = sc_train_rates[sid]

        epoch_train_rate = ep_rate_sum / max(1, ep_rate_cnt)

        dt = time.time()-t0
        if (epoch+1)%10==0 or epoch==0:
            tr_str = " ".join([f"S{sc.sid}:{sc_train_rates.get(sc.sid,0.0):.2f}" for sc in cfg.scenarios])
            te_str = " ".join([f"S{sc.sid}:{sc_rates.get(sc.sid,0.0):.2f}" for sc in cfg.scenarios])
            print(
                f"{epoch+1:3d} ph{phase} r={r:.2f} | loss={ep_loss/max(1,ep_n):.4f} train={epoch_train_rate:.2f} "
                f"| avg_test={avg_rate:.2f} best={best_avg:.2f} | DS={mds.total_size} +{ep_add} -{nd} ({dt:.1f}s)\n"
                f"      Train[S0..]: {tr_str}\n"
                f"      Test [S0..]: {te_str}",
                flush=True,
            )

    # Final summary
    print("\n"+"="*75)
    print("  MULTI-SCENARIO (p,lambda) TRAINING COMPLETE")
    print("="*75)
    for sc in cfg.scenarios:
        ds=mds.get(sc.sid); tr=sc_rates.get(sc.sid,0.0)
        print(f"  S{sc.sid} {sc.name:<18} test={tr:.2f} "
              f"MMSE-P={bl[sc.sid]['mmse_perf']:.2f} MMSE-E={bl[sc.sid]['mmse_imp']:.2f} "
              f"Opt-P={bl[sc.sid]['opt_perf']:.2f} Opt-E={bl[sc.sid]['opt_imp']:.2f} DS={ds.size}")
    print(f"\n  Average: {avg_rate:.3f} (best: {best_avg:.3f})")
    return model, mds, bl


###############################################################################
# 11. ENTRY POINT
###############################################################################
if __name__ == "__main__":
    cfg = Config(
        K=32, N=32, L_p=20, P_max=1.0, SNR_dB=20,
        encoder_hidden=256,
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
        max_ds_per_scenario=20000,
        prune_every=10, prune_drop_start=0.0, prune_drop_end=0.10, prune_min_unsup=1024,
        n_test_per_scenario=100,
    )
    train(cfg)
