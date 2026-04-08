"""
Pilot-Based ICL Precoding v6 — Multi-Scenario Extension
8 channel scenarios, ONE shared ICL Transformer, per-scenario datasets.

DESIGN PHILOSOPHY:
  - NO scenario tag injected into the model. The ICL Transformer identifies
    the scenario PURELY from the demonstration context (pilot-BF pairs).
    This is the core ICL capability: adaptation from context, not labels.
  - Scenario tags are used ONLY for dataset routing (context selection,
    bootstrapping admission, pruning) and per-scenario evaluation.

8 SCENARIOS:
  S0: Dense Urban     (3 cl, 5 rays, 10° spread)
  S1: LoS-Dominant    (1 cl, 10 rays, 3° spread)
  S2: Rich Scatter    (6 cl, 3 rays, 5° spread)
  S3: Suburban         (2 cl, 8 rays, 15° spread)
  S4: Indoor Office   (5 cl, 2 rays, 20° spread)
  S5: Near-LoS        (1 cl, 15 rays, 2° spread)
  S6: Moderate Urban  (4 cl, 4 rays, 8° spread)
  S7: Rayleigh iid    (no sparsity — structurally different)

KEY CHANGES FROM SINGLE-SCENARIO v5:
  1. ScenarioDef dataclass defines each scenario's channel parameters
  2. generate_channel_scenario() dispatches to sparse or Rayleigh
  3. MultiScenarioDataset wraps N per-scenario DynDataset instances
  4. Phase 0 encoders trained on MIXED data from ALL scenarios
  5. Training loop samples queries uniformly across scenarios
  6. Context selection restricted to query's scenario
  7. Self-bootstrapping routes to correct scenario dataset
  8. Per-scenario test sets, baselines, and evaluation
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
    """Defines a single channel scenario."""
    name: str
    sid: int            # scenario integer ID (0..N-1)
    ch_type: str        # 'sparse' or 'rayleigh'
    n_clusters: int = 3
    n_rays: int = 5
    spread_deg: float = 10.0

# The 8 preset scenarios
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
        if len(set(self.scenario_ids)) != self.n_scenarios:
            raise ValueError("Scenario IDs must be unique.")
        self.sid_to_scenario = {sc.sid: sc for sc in self.scenarios}

        self.D_tok = kw.get('D_tok', 256)
        self.edn_hidden = kw.get('edn_hidden', 256)
        self.edn_epochs = kw.get('edn_epochs', 500)
        self.edn_lr = kw.get('edn_lr', 1e-3)
        self.edn_batch = kw.get('edn_batch', 128)
        # Total samples for Phase 0 pretraining (split equally across scenarios)
        self.edn_n_samples = kw.get('edn_n_samples', 8000)

        self.n_demos = kw.get('n_demos', 5)
        self.d_model = kw.get('d_model', 512); self.n_heads = kw.get('n_heads', 8)
        self.n_layers = kw.get('n_layers', 6); self.d_ff = kw.get('d_ff', 1024)
        self.dropout = kw.get('dropout', 0.0)

        self.batch_size = kw.get('batch_size', 64)
        self.lr = kw.get('lr', 1e-4); self.lr_min = kw.get('lr_min', 5e-5)
        self.weight_decay = kw.get('weight_decay', 1e-4)
        # Per-scenario initial labeled dataset size
        self.init_ds_per_scenario = kw.get('init_ds_per_scenario', 256)
        self.wmmse_iters = kw.get('wmmse_iters', 100)
        self.wmmse_lr = kw.get('wmmse_lr', 0.03)
        self.unsup_scale = kw.get('unsup_scale', 0.01)

        # Hybrid loss
        self.hybrid_calib_steps = kw.get('hybrid_calib_steps', 4)
        self.hybrid_calib_batch = kw.get('hybrid_calib_batch', 16)
        self.hybrid_rate_gain = kw.get('hybrid_rate_gain', 1.0)
        self.hybrid_rate_scale_min = kw.get('hybrid_rate_scale_min', 0.01)
        self.hybrid_rate_scale_max = kw.get('hybrid_rate_scale_max', 1000.0)
        self.hybrid_switch_power = kw.get('hybrid_switch_power', 1.0)
        self.new_unsup_rate_ratio = kw.get('new_unsup_rate_ratio', 0.5)
        self.loss_mse_weight = kw.get('loss_mse_weight', 1.0)
        self.loss_rate_ds_weight = kw.get('loss_rate_ds_weight', 1.0)
        self.loss_rate_new_weight = kw.get('loss_rate_new_weight', 1.0)
        self.lazy_residual_min_ratio = kw.get('lazy_residual_min_ratio', 0.10)
        self.lazy_residual_weight = kw.get('lazy_residual_weight', 0.05)
        self.sup_residual_target_weight = kw.get('sup_residual_target_weight', 0.25)

        self.phase1_epochs = kw.get('phase1_epochs', 150)
        self.phase2_epochs = kw.get('phase2_epochs', 500)
        self.total_epochs = self.phase1_epochs + self.phase2_epochs
        self.steps_per_epoch = kw.get('steps_per_epoch', 80)

        self.boot_alpha_start = kw.get('boot_alpha_start', 0.60)
        self.boot_alpha_end = kw.get('boot_alpha_end', 0.90)
        self.boot_beta_start = kw.get('boot_beta_start', 0.60)
        self.boot_beta_end = kw.get('boot_beta_end', 0.90)
        self.max_ds_per_scenario = kw.get('max_ds_per_scenario', 20000)

        self.prune_every = kw.get('prune_every', 10)
        self.prune_drop_start = kw.get('prune_drop_start', 0.0)
        self.prune_drop_end = kw.get('prune_drop_end', 0.10)
        self.prune_min_unsup = kw.get('prune_min_unsup', 1024)

        self.n_test_per_scenario = kw.get('n_test_per_scenario', 100)
        self.rate_save_every = kw.get('rate_save_every', 10)


###############################################################################
# 3. CHANNEL GENERATION (dispatches by scenario type)
###############################################################################
def generate_sparse_channel(B, K, N, n_cl, n_ray, spread_deg):
    """Cluster-based sparse mmWave channel."""
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
    """iid Rayleigh fading — no angular structure."""
    return (torch.randn(B,K,N,device=device) + 1j*torch.randn(B,K,N,device=device)) / math.sqrt(2)


def generate_channel_scenario(B, K, N, sc: ScenarioDef):
    """Generate channel samples for a given scenario definition."""
    if sc.ch_type == 'rayleigh':
        return generate_rayleigh_channel(B, K, N)
    else:
        return generate_sparse_channel(B, K, N, sc.n_clusters, sc.n_rays, sc.spread_deg)


###############################################################################
# 4. SIGNAL PROCESSING (unchanged)
###############################################################################
def generate_pilot_dft(K, L_p):
    return (torch.fft.fft(torch.eye(K,device=device))/math.sqrt(K))[:,:L_p].contiguous()

def pilot_observe(H, Phi, sigma2):
    B,K,N = H.shape; Lp = Phi.size(1)
    Y = H.transpose(-1,-2) @ Phi.unsqueeze(0).expand(B,-1,-1)
    nr = torch.randn(B,N,Lp,device=device)*math.sqrt(sigma2/2)
    ni = torch.randn(B,N,Lp,device=device)*math.sqrt(sigma2/2)
    return Y + torch.complex(nr, ni)

def pilot_to_real(Y): return torch.cat([Y.real,Y.imag],dim=1).reshape(Y.size(0),-1)
def bf_to_real(W): return torch.cat([W.real,W.imag],dim=1).reshape(W.size(0),-1)
def real_to_bf(x,N,K): B=x.size(0);x=x.view(B,2*N,K);return torch.complex(x[:,:N,:],x[:,N:,:])
def channel_to_real(H): return torch.cat([H.real,H.imag],dim=-1).reshape(H.size(0),-1)

def compute_sum_rate(H,W,sigma2):
    HW=H@W; sig=torch.abs(torch.diagonal(HW,dim1=-2,dim2=-1))**2
    tot=torch.sum(torch.abs(HW)**2,dim=-1); SINR=sig/(tot-sig+sigma2)
    return torch.log2(1+SINR).sum(-1)

def mmse_beamformer(H,P_max,sigma2):
    B,K,N=H.shape; HH=H.conj().transpose(-1,-2)
    A=HH@H+sigma2*torch.eye(N,device=device,dtype=H.dtype).unsqueeze(0)
    W=torch.linalg.solve(A,HH); pw=torch.sum(torch.abs(W)**2,dim=(1,2)).real
    return W*torch.sqrt(P_max/(pw+1e-8)).view(B,1,1)

def power_normalize(W,P_max):
    pw=torch.sum(torch.abs(W)**2,dim=(1,2),keepdim=True).real
    return W*torch.sqrt(P_max/(pw+1e-8))

def ls_channel_est(Y,Phi):
    return (Y@torch.linalg.pinv(Phi).unsqueeze(0)).transpose(-1,-2).contiguous()

def mmse_channel_est(Y,Phi,sigma2):
    K,Lp=Phi.shape; A=Phi.T@Phi.conj()+sigma2*torch.eye(Lp,device=device,dtype=Phi.dtype)
    return torch.matmul((Phi.conj()@torch.linalg.inv(A)).unsqueeze(0),Y.transpose(-1,-2))


###############################################################################
# 5. NETWORK MODULES (unchanged from single-scenario)
###############################################################################
class FiLMLayer(nn.Module):
    def __init__(self, n_ch):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1,64),nn.GELU(),nn.Linear(64,2*n_ch))
    def forward(self,x,sigma2):
        B=x.size(0);C=x.size(1)
        p=self.net(torch.full((B,1),math.log(sigma2+1e-10),device=x.device))
        g=p[:,:C];b=p[:,C:]
        if x.dim()==3: g=g.unsqueeze(-1);b=b.unsqueeze(-1)
        return g*x+b

class PilotEncoder(nn.Module):
    def __init__(self,N,L_p,D_tok,hidden=256):
        super().__init__()
        self.N,self.L_p=N,L_p
        self.conv1=nn.Conv1d(2*N,hidden,3,padding=1)
        self.conv2=nn.Conv1d(hidden,hidden,3,padding=1)
        self.ln=nn.LayerNorm(hidden)
        self.attn_q=nn.Parameter(torch.randn(1,1,hidden)*0.02)
        self.attn_k=nn.Linear(hidden,hidden);self.attn_v=nn.Linear(hidden,hidden)
        self.proj=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,D_tok))
    def forward(self,x,sigma2=None):
        B=x.size(0);x=x.view(B,2*self.N,self.L_p)
        x=F.gelu(self.conv1(x));x=F.gelu(self.conv2(x))
        x=self.ln(x.transpose(1,2))
        q=self.attn_q.expand(B,-1,-1);k,v=self.attn_k(x),self.attn_v(x)
        w=F.softmax(torch.bmm(q,k.transpose(1,2))/math.sqrt(k.size(-1)),-1)
        return self.proj(torch.bmm(w,v).squeeze(1))

class ChannelDecoder(nn.Module):
    def __init__(self,K,N,D_tok,hidden=256):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(D_tok,hidden),nn.GELU(),
                               nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,2*K*N))
    def forward(self,z): return self.net(z)

class BFEncoder(nn.Module):
    def __init__(self,N,K,D_tok,hidden=256):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(2*N*K,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU())
        self.film=FiLMLayer(hidden);self.proj=nn.Linear(hidden,D_tok)
    def forward(self,W_real,sigma2=None):
        h=self.net(W_real)
        if sigma2 is not None: h=self.film(h,sigma2)
        return self.proj(h)

class BFDecoder(nn.Module):
    def __init__(self,N,K,D_tok,P_max,hidden=256):
        super().__init__()
        self.N,self.K,self.P_max=N,K,P_max
        self.net=nn.Sequential(nn.Linear(D_tok,hidden),nn.GELU(),
                               nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,2*N*K))
    def forward(self,c,normalize=True):
        W=real_to_bf(self.net(c),self.N,self.K)
        return power_normalize(W,self.P_max) if normalize else W

class CausalBlock(nn.Module):
    def __init__(self,d,heads,d_ff,drop=0.0):
        super().__init__()
        self.ln1=nn.LayerNorm(d);self.attn=nn.MultiheadAttention(d,heads,dropout=drop,batch_first=True)
        self.ln2=nn.LayerNorm(d);self.ff=nn.Sequential(nn.Linear(d,d_ff),nn.GELU(),nn.Linear(d_ff,d),nn.Dropout(drop))
    def forward(self,x,mask):
        h=self.ln1(x);x=x+self.attn(h,h,h,attn_mask=mask)[0];return x+self.ff(self.ln2(x))

class ICLTransformer(nn.Module):
    def __init__(self,D_tok,d,heads,layers,d_ff,drop=0.0):
        super().__init__()
        self.proj_in=nn.Linear(D_tok,d);self.ln_in=nn.LayerNorm(d)
        self.blocks=nn.ModuleList([CausalBlock(d,heads,d_ff,drop) for _ in range(layers)])
        self.ln_out=nn.LayerNorm(d);self.proj_out=nn.Linear(d,D_tok)
    def forward(self,seq):
        L=seq.size(1);mask=torch.triu(torch.ones(L,L,device=seq.device,dtype=torch.bool),diagonal=1)
        x=self.ln_in(self.proj_in(seq))
        for blk in self.blocks: x=blk(x,mask)
        return self.proj_out(self.ln_out(x))

class PilotICLModel(nn.Module):
    """Shared ICL model — identical architecture for all scenarios."""
    def __init__(self,pilot_enc,bf_enc,bf_dec,cfg):
        super().__init__()
        self.pilot_enc=pilot_enc;self.bf_enc=bf_enc;self.bf_dec=bf_dec
        self.transformer=ICLTransformer(cfg.D_tok,cfg.d_model,cfg.n_heads,cfg.n_layers,cfg.d_ff,cfg.dropout)
        self.D_tok=cfg.D_tok
    def forward(self,demo_pil,demo_W,query_pil,sigma2=None):
        B,l,_=demo_pil.shape
        all_pil=torch.cat([demo_pil.reshape(B*l,-1),query_pil],0)
        with torch.no_grad():
            all_z=self.pilot_enc(all_pil,sigma2=sigma2)
            demo_c=self.bf_enc(demo_W.reshape(B*l,-1),sigma2=sigma2).reshape(B,l,self.D_tok)
        demo_z=all_z[:B*l].reshape(B,l,self.D_tok);query_z=all_z[B*l:]
        tokens=[]
        for i in range(l): tokens.append(demo_z[:,i]);tokens.append(demo_c[:,i])
        tokens.append(query_z)
        out=self.transformer(torch.stack(tokens,dim=1))
        return self.bf_dec(out[:,-1],normalize=False)


###############################################################################
# 6. PER-SCENARIO DATASET
###############################################################################
class DynDataset:
    """Single-scenario dataset with self-bootstrapping support."""
    def __init__(self, max_sz=20000):
        self.max_sz=max_sz
        self.H=self.Y_real=self.W_real=self.rates=self.mmse_rates=self.is_sup=None
        self._n=0;self.n_sup=0;self.n_unsup=0
    @property
    def size(self): return self._n
    def add(self,H,Y_real,W_real,rates,mmse_rates=None,supervised=True):
        H,Y_real,W_real,rates=[x.detach() for x in [H,Y_real,W_real,rates]]
        flag=torch.full((H.size(0),),bool(supervised),device=device,dtype=torch.bool)
        mr=mmse_rates.detach() if mmse_rates is not None else torch.zeros_like(rates)
        if self.H is None:
            self.H,self.Y_real,self.W_real,self.rates,self.mmse_rates,self.is_sup=H,Y_real,W_real,rates,mr,flag
        else:
            self.H=torch.cat([self.H,H]);self.Y_real=torch.cat([self.Y_real,Y_real])
            self.W_real=torch.cat([self.W_real,W_real]);self.rates=torch.cat([self.rates,rates])
            self.mmse_rates=torch.cat([self.mmse_rates,mr]);self.is_sup=torch.cat([self.is_sup,flag])
        if supervised: self.n_sup+=H.size(0)
        else: self.n_unsup+=H.size(0)
        if self.H.size(0)>self.max_sz:
            sup_idx=torch.where(self.is_sup)[0];uns_idx=torch.where(~self.is_sup)[0]
            if sup_idx.numel()>=self.max_sz:
                keep=sup_idx[torch.topk(self.rates[sup_idx],self.max_sz).indices]
            else:
                nk=self.max_sz-sup_idx.numel()
                ku=uns_idx[torch.topk(self.rates[uns_idx],min(nk,uns_idx.numel())).indices] if uns_idx.numel()>nk else uns_idx
                keep=torch.cat([sup_idx,ku])
            for a in ['H','Y_real','W_real','rates','mmse_rates','is_sup']:
                setattr(self,a,getattr(self,a)[keep])
        self._n=self.H.size(0);self.n_sup=int(self.is_sup.sum().item());self.n_unsup=int((~self.is_sup).sum().item())
    def prune_unsup_bottom(self,drop_ratio,min_keep=0):
        if self._n==0 or drop_ratio<=0: return 0
        ui=torch.where(~self.is_sup)[0];nu=ui.numel()
        if nu==0: return 0
        nd=min(int(nu*drop_ratio),max(0,nu-min_keep))
        if nd<=0: return 0
        worst=ui[torch.topk(self.rates[ui],k=nd,largest=False).indices]
        keep=torch.ones(self._n,device=device,dtype=torch.bool);keep[worst]=False
        for a in ['H','Y_real','W_real','rates','mmse_rates','is_sup']:
            setattr(self,a,getattr(self,a)[keep])
        self._n=self.H.size(0);self.n_sup=int(self.is_sup.sum().item());self.n_unsup=int((~self.is_sup).sum().item())
        return nd


class MultiScenarioDataset:
    """Wraps N per-scenario DynDataset instances. Routes operations by scenario ID."""
    def __init__(self, scenarios: List[ScenarioDef], max_per_scenario=20000):
        self.scenarios = scenarios
        self.datasets = {sc.sid: DynDataset(max_sz=max_per_scenario) for sc in scenarios}
        self.n_scenarios = len(scenarios)

    def get(self, sid: int) -> DynDataset:
        return self.datasets[sid]

    @property
    def total_size(self):
        return sum(ds.size for ds in self.datasets.values())

    def summary(self):
        lines = []
        for sc in self.scenarios:
            ds = self.datasets[sc.sid]
            lines.append(f"  S{sc.sid} {sc.name:<18} size={ds.size:5d} (sup={ds.n_sup}, unsup={ds.n_unsup})")
        return "\n".join(lines)


###############################################################################
# 7. PHASE 0: BUILD MIXED DATA + PRETRAIN ENCODERS
###############################################################################
def build_mixed_labeled_cache(cfg, Phi):
    """
    Generate labeled data from ALL scenarios (equal proportions).
    Used for Phase 0 encoder pretraining and initial per-scenario datasets.
    """
    print("\n[Mixed Cache] Building multi-scenario labeled data...")
    per_sc = cfg.edn_n_samples // cfg.n_scenarios
    cache = {k: [] for k in ['H','Y_real','H_real','W_real','label_rate','mmse_rate','scenario_id']}

    for sc in cfg.scenarios:
        print(f"  S{sc.sid} {sc.name}: {per_sc} samples...", end=" ", flush=True)
        for s in range(0, per_sc, cfg.edn_batch):
            e = min(s+cfg.edn_batch, per_sc); b = e-s
            H = generate_channel_scenario(b, cfg.K, cfg.N, sc)
            Y = pilot_observe(H, Phi, cfg.sigma2)
            W = mmse_beamformer(H, cfg.P_max, cfg.sigma2)
            with torch.no_grad():
                r = compute_sum_rate(H, W, cfg.sigma2)
            cache['H'].append(H)
            cache['Y_real'].append(pilot_to_real(Y))
            cache['H_real'].append(channel_to_real(H))
            cache['W_real'].append(bf_to_real(W))
            cache['label_rate'].append(r)
            cache['mmse_rate'].append(r)
            cache['scenario_id'].append(torch.full((b,), sc.sid, device=device, dtype=torch.long))
        print(f"done", flush=True)

    out = {k: torch.cat(v, dim=0) for k, v in cache.items()}
    print(f"  Total: {out['H'].size(0)} samples across {cfg.n_scenarios} scenarios")
    return out


def pretrain_pilot_edn(cfg, cache):
    """Phase 0a: PilotEncoder + ChannelDecoder on ALL scenarios."""
    print("\n[Phase 0a] Pretraining PilotEncoder (multi-scenario)...")
    enc = PilotEncoder(cfg.N, cfg.L_p, cfg.D_tok, cfg.edn_hidden).to(device)
    dec = ChannelDecoder(cfg.K, cfg.N, cfg.D_tok, cfg.edn_hidden).to(device)
    opt = Adam(list(enc.parameters())+list(dec.parameters()), lr=cfg.edn_lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, cfg.edn_epochs, eta_min=0)
    n = cache['Y_real'].size(0)
    for ep in range(cfg.edn_epochs):
        el=0.;nb=0;perm=torch.randperm(n,device=device)
        for s in range(0,n,cfg.edn_batch):
            e=min(s+cfg.edn_batch,n);idx=perm[s:e]
            loss=F.mse_loss(dec(enc(cache['Y_real'][idx],sigma2=cfg.sigma2)),cache['H_real'][idx])
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters())+list(dec.parameters()),5.0)
            opt.step();el+=loss.item();nb+=1
        sched.step()
        if (ep+1)%100==0 or ep==0: print(f"  Ep {ep+1}/{cfg.edn_epochs} MSE={el/nb:.6f}")
    return enc


def pretrain_bf_edn(cfg, cache):
    """Phase 0b: BFEncoder + BFDecoder on ALL scenarios."""
    print("\n[Phase 0b] Pretraining BFEncoder+BFDecoder (multi-scenario)...")
    enc = BFEncoder(cfg.N, cfg.K, cfg.D_tok, cfg.edn_hidden).to(device)
    dec = BFDecoder(cfg.N, cfg.K, cfg.D_tok, cfg.P_max, cfg.edn_hidden).to(device)
    opt = Adam(list(enc.parameters())+list(dec.parameters()), lr=cfg.edn_lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, cfg.edn_epochs, eta_min=0)
    n = cache['W_real'].size(0)
    for ep in range(cfg.edn_epochs):
        el=0.;nb=0;perm=torch.randperm(n,device=device)
        for s in range(0,n,cfg.edn_batch):
            e=min(s+cfg.edn_batch,n);idx=perm[s:e]
            loss=F.mse_loss(bf_to_real(dec(enc(cache['W_real'][idx],sigma2=cfg.sigma2))),cache['W_real'][idx])
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters())+list(dec.parameters()),5.0)
            opt.step();el+=loss.item();nb+=1
        sched.step()
        if (ep+1)%100==0 or ep==0: print(f"  Ep {ep+1}/{cfg.edn_epochs} MSE={el/nb:.6f}")
    return enc, dec


###############################################################################
# 8. BASELINES (per-scenario)
###############################################################################
def compute_per_scenario_baselines(H_tests, Phi, cfg):
    """Compute baselines for each scenario's test set."""
    all_bl = {}
    for sc in cfg.scenarios:
        H = H_tests[sc.sid]
        s2 = cfg.sigma2; Pm = cfg.P_max
        Y = pilot_observe(H, Phi, s2)
        Hh = mmse_channel_est(Y, Phi, s2)
        with torch.no_grad():
            b1 = compute_sum_rate(H, mmse_beamformer(H, Pm, s2), s2).mean().item()
            b2 = compute_sum_rate(H, mmse_beamformer(Hh, Pm, s2), s2).mean().item()
        all_bl[sc.sid] = {'mmse_perf': b1, 'mmse_imp': b2, 'name': sc.name}
        print(f"  S{sc.sid} {sc.name:<18} MMSE-P={b1:.2f} MMSE-E={b2:.2f}")
    return all_bl


###############################################################################
# 9. EVALUATION (per-scenario)
###############################################################################
@torch.no_grad()
def evaluate_per_scenario(model, mds, H_tests, Phi, cfg):
    """Evaluate model on each scenario's test set. Context drawn from same scenario."""
    model.eval()
    results = {}
    for sc in cfg.scenarios:
        ds = mds.get(sc.sid)
        if ds.size < cfg.n_demos: continue
        H = H_tests[sc.sid]; B = H.size(0); bs = min(cfg.batch_size, B); rates = []
        for s in range(0, B, bs):
            e = min(s+bs, B); Hb = H[s:e]; b = Hb.size(0)
            Y = pilot_observe(Hb, Phi, cfg.sigma2); pil = pilot_to_real(Y)
            # Deterministic context for reproducible evaluation.
            # Each query index maps to fixed demo indices in the same scenario pool.
            q_idx = torch.arange(s, e, device=device).unsqueeze(1)
            demo_offset = torch.arange(cfg.n_demos, device=device).unsqueeze(0)
            d_idx = (q_idx * (cfg.n_demos + 3) + demo_offset + sc.sid) % ds.size
            dp = ds.Y_real[d_idx]; dw = ds.W_real[d_idx]
            H_ls = ls_channel_est(Y, Phi)
            W_base = mmse_beamformer(H_ls, cfg.P_max, cfg.sigma2)
            dW = model(dp, dw, pil, sigma2=cfg.sigma2)
            W_hat = power_normalize(W_base + dW, cfg.P_max)
            rates.append(compute_sum_rate(Hb, W_hat, cfg.sigma2))
        results[sc.sid] = torch.cat(rates).mean().item()
    model.train()
    return results


@torch.no_grad()
def calibrate_hybrid_rate_scale_multi(model, mds, Phi, cfg):
    """Estimate a stable scale between supervised MSE and unsupervised rate losses."""
    model.eval()
    n_steps = max(1, cfg.hybrid_calib_steps)
    B = max(1, min(cfg.hybrid_calib_batch, cfg.batch_size))

    eligible_sup_sids = []
    for sid in cfg.scenario_ids:
        ds = mds.get(sid)
        if ds.size >= cfg.n_demos and ds.is_sup is not None and ds.is_sup.any():
            eligible_sup_sids.append(sid)
    if len(eligible_sup_sids) == 0:
        model.train()
        return 1.0, 0.0, 1.0

    mse_vals, rate_vals = [], []
    for _ in range(n_steps):
        q_H_list, q_W_gt_list, q_pil_list = [], [], []
        dp_list, dw_list = [], []
        for _b in range(B):
            sid = int(eligible_sup_sids[torch.randint(0, len(eligible_sup_sids), (1,), device=device).item()])
            ds = mds.get(sid)
            sup_idx = torch.where(ds.is_sup)[0]
            qi = sup_idx[torch.randint(0, sup_idx.numel(), (1,), device=device)]

            H_q = ds.H[qi]
            W_q_gt = ds.W_real[qi]
            Y_q = pilot_observe(H_q, Phi, cfg.sigma2)
            pil_q = pilot_to_real(Y_q)

            d_idx = torch.randint(0, ds.size, (1, cfg.n_demos), device=device)
            dp_list.append(ds.Y_real[d_idx])
            dw_list.append(ds.W_real[d_idx])

            q_H_list.append(H_q)
            q_W_gt_list.append(W_q_gt)
            q_pil_list.append(pil_q)

        q_H = torch.cat(q_H_list, 0)
        q_W_gt = torch.cat(q_W_gt_list, 0)
        q_pil = torch.cat(q_pil_list, 0)
        dp = torch.cat(dp_list, 0)
        dw = torch.cat(dw_list, 0)

        q_Y = pilot_observe(q_H, Phi, cfg.sigma2)
        H_ls = ls_channel_est(q_Y, Phi)
        W_base = mmse_beamformer(H_ls, cfg.P_max, cfg.sigma2)
        dW = model(dp, dw, q_pil, sigma2=cfg.sigma2)
        W_hat = power_normalize(W_base + dW, cfg.P_max)

        mse_vals.append(F.mse_loss(bf_to_real(W_hat), q_W_gt, reduction='none').sum(-1).mean())
        rate_vals.append(compute_sum_rate(q_H, W_hat, cfg.sigma2).mean())

    mse_ref = torch.stack(mse_vals).mean().item()
    rate_ref = abs(torch.stack(rate_vals).mean().item())
    raw = (mse_ref / (rate_ref + 1e-8)) * cfg.hybrid_rate_gain
    scale = float(np.clip(raw, cfg.hybrid_rate_scale_min, cfg.hybrid_rate_scale_max))
    model.train()
    return scale, mse_ref, rate_ref


###############################################################################
# 10. MAIN TRAINING LOOP
###############################################################################
def train(cfg):
    set_global_seed(cfg.seed)
    print("="*75)
    print("  MULTI-SCENARIO ICL PRECODING v6")
    print("="*75)
    print(f"  K={cfg.K} N={cfg.N} L_p={cfg.L_p} SNR={cfg.SNR_dB}dB")
    print(f"  {cfg.n_scenarios} scenarios:")
    for sc in cfg.scenarios:
        tag = f"({sc.n_clusters}cl,{sc.n_rays}r,{sc.spread_deg}°)" if sc.ch_type=='sparse' else "(iid)"
        print(f"    S{sc.sid}: {sc.name:<18} {sc.ch_type:8s} {tag}")

    Phi = generate_pilot_dft(cfg.K, cfg.L_p)

    # Build mixed labeled cache for ALL scenarios
    cache = build_mixed_labeled_cache(cfg, Phi)

    # Phase 0: Pretrain encoders on mixed data
    pilot_enc = pretrain_pilot_edn(cfg, cache)
    bf_enc, bf_dec = pretrain_bf_edn(cfg, cache)
    for p in pilot_enc.parameters(): p.requires_grad_(False)
    for p in bf_enc.parameters(): p.requires_grad_(False)

    # Per-scenario test sets
    H_tests = {}
    for sc in cfg.scenarios:
        H_tests[sc.sid] = generate_channel_scenario(cfg.n_test_per_scenario, cfg.K, cfg.N, sc)
    print("\nPer-scenario baselines:")
    bl = compute_per_scenario_baselines(H_tests, Phi, cfg)

    # Initialize per-scenario datasets from cache
    mds = MultiScenarioDataset(cfg.scenarios, max_per_scenario=cfg.max_ds_per_scenario)
    sids = cache['scenario_id']
    for sc in cfg.scenarios:
        mask = sids == sc.sid
        idx = torch.where(mask)[0][:cfg.init_ds_per_scenario]
        if idx.numel() > 0:
            ds = mds.get(sc.sid)
            ds.add(cache['H'][idx], cache['Y_real'][idx], cache['W_real'][idx],
                   cache['label_rate'][idx], mmse_rates=cache['mmse_rate'][idx], supervised=True)

    # Safety: ensure every scenario can always build context during training/evaluation.
    for sc in cfg.scenarios:
        ds = mds.get(sc.sid)
        if ds.size >= cfg.n_demos:
            continue
        need = cfg.n_demos - ds.size
        H_extra = generate_channel_scenario(need, cfg.K, cfg.N, sc)
        Y_extra = pilot_observe(H_extra, Phi, cfg.sigma2)
        W_extra = mmse_beamformer(H_extra, cfg.P_max, cfg.sigma2)
        with torch.no_grad():
            r_extra = compute_sum_rate(H_extra, W_extra, cfg.sigma2)
        ds.add(
            H_extra,
            pilot_to_real(Y_extra),
            bf_to_real(W_extra),
            r_extra,
            mmse_rates=r_extra,
            supervised=True,
        )
    print(f"\nInitial datasets:\n{mds.summary()}")

    # Model
    model = PilotICLModel(pilot_enc, bf_enc, bf_dec, cfg).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.total_epochs, eta_min=cfg.lr_min)

    print("\n"+"="*75)
    best_avg = 0.0
    hist = {f'test_s{sc.sid}': [] for sc in cfg.scenarios}
    hist['test_avg'] = []; hist['ds_total'] = []
    hybrid_rate_scale = None

    for epoch in range(cfg.total_epochs):
        model.train(); t0 = time.time()
        if epoch < cfg.phase1_epochs: phase, r = 1, 0.0
        else:
            phase = 2
            p2e = epoch - cfg.phase1_epochs
            if p2e < 50: r = 0.25
            elif p2e < 100: r = 0.50
            elif p2e < 150: r = 0.75
            else: r = 1.0

        ep_prog = epoch / max(1, cfg.total_epochs-1)
        alpha_t = cfg.boot_alpha_start + (cfg.boot_alpha_end-cfg.boot_alpha_start)*ep_prog
        beta_t = cfg.boot_beta_start + (cfg.boot_beta_end-cfg.boot_beta_start)*ep_prog

        ep_loss_sum, ep_rate_sum, ep_add_total, ep_n = 0., 0., 0, 0

        n_steps = cfg.steps_per_epoch

        for step in range(n_steps):
            q_H_list, q_pil_list, q_W_gt_list = [], [], []
            dp_list, dw_list = [], []
            sc_per_sample, src_type_list = [], []

            eligible_demo_sids = [sid for sid in cfg.scenario_ids if mds.get(sid).size >= cfg.n_demos]
            if len(eligible_demo_sids) == 0:
                raise RuntimeError("No scenario has enough demos to build context.")

            if phase == 1:
                eligible_sup_sids = []
                for sid in eligible_demo_sids:
                    ds_cand = mds.get(sid)
                    if ds_cand.is_sup is not None and bool(ds_cand.is_sup.any().item()):
                        eligible_sup_sids.append(sid)
                if len(eligible_sup_sids) == 0:
                    raise RuntimeError("Phase 1 requires supervised samples in at least one scenario.")

                # Round-robin at batch-level across scenarios.
                batch_sid = eligible_sup_sids[(epoch * n_steps + step) % len(eligible_sup_sids)]
                ds_batch = mds.get(batch_sid)
                sample_plan = ['sup'] * cfg.batch_size
            else:
                B = cfg.batch_size
                n_uns = int(round(B * r))
                n_uns = max(0, min(B, n_uns))
                n_sup = B - n_uns

                # Single-scenario batch, scenario rotates in round-robin.
                batch_sid = eligible_demo_sids[(epoch * n_steps + step) % len(eligible_demo_sids)]
                ds_batch = mds.get(batch_sid)

                if n_uns == 0:
                    n_ds_uns = 0
                    n_new_uns = 0
                elif ds_batch.n_unsup == 0:
                    n_ds_uns = 0
                    n_new_uns = n_uns
                else:
                    n_ds_uns = n_uns // 2
                    n_new_uns = n_uns - n_ds_uns

                sample_plan = ['sup'] * n_sup + ['ds_uns'] * n_ds_uns + ['new_uns'] * n_new_uns
                if len(sample_plan) == 0:
                    continue
                plan_perm = torch.randperm(len(sample_plan), device=device)
                sample_plan = [sample_plan[int(i.item())] for i in plan_perm]

            for i, sample_type in enumerate(sample_plan):
                if phase == 1:
                    sid = int(batch_sid)
                    ds = ds_batch
                    assert ds.H is not None and ds.W_real is not None and ds.Y_real is not None and ds.is_sup is not None
                    sup_idx = torch.where(ds.is_sup)[0]
                    q_idx = sup_idx[torch.randint(0, sup_idx.numel(), (1,), device=device)]
                    H_q = ds.H[q_idx]
                    W_q_gt = ds.W_real[q_idx]
                    src_type = 0  # supervised
                else:
                    sid = int(batch_sid)
                    ds = ds_batch
                    if sample_type == 'sup':
                        if ds.is_sup is None or not bool(ds.is_sup.any().item()):
                            sample_type = 'new_uns'
                        else:
                            assert ds.H is not None and ds.W_real is not None and ds.Y_real is not None and ds.is_sup is not None
                            sup_idx = torch.where(ds.is_sup)[0]
                            q_idx = sup_idx[torch.randint(0, sup_idx.numel(), (1,), device=device)]
                            H_q = ds.H[q_idx]
                            W_q_gt = ds.W_real[q_idx]
                            src_type = 0

                    if sample_type == 'ds_uns':
                        if ds.n_unsup <= 0:
                            sample_type = 'new_uns'
                        else:
                            assert ds.H is not None and ds.W_real is not None and ds.Y_real is not None and ds.is_sup is not None
                            uns_idx = torch.where(~ds.is_sup)[0]
                            q_idx = uns_idx[torch.randint(0, uns_idx.numel(), (1,), device=device)]
                            H_q = ds.H[q_idx]
                            W_q_gt = None
                            src_type = 1  # in-dataset unsup

                    if sample_type == 'new_uns':
                        sc = cfg.sid_to_scenario[sid]
                        assert ds.Y_real is not None and ds.W_real is not None
                        H_q = generate_channel_scenario(1, cfg.K, cfg.N, sc)
                        W_q_gt = None
                        src_type = 2  # fresh unsup

                Y_q = pilot_observe(H_q, Phi, cfg.sigma2)
                pil_q = pilot_to_real(Y_q)
                assert ds.Y_real is not None and ds.W_real is not None
                d_idx = torch.randint(0, ds.size, (1, cfg.n_demos), device=device)

                q_H_list.append(H_q)
                q_pil_list.append(pil_q)
                q_W_gt_list.append(W_q_gt if W_q_gt is not None else torch.zeros(1, 2 * cfg.N * cfg.K, device=device))
                dp_list.append(ds.Y_real[d_idx])
                dw_list.append(ds.W_real[d_idx])
                sc_per_sample.append(sid)
                src_type_list.append(src_type)

            q_H = torch.cat(q_H_list, 0)
            q_pil = torch.cat(q_pil_list, 0)
            q_W_gt = torch.cat(q_W_gt_list, 0)
            dp = torch.cat(dp_list, 0)
            dw = torch.cat(dw_list, 0)
            src_type = torch.tensor(src_type_list, device=device, dtype=torch.long)

            sup_mask = src_type == 0
            ds_uns_mask = src_type == 1
            new_uns_mask = src_type == 2

            q_Y = pilot_observe(q_H, Phi, cfg.sigma2)
            H_ls = ls_channel_est(q_Y, Phi)
            W_base = mmse_beamformer(H_ls, cfg.P_max, cfg.sigma2)
            dW = model(dp, dw, q_pil, sigma2=cfg.sigma2)
            W_hat = power_normalize(W_base + dW, cfg.P_max)
            rate_pred = compute_sum_rate(q_H, W_hat, cfg.sigma2)

            if sup_mask.any():
                mse_sup = F.mse_loss(bf_to_real(W_hat)[sup_mask], q_W_gt[sup_mask], reduction='none').sum(-1).mean()
            else:
                mse_sup = torch.zeros((), device=device, dtype=rate_pred.dtype)

            if phase == 1:
                loss = cfg.loss_mse_weight * mse_sup
            else:
                if ds_uns_mask.any():
                    rate_loss_ds_unsup = -rate_pred[ds_uns_mask].mean()
                else:
                    rate_loss_ds_unsup = torch.zeros((), device=device, dtype=rate_pred.dtype)

                if new_uns_mask.any():
                    rate_loss_new_unsup = -rate_pred[new_uns_mask].mean()
                else:
                    rate_loss_new_unsup = torch.zeros((), device=device, dtype=rate_pred.dtype)

                if hybrid_rate_scale is None:
                    hybrid_rate_scale, mse_ref, rate_ref = calibrate_hybrid_rate_scale_multi(model, mds, Phi, cfg)
                    print(f"  [Hybrid calibration] mse_ref={mse_ref:.4f} rate_ref={rate_ref:.4f} "
                          f"scale={hybrid_rate_scale:.4f}", flush=True)

                switch = r ** cfg.hybrid_switch_power
                rate_combo = (
                    cfg.loss_rate_ds_weight * rate_loss_ds_unsup
                    + cfg.loss_rate_new_weight * rate_loss_new_unsup
                )
                loss = cfg.loss_mse_weight * mse_sup + switch * hybrid_rate_scale * rate_combo

            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()

            ep_loss_sum += loss.item()
            ep_rate_sum += rate_pred.mean().item()
            ep_n += 1

            if phase == 2 and new_uns_mask.any():
                with torch.no_grad():
                    for sid in set(sc_per_sample):
                        sid_mask = torch.tensor([s == sid for s in sc_per_sample], device=device)
                        sc_mask = sid_mask & new_uns_mask
                        if not sc_mask.any():
                            continue

                        ur = rate_pred[sc_mask]
                        qH_s = q_H[sc_mask]
                        qpil_s = q_pil[sc_mask]
                        What_s = W_hat[sc_mask]

                        mmse_r = compute_sum_rate(qH_s, mmse_beamformer(qH_s, cfg.P_max, cfg.sigma2), cfg.sigma2)
                        per_ok = ur > alpha_t * mmse_r
                        if ur.numel() > 1:
                            cross_ok = ur >= torch.quantile(ur, beta_t)
                        else:
                            cross_ok = torch.ones_like(ur, dtype=torch.bool)

                        good = per_ok & cross_ok
                        if good.any():
                            mds.get(sid).add(
                                qH_s[good], qpil_s[good], bf_to_real(What_s[good]),
                                ur[good], mmse_rates=mmse_r[good], supervised=False)
                            ep_add_total += good.sum().item()

        scheduler.step()

        # Per-scenario bottom pruning
        nd_total = 0
        if phase == 2 and (epoch+1) % cfg.prune_every == 0:
            fp = (epoch-cfg.phase1_epochs)/max(1, cfg.phase2_epochs-1)
            dr = cfg.prune_drop_start + (cfg.prune_drop_end-cfg.prune_drop_start)*fp
            for sc in cfg.scenarios:
                nd_total += mds.get(sc.sid).prune_unsup_bottom(dr, cfg.prune_min_unsup)

        # Per-scenario evaluation
        sc_rates = evaluate_per_scenario(model, mds, H_tests, Phi, cfg)
        avg_rate = np.mean([v for v in sc_rates.values()]) if sc_rates else 0.0
        best_avg = max(best_avg, avg_rate)

        for sc in cfg.scenarios:
            hist[f'test_s{sc.sid}'].append(sc_rates.get(sc.sid, 0.0))
        hist['test_avg'].append(avg_rate)
        hist['ds_total'].append(mds.total_size)

        dt = time.time() - t0
        rate_str = " ".join([f"S{sc.sid}={sc_rates.get(sc.sid,0):.1f}" for sc in cfg.scenarios])
        print(f"{epoch+1:3d} ph{phase} r={r:.2f} | loss={ep_loss_sum/max(1,ep_n):.4f} | "
              f"avg={avg_rate:.2f} best={best_avg:.2f} | DS={mds.total_size} +{ep_add_total} -{nd_total} | "
              f"{rate_str} ({dt:.1f}s)", flush=True)

    # Final summary
    print("\n"+"="*75)
    print("  MULTI-SCENARIO TRAINING COMPLETE")
    print("="*75)
    for sc in cfg.scenarios:
        ds = mds.get(sc.sid)
        tr = sc_rates.get(sc.sid, 0.0)
        b_mmse = bl[sc.sid]['mmse_perf']
        b_mmse_e = bl[sc.sid]['mmse_imp']
        print(f"  S{sc.sid} {sc.name:<18} test={tr:.2f} "
              f"MMSE-P={b_mmse:.2f} MMSE-E={b_mmse_e:.2f} "
              f"DS={ds.size}({ds.n_sup}s+{ds.n_unsup}u)")
    print(f"\n  Average test rate: {avg_rate:.3f} (best: {best_avg:.3f})")
    print(f"  Total dataset: {mds.total_size}")

    return model, mds, bl, hist


###############################################################################
# 11. ENTRY POINT
###############################################################################
if __name__ == "__main__":
    cfg = Config(
        K=32, N=32, L_p=20, P_max=1.0, SNR_dB=20,
        D_tok=256, edn_hidden=256, edn_epochs=500, edn_lr=1e-3, edn_batch=128,
        edn_n_samples=8000,  # 1000 per scenario
        n_demos=5, d_model=512, n_heads=8, n_layers=6, d_ff=1024,
        batch_size=64, lr=1e-4, lr_min=5e-5, weight_decay=1e-4,
        init_ds_per_scenario=256,
        wmmse_iters=100, wmmse_lr=0.03, unsup_scale=0.01,
        phase1_epochs=150, phase2_epochs=500, steps_per_epoch=80,
        boot_alpha_start=0.60, boot_alpha_end=0.90,
        boot_beta_start=0.60, boot_beta_end=0.90,
        max_ds_per_scenario=20000,
        prune_every=10, prune_drop_start=0.0, prune_drop_end=0.10, prune_min_unsup=1024,
        n_test_per_scenario=100,
    )
    train(cfg)
