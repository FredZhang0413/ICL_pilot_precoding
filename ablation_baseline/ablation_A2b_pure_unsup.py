"""
ABLATION A2b: Pure Unsupervised ICL (No Supervised Warm-Start)

PURPOSE: Show that the supervised warm-start (Phase 1) is essential.
Without it, the model optimizes sum-rate from random initialization,
which is a hard non-convex problem leading to slow/poor convergence.

SETUP: Full ICL architecture. NO labeled data. Training uses ONLY sum-rate
loss from epoch 1. Self-bootstrapping is active (dataset starts empty,
grows purely from model-generated solutions). Initial context uses
MMSE BF solutions as seed demos (cheaply computed, no WMMSE needed).

EXPECTED RESULT: Very slow convergence or convergence to poor local optimum.
"""

import math, torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
import numpy as np
from typing import Dict, List
import warnings, time
import matplotlib; matplotlib.use('Agg')

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_global_seed(seed,det=False):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# ============================================================================
# SHARED UTILITIES (identical to proposed — omitted comments for brevity)
# ============================================================================
class Config:
    def __init__(self,**kw):
        self.seed=kw.get('seed',2026);self.K=kw.get('K',32);self.N=kw.get('N',32)
        self.L_p=kw.get('L_p',20);self.P_max=kw.get('P_max',1.0)
        self.SNR_dB=kw.get('SNR_dB',20);self.sigma2=self.P_max/(10**(self.SNR_dB/10))
        self.ch_n_clusters=kw.get('ch_n_clusters',3);self.ch_n_rays=kw.get('ch_n_rays',5)
        self.ch_spread_deg=kw.get('ch_spread_deg',10.0)
        self.D_tok=kw.get('D_tok',256);self.edn_hidden=kw.get('edn_hidden',512)
        self.edn_epochs=kw.get('edn_epochs',200);self.edn_lr=kw.get('edn_lr',1e-3)
        self.edn_batch=kw.get('edn_batch',128)
        self.n_demos=kw.get('n_demos',5)
        self.d_model=kw.get('d_model',512);self.n_heads=kw.get('n_heads',8)
        self.n_layers=kw.get('n_layers',6);self.d_ff=kw.get('d_ff',1024)
        self.dropout=kw.get('dropout',0.0)
        self.batch_size=kw.get('batch_size',64);self.lr=kw.get('lr',2e-4)
        self.lr_min=kw.get('lr_min',5e-5);self.weight_decay=kw.get('weight_decay',1e-4)
        self.unsup_scale=kw.get('unsup_scale',0.01)
        # Seed dataset: small set of MMSE BF solutions (cheap to compute)
        self.seed_ds_size=kw.get('seed_ds_size',200)
        self.total_epochs=kw.get('total_epochs',550)
        self.steps_per_epoch=kw.get('steps_per_epoch',80)
        self.n_test=kw.get('n_test',200)
        self.max_ds_size=kw.get('max_ds_size',50000)
        self.boot_alpha_start=kw.get('boot_alpha_start',0.60)
        self.boot_alpha_end=kw.get('boot_alpha_end',0.90)
        self.boot_beta_start=kw.get('boot_beta_start',0.50)
        self.boot_beta_end=kw.get('boot_beta_end',0.80)
        self.prune_every=kw.get('prune_every',10)
        self.prune_drop_end=kw.get('prune_drop_end',0.10)
        self.prune_min_unsup=kw.get('prune_min_unsup',4096)
        self.wmmse_iters=kw.get('wmmse_iters',500);self.wmmse_lr=kw.get('wmmse_lr',0.03)
        # Pretraining dataset size for encoders
        self.edn_ds_size=kw.get('edn_ds_size',5000)

def generate_channel(B,K,N,n_cl=3,n_ray=5,spread=10.0):
    L=n_cl*n_ray;asp=math.radians(spread)
    cm=(torch.rand(B,K,n_cl,1,device=device)-0.5)*math.pi
    ro=torch.randn(B,K,n_cl,n_ray,device=device)*asp
    ang=(cm+ro).clamp(-math.pi/2,math.pi/2).reshape(B,K,L)
    alp=(torch.randn(B,K,L,device=device)+1j*torch.randn(B,K,L,device=device))/math.sqrt(2)
    idx=torch.arange(N,device=device,dtype=torch.float32).view(1,1,1,N)
    ph=math.pi*idx*torch.sin(ang).unsqueeze(-1)
    st=torch.polar(torch.ones_like(ph),ph)/math.sqrt(N)
    return math.sqrt(N/L)*torch.sum(alp.unsqueeze(-1)*st,dim=2)

def generate_pilot_dft(K,L_p):
    return (torch.fft.fft(torch.eye(K,device=device))/math.sqrt(K))[:,:L_p].contiguous()
def pilot_observe(H,Phi,sigma2):
    B,K,N=H.shape;Lp=Phi.size(1);Y=H.transpose(-1,-2)@Phi.unsqueeze(0).expand(B,-1,-1)
    return Y+torch.complex(torch.randn(B,N,Lp,device=device)*math.sqrt(sigma2/2),torch.randn(B,N,Lp,device=device)*math.sqrt(sigma2/2))
def pilot_to_real(Y): return torch.cat([Y.real,Y.imag],dim=1).reshape(Y.size(0),-1)
def bf_to_real(W): return torch.cat([W.real,W.imag],dim=1).reshape(W.size(0),-1)
def real_to_bf(x,N,K): B=x.size(0);x=x.view(B,2*N,K);return torch.complex(x[:,:N,:],x[:,N:,:])
def channel_to_real(H): return torch.cat([H.real,H.imag],dim=-1).reshape(H.size(0),-1)
def compute_sum_rate(H,W,sigma2):
    HW=H@W;sig=torch.abs(torch.diagonal(HW,dim1=-2,dim2=-1))**2
    tot=torch.sum(torch.abs(HW)**2,dim=-1);return torch.log2(1+sig/(tot-sig+sigma2)).sum(-1)
def mmse_beamformer(H,P_max,sigma2):
    B,K,N=H.shape;HH=H.conj().transpose(-1,-2)
    A=HH@H+sigma2*torch.eye(N,device=device,dtype=H.dtype).unsqueeze(0)
    W=torch.linalg.solve(A,HH);pw=torch.sum(torch.abs(W)**2,dim=(1,2)).real
    return W*torch.sqrt(P_max/(pw+1e-8)).view(B,1,1)
def power_normalize(W,P_max):
    pw=torch.sum(torch.abs(W)**2,dim=(1,2),keepdim=True).real;return W*torch.sqrt(P_max/(pw+1e-8))
def ls_channel_est(Y,Phi): return (Y@torch.linalg.pinv(Phi).unsqueeze(0)).transpose(-1,-2).contiguous()
def mmse_channel_est(Y,Phi,sigma2):
    K,Lp=Phi.shape;A=Phi.T@Phi.conj()+sigma2*torch.eye(Lp,device=device,dtype=Phi.dtype)
    return torch.matmul((Phi.conj()@torch.linalg.inv(A)).unsqueeze(0),Y.transpose(-1,-2))
def generate_wmmse_labels(H,P_max,sigma2,n_iters=500,lr=0.03,n_restarts=2):
    B,K,N=H.shape;Hd=H.detach();best_r=torch.full((B,),-float('inf'),device=device)
    best_W=torch.zeros(B,N,K,device=device,dtype=torch.cfloat)
    for _ in range(n_restarts):
        Wr=(torch.randn(B,N,K,device=device)*0.05).requires_grad_(True)
        Wi=(torch.randn(B,N,K,device=device)*0.05).requires_grad_(True)
        opt=Adam([Wr,Wi],lr=lr)
        for _ in range(n_iters):
            W=power_normalize(torch.complex(Wr,Wi),P_max);(-compute_sum_rate(Hd,W,sigma2).sum()).backward();opt.step();opt.zero_grad()
        with torch.no_grad():
            Ws=power_normalize(torch.complex(Wr,Wi),P_max);rs=compute_sum_rate(Hd,Ws,sigma2)
            imp=rs>best_r
            if imp.any(): best_r[imp]=rs[imp];best_W[imp]=Ws[imp]
    return best_W.detach(),best_r.detach()
def compute_baselines(H_test,Phi,cfg):
    B=H_test.size(0);s2=cfg.sigma2;Pm=cfg.P_max;bs=min(64,B)
    res={k:[] for k in ['mmse_perf','mmse_imp','wmmse_perf','wmmse_imp']}
    for s in range(0,B,bs):
        e=min(s+bs,B);H=H_test[s:e];Y=pilot_observe(H,Phi,s2);Hh=mmse_channel_est(Y,Phi,s2)
        with torch.no_grad():
            res['mmse_perf'].append(compute_sum_rate(H,mmse_beamformer(H,Pm,s2),s2))
            res['mmse_imp'].append(compute_sum_rate(H,mmse_beamformer(Hh,Pm,s2),s2))
        W3,_=generate_wmmse_labels(H,Pm,s2,n_restarts=2)
        with torch.no_grad(): res['wmmse_perf'].append(compute_sum_rate(H,W3,s2))
        W4,_=generate_wmmse_labels(Hh,Pm,s2,n_restarts=2)
        with torch.no_grad(): res['wmmse_imp'].append(compute_sum_rate(H,W4,s2))
    return {k:torch.cat(v).mean().item() for k,v in res.items()}

# ============================================================================
# FULL ICL MODEL + ENCODERS (identical to proposed)
# ============================================================================
class FiLMLayer(nn.Module):
    def __init__(self,n_ch):
        super().__init__();self.net=nn.Sequential(nn.Linear(1,64),nn.GELU(),nn.Linear(64,2*n_ch))
    def forward(self,x,sigma2):
        B=x.size(0);C=x.size(1);p=self.net(torch.full((B,1),math.log(sigma2+1e-10),device=x.device))
        g=p[:,:C];b=p[:,C:]
        if x.dim()==3: g=g.unsqueeze(-1);b=b.unsqueeze(-1)
        return g*x+b

class PilotEncoder(nn.Module):
    def __init__(self,N,L_p,D_tok,hidden=512):
        super().__init__();self.N,self.L_p=N,L_p
        self.conv1=nn.Conv1d(2*N,hidden,3,padding=1);self.conv2=nn.Conv1d(hidden,hidden,3,padding=1)
        self.film=FiLMLayer(hidden);self.ln=nn.LayerNorm(hidden)
        self.attn_q=nn.Parameter(torch.randn(1,1,hidden)*0.02)
        self.attn_k=nn.Linear(hidden,hidden);self.attn_v=nn.Linear(hidden,hidden)
        self.proj=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,D_tok))
    def forward(self,x,sigma2=None):
        B=x.size(0);x=x.view(B,2*self.N,self.L_p);x=F.gelu(self.conv1(x));x=F.gelu(self.conv2(x))
        if sigma2 is not None: x=self.film(x,sigma2)
        x=self.ln(x.transpose(1,2));q=self.attn_q.expand(B,-1,-1);k,v=self.attn_k(x),self.attn_v(x)
        w=F.softmax(torch.bmm(q,k.transpose(1,2))/math.sqrt(k.size(-1)),-1)
        return self.proj(torch.bmm(w,v).squeeze(1))

class ChannelDecoder(nn.Module):
    def __init__(self,K,N,D_tok,hidden=512):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(D_tok,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,2*K*N))
    def forward(self,z): return self.net(z)

class BFEncoder(nn.Module):
    def __init__(self,N,K,D_tok,hidden=512):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(2*N*K,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU())
        self.film=FiLMLayer(hidden);self.proj=nn.Linear(hidden,D_tok)
    def forward(self,W_real,sigma2=None):
        h=self.net(W_real);
        if sigma2 is not None: h=self.film(h,sigma2)
        return self.proj(h)

class BFDecoder(nn.Module):
    def __init__(self,N,K,D_tok,P_max,hidden=512):
        super().__init__();self.N,self.K,self.P_max=N,K,P_max
        self.net=nn.Sequential(nn.Linear(D_tok,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,2*N*K))
    def forward(self,c,normalize=True):
        W=real_to_bf(self.net(c),self.N,self.K);return power_normalize(W,self.P_max) if normalize else W

class CausalBlock(nn.Module):
    def __init__(self,d,heads,d_ff,drop=0.0):
        super().__init__();self.ln1=nn.LayerNorm(d);self.attn=nn.MultiheadAttention(d,heads,dropout=drop,batch_first=True)
        self.ln2=nn.LayerNorm(d);self.ff=nn.Sequential(nn.Linear(d,d_ff),nn.GELU(),nn.Linear(d_ff,d),nn.Dropout(drop))
    def forward(self,x,mask):
        h=self.ln1(x);x=x+self.attn(h,h,h,attn_mask=mask)[0];return x+self.ff(self.ln2(x))

class ICLTransformer(nn.Module):
    def __init__(self,D_tok,d,heads,layers,d_ff,drop=0.0):
        super().__init__();self.proj_in=nn.Linear(D_tok,d);self.ln_in=nn.LayerNorm(d)
        self.blocks=nn.ModuleList([CausalBlock(d,heads,d_ff,drop) for _ in range(layers)])
        self.ln_out=nn.LayerNorm(d);self.proj_out=nn.Linear(d,D_tok)
    def forward(self,seq):
        L=seq.size(1);mask=torch.triu(torch.ones(L,L,device=seq.device,dtype=torch.bool),diagonal=1)
        x=self.ln_in(self.proj_in(seq))
        for blk in self.blocks: x=blk(x,mask)
        return self.proj_out(self.ln_out(x))

class ICLModelFull(nn.Module):
    def __init__(self,pilot_enc,bf_enc,bf_dec,cfg):
        super().__init__();self.pilot_enc=pilot_enc;self.bf_enc=bf_enc;self.bf_dec=bf_dec
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

def pretrain_pilot_edn(cfg,Phi):
    enc=PilotEncoder(cfg.N,cfg.L_p,cfg.D_tok,cfg.edn_hidden).to(device)
    dec=ChannelDecoder(cfg.K,cfg.N,cfg.D_tok,cfg.edn_hidden).to(device)
    opt=Adam(list(enc.parameters())+list(dec.parameters()),lr=cfg.edn_lr)
    sched=optim.lr_scheduler.CosineAnnealingLR(opt,cfg.edn_epochs,eta_min=0)
    print("\n[Phase 0a] Pretraining PilotEncoder...")
    for ep in range(cfg.edn_epochs):
        el=0.;nb=0
        for _ in range(cfg.edn_ds_size//cfg.edn_batch):
            H=generate_channel(cfg.edn_batch,cfg.K,cfg.N,cfg.ch_n_clusters,cfg.ch_n_rays,cfg.ch_spread_deg)
            Y=pilot_observe(H,Phi,cfg.sigma2)
            loss=F.mse_loss(dec(enc(pilot_to_real(Y),sigma2=cfg.sigma2)),channel_to_real(H))
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(list(enc.parameters())+list(dec.parameters()),5.0)
            opt.step();el+=loss.item();nb+=1
        sched.step()
        if (ep+1)%50==0 or ep==0: print(f"  Ep {ep+1}/{cfg.edn_epochs} MSE={el/nb:.6f}")
    return enc

def pretrain_bf_edn(cfg,Phi):
    enc=BFEncoder(cfg.N,cfg.K,cfg.D_tok,cfg.edn_hidden).to(device)
    dec=BFDecoder(cfg.N,cfg.K,cfg.D_tok,cfg.P_max,cfg.edn_hidden).to(device)
    opt=Adam(list(enc.parameters())+list(dec.parameters()),lr=cfg.edn_lr)
    sched=optim.lr_scheduler.CosineAnnealingLR(opt,cfg.edn_epochs,eta_min=0)
    print("\n[Phase 0b] Pretraining BFEncoder+BFDecoder...")
    for ep in range(cfg.edn_epochs):
        el=0.;nb=0
        for _ in range(cfg.edn_ds_size//cfg.edn_batch):
            H=generate_channel(cfg.edn_batch,cfg.K,cfg.N,cfg.ch_n_clusters,cfg.ch_n_rays,cfg.ch_spread_deg)
            W=mmse_beamformer(H,cfg.P_max,cfg.sigma2);Wr=bf_to_real(W)
            loss=F.mse_loss(bf_to_real(dec(enc(Wr,sigma2=cfg.sigma2))),Wr)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(list(enc.parameters())+list(dec.parameters()),5.0)
            opt.step();el+=loss.item();nb+=1
        sched.step()
        if (ep+1)%50==0 or ep==0: print(f"  Ep {ep+1}/{cfg.edn_epochs} MSE={el/nb:.6f}")
    return enc,dec

# ============================================================================
# DATASET (same as proposed, for self-bootstrapping)
# ============================================================================
class DynDataset:
    def __init__(self,max_sz=50000):
        self.max_sz=max_sz;self.H=self.Y_real=self.W_real=self.rates=self.mmse_rates=self.is_sup=None
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
            _,idx=torch.topk(self.rates,self.max_sz)
            for a in ['H','Y_real','W_real','rates','mmse_rates','is_sup']:setattr(self,a,getattr(self,a)[idx])
        self._n=self.H.size(0);self.n_sup=int(self.is_sup.sum().item());self.n_unsup=int((~self.is_sup).sum().item())
    def prune_unsup_bottom(self,drop_ratio,min_keep=0):
        if self._n==0 or drop_ratio<=0: return 0
        ui=torch.where(~self.is_sup)[0];nu=ui.numel()
        if nu==0: return 0
        nd=min(int(nu*drop_ratio),max(0,nu-min_keep))
        if nd<=0: return 0
        worst=ui[torch.topk(self.rates[ui],k=nd,largest=False).indices]
        keep=torch.ones(self._n,device=device,dtype=torch.bool);keep[worst]=False
        for a in ['H','Y_real','W_real','rates','mmse_rates','is_sup']:setattr(self,a,getattr(self,a)[keep])
        self._n=self.H.size(0);self.n_sup=int(self.is_sup.sum().item());self.n_unsup=int((~self.is_sup).sum().item())
        return nd

# ============================================================================
# A2b TRAINING: PURE UNSUPERVISED (sum-rate only, no MSE, no labeled data)
# ============================================================================
def train(cfg):
    set_global_seed(cfg.seed)
    print("="*70)
    print("  ABLATION A2b: Pure Unsupervised ICL (No Warm-Start)")
    print("="*70)

    Phi=generate_pilot_dft(cfg.K,cfg.L_p)
    pilot_enc=pretrain_pilot_edn(cfg,Phi); bf_enc,bf_dec=pretrain_bf_edn(cfg,Phi)
    for p in pilot_enc.parameters(): p.requires_grad_(False)
    for p in bf_enc.parameters(): p.requires_grad_(False)

    # Seed dataset: small MMSE BF solutions (cheap, no WMMSE needed)
    # These provide initial context for the ICL sequence
    print(f"\nSeed dataset ({cfg.seed_ds_size} MMSE BF samples)...")
    ds=DynDataset(max_sz=cfg.max_ds_size)
    gbs=min(64,cfg.seed_ds_size)
    for s in range(0,cfg.seed_ds_size,gbs):
        e=min(s+gbs,cfg.seed_ds_size)
        H=generate_channel(e-s,cfg.K,cfg.N,cfg.ch_n_clusters,cfg.ch_n_rays,cfg.ch_spread_deg)
        W=mmse_beamformer(H,cfg.P_max,cfg.sigma2); Y=pilot_observe(H,Phi,cfg.sigma2)
        with torch.no_grad(): r=compute_sum_rate(H,W,cfg.sigma2)
        ds.add(H,pilot_to_real(Y),bf_to_real(W),r,mmse_rates=r,supervised=True)

    H_test=generate_channel(cfg.n_test,cfg.K,cfg.N,cfg.ch_n_clusters,cfg.ch_n_rays,cfg.ch_spread_deg)
    bl=compute_baselines(H_test,Phi,cfg)
    print("\nBaselines:")
    for k,v in bl.items(): print(f"  {k}: {v:.4f}")

    model=ICLModelFull(pilot_enc,bf_enc,bf_dec,cfg).to(device)
    trainable=[p for p in model.parameters() if p.requires_grad]
    optimizer=AdamW(trainable,lr=cfg.lr,weight_decay=cfg.weight_decay)
    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,cfg.total_epochs,eta_min=cfg.lr_min)

    print("\n"+"="*70)
    best_test=0.0

    for epoch in range(cfg.total_epochs):
        model.train();t0=time.time();ep_rate=0.;ep_add=0;ep_n=0
        ep_prog=epoch/max(1,cfg.total_epochs-1)
        alpha_t=cfg.boot_alpha_start+(cfg.boot_alpha_end-cfg.boot_alpha_start)*ep_prog
        beta_t=cfg.boot_beta_start+(cfg.boot_beta_end-cfg.boot_beta_start)*ep_prog

        for step in range(cfg.steps_per_epoch):
            B=cfg.batch_size;l=cfg.n_demos
            # ALL queries unsupervised (fresh channels)
            q_H=generate_channel(B,cfg.K,cfg.N,cfg.ch_n_clusters,cfg.ch_n_rays,cfg.ch_spread_deg)
            q_Y=pilot_observe(q_H,Phi,cfg.sigma2);q_pil=pilot_to_real(q_Y)

            # Random context from dataset
            d_idx=torch.randint(0,ds.size,(B,l),device=device)
            dp=ds.Y_real[d_idx]; dw=ds.W_real[d_idx]

            H_ls=ls_channel_est(q_Y,Phi);W_base=mmse_beamformer(H_ls,cfg.P_max,cfg.sigma2)
            dW=model(dp,dw,q_pil,sigma2=cfg.sigma2)
            W_hat=power_normalize(W_base+dW,cfg.P_max)

            # PURE sum-rate loss (no MSE)
            rate=compute_sum_rate(q_H,W_hat,cfg.sigma2)
            loss=-rate.mean()*cfg.unsup_scale

            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable,5.0);optimizer.step()
            ep_rate+=rate.mean().item();ep_n+=1

            # Self-bootstrapping (same dual-threshold as proposed)
            with torch.no_grad():
                mmse_r=compute_sum_rate(q_H,mmse_beamformer(q_H,cfg.P_max,cfg.sigma2),cfg.sigma2)
                per_ok=rate>alpha_t*mmse_r
                cross_ok=rate>=torch.quantile(rate,beta_t) if rate.numel()>1 else torch.ones_like(rate,dtype=torch.bool)
                good=per_ok&cross_ok
                if good.any():
                    gi=torch.where(good)[0]
                    ds.add(q_H[gi],q_pil[gi],bf_to_real(W_hat[gi]),rate[gi],mmse_rates=mmse_r[gi],supervised=False)
                    ep_add+=good.sum().item()

        scheduler.step()
        nd=0
        if (epoch+1)%cfg.prune_every==0:
            dr=cfg.prune_drop_end*epoch/max(1,cfg.total_epochs-1)
            nd=ds.prune_unsup_bottom(dr,cfg.prune_min_unsup)

        model.eval();test_r=[]
        with torch.no_grad():
            for s in range(0,H_test.size(0),cfg.batch_size):
                e=min(s+cfg.batch_size,H_test.size(0));H=H_test[s:e];b=H.size(0)
                Y=pilot_observe(H,Phi,cfg.sigma2);pil=pilot_to_real(Y)
                d_idx=torch.randint(0,ds.size,(b,cfg.n_demos),device=device)
                dp=ds.Y_real[d_idx];dw=ds.W_real[d_idx]
                H_ls=ls_channel_est(Y,Phi);W_base=mmse_beamformer(H_ls,cfg.P_max,cfg.sigma2)
                dW=model(dp,dw,pil,sigma2=cfg.sigma2)
                W_hat=power_normalize(W_base+dW,cfg.P_max)
                test_r.append(compute_sum_rate(H,W_hat,cfg.sigma2))
        tr=torch.cat(test_r).mean().item();best_test=max(best_test,tr)
        model.train();dt=time.time()-t0
        print(f"{epoch+1:3d} | rate={ep_rate/max(1,ep_n):.2f} | test={tr:.3f} best={best_test:.3f} | DS={ds.size} +{ep_add} -{nd} ({dt:.1f}s)",flush=True)

    b4=bl['wmmse_imp']
    print("\n"+"="*68+"\n  A2b COMPLETE (Pure Unsupervised)")
    for k,v in [("B1",bl['mmse_perf']),("B2",bl['mmse_imp']),("B3",bl['wmmse_perf']),("B4",b4),("A2b best",best_test)]:
        print(f"  {k:<20} {v:8.3f}  ({100*v/b4:.1f}%)" if b4>0 else f"  {k} {v}")
    return model,bl

if __name__=="__main__":
    cfg=Config(K=32,N=32,L_p=20,SNR_dB=20,D_tok=256,edn_hidden=128,edn_epochs=200,
               n_demos=5,d_model=512,n_heads=8,n_layers=6,d_ff=1024,
               seed_ds_size=200,total_epochs=550,steps_per_epoch=80,batch_size=64,lr=2e-4,
               max_ds_size=50000,n_test=200,edn_ds_size=5000)
    train(cfg)
