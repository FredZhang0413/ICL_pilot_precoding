"""
ABLATION A2a: Pure Supervised ICL (No Unsupervised, No Self-Bootstrapping)

PURPOSE: Show that supervised MSE alone cannot discover solutions beyond label
quality. The model can at best replicate the MMSE BF labels, never outperform them.

SETUP: Full ICL architecture (identical to proposed). Large labeled dataset
(same size as the final evolved dataset of the proposed method). Training uses
ONLY MSE loss on labeled samples for the entire duration. No sum-rate loss,
no self-bootstrapping, no curriculum transition.

EXPECTED RESULT: Performance plateaus at/below the MMSE BF + Perfect CSI level,
since MSE training can only learn to reproduce labels, not improve beyond them.
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
# SHARED UTILITIES (identical to proposed)
# ============================================================================
class Config:
    def __init__(self,**kw):
        self.seed=kw.get('seed',2026); self.K=kw.get('K',32); self.N=kw.get('N',32)
        self.L_p=kw.get('L_p',20); self.P_max=kw.get('P_max',1.0)
        self.SNR_dB=kw.get('SNR_dB',20); self.sigma2=self.P_max/(10**(self.SNR_dB/10))
        self.ch_n_clusters=kw.get('ch_n_clusters',3); self.ch_n_rays=kw.get('ch_n_rays',5)
        self.ch_spread_deg=kw.get('ch_spread_deg',10.0)
        self.D_tok=kw.get('D_tok',256); self.edn_hidden=kw.get('edn_hidden',512)
        self.edn_epochs=kw.get('edn_epochs',200); self.edn_lr=kw.get('edn_lr',1e-3)
        self.edn_batch=kw.get('edn_batch',128)
        self.n_demos=kw.get('n_demos',5)
        self.d_model=kw.get('d_model',512); self.n_heads=kw.get('n_heads',8)
        self.n_layers=kw.get('n_layers',6); self.d_ff=kw.get('d_ff',1024)
        self.dropout=kw.get('dropout',0.0)
        self.batch_size=kw.get('batch_size',64); self.lr=kw.get('lr',2e-4)
        self.lr_min=kw.get('lr_min',5e-5); self.weight_decay=kw.get('weight_decay',1e-4)
        # Large labeled dataset — matches final evolved dataset size of proposed method
        self.ds_size=kw.get('ds_size',10000)
        self.total_epochs=kw.get('total_epochs',550)
        self.steps_per_epoch=kw.get('steps_per_epoch',80)
        self.n_test=kw.get('n_test',200)
        self.wmmse_iters=kw.get('wmmse_iters',500); self.wmmse_lr=kw.get('wmmse_lr',0.03)

def generate_channel(B,K,N,n_cl=3,n_ray=5,spread=10.0):
    L=n_cl*n_ray; asp=math.radians(spread)
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
    B,K,N=H.shape; Lp=Phi.size(1)
    Y=H.transpose(-1,-2)@Phi.unsqueeze(0).expand(B,-1,-1)
    nr=torch.randn(B,N,Lp,device=device)*math.sqrt(sigma2/2)
    ni=torch.randn(B,N,Lp,device=device)*math.sqrt(sigma2/2)
    return Y+torch.complex(nr,ni)

def pilot_to_real(Y): return torch.cat([Y.real,Y.imag],dim=1).reshape(Y.size(0),-1)
def bf_to_real(W): return torch.cat([W.real,W.imag],dim=1).reshape(W.size(0),-1)
def real_to_bf(x,N,K):
    B=x.size(0); x=x.view(B,2*N,K); return torch.complex(x[:,:N,:],x[:,N:,:])
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

def generate_wmmse_labels(H,P_max,sigma2,n_iters=500,lr=0.03,n_restarts=2):
    B,K,N=H.shape; Hd=H.detach()
    best_r=torch.full((B,),-float('inf'),device=device)
    best_W=torch.zeros(B,N,K,device=device,dtype=torch.cfloat)
    for _ in range(n_restarts):
        Wr=(torch.randn(B,N,K,device=device)*0.05).requires_grad_(True)
        Wi=(torch.randn(B,N,K,device=device)*0.05).requires_grad_(True)
        opt=Adam([Wr,Wi],lr=lr)
        for _ in range(n_iters):
            W=power_normalize(torch.complex(Wr,Wi),P_max)
            (-compute_sum_rate(Hd,W,sigma2).sum()).backward(); opt.step(); opt.zero_grad()
        with torch.no_grad():
            Ws=power_normalize(torch.complex(Wr,Wi),P_max); rs=compute_sum_rate(Hd,Ws,sigma2)
            imp=rs>best_r
            if imp.any(): best_r[imp]=rs[imp]; best_W[imp]=Ws[imp]
    return best_W.detach(), best_r.detach()

def compute_baselines(H_test,Phi,cfg):
    B=H_test.size(0); s2=cfg.sigma2; Pm=cfg.P_max; bs=min(64,B)
    res={k:[] for k in ['mmse_perf','mmse_imp','wmmse_perf','wmmse_imp']}
    for s in range(0,B,bs):
        e=min(s+bs,B); H=H_test[s:e]
        Y=pilot_observe(H,Phi,s2); Hh=mmse_channel_est(Y,Phi,s2)
        with torch.no_grad():
            res['mmse_perf'].append(compute_sum_rate(H,mmse_beamformer(H,Pm,s2),s2))
            res['mmse_imp'].append(compute_sum_rate(H,mmse_beamformer(Hh,Pm,s2),s2))
        W3,_=generate_wmmse_labels(H,Pm,s2,n_restarts=2)
        with torch.no_grad(): res['wmmse_perf'].append(compute_sum_rate(H,W3,s2))
        W4,_=generate_wmmse_labels(Hh,Pm,s2,n_restarts=2)
        with torch.no_grad(): res['wmmse_imp'].append(compute_sum_rate(H,W4,s2))
    return {k:torch.cat(v).mean().item() for k,v in res.items()}

# ============================================================================
# FULL ICL MODEL (identical to proposed)
# ============================================================================
class FiLMLayer(nn.Module):
    def __init__(self,n_ch):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(1,64),nn.GELU(),nn.Linear(64,2*n_ch))
    def forward(self,x,sigma2):
        B=x.size(0);C=x.size(1)
        p=self.net(torch.full((B,1),math.log(sigma2+1e-10),device=x.device))
        g=p[:,:C];b=p[:,C:]
        if x.dim()==3: g=g.unsqueeze(-1);b=b.unsqueeze(-1)
        return g*x+b

class PilotEncoder(nn.Module):
    def __init__(self,N,L_p,D_tok,hidden=512):
        super().__init__()
        self.N,self.L_p=N,L_p
        self.conv1=nn.Conv1d(2*N,hidden,3,padding=1);self.conv2=nn.Conv1d(hidden,hidden,3,padding=1)
        self.film=FiLMLayer(hidden);self.ln=nn.LayerNorm(hidden)
        self.attn_q=nn.Parameter(torch.randn(1,1,hidden)*0.02)
        self.attn_k=nn.Linear(hidden,hidden);self.attn_v=nn.Linear(hidden,hidden)
        self.proj=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,D_tok))
    def forward(self,x,sigma2=None):
        B=x.size(0);x=x.view(B,2*self.N,self.L_p)
        x=F.gelu(self.conv1(x));x=F.gelu(self.conv2(x))
        if sigma2 is not None: x=self.film(x,sigma2)
        x=self.ln(x.transpose(1,2))
        q=self.attn_q.expand(B,-1,-1);k,v=self.attn_k(x),self.attn_v(x)
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
        h=self.net(W_real)
        if sigma2 is not None: h=self.film(h,sigma2)
        return self.proj(h)

class BFDecoder(nn.Module):
    def __init__(self,N,K,D_tok,P_max,hidden=512):
        super().__init__()
        self.N,self.K,self.P_max=N,K,P_max
        self.net=nn.Sequential(nn.Linear(D_tok,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,2*N*K))
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

class ICLModelFull(nn.Module):
    """Full ICL model — identical to proposed, used by A2a/A2b/A2c."""
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

# ============================================================================
# PRETRAIN ENCODERS (Phase 0 — identical to proposed)
# ============================================================================
def pretrain_pilot_edn(cfg,Phi):
    print("\n[Phase 0a] Pretraining PilotEncoder...")
    enc=PilotEncoder(cfg.N,cfg.L_p,cfg.D_tok,cfg.edn_hidden).to(device)
    dec=ChannelDecoder(cfg.K,cfg.N,cfg.D_tok,cfg.edn_hidden).to(device)
    opt=Adam(list(enc.parameters())+list(dec.parameters()),lr=cfg.edn_lr)
    sched=optim.lr_scheduler.CosineAnnealingLR(opt,cfg.edn_epochs,eta_min=0)
    for ep in range(cfg.edn_epochs):
        el=0.;nb=0
        for _ in range(cfg.ds_size//cfg.edn_batch):
            H=generate_channel(cfg.edn_batch,cfg.K,cfg.N,cfg.ch_n_clusters,cfg.ch_n_rays,cfg.ch_spread_deg)
            Y=pilot_observe(H,Phi,cfg.sigma2)
            loss=F.mse_loss(dec(enc(pilot_to_real(Y),sigma2=cfg.sigma2)),channel_to_real(H))
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(list(enc.parameters())+list(dec.parameters()),5.0)
            opt.step();el+=loss.item();nb+=1
        sched.step()
        if (ep+1)%50==0 or ep==0: print(f"  Ep {ep+1}/{cfg.edn_epochs} MSE={el/nb:.6f}")
    return enc

def pretrain_bf_edn(cfg,Phi):
    print("\n[Phase 0b] Pretraining BFEncoder+BFDecoder...")
    enc=BFEncoder(cfg.N,cfg.K,cfg.D_tok,cfg.edn_hidden).to(device)
    dec=BFDecoder(cfg.N,cfg.K,cfg.D_tok,cfg.P_max,cfg.edn_hidden).to(device)
    opt=Adam(list(enc.parameters())+list(dec.parameters()),lr=cfg.edn_lr)
    sched=optim.lr_scheduler.CosineAnnealingLR(opt,cfg.edn_epochs,eta_min=0)
    for ep in range(cfg.edn_epochs):
        el=0.;nb=0
        for _ in range(cfg.ds_size//cfg.edn_batch):
            H=generate_channel(cfg.edn_batch,cfg.K,cfg.N,cfg.ch_n_clusters,cfg.ch_n_rays,cfg.ch_spread_deg)
            W=mmse_beamformer(H,cfg.P_max,cfg.sigma2);Wr=bf_to_real(W)
            loss=F.mse_loss(bf_to_real(dec(enc(Wr,sigma2=cfg.sigma2))),Wr)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(list(enc.parameters())+list(dec.parameters()),5.0)
            opt.step();el+=loss.item();nb+=1
        sched.step()
        if (ep+1)%50==0 or ep==0: print(f"  Ep {ep+1}/{cfg.edn_epochs} MSE={el/nb:.6f}")
    return enc,dec

# ============================================================================
# A2a TRAINING: PURE SUPERVISED (MSE only, no sum-rate, no bootstrapping)
# ============================================================================
def train(cfg):
    set_global_seed(cfg.seed)
    print("="*70)
    print("  ABLATION A2a: Pure Supervised ICL (MSE only)")
    print("="*70)

    Phi=generate_pilot_dft(cfg.K,cfg.L_p)
    pilot_enc=pretrain_pilot_edn(cfg,Phi); bf_enc,bf_dec=pretrain_bf_edn(cfg,Phi)
    for p in pilot_enc.parameters(): p.requires_grad_(False)
    for p in bf_enc.parameters(): p.requires_grad_(False)

    # Large labeled dataset (MMSE BF on perfect CSI)
    print(f"\nGenerating LARGE labeled dataset ({cfg.ds_size} samples)...")
    H_all,W_all,Y_all=[],[],[]
    gbs=min(64,cfg.ds_size)
    for s in range(0,cfg.ds_size,gbs):
        e=min(s+gbs,cfg.ds_size)
        H=generate_channel(e-s,cfg.K,cfg.N,cfg.ch_n_clusters,cfg.ch_n_rays,cfg.ch_spread_deg)
        W=mmse_beamformer(H,cfg.P_max,cfg.sigma2)
        Y=pilot_observe(H,Phi,cfg.sigma2)
        H_all.append(H);W_all.append(bf_to_real(W));Y_all.append(pilot_to_real(Y))
        if e%1000==0: print(f"  [{e}/{cfg.ds_size}]",flush=True)
    H_ds=torch.cat(H_all);W_ds=torch.cat(W_all);Y_ds=torch.cat(Y_all)
    print(f"  Dataset: {H_ds.size(0)} samples")

    H_test=generate_channel(cfg.n_test,cfg.K,cfg.N,cfg.ch_n_clusters,cfg.ch_n_rays,cfg.ch_spread_deg)
    print("\nComputing baselines...")
    bl=compute_baselines(H_test,Phi,cfg)
    for k,v in bl.items(): print(f"  {k}: {v:.4f}")

    model=ICLModelFull(pilot_enc,bf_enc,bf_dec,cfg).to(device)
    trainable=[p for p in model.parameters() if p.requires_grad]
    optimizer=AdamW(trainable,lr=cfg.lr,weight_decay=cfg.weight_decay)
    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,cfg.total_epochs,eta_min=cfg.lr_min)

    print("\n"+"="*70)
    best_test=0.0

    for epoch in range(cfg.total_epochs):
        model.train(); t0=time.time(); ep_mse=0.; ep_n=0

        for step in range(cfg.steps_per_epoch):
            B=cfg.batch_size; l=cfg.n_demos
            # ALL queries are supervised — sample from labeled dataset
            idx=torch.randint(0,H_ds.size(0),(B,),device=device)
            q_H=H_ds[idx]; q_W_gt=W_ds[idx]; q_pil=Y_ds[idx]

            # Random context (from same labeled dataset)
            d_idx=torch.randint(0,H_ds.size(0),(B,l),device=device)
            dp=Y_ds[d_idx]; dw=W_ds[d_idx]

            q_Y=pilot_observe(q_H,Phi,cfg.sigma2)
            H_ls=ls_channel_est(q_Y,Phi)
            W_base=mmse_beamformer(H_ls,cfg.P_max,cfg.sigma2)
            dW=model(dp,dw,q_pil,sigma2=cfg.sigma2)
            W_hat=power_normalize(W_base+dW,cfg.P_max)

            # PURE MSE loss — no sum-rate
            loss=F.mse_loss(bf_to_real(W_hat),q_W_gt,reduction='none').sum(-1).mean()

            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable,5.0);optimizer.step()
            ep_mse+=loss.item();ep_n+=1

        scheduler.step()

        # Evaluate
        model.eval(); test_r=[]
        with torch.no_grad():
            for s in range(0,H_test.size(0),cfg.batch_size):
                e=min(s+cfg.batch_size,H_test.size(0));H=H_test[s:e];b=H.size(0)
                Y=pilot_observe(H,Phi,cfg.sigma2);pil=pilot_to_real(Y)
                d_idx=torch.randint(0,H_ds.size(0),(b,cfg.n_demos),device=device)
                dp=Y_ds[d_idx];dw=W_ds[d_idx]
                H_ls=ls_channel_est(Y,Phi);W_base=mmse_beamformer(H_ls,cfg.P_max,cfg.sigma2)
                dW=model(dp,dw,pil,sigma2=cfg.sigma2)
                W_hat=power_normalize(W_base+dW,cfg.P_max)
                test_r.append(compute_sum_rate(H,W_hat,cfg.sigma2))
        tr=torch.cat(test_r).mean().item();best_test=max(best_test,tr)
        model.train();dt=time.time()-t0
        print(f"{epoch+1:3d} | mse={ep_mse/max(1,ep_n):.5f} | test={tr:.3f} best={best_test:.3f} ({dt:.1f}s)",flush=True)

    b4=bl['wmmse_imp']
    print("\n"+"="*68+"\n  A2a COMPLETE (Pure Supervised ICL)")
    for k,v in [("B1",bl['mmse_perf']),("B2",bl['mmse_imp']),("B3",bl['wmmse_perf']),
                ("B4",b4),("A2a best",best_test)]:
        print(f"  {k:<20} {v:8.3f}  ({100*v/b4:.1f}%)" if b4>0 else f"  {k} {v}")
    return model,bl

if __name__=="__main__":
    cfg=Config(K=32,N=32,L_p=20,SNR_dB=20,D_tok=256,edn_hidden=128,edn_epochs=200,
               n_demos=5,d_model=512,n_heads=8,n_layers=6,d_ff=1024,
               ds_size=10000,total_epochs=550,steps_per_epoch=80,batch_size=64,lr=1e-4,n_test=200)
    train(cfg)
