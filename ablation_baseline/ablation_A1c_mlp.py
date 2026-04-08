"""
ABLATION A1c: No-Transformer Baseline — Equal-Parameter MLP

PURPOSE: Prove that the Transformer's sequence-processing capability (not just
parameter count) is essential for ICL. This baseline replaces the ICL Transformer
with a 5-layer MLP of matched parameter count (~13M), processing ONLY the
compressed query pilot embedding z — no context, no demos.

ARCHITECTURE:
  PilotEncoder(frozen) → z ∈ R^{D_tok} → MLP(~13M params) → c_hat → BFDecoder → W_hat

TRAINING: Same curriculum (Phase 1 MSE + Phase 2 sum-rate). No self-bootstrapping.

EXPECTED RESULT: Worse than ICL Transformer, confirming that sequence-level
attention over demo pairs is what enables ICL, not raw model capacity.
"""

import math, torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
import numpy as np
from typing import Dict, List
import warnings, time
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_and_plot_rates(train_rates, test_rates, tag):
    train_path = f"{tag}_train_rate.pt"
    test_path = f"{tag}_test_rate.pt"
    fig_path = f"{tag}_train_test_rate_vs_epochs.png"

    torch.save(torch.tensor(train_rates, dtype=torch.float32), train_path)
    torch.save(torch.tensor(test_rates, dtype=torch.float32), test_path)

    train_loaded = torch.load(train_path, map_location='cpu').cpu().numpy()
    test_loaded = torch.load(test_path, map_location='cpu').cpu().numpy()
    epochs = np.arange(1, len(train_loaded) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loaded, label='Train Rate', linewidth=2)
    plt.plot(epochs, test_loaded, label='Test Rate', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Sum Rate')
    plt.title(f'{tag}: Train/Test Rate vs Epochs')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved: {train_path}, {test_path}, {fig_path}")

def set_global_seed(seed, det=False):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# ============================================================================
# SHARED UTILITIES
# ============================================================================
class Config:
    def __init__(self, **kw):
        self.seed=kw.get('seed',2026); self.K=kw.get('K',32); self.N=kw.get('N',32)
        self.L_p=kw.get('L_p',20); self.P_max=kw.get('P_max',1.0)
        self.SNR_dB=kw.get('SNR_dB',20); self.sigma2=self.P_max/(10**(self.SNR_dB/10))
        self.ch_n_clusters=kw.get('ch_n_clusters',3); self.ch_n_rays=kw.get('ch_n_rays',5)
        self.ch_spread_deg=kw.get('ch_spread_deg',10.0)
        self.D_tok=kw.get('D_tok',256); self.edn_hidden=kw.get('edn_hidden',512)
        self.edn_epochs=kw.get('edn_epochs',200); self.edn_lr=kw.get('edn_lr',1e-3)
        self.edn_batch=kw.get('edn_batch',128)
        self.mlp_hidden=kw.get('mlp_hidden',2048)  # sized to match TF param count
        self.batch_size=kw.get('batch_size',64); self.lr=kw.get('lr',2e-4)
        self.lr_min=kw.get('lr_min',5e-5); self.weight_decay=kw.get('weight_decay',1e-4)
        self.ds_size=kw.get('ds_size',5000); self.wmmse_iters=kw.get('wmmse_iters',500)
        self.wmmse_lr=kw.get('wmmse_lr',0.03); self.unsup_scale=kw.get('unsup_scale',0.01)
        self.phase1_epochs=kw.get('phase1_epochs',50)
        self.phase2_epochs=kw.get('phase2_epochs',500)
        self.total_epochs=self.phase1_epochs+self.phase2_epochs
        self.steps_per_epoch=kw.get('steps_per_epoch',80)
        self.n_test=kw.get('n_test',200)

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
# PILOT ENCODER (pretrained, frozen) — same as proposed
# ============================================================================
class FiLMLayer(nn.Module):
    def __init__(self, n_ch):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(1,64),nn.GELU(),nn.Linear(64,2*n_ch))
    def forward(self,x,sigma2):
        B=x.size(0); C=x.size(1)
        log_s=torch.full((B,1),math.log(sigma2+1e-10),device=x.device)
        p=self.net(log_s); g=p[:,:C]; b=p[:,C:]
        if x.dim()==3: g=g.unsqueeze(-1); b=b.unsqueeze(-1)
        return g*x+b

class PilotEncoder(nn.Module):
    def __init__(self,N,L_p,D_tok,hidden=512):
        super().__init__()
        self.N,self.L_p=N,L_p
        self.conv1=nn.Conv1d(2*N,hidden,3,padding=1); self.conv2=nn.Conv1d(hidden,hidden,3,padding=1)
        self.film=FiLMLayer(hidden); self.ln=nn.LayerNorm(hidden)
        self.attn_q=nn.Parameter(torch.randn(1,1,hidden)*0.02)
        self.attn_k=nn.Linear(hidden,hidden); self.attn_v=nn.Linear(hidden,hidden)
        self.proj=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,D_tok))
    def forward(self,x,sigma2=None):
        B=x.size(0); x=x.view(B,2*self.N,self.L_p)
        x=F.gelu(self.conv1(x)); x=F.gelu(self.conv2(x))
        if sigma2 is not None: x=self.film(x,sigma2)
        x=self.ln(x.transpose(1,2))
        q=self.attn_q.expand(B,-1,-1); k,v=self.attn_k(x),self.attn_v(x)
        w=F.softmax(torch.bmm(q,k.transpose(1,2))/math.sqrt(k.size(-1)),-1)
        return self.proj(torch.bmm(w,v).squeeze(1))

class ChannelDecoder(nn.Module):
    def __init__(self,K,N,D_tok,hidden=512):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(D_tok,hidden),nn.GELU(),
                               nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,2*K*N))
    def forward(self,z): return self.net(z)

class BFDecoder(nn.Module):
    def __init__(self,N,K,D_tok,P_max,hidden=512):
        super().__init__()
        self.N,self.K,self.P_max=N,K,P_max
        self.net=nn.Sequential(nn.Linear(D_tok,hidden),nn.GELU(),
                               nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,2*N*K))
    def forward(self,c,normalize=True):
        x=self.net(c); W=real_to_bf(x,self.N,self.K)
        return power_normalize(W,self.P_max) if normalize else W

# ============================================================================
# PRETRAIN PILOT ENCODER (Phase 0a — same as proposed)
# ============================================================================
def pretrain_pilot_enc(cfg, Phi):
    print("\n[Phase 0a] Pretraining PilotEncoder...")
    enc=PilotEncoder(cfg.N,cfg.L_p,cfg.D_tok,cfg.edn_hidden).to(device)
    dec=ChannelDecoder(cfg.K,cfg.N,cfg.D_tok,cfg.edn_hidden).to(device)
    opt=Adam(list(enc.parameters())+list(dec.parameters()),lr=cfg.edn_lr)
    sched=optim.lr_scheduler.CosineAnnealingLR(opt,cfg.edn_epochs,eta_min=0)
    for ep in range(cfg.edn_epochs):
        el=0.; nb=0
        for _ in range(cfg.ds_size//cfg.edn_batch):
            H=generate_channel(cfg.edn_batch,cfg.K,cfg.N,cfg.ch_n_clusters,cfg.ch_n_rays,cfg.ch_spread_deg)
            Y=pilot_observe(H,Phi,cfg.sigma2); Yr=pilot_to_real(Y); Hr=channel_to_real(H)
            loss=F.mse_loss(dec(enc(Yr,sigma2=cfg.sigma2)),Hr)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters())+list(dec.parameters()),5.0)
            opt.step(); el+=loss.item(); nb+=1
        sched.step()
        if (ep+1)%50==0 or ep==0: print(f"  Ep {ep+1}/{cfg.edn_epochs} MSE={el/nb:.6f}")
    return enc

# ============================================================================
# A1c MODEL: Equal-Parameter MLP (no Transformer, no ICL)
# ============================================================================
class MLPModel(nn.Module):
    """
    Replaces ICL Transformer with a pure MLP of matched parameter count.
    Input: z (D_tok) from frozen PilotEncoder → MLP → c_hat (D_tok) → BFDecoder (direct W).
    """
    def __init__(self, pilot_enc, bf_dec, cfg):
        super().__init__()
        self.pilot_enc = pilot_enc  # frozen
        self.bf_dec = bf_dec        # trainable
        h = cfg.mlp_hidden
        D = cfg.D_tok
        # 5-layer MLP: D→h→h→h→h→D (param count ≈ Dh + 3h² + hD ≈ 13M for h=2048)
        self.mlp = nn.Sequential(
            nn.Linear(D, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(),
            nn.Linear(h, D),
        )
        n_mlp = sum(p.numel() for p in self.mlp.parameters())
        n_dec = sum(p.numel() for p in self.bf_dec.parameters())
        print(f"[A1c] MLP params: {n_mlp:,}, BFDecoder: {n_dec:,}, total trainable: {n_mlp+n_dec:,}")

    def forward(self, query_pil_real, sigma2=None):
        with torch.no_grad():
            z = self.pilot_enc(query_pil_real, sigma2=sigma2)
        c_hat = self.mlp(z)
        return self.bf_dec(c_hat, normalize=True)  # returns direct W_hat


# ============================================================================
# TRAINING (no ICL, no context, no bootstrapping)
# ============================================================================
def train(cfg):
    set_global_seed(cfg.seed)
    print("="*70)
    print("  ABLATION A1c: Equal-Parameter MLP (No Transformer, No ICL)")
    print("="*70)

    Phi = generate_pilot_dft(cfg.K, cfg.L_p)

    # Pretrain pilot encoder
    pilot_enc = pretrain_pilot_enc(cfg, Phi)
    for p in pilot_enc.parameters(): p.requires_grad_(False)

    # BF decoder (fresh, trainable)
    bf_dec = BFDecoder(cfg.N, cfg.K, cfg.D_tok, cfg.P_max, cfg.edn_hidden).to(device)

    # Labeled dataset
    print(f"\nGenerating labeled dataset ({cfg.ds_size} samples)...")
    H_all,W_all,Y_all,R_all=[],[],[],[]
    gbs=min(64,cfg.ds_size)
    for s in range(0,cfg.ds_size,gbs):
        e=min(s+gbs,cfg.ds_size)
        H=generate_channel(e-s,cfg.K,cfg.N,cfg.ch_n_clusters,cfg.ch_n_rays,cfg.ch_spread_deg)
        W_star=mmse_beamformer(H,cfg.P_max,cfg.sigma2)
        Y=pilot_observe(H,Phi,cfg.sigma2)
        with torch.no_grad(): r=compute_sum_rate(H,W_star,cfg.sigma2)
        H_all.append(H); W_all.append(bf_to_real(W_star)); Y_all.append(pilot_to_real(Y)); R_all.append(r)
    H_ds=torch.cat(H_all); W_ds=torch.cat(W_all); Y_ds=torch.cat(Y_all)

    H_test=generate_channel(cfg.n_test,cfg.K,cfg.N,cfg.ch_n_clusters,cfg.ch_n_rays,cfg.ch_spread_deg)
    print("\nComputing baselines...")
    bl=compute_baselines(H_test,Phi,cfg)
    for k,v in bl.items(): print(f"  {k}: {v:.4f}")

    model=MLPModel(pilot_enc,bf_dec,cfg).to(device)
    trainable=[p for p in model.parameters() if p.requires_grad]
    optimizer=AdamW(trainable,lr=cfg.lr,weight_decay=cfg.weight_decay)
    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,cfg.total_epochs,eta_min=cfg.lr_min)

    print("\n"+"="*70)
    best_test=0.0
    train_hist=[]
    test_hist=[]

    for epoch in range(cfg.total_epochs):
        model.train(); t0=time.time()
        phase = 1 if epoch < cfg.phase1_epochs else 2
        ep_mse,ep_rate,ep_n=0.,0.,0

        for step in range(cfg.steps_per_epoch):
            B=cfg.batch_size
            if phase==1:
                idx=torch.randint(0,H_ds.size(0),(B,),device=device)
                q_H=H_ds[idx]; q_W_gt=W_ds[idx]; q_pil=Y_ds[idx]
            else:
                q_H=generate_channel(B,cfg.K,cfg.N,cfg.ch_n_clusters,cfg.ch_n_rays,cfg.ch_spread_deg)
                q_Y=pilot_observe(q_H,Phi,cfg.sigma2); q_pil=pilot_to_real(q_Y)
                idx=torch.randint(0,H_ds.size(0),(B,),device=device); q_W_gt=W_ds[idx]

            W_hat=model(q_pil,sigma2=cfg.sigma2)
            rate=compute_sum_rate(q_H,W_hat,cfg.sigma2)
            ep_rate+=rate.mean().item()

            if phase==1:
                loss=F.mse_loss(bf_to_real(W_hat),q_W_gt,reduction='none').sum(-1).mean()
                ep_mse+=loss.item()
            else:
                loss=-rate.mean()*cfg.unsup_scale
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable,5.0); optimizer.step(); ep_n+=1

        scheduler.step()
        model.eval()
        test_r=[]
        with torch.no_grad():
            for s in range(0,H_test.size(0),cfg.batch_size):
                e=min(s+cfg.batch_size,H_test.size(0)); H=H_test[s:e]
                Y=pilot_observe(H,Phi,cfg.sigma2); pil=pilot_to_real(Y)
                W_hat=model(pil,sigma2=cfg.sigma2)
                test_r.append(compute_sum_rate(H,W_hat,cfg.sigma2))
                tr=torch.cat(test_r).mean().item(); best_test=max(best_test,tr)
                train_hist.append(ep_rate/max(1,ep_n))
                test_hist.append(tr)
        model.train(); dt=time.time()-t0
        print(f"{epoch+1:3d} ph{phase} | mse={ep_mse/max(1,ep_n):.5f} rate={ep_rate/max(1,ep_n):.2f} | "
              f"test={tr:.3f} best={best_test:.3f} ({dt:.1f}s)",flush=True)

    b4=bl['wmmse_imp']
    print("\n"+"="*68+"\n  A1c COMPLETE (MLP, No ICL)")
    for k,v in [("B1",bl['mmse_perf']),("B2",bl['mmse_imp']),("B3",bl['wmmse_perf']),
                ("B4",b4),("A1c best",best_test)]:
        print(f"  {k:<20} {v:8.3f}  ({100*v/b4:.1f}%)" if b4>0 else f"  {k} {v}")

    save_and_plot_rates(train_hist, test_hist, tag='a1c_mlp_no_icl')
    return model,bl

if __name__=="__main__":
    cfg=Config(K=32,N=32,L_p=20,SNR_dB=20,D_tok=256,edn_hidden=512,edn_epochs=200,
               mlp_hidden=2048,ds_size=5000,phase1_epochs=50,phase2_epochs=500,
               steps_per_epoch=80,batch_size=64,lr=2e-4,n_test=200)
    train(cfg)
