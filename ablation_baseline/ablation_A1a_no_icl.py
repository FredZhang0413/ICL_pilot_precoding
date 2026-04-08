"""
ABLATION A1a: No-ICL Baseline — Direct Pilot-to-Beamformer Transformer

PURPOSE: Prove that ICL (demonstration context) is necessary.
This baseline uses the SAME Transformer architecture but processes ONLY the
query pilot signal — no demo pairs, no context. The pilot observation Y is
treated as a natural token sequence: N antenna positions × L_p pilot slots.

ARCHITECTURE:
  Y ∈ C^{N×L_p} → real-valued (2N, L_p) → N tokens of dim 2*L_p
  → Transformer (same depth/heads as proposed) → linear output head → W_hat

TRAINING: Same curriculum (Phase 1 MSE + Phase 2 sum-rate).
No self-bootstrapping (no context to enrich).

EXPECTED RESULT: Significantly worse than ICL model, proving that context
resolves the ill-posedness of the pilot→BF mapping.
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

def set_global_seed(seed, det=False):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if det: torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

# ============================================================================
# SHARED UTILITIES (identical to proposed method)
# ============================================================================
class Config:
    def __init__(self, **kw):
        self.seed=kw.get('seed',2026); self.K=kw.get('K',32); self.N=kw.get('N',32)
        self.L_p=kw.get('L_p',20); self.P_max=kw.get('P_max',1.0)
        self.SNR_dB=kw.get('SNR_dB',20); self.sigma2=self.P_max/(10**(self.SNR_dB/10))
        self.ch_n_clusters=kw.get('ch_n_clusters',3); self.ch_n_rays=kw.get('ch_n_rays',5)
        self.ch_spread_deg=kw.get('ch_spread_deg',10.0)
        # Transformer (matched to proposed ICL model)
        self.d_model=kw.get('d_model',512); self.n_heads=kw.get('n_heads',8)
        self.n_layers=kw.get('n_layers',6); self.d_ff=kw.get('d_ff',1024)
        self.dropout=kw.get('dropout',0.0)
        # Training
        self.batch_size=kw.get('batch_size',64); self.lr=kw.get('lr',2e-4)
        self.lr_min=kw.get('lr_min',5e-5); self.weight_decay=kw.get('weight_decay',1e-4)
        self.ds_size=kw.get('ds_size',5000); self.wmmse_iters=kw.get('wmmse_iters',500)
        self.wmmse_lr=kw.get('wmmse_lr',0.03); self.unsup_scale=kw.get('unsup_scale',0.01)
        self.phase1_epochs=kw.get('phase1_epochs',50)
        self.phase2_epochs=kw.get('phase2_epochs',500)
        self.total_epochs=self.phase1_epochs+self.phase2_epochs
        self.steps_per_epoch=kw.get('steps_per_epoch',80)
        self.n_test=kw.get('n_test',200)
        self.rate_save_every=kw.get('rate_save_every',10)
        self.train_rate_pt=kw.get('train_rate_pt','a1a_training_rate.pt')
        self.train_curve_png=kw.get('train_curve_png','a1a_training_curve.png')

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

def bf_to_real(W): return torch.cat([W.real,W.imag],dim=1).reshape(W.size(0),-1)
def real_to_bf(x,N,K):
    B=x.size(0); x=x.view(B,2*N,K); return torch.complex(x[:,:N,:],x[:,N:,:])

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
    Phi_pinv=torch.linalg.pinv(Phi); return (Y@Phi_pinv.unsqueeze(0)).transpose(-1,-2).contiguous()

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
# A1a MODEL: Direct Pilot-to-BF Transformer (NO ICL, NO CONTEXT)
# ============================================================================
class CausalBlock(nn.Module):
    """Standard Transformer block (same as proposed)."""
    def __init__(self, d, heads, d_ff, drop=0.0):
        super().__init__()
        self.ln1=nn.LayerNorm(d); self.attn=nn.MultiheadAttention(d,heads,dropout=drop,batch_first=True)
        self.ln2=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,d_ff),nn.GELU(),nn.Linear(d_ff,d),nn.Dropout(drop))
    def forward(self,x,mask=None):
        h=self.ln1(x); x=x+self.attn(h,h,h,attn_mask=mask)[0]
        return x+self.ff(self.ln2(x))


class DirectPilotBFModel(nn.Module):
    """
    No-ICL baseline: Transformer processes pilot Y as a token sequence.

    Input: Y ∈ C^{N×L_p} → real-valued (N, 2*L_p) → N tokens of dim 2*L_p
    The Transformer treats each antenna's pilot observation as one token.
    A learnable [CLS]-style readout token is appended to aggregate information.
    Output head: projects aggregated representation to full BF W ∈ C^{N×K}.

    Uses residual structure: W = W_base(LS) + DeltaW(Transformer)
    """
    def __init__(self, cfg):
        super().__init__()
        self.N, self.K, self.P_max = cfg.N, cfg.K, cfg.P_max
        self.L_p = cfg.L_p
        tok_dim = 2 * cfg.L_p  # each antenna's pilot = 2*L_p real values

        # Input projection: 2*L_p → d_model
        self.proj_in = nn.Linear(tok_dim, cfg.d_model)
        self.ln_in = nn.LayerNorm(cfg.d_model)

        # Learnable readout token (aggregates all antenna info)
        self.readout = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)

        # Transformer blocks (same config as proposed ICL model)
        self.blocks = nn.ModuleList([
            CausalBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.ln_out = nn.LayerNorm(cfg.d_model)

        # Output head: d_model → 2*N*K (real+imag beamformer)
        self.out_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(),
            nn.Linear(cfg.d_ff, 2 * cfg.N * cfg.K)
        )

        self.apply(self._init)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[A1a] DirectPilotBFModel params: {n_params:,}")

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, Y_complex, sigma2=None):
        """
        Y_complex: (B, N, L_p) complex pilot observation
        Returns: dW (B, N, K) complex — residual beamformer (unnormalized)
        """
        B = Y_complex.size(0)
        # Reshape to (B, N, 2*L_p): each antenna = one token
        Y_real = torch.cat([Y_complex.real, Y_complex.imag], dim=-1)  # (B, N, 2*Lp)

        x = self.ln_in(self.proj_in(Y_real))  # (B, N, d_model)

        # Append readout token
        ro = self.readout.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([x, ro], dim=1)  # (B, N+1, d_model)

        # No causal mask needed — this is not autoregressive
        for blk in self.blocks:
            x = blk(x, mask=None)

        x = self.ln_out(x)
        out = x[:, -1, :]  # readout position: (B, d_model)

        # Project to full BF
        w_real = self.out_head(out)  # (B, 2NK)
        dW = real_to_bf(w_real, self.N, self.K)
        return dW


# ============================================================================
# TRAINING LOOP (No ICL — uses same curriculum but no context/bootstrapping)
# ============================================================================
def train(cfg):
    set_global_seed(cfg.seed)
    print("="*70)
    print("  ABLATION A1a: No-ICL Direct Pilot→BF Transformer")
    print("="*70)
    print(f"  K={cfg.K} N={cfg.N} L_p={cfg.L_p} SNR={cfg.SNR_dB}dB")

    Phi = generate_pilot_dft(cfg.K, cfg.L_p)

    # Generate labeled dataset (MMSE BF on perfect CSI as labels)
    print(f"\nGenerating labeled dataset ({cfg.ds_size} samples)...")
    H_all, W_all, Y_all, R_all = [], [], [], []
    gbs = min(64, cfg.ds_size)
    for s in range(0, cfg.ds_size, gbs):
        e = min(s+gbs, cfg.ds_size); b = e-s
        H = generate_channel(b, cfg.K, cfg.N, cfg.ch_n_clusters, cfg.ch_n_rays, cfg.ch_spread_deg)
        W_star = mmse_beamformer(H, cfg.P_max, cfg.sigma2)
        Y = pilot_observe(H, Phi, cfg.sigma2)
        with torch.no_grad(): r = compute_sum_rate(H, W_star, cfg.sigma2)
        H_all.append(H); W_all.append(bf_to_real(W_star)); Y_all.append(Y); R_all.append(r)
    H_ds = torch.cat(H_all); W_ds = torch.cat(W_all)
    Y_ds = torch.cat(Y_all); R_ds = torch.cat(R_all)
    print(f"  Dataset ready: {H_ds.size(0)} samples, avg rate: {R_ds.mean():.2f}")

    # Test set & baselines
    H_test = generate_channel(cfg.n_test, cfg.K, cfg.N, cfg.ch_n_clusters,
                              cfg.ch_n_rays, cfg.ch_spread_deg)
    print("\nComputing baselines...")
    bl = compute_baselines(H_test, Phi, cfg)
    for k,v in bl.items(): print(f"  {k}: {v:.4f}")

    # Model
    model = DirectPilotBFModel(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.total_epochs, eta_min=cfg.lr_min)

    print("\n"+"="*70)
    best_test = 0.0
    hist = {'test': [], 'train': [], 'ph': []}

    for epoch in range(cfg.total_epochs):
        model.train(); t0 = time.time()
        if epoch < cfg.phase1_epochs: phase = 1
        else: phase = 2

        ep_mse, ep_rate, ep_n = 0., 0., 0

        for step in range(cfg.steps_per_epoch):
            B = cfg.batch_size
            if phase == 1:
                # Supervised: sample from dataset
                idx = torch.randint(0, H_ds.size(0), (B,), device=device)
                q_H = H_ds[idx]; q_W_gt = W_ds[idx]; q_Y = Y_ds[idx]
            else:
                # Unsupervised: fresh channels
                q_H = generate_channel(B, cfg.K, cfg.N, cfg.ch_n_clusters,
                                       cfg.ch_n_rays, cfg.ch_spread_deg)
                q_Y = pilot_observe(q_H, Phi, cfg.sigma2)
                idx = torch.randint(0, H_ds.size(0), (B,), device=device)
                q_W_gt = W_ds[idx]  # dummy, not used in phase 2

            # Residual: W = W_base(LS) + dW(model)
            H_ls = ls_channel_est(q_Y, Phi)
            W_base = mmse_beamformer(H_ls, cfg.P_max, cfg.sigma2)
            dW = model(q_Y, sigma2=cfg.sigma2)
            W_hat = power_normalize(W_base + dW, cfg.P_max)

            rate = compute_sum_rate(q_H, W_hat, cfg.sigma2)
            W_hat_real = bf_to_real(W_hat)

            if phase == 1:
                loss = F.mse_loss(W_hat_real, q_W_gt, reduction='none').sum(-1).mean()
                ep_mse += loss.item()
            else:
                loss = -rate.mean() * cfg.unsup_scale

            # Train-rate metric should always be tracked (both phases).
            ep_rate += rate.mean().item()

            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); ep_n += 1

        scheduler.step()

        # Evaluate
        model.eval()
        test_rates = []
        with torch.no_grad():
            for s in range(0, H_test.size(0), cfg.batch_size):
                e = min(s+cfg.batch_size, H_test.size(0)); H = H_test[s:e]
                Y = pilot_observe(H, Phi, cfg.sigma2)
                H_ls = ls_channel_est(Y, Phi)
                W_base = mmse_beamformer(H_ls, cfg.P_max, cfg.sigma2)
                dW = model(Y, sigma2=cfg.sigma2)
                W_hat = power_normalize(W_base + dW, cfg.P_max)
                test_rates.append(compute_sum_rate(H, W_hat, cfg.sigma2))
        tr = torch.cat(test_rates).mean().item()
        best_test = max(best_test, tr)
        model.train()

        am = ep_mse/max(1,ep_n); ar = ep_rate/max(1,ep_n); dt = time.time()-t0
        hist['test'].append(tr); hist['train'].append(ar); hist['ph'].append(phase)

        if ar <= 0:
            print("  [warn] train rate <= 0, please check channel generation/SNR/loss scaling.", flush=True)

        # Save train-rate snapshots every N epochs.
        if (epoch + 1) % cfg.rate_save_every == 0:
            ep_tensor = torch.arange(1, len(hist['train']) + 1)
            torch.save({
                'epochs': ep_tensor,
                'train_rate': torch.tensor(hist['train'], dtype=torch.float32),
            }, cfg.train_rate_pt)

        print(f"{epoch+1:3d} ph{phase} | mse={am:.5f} rate={ar:.2f} | "
              f"test={tr:.3f} best={best_test:.3f} ({dt:.1f}s)", flush=True)

    b4 = bl['wmmse_imp']
    print("\n"+"="*68)
    print("  A1a COMPLETE (No ICL)")
    for k,v in [("B1 MMSE+P",bl['mmse_perf']),("B2 MMSE+E",bl['mmse_imp']),
                ("B3 WMMSE+P",bl['wmmse_perf']),("B4 WMMSE+E",b4),
                ("A1a best",best_test),("A1a final",tr)]:
        print(f"  {k:<20} {v:8.3f}  ({100*v/b4:.1f}%)" if b4>0 else f"  {k} {v}")

    # Final overwrite save to ensure full history is persisted.
    ep_tensor = torch.arange(1, len(hist['train']) + 1)
    torch.save({
        'epochs': ep_tensor,
        'train_rate': torch.tensor(hist['train'], dtype=torch.float32),
    }, cfg.train_rate_pt)

    # Plot training curve based on saved pt file.
    rate_payload = torch.load(cfg.train_rate_pt, map_location='cpu')
    ep_arr = rate_payload['epochs'].cpu().numpy()
    tr_arr = rate_payload['train_rate'].cpu().numpy()
    plt.figure(figsize=(8, 5))
    plt.plot(ep_arr, tr_arr, linewidth=1.8, label='Train rate')
    plt.xlabel('Epoch')
    plt.ylabel('Sum rate')
    plt.title('A1a Training Rate Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.train_curve_png, dpi=180)
    plt.close()
    print(f"  Saved: {cfg.train_rate_pt}, {cfg.train_curve_png}")
    return model, bl

if __name__=="__main__":
    cfg=Config(K=32,N=32,L_p=20,SNR_dB=20,d_model=512,n_heads=8,n_layers=6,d_ff=1024,
               ds_size=5000,phase1_epochs=50,phase2_epochs=500,steps_per_epoch=80,
               batch_size=64,lr=2e-4,n_test=200,rate_save_every=10,
               train_rate_pt='a1a_training_rate.pt',train_curve_png='a1a_training_curve.png')
    train(cfg)
