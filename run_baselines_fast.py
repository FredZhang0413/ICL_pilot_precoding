"""
Baseline Benchmarks: Cluster-Sparse Channel  (OMP-DFT vs IID-MMSE)
=====================================================================

System: MU-MISO downlink,  K=16 users,  N=16 BS antennas
Channel: Clustered-sparse ULA  (3 clusters x 5 rays, spread=10 deg)
Pilot:   random Gaussian  Phi (K, L_p)

Observation model:  Y = H^T Phi + N      shape (B, N, L_p)

Estimators compared
-------------------
  IID-MMSE   H_hat = Phi*(Phi^T Phi* + sigma2 I)^{-1} Y^T
             Assumes R=I_K prior.  When L_p < K: NMSE floor = (K-L_p)/K
             (floor is SNR-independent; no post-processing can remove it).

  DFT-MMSE   Two-step: IID-MMSE pilot inversion -> FFT top-k thresholding.
             ONLY helps when L_p >= K.  Reason: when L_p < K the MMSE error
             is dominated by inter-user leakage which is DFT-white (uniform
             spectrum), so thresholding removes signal and error in equal
             proportions -> zero net NMSE gain.  When L_p >= K the error is
             purely thermal noise (also DFT-white), but the SIGNAL is sparse
             in the DFT domain (top-s bins), so thresholding improves NMSE
             by ~10*log10(N/s) dB  (e.g. 10*log10(32/9) ~ 5.5 dB).

Test conditions
---------------
  Under  Cluster SNR=15 dB, L_p=10  (IID-MMSE, underdetermined)
  Under  Cluster SNR=25 dB, L_p=10  (IID-MMSE, underdetermined)
  Full   Cluster SNR=15 dB, L_p=K=32  (IID-MMSE, full-rank pilot)
  DFT    Cluster SNR=15 dB, L_p=K=32  (DFT-MMSE, full-rank + sparsity)
  Over   Cluster SNR=15 dB, L_p=2K=64 (IID-MMSE, overdetermined ref)

Precoder baselines (same in every condition)
--------------------------------------------
  B1  MMSE-BF    + perfect CSI        upper bound
  B2  MMSE-BF    + estimated CSI
  B3  Opt(p,lam) + perfect CSI        optimal structure, perfect
  B4  Opt(p,lam) + estimated CSI      <- ICL model 1-shot target
"""

import math
import time
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
K       = 32      # users
N       = 32      # BS antennas
P_max   = 1.0
N_TEST  = 200     # total test samples
BATCH   = 20      # batch size
N_ITERS = 400     # Adam iterations per restart
N_RST   = 2       # random restarts for (p, lam) optimisation

# DFT thresholding hyper-parameter
# With N=32 antennas, 3 clusters x 5 rays, spread=10 deg:
#   - top-9 bins capture ~90% of per-user energy  (n_cl=3, ~3 bins/cluster)
#   - only useful when L_p >= K (pilot is full-rank)
OMP_SPARSITY = 9     # DFT bins retained per user  (n_cl * ~3 bins/cluster)


# ==================================================================
# 1.  Channel model  (cluster-sparse only)
# ==================================================================

def gen_cluster(B, K, N, n_cl=3, n_ray=5, spread_deg=10.0):
    """
    Clustered-sparse ULA channel (geometry-based stochastic model).

    h_k = sum_{c,r} g_{c,r} * a(theta_{c,r})
        g_{c,r}  ~ CN(0, 1/(n_cl*n_ray))    -> E[||h_k||^2] = N
        theta_{c,r}  = theta_c + D_{c,r},   D ~ N(0, spread^2)
        a(theta)[n]  = exp(j*pi*n*sin(theta))    (half-wavelength ULA)

    Returns (B, K, N) complex.
    """
    s     = spread_deg * math.pi / 180.0
    n_idx = torch.arange(N, device=device, dtype=torch.float)

    cl_ang  = (torch.rand(B, K, n_cl, device=device) - 0.5) * (2 * math.pi / 3)
    ray_ang = cl_ang.unsqueeze(-1) + \
              torch.randn(B, K, n_cl, n_ray, device=device) * s

    steer = torch.exp(
        1j * math.pi * torch.sin(ray_ang).unsqueeze(-1) * n_idx)

    scale = 1.0 / math.sqrt(2 * n_cl * n_ray)
    gains = torch.complex(
        torch.randn(B, K, n_cl, n_ray, device=device) * scale,
        torch.randn(B, K, n_cl, n_ray, device=device) * scale).unsqueeze(-1)

    return (gains * steer).sum(dim=(2, 3))          # (B, K, N)


# ==================================================================
# 2.  Pilot operations
# ==================================================================

def gen_pilot(K, L_p):
    """Random Gaussian pilot matrix (K, L_p) complex."""
    scale = 1.0 / math.sqrt(2 * L_p)
    return torch.complex(
        torch.randn(K, L_p, device=device) * scale,
        torch.randn(K, L_p, device=device) * scale)


def observe(H, Phi, sigma2):
    """Y = H^T Phi + N.  Returns (B, N, L_p) complex."""
    B, _, N_ = H.shape
    L_p_     = Phi.shape[1]
    Y  = H.transpose(-1, -2) @ Phi.unsqueeze(0)
    nr = torch.randn(B, N_, L_p_, device=device) * math.sqrt(sigma2 / 2)
    ni = torch.randn(B, N_, L_p_, device=device) * math.sqrt(sigma2 / 2)
    return Y + torch.complex(nr, ni)


# ==================================================================
# 3.  Channel estimators
# ==================================================================

def mmse_est(Y, Phi, sigma2):
    """
    IID-MMSE: H_hat = Phi* (Phi^T Phi* + sigma2 I)^{-1} Y^T
    Returns (B, K, N).
    """
    L_p_ = Phi.shape[1]
    A  = Phi.T @ Phi.conj() + sigma2 * torch.eye(L_p_, dtype=Phi.dtype, device=device)
    PA = Phi.conj() @ torch.linalg.inv(A)
    return PA.unsqueeze(0) @ Y.transpose(-1, -2)


def omp_est(Y, Phi, sigma2, sparsity=OMP_SPARSITY, n_grid=None):
    """
    Two-step DFT-MMSE channel estimator.

    Step 1 -- IID-MMSE pilot inversion:
        H_init = Phi*(Phi^T Phi* + sigma2 I)^{-1} Y^T     (B, K, N)

    Step 2 -- Per-user N-point FFT hard thresholding:
        C_k    = FFT(h_k_init)             angular domain
        C_k'   = top-s mask on |C_k|       keep 'sparsity' strongest bins
        h_k'   = IFFT(C_k')               back to spatial domain

    IMPORTANT -- when this helps and when it does not:

    L_p >= K  (full-rank pilot, e.g. L_p=K=32):
        MMSE error is dominated by thermal noise, which is DFT-white.
        True h_k occupies ~sparsity bins out of N.  Thresholding keeps
        (sparsity/N) of the noise power while retaining ~90% signal energy
        => NMSE improves by roughly 10*log10(N/sparsity) ~ 5.5 dB.
        This is where the DFT step is worthwhile.

    L_p < K  (underdetermined, e.g. L_p=10, K=32):
        MMSE error is dominated by inter-user leakage (K-L_p null-space).
        Leakage = sum of other users' channels at all random angles
        => DFT-white error spectrum.  Thresholding removes signal and
        leakage in equal proportion => near-zero NMSE improvement.
        Verified experimentally: MMSE=-1.51 dB, DFT(s=9)=-1.32 dB.

    Y        : (B, N, L_p)  pilot observations
    Phi      : (K, L_p)     pilot matrix
    sparsity : DFT bins to keep per user  (n_cl * ~3, default=9)
    Returns H_hat (B, K, N) complex.
    """
    # Step 1: MMSE user demixing / pilot inversion
    H_init = mmse_est(Y, Phi, sigma2)               # (B, K, N)

    # Step 2: angular-domain hard thresholding (vectorised, no loops)
    C      = torch.fft.fft(H_init, dim=-1)          # (B, K, N)
    amp    = C.abs()
    thr    = torch.topk(amp, sparsity, dim=-1).values[..., -1:]
    return torch.fft.ifft(C * (amp >= thr), dim=-1)  # (B, K, N) complex


# ==================================================================
# 4.  Beamformers, optimal precoder structure, and sum-rate
# ==================================================================

def mmse_bf(H, P_max, sigma2):
    """Regularised ZF / MMSE beamformer.  H:(B,K,N) -> W:(B,N,K)."""
    B, K_, N_ = H.shape
    Hh = H.conj().transpose(-1, -2)
    A  = Hh @ H + sigma2 * torch.eye(N_, dtype=H.dtype, device=device)
    W  = torch.linalg.solve(A, Hh)
    pw = W.abs().pow(2).sum(dim=(1, 2)).real
    return W * torch.sqrt(P_max / (pw + 1e-8)).view(B, 1, 1)


def recon_w(H, p, lam, sigma2):
    """
    Optimal precoder structure:
        w_k = sqrt(p_k) * (A^{-1} h_k*) / ||A^{-1} h_k*||
        A   = I_N + H^H diag(lam/sigma2) H
    H:(B,K,N), p:(B,K), lam:(B,K) -> W:(B,N,K)
    """
    B, K_, N_ = H.shape
    Hh  = H.conj().transpose(-1, -2)
    ld  = torch.diag_embed(lam / sigma2).to(torch.cfloat)
    eye = torch.eye(N_, dtype=torch.cfloat, device=device).unsqueeze(0)
    A   = eye + Hh @ ld @ Hh.conj().transpose(-1, -2)
    v   = torch.linalg.solve(A, Hh)
    v   = v / (v.norm(dim=1, keepdim=True).real + 1e-8)
    return v * p.unsqueeze(1).sqrt().to(torch.cfloat)


def sum_rate(H, W, sigma2):
    """MU-MISO sum rate.  H:(B,K,N), W:(B,N,K) -> (B,)."""
    HW   = H @ W
    sig  = HW.diagonal(dim1=-2, dim2=-1).abs().pow(2)
    tot  = HW.abs().pow(2).sum(-1)
    SINR = sig / (tot - sig + sigma2)
    return torch.log2(1 + SINR).sum(-1)


# ==================================================================
# 5.  Adam-based (p, lam) optimiser
# ==================================================================

def opt_plam(H, P_max, sigma2, n_iters=400, lr=0.03, n_restarts=2):
    """
    Maximise sum rate w.r.t. (p, lam) via Adam with multiple random restarts.
    p = softmax(p_log) * P_max,  lam = softplus(l_log).
    Returns best_p (B, K), best_lam (B, K).
    """
    B, K_, _ = H.shape
    Hd       = H.detach()
    best_r   = torch.full((B,), -1e9, device=device)
    best_p   = torch.zeros(B, K_, device=device)
    best_lam = torch.zeros(B, K_, device=device)

    for _ in range(n_restarts):
        pl  = (torch.randn(B, K_, device=device) * 0.1).requires_grad_(True)
        ll  = (torch.randn(B, K_, device=device) * 0.1).requires_grad_(True)
        opt = optim.Adam([pl, ll], lr=lr)

        for _ in range(n_iters):
            p_   = F.softmax(pl, dim=-1) * P_max
            l_   = F.softplus(ll)
            W_   = recon_w(Hd, p_, l_, sigma2)
            loss = -sum_rate(Hd, W_, sigma2).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        with torch.no_grad():
            p_f  = F.softmax(pl, dim=-1) * P_max
            l_f  = F.softplus(ll)
            r_f  = sum_rate(Hd, recon_w(Hd, p_f, l_f, sigma2), sigma2)
            mask = r_f > best_r
            best_r[mask]   = r_f[mask]
            best_p[mask]   = p_f[mask]
            best_lam[mask] = l_f[mask]

    return best_p.detach(), best_lam.detach()


# ==================================================================
# 6.  Per-condition benchmark runner
# ==================================================================

def run_condition(label, K, N, L_p, P_max, SNR_dB,
                  estimator='mmse', n_cl=3, n_ray=5,
                  n_test=200, bs=20, n_iters=400, n_restarts=2):
    """
    Evaluate all 4 baselines for one condition.

    estimator : 'mmse' -> IID-MMSE   |   'omp' -> DFT-basis OMP
    """
    sigma2    = P_max / (10 ** (SNR_dB / 10.0))
    ch_sparsity = n_cl * n_ray          # true number of channel paths
    Phi       = gen_pilot(K, L_p)

    est_label = ('IID-MMSE' if estimator == 'mmse'
                 else 'DFT-OMP s={}'.format(OMP_SPARSITY))

    print('\n' + '-'*72)
    print('  {}'.format(label))
    print('  Estimator: {}  sigma2={:.6f}  L_p={}  K={}  N={}'.format(
        est_label, sigma2, L_p, K, N))
    print('-'*72)

    b1, b2, b3, b4, nmse_l = [], [], [], [], []
    t0 = time.time()

    for s in range(0, n_test, bs):
        e = min(s + bs, n_test)
        b = e - s

        H  = gen_cluster(b, K, N, n_cl=n_cl, n_ray=n_ray)
        Y  = observe(H, Phi, sigma2)

        if estimator == 'mmse':
            Hh = mmse_est(Y, Phi, sigma2)
        else:
            Hh = omp_est(Y, Phi, sigma2, sparsity=OMP_SPARSITY)

        nmse = ((Hh - H).norm(dim=(1, 2)).pow(2) /
                (H.norm(dim=(1, 2)).pow(2) + 1e-8)).real
        nmse_l.extend(nmse.tolist())

        with torch.no_grad():
            W1 = mmse_bf(H,  P_max, sigma2)
            b1.extend(sum_rate(H, W1, sigma2).tolist())
            W2 = mmse_bf(Hh, P_max, sigma2)
            b2.extend(sum_rate(H, W2, sigma2).tolist())

        p3, l3 = opt_plam(H,  P_max, sigma2, n_iters=n_iters, n_restarts=n_restarts)
        with torch.no_grad():
            b3.extend(sum_rate(H, recon_w(H,  p3, l3, sigma2), sigma2).tolist())

        p4, l4 = opt_plam(Hh, P_max, sigma2, n_iters=n_iters, n_restarts=n_restarts)
        with torch.no_grad():
            b4.extend(sum_rate(H, recon_w(Hh, p4, l4, sigma2), sigma2).tolist())

        elapsed = time.time() - t0
        eta     = elapsed / e * (n_test - e) if e < n_test else 0
        print('  [{:3d}/{}]  t={:.0f}s  ETA={:.0f}s  |  '
              'B1={:.3f}  B2={:.3f}  B3={:.3f}  B4={:.3f}'.format(
              e, n_test, elapsed, eta,
              np.mean(b1), np.mean(b2), np.mean(b3), np.mean(b4)), flush=True)

    r = dict(b1=np.mean(b1), b2=np.mean(b2),
             b3=np.mean(b3), b4=np.mean(b4),
             nmse_db=10 * np.log10(np.mean(nmse_l)))

    gap12 = 100 * (r['b1'] - r['b2']) / (r['b1'] + 1e-8)
    gap34 = 100 * (r['b3'] - r['b4']) / (r['b3'] + 1e-8)

    print('\n  -- Results ----------------------------------------------------')
    print('  NMSE: {:.1f} dB'.format(r['nmse_db']))
    print('  B1 MMSE-BF+Perfect  : {:.4f} bps/Hz'.format(r['b1']))
    print('  B2 MMSE-BF+{}  : {:.4f} bps/Hz  gap={:.1f}%'.format(
        est_label, r['b2'], gap12))
    print('  B3 Opt+Perfect      : {:.4f} bps/Hz'.format(r['b3']))
    print('  B4 Opt+{}      : {:.4f} bps/Hz  gap={:.1f}%'.format(
        est_label, r['b4'], gap34))
    return r


# ==================================================================
# 7.  Entry point
# ==================================================================
if __name__ == '__main__':
    print('='*72)
    print('  CLUSTER-SPARSE CHANNEL BASELINES: IID-MMSE vs OMP-DFT')
    print('  K={}  N={}  Device: {}'.format(K, N, device))
    print('  OMP: sparsity={} (FFT hard-threshold, N-pt DFT)'.format(OMP_SPARSITY))
    print('='*72)

    results = {}

    # Under-determined pilots: L_p=10 < K=32
    # DFT post-processing gives near-zero improvement here (error is DFT-white
    # due to inter-user leakage, not thermal noise). MMSE is near-optimal.
    results['Under_15'] = run_condition(
        'Under_15 -- Cluster SNR=15dB L_p=10 < K=32  IID-MMSE',
        K, N, L_p=10, P_max=P_max, SNR_dB=15, estimator='mmse',
        n_test=N_TEST, bs=BATCH, n_iters=N_ITERS, n_restarts=N_RST)

    results['Under_25'] = run_condition(
        'Under_25 -- Cluster SNR=25dB L_p=10 < K=32  IID-MMSE',
        K, N, L_p=10, P_max=P_max, SNR_dB=25, estimator='mmse',
        n_test=N_TEST, bs=BATCH, n_iters=N_ITERS, n_restarts=N_RST)

    # Full-rank pilots: L_p=K=32  (minimum for no null-space)
    # MMSE error is now thermal-noise dominated -> DFT step genuinely helps.
    results['Full_mmse'] = run_condition(
        'Full_mmse -- Cluster SNR=15dB L_p=K=32  IID-MMSE',
        K, N, L_p=K, P_max=P_max, SNR_dB=15, estimator='mmse',
        n_test=N_TEST, bs=BATCH, n_iters=N_ITERS, n_restarts=N_RST)

    results['Full_dft'] = run_condition(
        'Full_dft  -- Cluster SNR=15dB L_p=K=32  DFT-MMSE (s={})'.format(OMP_SPARSITY),
        K, N, L_p=K, P_max=P_max, SNR_dB=15, estimator='omp',
        n_test=N_TEST, bs=BATCH, n_iters=N_ITERS, n_restarts=N_RST)

    # Over-determined: L_p=2K=64 -- upper reference
    results['Over'] = run_condition(
        'Over      -- Cluster SNR=15dB L_p=2K=64  IID-MMSE (overdetermined ref)',
        K, N, L_p=2*K, P_max=P_max, SNR_dB=15, estimator='mmse',
        n_test=N_TEST, bs=BATCH, n_iters=N_ITERS, n_restarts=N_RST)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print('\n' + '='*80)
    print('  SUMMARY TABLE')
    print('='*80)
    print('  {:<52} {:>6} {:>7} {:>7} {:>7} {:>7} {:>8}'.format(
        'Condition', 'NMSE', 'B1', 'B2', 'B3', 'B4', 'B1-B2%'))
    print('  ' + '-'*84)

    cond_labels = {
        'Under_15':  'Under_15:  L_p=10<K  SNR=15dB  IID-MMSE  (underdetermined)',
        'Under_25':  'Under_25:  L_p=10<K  SNR=25dB  IID-MMSE  (underdetermined)',
        'Full_mmse': 'Full_mmse: L_p=K=32  SNR=15dB  IID-MMSE  (full-rank)',
        'Full_dft':  'Full_dft:  L_p=K=32  SNR=15dB  DFT-MMSE  (full-rank+DFT)',
        'Over':      'Over:      L_p=2K=64 SNR=15dB  IID-MMSE  (overdetermined)',
    }
    for key, lbl in cond_labels.items():
        r   = results[key]
        gap = 100 * (r['b1'] - r['b2']) / (r['b1'] + 1e-8)
        flg = 'OK' if 0 <= gap <= 50 else ('!! BUG' if gap < 0 else '(!!)>50%')
        print('  {:<52} {:>5.1f}dB {:>7.3f} {:>7.3f} {:>7.3f} {:>7.3f} {:>6.1f}%  {}'.format(
            lbl, r['nmse_db'], r['b1'], r['b2'], r['b3'], r['b4'], gap, flg))

    print()
    print('  B1=MMSE-BF+Perfect  B2=MMSE-BF+Est  B3=Opt+Perfect  B4=Opt+Est(ICL target)')
    print('  Key insight: Full_mmse vs Full_dft = DFT gain when L_p=K (thermal noise dominated)')
    print('               Under_15 vs Full_mmse = cost of L_p<K null-space floor')
    print('               DFT post-processing is ineffective when L_p<K (leakage-dominated)')
    print('='*80)