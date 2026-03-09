"""
Fast Baseline Benchmark: MMSE beamformer baselines.
(p,lam) optimization requires PyTorch for proper gradient-based optimization.
"""
import numpy as np
from numpy.linalg import inv, norm, solve
import time

np.random.seed(2026)

K = 16; N = 16; L_p = 10; P_max = 1.0; SNR_dB = 15
sigma2 = P_max / (10 ** (SNR_dB / 10))
n_test = 500

print("=" * 70)
print(f"BASELINE BENCHMARK (K={K}, N={N}, L_p={L_p}, SNR={SNR_dB}dB)")
print("=" * 70)

def gen_channel(K, N):
    return (np.random.randn(K, N) + 1j * np.random.randn(K, N)) / np.sqrt(2)

def gen_pilot_matrix(K, L_p):
    return (np.random.randn(K, L_p) + 1j * np.random.randn(K, L_p)) / np.sqrt(2 * L_p)

def gen_pilot_signal(H, Phi, sigma2):
    Y = H.T @ Phi
    noise = (np.random.randn(*Y.shape) + 1j * np.random.randn(*Y.shape)) * np.sqrt(sigma2/2)
    return Y + noise

def mmse_estimate(Y, Phi, sigma2):
    A = Phi.T @ Phi.conj() + sigma2 * np.eye(Phi.shape[1])
    return Phi.conj() @ inv(A) @ Y.T

def mmse_beamformer(H, P_max, sigma2):
    H_H = H.conj().T
    W = solve(H_H @ H + sigma2 * np.eye(H.shape[1]), H_H)
    W *= np.sqrt(P_max / (norm(W, 'fro')**2 + 1e-10))
    return W

def compute_rate(H, W, sigma2):
    HW = H @ W
    sig = np.abs(np.diag(HW))**2
    total = np.sum(np.abs(HW)**2, axis=1)
    interf = total - sig
    SINR = sig / (interf + sigma2)
    return np.sum(np.log2(1 + SINR))

def reconstruct_precoder(H, p, lam, sigma2):
    K_, N_ = H.shape
    h_cols = H.T
    A = np.eye(N_, dtype=complex)
    for j in range(K_):
        hj = h_cols[:, j:j+1]
        A += (lam[j] / sigma2) * (hj @ hj.conj().T)
    A_inv = inv(A)
    W = np.zeros((N_, K_), dtype=complex)
    for k in range(K_):
        vk = A_inv @ h_cols[:, k:k+1]
        vk /= (norm(vk) + 1e-10)
        W[:, k:k+1] = np.sqrt(p[k]) * vk
    return W

# Find MMSE-equivalent (p,lam) by matching the MMSE beamformer structure
def find_mmse_equivalent_plam(H, P_max, sigma2, n_search=200):
    """
    Search for (p,lam) that best matches MMSE beamformer performance.
    The optimal structure WITH the right (p,lam) should match MMSE.
    Search over a 1D family: lam_k = alpha * P_max/K for various alpha.
    """
    K_ = H.shape[0]
    best_rate = -np.inf
    best_p = np.ones(K_) * P_max / K_
    best_lam = np.ones(K_) * P_max / K_
    
    for alpha in np.linspace(0.01, 5.0, n_search):
        lam = np.ones(K_) * alpha * P_max / K_
        # With fixed lam, optimize power allocation
        W = reconstruct_precoder(H, np.ones(K_) * P_max / K_, lam, sigma2)
        # Compute per-user effective channel gains
        HW = H @ W
        gains = np.abs(np.diag(HW))**2
        # Water-filling style: allocate more power to stronger channels
        p = gains / (gains.sum() + 1e-10) * P_max
        p = np.maximum(p, 1e-6)
        p = p / p.sum() * P_max
        
        W2 = reconstruct_precoder(H, p, lam, sigma2)
        rate = compute_rate(H, W2, sigma2)
        if rate > best_rate:
            best_rate = rate
            best_p = p.copy()
            best_lam = lam.copy()
    
    return best_p, best_lam, best_rate

# =============================================================================
Phi = gen_pilot_matrix(K, L_p)

rates_mmse_p, rates_mmse_e = [], []
rates_opt_p, rates_opt_e = [], []
nmses = []

t0 = time.time()
print(f"\nRunning {n_test} samples...")
print("-" * 70)

for i in range(n_test):
    H = gen_channel(K, N)
    Y = gen_pilot_signal(H, Phi, sigma2)
    H_hat = mmse_estimate(Y, Phi, sigma2)
    nmses.append(norm(H_hat - H, 'fro')**2 / norm(H, 'fro')**2)
    
    # 1. MMSE BF + Perfect CSI
    W1 = mmse_beamformer(H, P_max, sigma2)
    rates_mmse_p.append(compute_rate(H, W1, sigma2))
    
    # 2. MMSE BF + Imperfect CSI (estimated channel)
    W2 = mmse_beamformer(H_hat, P_max, sigma2)
    rates_mmse_e.append(compute_rate(H, W2, sigma2))
    
    # 3. Optimized (p,lam) + Perfect CSI (grid search)
    _, _, r3 = find_mmse_equivalent_plam(H, P_max, sigma2, n_search=300)
    rates_opt_p.append(r3)
    
    # 4. Optimized (p,lam) + Imperfect CSI
    _, _, r4 = find_mmse_equivalent_plam(H_hat, P_max, sigma2, n_search=300)
    # Evaluate with TRUE channel
    p4, lam4, _ = find_mmse_equivalent_plam(H_hat, P_max, sigma2, n_search=300)
    W4 = reconstruct_precoder(H_hat, p4, lam4, sigma2)
    rates_opt_e.append(compute_rate(H, W4, sigma2))
    
    if (i+1) % 50 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (i+1) * (n_test - i - 1)
        print(f"  [{i+1:4d}/{n_test}] ({elapsed:.0f}s, ETA {eta:.0f}s)  "
              f"MMSE-P: {np.mean(rates_mmse_p):.2f}  "
              f"MMSE-E: {np.mean(rates_mmse_e):.2f}  "
              f"OPT-P:  {np.mean(rates_opt_p):.2f}  "
              f"OPT-E:  {np.mean(rates_opt_e):.2f}")

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s")

print("\n" + "=" * 70)
print("BASELINE RESULTS")
print("=" * 70)
print(f"  MMSE Channel Estimation NMSE: {10*np.log10(np.mean(nmses)):.2f} dB\n")

labels = [
    ('MMSE-P', '1. MMSE Beamformer + Perfect CSI',             rates_mmse_p),
    ('MMSE-E', '2. MMSE Beamformer + Imperfect CSI',           rates_mmse_e),
    ('OPT-P',  '3. Opt (p,lam) Precoder + Perfect CSI',        rates_opt_p),
    ('OPT-E',  '4. Opt (p,lam) Precoder + Imperfect CSI',      rates_opt_e),
]
for tag, label, vals in labels:
    print(f"  {label:<50s}  {np.mean(vals):>8.4f} bps/Hz  (std: {np.std(vals):.2f})")

print(f"\n  CSI estimation loss (MMSE BF):      {np.mean(rates_mmse_p)-np.mean(rates_mmse_e):.2f} bps/Hz")
print(f"  CSI estimation loss (Opt precoder): {np.mean(rates_opt_p)-np.mean(rates_opt_e):.2f} bps/Hz")
print(f"  Structure gap (Perfect CSI):        {np.mean(rates_mmse_p)-np.mean(rates_opt_p):.2f} bps/Hz")
print()
print("  NOTE: Opt (p,lam) with grid search is a LOWER BOUND on true WMMSE.")
print("        PyTorch Adam optimization (in training code) will achieve higher rates.")
print("        The ICL model target is between Baseline 4 and Baseline 1.")
print("=" * 70)

# Save
np.savez('/home/claude/baseline_results.npz',
         mmse_perfect=rates_mmse_p, mmse_imperfect=rates_mmse_e,
         opt_perfect=rates_opt_p, opt_imperfect=rates_opt_e,
         nmse=nmses, K=K, N=N, L_p=L_p, SNR_dB=SNR_dB, sigma2=sigma2)
