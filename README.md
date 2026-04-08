<div align="center">

# Self-Evolving In-Context Learning for<br>Pilot-to-Precoder Design in Multi-User MIMO

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Direct pilot-to-beamformer design via in-context learning — no channel estimation needed.**

[Overview](#overview) •
[Architecture](#architecture) •
[Quick Start](#quick-start) •
[File Structure](#file-structure) •
[Ablation Baselines](#ablation-baselines) •
[Configuration](#configuration) •
[Citation](#citation)

</div>

---

## Overview

This repository implements a **self-evolving in-context learning (ICL) framework** for multi-user MISO downlink precoding that operates directly on noisy pilot observations. Unlike existing approaches that rely on perfect channel state information (CSI) or large-scale pre-computed WMMSE datasets, our method:

- **Bypasses channel estimation entirely** at inference time
- Requires only **~1,000 labeled samples** for warm-start (vs. ~80,000 in prior ICL-based wireless models)
- **Self-evolves** the demonstration pool during training via dual-threshold bootstrapping
- Adapts to **multiple channel scenarios** with a single shared model, through context alone
- Achieves **forward-pass-only adaptation** to unseen scenarios — no gradient updates needed

### Key Idea

The pilot-to-beamformer mapping is fundamentally **ill-posed**: the same noisy pilot observation can correspond to infinitely many channels, each requiring a different optimal precoder. ICL resolves this ambiguity by providing **demonstration context** — a few pilot-beamformer pairs from the current environment — that implicitly communicates the channel distribution to the Transformer.

<div align="center">

```
  ┌─────────────┐      ┌──────────────┐      ┌─────────────────┐      ┌──────────┐      ┌──────┐
  │  Context     │      │   Frozen     │      │                 │      │ Trainable│      │      │
  │  {(Yᵢ,W*ᵢ)} │─────▶│  Encoders    │─────▶│ ICL Transformer │─────▶│BF Decoder│──+──▶│  Ŵ   │
  │  + Query Yq  │      │ Pilot + BF   │      │ (6L, 8H, d=512) │      │          │  │   │      │
  └─────────────┘      └──────────────┘      └─────────────────┘      └──────────┘  │   └──────┘
                                                                                     │
                                                                        W_base ──────┘
                                                                       (LS + MMSE)
```

</div>

---

## Architecture

### Training Pipeline

The framework trains in three phases:

| Phase | Description | Loss | Dataset |
|:-----:|:------------|:----:|:-------:|
| **0** | Pretrain two encoder-decoder networks (EDNs) on mixed-scenario data. Freeze encoders after convergence. | Channel MSE / BF MSE | Static |
| **1** | Supervised warm-start of ICL Transformer + BF Decoder | Beamformer MSE | Small labeled D₀ |
| **2** | Curriculum self-evolution with staircase schedule (0.25→0.50→0.75→1.0 unsup ratio) | Hybrid MSE + Sum-Rate | Self-growing D |

### Model Components

| Component | Input → Output | Role | Trainable? |
|:----------|:---------------|:-----|:----------:|
| **PilotEncoder** | Y ∈ ℂ^{N×Lp} → z ∈ ℝ^{D} | CNN + attention pooling, compresses pilot to token | ❄️ Frozen |
| **BFEncoder** | W ∈ ℂ^{N×K} → c ∈ ℝ^{D} | MLP + FiLM(σ²), compresses beamformer to token | ❄️ Frozen |
| **ICL Transformer** | {z₁,c₁,...,zₗ,cₗ,z_q} → ĉ | Causal decoder-only, no positional encoding | ✅ Trainable |
| **BF Decoder** | ĉ → ΔW ∈ ℂ^{N×K} | MLP, recovers residual beamformer | ✅ Trainable |

### Dual-Threshold Self-Bootstrapping

During Phase 2, model-generated solutions are admitted to the dataset only if they pass **both** gates:

1. **Per-instance**: `rate > α(e) × MMSE_rate` — ensures absolute quality
2. **Cross-instance**: `rate ≥ β(e)-percentile in batch` — prevents flood admission

Both thresholds schedule upward during training (α: 0.60→0.90, β: 0.60→0.90).

---

## Quick Start

### Requirements

```bash
pip install torch>=2.0 numpy matplotlib tqdm
```

### Run the Proposed Method (Single Scenario)

```bash
# Latest version with all features
python pilot_icl_4_6.py
```

### Run Multi-Scenario Extension

```bash
python pilot_icl_multi_task_4_5.py
```

### Run Ablation Baselines

```bash
# No-ICL baseline (direct pilot-to-BF Transformer)
python ablation_baseline/ablation_A1a_no_icl.py

# Pure supervised ICL (no sum-rate loss)
python ablation_baseline/ablation_A2a_pure_sup.py

# No self-bootstrapping (static dataset)
python ablation_baseline/ablation_A3a_no_bootstrap.py
```

### Default Configuration

| Parameter | Value | Description |
|:----------|:-----:|:------------|
| K, N | 32, 32 | Users, antennas |
| L_p | 20 | Pilot length (< K) |
| SNR | 20 dB | Signal-to-noise ratio |
| D_tok | 256 | Compressed token dimension |
| n_demos | 5 | Context length (demo pairs) |
| d_model | 512 | Transformer model dimension |
| n_layers | 6 | Transformer depth |
| Phase 1 | 50 epochs | Supervised warm-start |
| Phase 2 | 1000 epochs | Curriculum self-evolution |

---

## File Structure

```
ICL_pilot_precoding/
├── pilot_icl_4_6.py              # 🔥 Latest proposed method (recommended)
├── pilot_icl_4_5.py              # Previous stable version
├── pilot_icl_4_3_3.py            # Earlier version with anti-lazy regularization
├── pilot_icl_sparse_3_25.py      # Original sparse channel version
├── pilot_icl_multi_task_4_5.py   # Multi-scenario (8 channels) extension
├── baseline_sallom_pilot.py      # SALLO-M Transformer baseline (no CSI)
├── plot_4_6.py                   # Plotting utilities for training curves
├── change_record.txt             # Version changelog
├── ablation_baseline/
│   ├── ablation_A1a_no_icl.py        # A1a: No ICL context
│   ├── ablation_A1c_mlp.py           # A1c: Equal-param MLP (no Transformer)
│   ├── ablation_A2a_pure_sup.py      # A2a: Pure supervised (MSE only)
│   ├── ablation_A2b_pure_unsup.py    # A2b: Pure unsupervised (no warm-start)
│   ├── ablation_A2c_hard_switch.py   # A2c: Hard switch (no curriculum)
│   ├── ablation_A3a_no_bootstrap.py  # A3a: No self-bootstrapping
│   ├── ablation_A3b_accept_all.py    # A3b: Accept all (no threshold)
│   └── ablation_A3c_fixed_thresh.py  # A3c: Fixed threshold
└── LICENSE
```

### Version Evolution

```
v3.25 (sparse_3_25)  →  Original sparse channel + (p,λ) parameterization
v4.3.3               →  Compressed-BF ICL + residual structure + anti-lazy reg
v4.5                 →  Refined loss balancing + staircase schedule
v4.6                 →  Latest: improved bootstrapping + evaluation pipeline
multi_task_4.5       →  8-scenario extension with round-robin training
```

---

## Ablation Baselines

The ablation study validates every key component. Each baseline modifies **exactly one** aspect while keeping everything else identical.

### Group 1: Is ICL Necessary?

| ID | Baseline | What Changes | What It Proves |
|:--:|:---------|:-------------|:---------------|
| A1a | No ICL | Remove all demo context | Context resolves pilot→BF ill-posedness |
| A1c | MLP | Replace Transformer with equal-param MLP | Sequence processing matters, not just capacity |
| — | SALLO-M | State-of-the-art L2O Transformer (adapted) | ICL outperforms non-ICL architectures |

### Group 2: Training Strategy

| ID | Baseline | What Changes | What It Proves |
|:--:|:---------|:-------------|:---------------|
| A2a | Pure Supervised | MSE only, no sum-rate | Can't exceed label quality |
| A2b | Pure Unsupervised | Sum-rate from epoch 1 | Warm-start is essential |
| A2c | Hard Switch | Instant 100% unsup at Phase 2 | Gradual curriculum needed |

### Group 3: Self-Bootstrapping

| ID | Baseline | What Changes | What It Proves |
|:--:|:---------|:-------------|:---------------|
| A3a | No Bootstrap | Dataset stays static | Context diversity crucial |
| A3b | Accept All | No quality threshold | Dataset pollution → negative feedback |
| A3c | Fixed Threshold | Non-adaptive α, no β | Too strict OR too loose |

---

## Multi-Scenario Extension

The framework extends to **8 diverse channel scenarios** using a single shared ICL Transformer:

| ID | Scenario | Clusters | Rays | Spread | Character |
|:--:|:---------|:--------:|:----:|:------:|:----------|
| S0 | Dense Urban | 3 | 5 | 10° | Default — moderate sparsity |
| S1 | LoS-Dominant | 1 | 10 | 3° | Nearly rank-1, tight angular |
| S2 | Rich Scatter | 6 | 3 | 5° | Many clusters, high rank |
| S3 | Suburban | 2 | 8 | 15° | Wide spread |
| S4 | Indoor Office | 5 | 2 | 20° | Many weak clusters |
| S5 | Near-LoS | 1 | 15 | 2° | Almost pure line-of-sight |
| S6 | Moderate Urban | 4 | 4 | 8° | Balanced |
| S7 | Rayleigh iid | — | — | — | No sparsity (stress test) |

**No scenario tag is provided to the model.** The Transformer identifies the environment purely from the demonstration context — this is the core ICL capability.

### Online Adaptation

- **Seen scenarios**: Use self-evolved per-scenario dataset as context. Forward-pass only.
- **Unseen scenarios**: Generate ~50 LMMSE demo pairs → use as context → self-evolve through inference. No gradients needed.

---

## Configuration

All hyperparameters are controlled via the `Config` class. Key groups:

```python
cfg = Config(
    # System
    K=32, N=32, L_p=20, SNR_dB=20,
    # Channel
    ch_n_clusters=3, ch_n_rays=5, ch_spread_deg=10.0,
    # Architecture
    D_tok=256, d_model=512, n_heads=8, n_layers=6, d_ff=1024,
    n_demos=5, edn_hidden=128,
    # Training
    phase1_epochs=50, phase2_epochs=1000, steps_per_epoch=80,
    batch_size=64, lr=1e-4,
    # Bootstrapping
    boot_alpha_start=0.60, boot_alpha_end=0.90,
    boot_beta_start=0.60, boot_beta_end=0.90,
    # Pruning
    prune_every=10, prune_drop_end=0.10,
)
```

---

## Citation

If you find this code useful, please cite:

```bibtex
@article{zhang2026self_evolving_icl,
  title={Self-Evolving In-Context Learning for Direct Pilot-to-Precoder 
         Design in Multi-User {MIMO}},
  author={Zhang, Yubo and Liu, Xiao-Yang and Wang, Xiaodong},
  journal={submitted to IEEE Trans. Commun.},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

<div align="center">
<sub>Department of Electrical Engineering, Columbia University</sub>
</div>
