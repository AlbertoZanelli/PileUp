#!/usr/bin/env python3
"""Self-check di reliability_R (src/analysis.py).

  /opt/anaconda3/envs/pyrootAlbi/bin/python test/test_reliability_R.py
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.analysis import reliability_R, compute_W_torch

n, N = 64, 38
rng = np.random.default_rng(0)
S = torch.tensor(np.fft.fft(np.exp(-np.arange(n) / 5.0)), dtype=torch.cfloat)
h = rng.uniform(1e-3, 1e-2, n // 2 + 1)                 # NPS: meta' + mirror, come nei dati
nps = torch.tensor(np.concatenate([h, h[-2:0:-1]]), dtype=torch.float32)

R = reliability_R(S, nps, N, beta=2.0, eps_frac=0.0)
assert R.dtype == torch.float32 and R.shape == S.shape
assert torch.all(R >= 0) and torch.all(R <= 1)

# R=0.5 quando |S|^2_n = 2 * beta*NPS_n/N (SNR in potenza del template = 2*beta):
# con |S| ~ sqrt(NPS) e N = 2*beta la condizione vale in OGNI bin.
S_half = torch.sqrt(nps).to(torch.cfloat)
assert torch.allclose(reliability_R(S_half, nps, N_events=4, beta=2.0, eps_frac=0.0),
                      torch.full((n,), 0.5), atol=1e-5)

# invariante a un riscalamento costante dell'NPS (come W)
assert torch.allclose(R, reliability_R(S, 137.0 * nps, N, 2.0, 0.0), atol=1e-6)

# beta piu' grande => gate piu' stretto ; N piu' grande => gate piu' largo
assert torch.all(reliability_R(S, nps, N, 8.0, 0.0) <= R + 1e-6)
assert torch.all(reliability_R(S, nps, 4 * N, 2.0, 0.0) >= R - 1e-6)

# hermitiana: R*W resta un kernel a impulso reale
W = compute_W_torch(S, nps, torch.tensor(1.0))
k = np.fft.ifft((R * W).numpy())
assert np.max(np.abs(k.imag)) < 1e-6 * np.max(np.abs(k.real))

print("reliability_R OK")
