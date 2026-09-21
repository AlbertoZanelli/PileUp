"""
check_phase_filters_m205.py
===========================
Controllo dei filtri di banda a FASE LIBERA (PHASE in analysis_BI_m205_wiener_regolarized.py,
phase= in src.analysis.optimize_filters_wiener).

Verifica le quattro cose che possono rompersi:
  1. f REALE: compute_vars_wiener non cambia di una virgola (ramo storico, bit per bit).
  2. mirror hermitiano coniugato -> il filtro complesso resta REALE nel tempo.
  3. RITARDO PURO (f = exp(-i w tau)): risposta e varianze INVARIATE -> nessun guadagno
     fasullo da una traslazione (finche' il picco resta nella banda +-jitter_max).
  4. fase CASUALE: il picco del template filtrato puo' solo ABBASSARSI, quindi la varianza
     puo' solo SALIRE. E' la guardia contro il guadagno finto: la vecchia formula
     sum(|f|*Re(WS)) avrebbe dato la risposta a fase nulla con il rumore della fase libera.

    KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/bin/python3.13 test/check_phase_filters_m205.py
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.analysis as an

N = 1024
rng = np.random.default_rng(0)
t = np.arange(N)
pulse = np.exp(-(t - N // 2) / 60.0) - np.exp(-(t - N // 2) / 8.0)
pulse[t < N // 2] = 0.0
S = torch.tensor(np.fft.fft(pulse * np.hanning(N)), dtype=torch.cfloat)
nps = torch.tensor(1e-4 + 1e-2 / (1 + (np.fft.fftfreq(N) * N / 20.0) ** 2), dtype=torch.float32)
W = an.compute_W_torch(S, nps, torch.tensor(1.0))

half = N // 2 + 1
mag1 = torch.rand(half) + 0.5
mag2 = torch.rand(half) + 0.5
mirror = lambda h: torch.cat([h, h[1:-1].flip(0).conj()])
f1r, f2r = mirror(mag1), mirror(mag2)

# 1. ramo reale invariato
W_nps = (W.abs() ** 2) * nps
resp = (W * S).real
ref = [(torch.sum(m ** 2 * W_nps) / torch.sum(m * resp) ** 2).item() for m in (f1r, f2r)]
v1, v2, _ = an.compute_vars_wiener(W, S, nps, f1r, f2r)
assert (v1.item(), v2.item()) == (ref[0], ref[1]), "il ramo con f reale e' cambiato"

# 2. filtro complesso -> ifft reale
ph = torch.zeros(half); ph[1:-1] = torch.rand(half - 2) * 6.0
f1c = mirror(mag1 * torch.exp(1j * ph))
assert torch.fft.ifft(f1c).imag.abs().max() < 1e-6, "mirror non hermitiano"

# 3. ritardo puro: tutto invariato
w = torch.tensor(2 * np.pi * np.fft.fftfreq(N)[:half], dtype=torch.float32)
f1d = mirror(mag1 * torch.exp(-1j * w * 3.0))          # 3 campioni, dentro jitter_max = 20
d1, d2, _ = an.compute_vars_wiener(W, S, nps, f1d, mirror(mag2 + 0j))
assert abs(d1.item() / v1.item() - 1) < 1e-4, f"un ritardo puro cambia var1: {d1.item()/v1.item()}"
assert abs(d2.item() / v2.item() - 1) < 1e-4, "var2 cambia senza motivo"

# 4. fase casuale -> risposta giu', varianza su
c1, _, _ = an.compute_vars_wiener(W, S, nps, f1c, f2r)
assert c1.item() > v1.item() * (1 + 1e-6), f"la fase casuale non penalizza: {c1.item()/v1.item()}"

# 5. il training con phase=True parte esattamente dall'ottimo reale (p = 0)
kw = dict(N_sigma=1.28, n_trials=2, activation_fct=torch.abs, f1_init=mag1.clone(),
          f2_init=mag2.clone(), verbose=False, use_interp=True, eta_min=1e-2)
tt = torch.tensor([0.0, 3e-4], dtype=torch.float32)
rr = torch.tensor([0.3, 0.5], dtype=torch.float32)
rd = torch.ones(2) / 2
wt = torch.tensor(2 * np.pi * np.fft.fftfreq(N), dtype=torch.cfloat)
args = (S, W, wt, tt, rr, nps, torch.tensor(1.0), rd)
Jr = an.optimize_filters_wiener(*args, phase=False, **kw)[2]
Jp = an.optimize_filters_wiener(*args, phase=True, **kw)[2]
# p = 0 -> stesso filtro; la J differisce solo perche' il ramo complesso calcola la risposta
# come picco della ifft invece che come somma (stessa quantita', float32).
assert abs(Jr[0] / Jp[0] - 1) < 1e-4, f"phase=True non parte dall'ottimo reale: {Jr[0]} vs {Jp[0]}"

print("OK  ramo reale invariato, mirror hermitiano, ritardo neutro, fase penalizzata, "
      f"stesso J iniziale ({Jr[0]:.6f})")
