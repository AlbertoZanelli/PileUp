"""
Controllo della generazione v2 (src/simulation.simulate_frequency_pulses_v2), su ch31 WP7.

1. Con le tre aggiunte spente l'uscita e' IDENTICA, bit per bit, a simulate_frequency_pulses.
2. Singoli (dt_max = 0) e pile-up con lo stesso seed ricevono gli stessi E, tau, ampiezza e r.
3. Energia dentro la ROI, tau dentro [0, 1 campione).
4. Lo shift sposta SOLO il segnale: il rumore resta quello estratto dall'originale.
5. Fluttuazione dei fotoni: la dispersione delle ampiezze (stimate senza rumore) e' 1/sqrt(N).
6. K_for(8e-4) e' il K di sempre.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 /opt/homebrew/bin/python3.13 test/mc/check_generator_v2_m205.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import src.analysis as an
import src.simulation as sim
import utility.functions as fn

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CH, WP, A, N = 31, 7, 1.6e-3, 400
tpl = np.load(f"{BASE}/residual_scan_bessel/fits_octopus/bestfit_ch{CH}_wp{WP}.npy")
nps = np.load(f"{BASE}/m205_NPS_clean/ch{CH}/nps_ch{CH}_wp{WP}.npy")
S, w, _ = an.compute_H(tpl / tpl.max(), nps, np.hanning, sampling_rate=1e4)
v2 = lambda seed, **kw: sim.simulate_frequency_pulses_v2(
    S, nps, w, N, np.random.default_rng(seed), np.random.default_rng([seed, 1]),
    signal_scale=A, dt_max=kw.pop("dt_max", fn.DT_MAX_V2), **kw)

# 1. aggiunte spente = originale
ref = sim.simulate_frequency_pulses(S, nps, 0.0, w, nsim=N, seed=np.random.default_rng(5),
                                    signal_scale=A, dt_max=fn.DT_MAX_V2)
fp, _ = v2(5)
assert np.array_equal(fp, ref[0]), "v2 senza aggiunte diverso dall'originale"

# 2. singoli e pile-up appaiati
on = dict(n_photons=fn.n_photons(CH), e_roi=fn.E_ROI_MEV, shift_max=1e-4)
_, i_s = v2(7, dt_max=0.0, **on)
_, i_p = v2(7, **on)
for k in ("energy", "tau", "amp", "sigma", "r"):
    assert np.array_equal(i_s[k], i_p[k]), f"singoli e pile-up con {k} diverso"
assert not i_s["dt"].any() and i_p["dt"].max() <= fn.DT_MAX_V2

# 3. intervalli
assert fn.E_ROI_MEV[0] <= i_p["energy"].min() and i_p["energy"].max() <= fn.E_ROI_MEV[1]
assert 0 <= i_p["tau"].min() and i_p["tau"].max() < 1e-4

# 4. lo shift tocca solo il segnale
orig = sim.simulate_frequency_pulses(S, nps, 0.0, w, nsim=N, seed=np.random.default_rng(9),
                                     signal_scale=A, dt_max=fn.DT_MAX_V2)
fp, info = v2(9, shift_max=1e-4)
expect = (orig[0] - orig[2]) * np.exp(-1j * w[None, :] * info["tau"][:, None]) + orig[2]
assert np.allclose(fp, expect, rtol=0, atol=1e-12 * np.abs(expect).max()), "lo shift non torna"

# 5. fotoni: ampiezza di ogni evento (singoli, rumore trascurabile) proiettata sul template
n_ph = fn.n_photons(CH)
fp, info = sim.simulate_frequency_pulses_v2(S, nps * 1e-20, w, 4000, np.random.default_rng(3),
                                            np.random.default_rng([3, 1]), signal_scale=A,
                                            dt_max=0.0, n_photons=n_ph)
a_hat = (fp @ np.conj(S)).real / np.sum(np.abs(S) ** 2)
rel = a_hat.std() / a_hat.mean()
assert abs(rel * np.sqrt(n_ph) - 1) < 0.05, f"sigma_A/A = {rel:.4f}, atteso {1 / np.sqrt(n_ph):.4f}"

# 6. K
assert fn.K_for(8e-4) == fn.K

print(f"OK: generazione v2 controllata (sigma_A/A = {rel:.4f}, atteso {1 / np.sqrt(n_ph):.4f}; "
      f"K(2.369 ms)/K(0.8 ms) = {fn.K_for(fn.DT_MAX_V2) / fn.K:.3f})")
