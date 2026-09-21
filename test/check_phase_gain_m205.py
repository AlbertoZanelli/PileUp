"""
check_phase_gain_m205.py
========================
La FASE LIBERA dei filtri di banda (PHASE in analysis_BI_m205_wiener_regolarized.py) conviene?
Verdetto sul MONTE CARLO, cioe' l'unico che conta: l'analitico premia i filtri che rompono il
suo stesso modello (cfr. lambda).

Si addestrano i filtri REALI fino a convergenza, poi si apre SOLA la fase per 10, 25, 50, 100
passi in piu' e si misura il BI Monte Carlo di ogni stadio sugli STESSI eventi (generati dal
fit, la "verita'"). Un punto solo, griglia ridotta 40x40: serve il confronto, non il numero.

Misurato (ch34 wp15, 20000 eventi/popolazione, 2026-09-21, dopo la correzione del taglio in
compute_A):
    real 500    BI_an 9.35e-5  BI_mc 9.49e-5  MC/an 1.01  dBI_mc   +0.0%  sigmaY an/mc 1.02
    +phase 10   BI_an 1.04e-4  BI_mc 1.08e-4  MC/an 1.04  dBI_mc  +14.3%
    +phase 25   BI_an 9.74e-5  BI_mc 1.05e-4  MC/an 1.07  dBI_mc  +10.2%
    +phase 50   BI_an 9.15e-5  BI_mc 1.04e-4  MC/an 1.14  dBI_mc   +9.6%
    +phase 100  BI_an 6.05e-5  BI_mc 1.04e-4  MC/an 1.71  dBI_mc   +9.1%
    phase 500   BI_an 3.78e-5  BI_mc 2.36e-4  MC/an 6.24  dBI_mc +149%    sigmaY an/mc 0.11
cioe' l'analitico migliora (-60% a 500 passi) e il BI VERO peggiora SEMPRE, gia' dai primi
passi. La penalita' su s NON se ne accorge (s resta 0.03-0.06): a rompersi non e' la
risoluzione delle singole bande ma la sigma_Y del RAPPORTO, che l'analitico sottostima di 8.7
volte (0.0019 contro 0.0166 misurata) perche' la fase la ottiene come RESIDUO di una
cancellazione fra numeratore e denominatore -- lo stesso meccanismo del collasso di lambda.
Il rapporto sigmaY an/mc sui singoli e' la guardia giusta: 1.02 con f reale, 0.11 con la fase.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 /opt/homebrew/bin/python3.13 \
        test/check_phase_gain_m205.py
(~6 minuti; OMP_NUM_THREADS=1 e' necessario, con piu' thread torch+scipy vanno in segfault)
"""
import sys, csv, time
import numpy as np, torch
BASE = "/Users/albertozanelli/Desktop/Tesi_Erasmus/PileUp"
sys.path.insert(0, BASE)
import src.analysis as an, src.simulation as sim, src.dataset as ds
import utility.functions as fn
from scipy.stats import norm
from utility.double_beta_spectrum import pdf_ratio2b

CH, WP, NGRID = 34, 15, 40
VB = np.array([0.6,1.0,1.4,1.8,2,3,4,5,6,8,10,20,26,30,40])
vb = round(float(VB[WP//2]), 3)
amp = {(int(r["channel"]), round(float(r["vbias_V"]),3)): float(r["amplitude_mV"])*1e-3
       for r in csv.DictReader(open(f"{BASE}/amplitudes_m205.csv")) if r["amplitude_mV"].strip()}
signal_amp = amp[(CH, vb)]
pulse = np.load(f"{BASE}/m205_AP_sim/ch{CH}/simAP_APsimfit10000led_ch{CH}_wp{WP}.npy")
nps = np.load(f"{BASE}/m205_NPS_clean/ch{CH}/nps_ch{CH}_wp{WP}.npy")
S, w, _ = an.compute_H(pulse, nps, np.hanning, sampling_rate=10000)
to = lambda a, d=torch.cfloat: torch.tensor(np.asarray(a), dtype=d)
r = np.linspace(0, 0.5, NGRID); rd = pdf_ratio2b(r); rd /= rd.mean()
S_t, w_t, nps_t = to(S), to(w), to(nps)
t_t, r_t, rd_t = to(np.linspace(0, 8e-4, NGRID)), to(r), to(rd)
amp_t = torch.tensor(signal_amp, dtype=torch.float32)
W_unit = an.compute_W_torch(S_t, torch.tensor(nps, dtype=torch.float32), torch.tensor(1.0))
N_sigma = float(norm.ppf(0.9))
common = dict(N_sigma=N_sigma, activation_fct=torch.abs, eta_min=1e-2,
              use_interp=True, verbose=False)
args = (S_t, W_unit, w_t, t_t, r_t, nps_t, amp_t, rd_t)

g = torch.Generator().manual_seed(0)
n = len(S)//2 + 1
i1, i2 = torch.abs(torch.rand(n, generator=g)), torch.abs(torch.rand(n, generator=g))
f1b, f2b, Jb = an.optimize_filters_wiener(*args, f1_init=i1.clone(), f2_init=i2.clone(),
                                          n_trials=500, phase=False, **common)
sets = [("real 500", f1b, f2b, Jb[-1])]
for k in (10, 25, 50, 100):
    a, b, J = an.optimize_filters_wiener(*args, f1_init=f1b.clone(), f2_init=f2b.clone(),
                                         n_trials=k, phase=True, **common)
    sets.append((f"+phase {k}", a, b, J[-1]))
# e il training COMPLETO con la fase, dallo stesso punto di partenza casuale dei filtri reali
a5, b5, J5 = an.optimize_filters_wiener(*args, f1_init=i1.clone(), f2_init=i2.clone(),
                                        n_trials=500, phase=True, **common)
sets.append(("phase 500", a5, b5, J5[-1]))

NSIM, CHUNK = 20000, 500
gen = np.load(f"{BASE}/residual_scan_bessel/fits_octopus/bestfit_ch{CH}_wp{WP}.npy")
Sg, wg, _ = an.compute_H(gen, nps, np.hanning, sampling_rate=10000)
Wn = W_unit.detach().cpu().numpy()
filt = [(Wn, a.detach().cpu().numpy(), b.detach().cpu().numpy()) for _, a, b, _ in sets]
def psd(dt_max, seed):
    rng = np.random.default_rng(seed); o = [[] for _ in filt]
    for k in range(0, NSIM, CHUNK):
        nn = min(CHUNK, NSIM - k)
        fp, *_ = sim.simulate_frequency_pulses(Sg, nps, 0.0, wg, nsim=nn, seed=rng,
                                               signal_scale=signal_amp, dt_max=dt_max,
                                               fold_ratio=False)
        p = np.fft.ifft(fp, axis=1).real.astype(np.float32)
        d = ds.NumpyDataset(p); d.win_length = p.shape[1]
        for oo, (H, a, b) in zip(o, filt):
            oo.append(np.asarray(an.get_PSD_interpole(d, H, a, b)[0]).ravel())
    return [np.concatenate(x) for x in o]
sg, pu = psd(0.0, 1234), psd(8e-4, 1234)
print(f"\nch{CH} wp{WP}, {NSIM} eventi/popolazione, griglia {NGRID}x{NGRID}")
base = None
S_H_del, S_H, S2n = an.precompute_constants(S_t, W_unit, w_t, t_t, nps_t)
for (lbl, a, b, J), ps, pp in zip(sets, sg, pu):
    rp = float(np.mean(pp < np.percentile(ps, 10)))
    bi = fn.K * (1 - rp); base = base or bi
    v1, v2, _ = an.compute_vars_wiener(W_unit, S_t, nps_t, a, b)
    # sigma_Y dei SINGOLI: analitica (1o ordine sul rapporto) contro quella VERA del MC.
    # E' la guardia che s1, s2 non danno: s e' la risoluzione di OGNI banda, mentre qui a
    # rompersi e' la CANCELLAZIONE fra numeratore e denominatore del rapporto.
    muY, sigY = an.compute_mu_sigma_wiener(a, b, S_H_del, r_t, S_H, S2n, W_unit, S_t, nps_t,
                                           amp_t, use_interp=True)
    sig_mc = (np.percentile(ps, 84) - np.percentile(ps, 16)) / 2
    print(f"{lbl:10s} BI_an={J*fn.K:.4g}  BI_mc={bi:.4g}  MC/an={bi/(J*fn.K):8.3g}  "
          f"dBI_mc={100*(bi/base-1):+6.1f}%  s=({float(v1)**.5/signal_amp:.3f},"
          f"{float(v2)**.5/signal_amp:.3f})  sigmaY an/mc={float(sigY[0,0])/sig_mc:5.2f}")
