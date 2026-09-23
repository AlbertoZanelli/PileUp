"""
width_estimator_mc_m205.py
==========================
Lo stimatore di LARGHEZZA (vedi test/estimator_headroom_m205.py) sul MONTE CARLO, appaiato con i
filtri Wwna _hist sugli STESSI eventi:
    T = Re<X e^{i w t_OF}, g>/||g||,   g = S (M2/M0 - w^2)
con t_OF dal massimo della correlazione OF (+-20 campioni) raffinato con una parabola. Lineare
nei dati a tempo fissato, a norma unitaria: sui singoli media 0 e deviazione standard 1, senza
training, senza rapporto -> il modello analitico e' esatto al prim'ordine. Taglio al 90% dei
singoli, si rigetta T grande (il pile-up e' piu' largo: T > 0).

Misurato (30 000 eventi appaiati per popolazione, 2026-09-23):
    ch34 wp15 (SNR 124): singoli T 0.00 +- 1.00;  BI larghezza 9.381e-5, Wwna 9.368e-5
                         -> +0.14 +- 0.33 %, e MC/analitico = 0.994 per la larghezza
    ch31 wp7  (SNR 20):  singoli T -0.05 +- 1.00; BI larghezza 7.798e-5, Wwna 6.609e-5
                         -> +18.0 +- 0.8 %  (a basso SNR si risolvono solo i dt grandi, dove lo
                         sviluppo al second'ordine non basta)

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 /opt/homebrew/bin/python3.13 \
        test/width_estimator_mc_m205.py <canale> <wp>          (~5 min)
"""
import sys, csv, glob, numpy as np
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
import src.analysis as an, src.simulation as sim, src.dataset as ds, utility.functions as fn
CH, WP, NSIM = int(sys.argv[1]), int(sys.argv[2]), 30000
WD = f"{BASE}/m205_results_wiener_APsimfit10000led_npsclean_swna1_hist"
A = {(int(r["channel"]), int(r["wp"])): float(r["signal_amp"]) for r in csv.DictReader(open(glob.glob(f"{WD}/BI_results_*.csv")[0]))}[(CH, WP)]
full = lambda h: np.concatenate([h, np.conj(h[-2:0:-1]) if np.iscomplexobj(h) else h[-2:0:-1]])
K, f1, f2 = (full(np.load(f"{WD}/trained_filters/{n}_ch{CH}_wp{WP}.npy")) for n in ("kernel", "f1", "f2"))
gen = np.load(f"{BASE}/residual_scan_bessel/fits_octopus/bestfit_ch{CH}_wp{WP}.npy")
nps = np.load(f"{BASE}/m205_NPS_clean/ch{CH}/nps_ch{CH}_wp{WP}.npy")
S, w, _ = an.compute_H(gen, nps, np.hanning, sampling_rate=10000)
p = np.abs(S)**2 / nps; M0, M2, M4 = p.sum(), (w**2*p).sum(), (w**4*p).sum()
g = S * (M2/M0 - w**2); gn = np.sqrt(M4 - M2**2/M0)
N = len(S); J = 20
def width_stat(x):                               # x: (n, N) impulsi nel tempo
    X = np.fft.fft(x, axis=1)
    y = np.fft.ifft(X * np.conj(S) / nps, axis=1).real          # correlazione OF
    idx = np.r_[N-J:N, 0:J+1]                                    # +-J campioni attorno a 0
    k = idx[y[:, idx].argmax(axis=1)]
    ym, y0, yp = y[np.arange(len(x)), (k-1) % N], y[np.arange(len(x)), k], y[np.arange(len(x)), (k+1) % N]
    frac = 0.5 * (ym - yp) / (ym - 2*y0 + yp)                   # vertice della parabola
    tau = ((k + N//2) % N - N//2 + frac) / 10000.0               # secondi
    return (X * np.exp(1j * w[None, :] * tau[:, None]) * np.conj(g)[None, :] / nps).real.sum(axis=1) / gn
T, PSD = ([], []), ([], [])
for pop, dt in ((0, 0.0), (1, 8e-4)):
    rng = np.random.default_rng(4242)
    for k in range(0, NSIM, 500):
        fp, *_ = sim.simulate_frequency_pulses(S, nps, 0.0, w, nsim=500, seed=rng, signal_scale=A, dt_max=dt, fold_ratio=False)
        x = np.fft.ifft(fp, axis=1).real
        T[pop].append(width_stat(x))
        d = ds.NumpyDataset(x.astype(np.float32)); d.win_length = N
        PSD[pop].append(np.asarray(an.get_PSD_interpole(d, K, f1, f2)[0]).ravel())
Ts, Tp = map(np.concatenate, T); Ps, Pp = map(np.concatenate, PSD)
bi_T = lambda ts, tp: fn.K * np.mean(tp < np.percentile(ts, 90))       # sopravvive se T sotto il taglio
bi_Y = lambda ps, pp: fn.K * (1 - np.mean(pp < np.percentile(ps, 10)))  # come simulate_BI_error: 1 - rigettati
BT, BY = bi_T(Ts, Tp), bi_Y(Ps, Pp)
rng = np.random.default_rng(0); D = []
for _ in range(300):
    i = rng.integers(0, NSIM, NSIM); D.append(100 * (bi_T(Ts[i], Tp[i]) / bi_Y(Ps[i], Pp[i]) - 1))
print(f"ch{CH} wp{WP}: singoli T media {Ts.mean():+.3f}  dev.std {Ts.std():.3f}  (attesi 0 e 1)")
print(f"  BI_mc larghezza  = {BT:.4e}")
print(f"  BI_mc Wwna       = {BY:.4e}")
print(f"  larghezza vs Wwna (appaiato): {100*(BT/BY-1):+.2f} +- {np.std(D):.2f} %")
