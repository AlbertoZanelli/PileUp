"""
likelihood_estimator_m205.py
============================
Un discriminante oltre al rapporto dei massimi: il RAPPORTO DI VEROSIMIGLIANZA MEDIATO sul modello
del pile-up (M). Per ogni evento chiede: quanto e' piu' verosimile come pile-up di due impulsi --
a qualsiasi dt e r, pesati con la loro distribuzione vera -- che come impulso singolo?

    y(t)      = <X, S e^{-iwt}>                  correlazione OF dell'evento (metrica del rumore)
    L1        = max_t y(t)^2 / R0                un impulso, ampiezza e tempo profilati
    L(dt, r)  = max_t max(0, (1-r) y(t) + r y(t+dt))^2 / ||P||^2,  P = S[(1-r) + r e^{-iw dt}]
    M         = log sum_{dt,r} pi(dt, r) exp((L(dt,r) - L1)/2),  pi = uniforme in dt x pdf_ratio2b(r)
Per il lemma di Neyman-Pearson e' il test ottimo per una figura di merito che MEDIA su (dt, r), come
il BI. Non si addestra niente: servono solo il template e la NPS. Il template e' l'AP SIMULATO,
lo stesso su cui sono addestrati i filtri di Y; gli eventi sono generati dal fit.

Protocollo: due campioni MC indipendenti (seme 1111 = scelta della combinazione Y oppure M,
seme 2222 = misura), taglio al 90% dei singoli del campione di misura, BI = K * frazione di pile-up
sotto il taglio, errori da bootstrap APPAIATO sugli eventi.

Misurato (30 000 eventi per popolazione, filtri Wwna a 500 passi, 2026-09-23):
    punto      SNR   M da solo vs Y     Y oppure M vs Y    limite teorico vs Y
    ch31 wp7    20   -2.32 +- 0.42 %    -2.52 +- 0.35 %    -5.6 %
    ch91 wp15   25   -1.75 +- 0.27 %    -1.74 +- 0.27 %    -5.1 %
    ch34 wp15  124   -0.41 +- 0.31 %    -0.28 +- 0.27 %    -2.4 %
Con il template VERO (il fit) al posto dell'AP simulato: -2.24 / -1.72 / -0.48 %, cioe' lo stesso.
Il guadagno sta a dt > 0.4 ms (ch31, 0.6-0.8 ms: sopravvivenza 1.9 % -> 0.3 %), dove il rapporto
dei massimi SATURA: quando i due impulsi si separano il massimo della banda alta si aggancia al solo
impulso maggiore e Y smette di crescere con dt. Il 78 % del BI viene da dt < 0.2 ms, dove Y e' gia'
sul limite teorico: a questo SNR sono irrisolvibili per qualsiasi stimatore.
Passi intermedi (in HANDOFF.md, punto 8): Y + larghezza non guadagna niente (correlazione 0.95-0.99
sui singoli: sono la stessa misura); Y oppure un GLRT a due impulsi (ampiezze libere) -1.9 % su ch31.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 /opt/homebrew/bin/python3.13 \\
        test/likelihood_estimator_m205.py <canale> <wp> [n_eventi=30000]     (~15 min a 30 000)
"""
import os, sys, csv, glob, time, numpy as np
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, BASE)
import src.analysis as an, src.simulation as sim, src.dataset as ds, utility.functions as fn
from scipy.stats import norm
from src.pileup_likelihood import PileupLikelihood

CH, WP = int(sys.argv[1]), int(sys.argv[2])
NSIM = int(sys.argv[3]) if len(sys.argv) > 3 else 30000
WD = f"{BASE}/m205_results_wiener_APsimfit10000led_npsclean_swna1"      # filtri di Y
A = {(int(r["channel"]), int(r["wp"])): float(r["signal_amp"])
     for r in csv.DictReader(open(glob.glob(f"{WD}/BI_results_*.csv")[0]))}[(CH, WP)]
full = lambda h: np.concatenate([h, np.conj(h[-2:0:-1]) if np.iscomplexobj(h) else h[-2:0:-1]])
K, f1, f2 = (full(np.load(f"{WD}/trained_filters/{n}_ch{CH}_wp{WP}.npy")) for n in ("kernel", "f1", "f2"))
nps = np.load(f"{BASE}/m205_NPS_clean/ch{CH}/nps_ch{CH}_wp{WP}.npy")
S, w, _ = an.compute_H(np.load(f"{BASE}/residual_scan_bessel/fits_octopus/bestfit_ch{CH}_wp{WP}.npy"),
                       nps, np.hanning, sampling_rate=10000)                  # GENERA gli eventi
S_l, _, _ = an.compute_H(np.load(f"{BASE}/m205_AP_sim/ch{CH}/simAP_APsimfit10000led_ch{CH}_wp{WP}.npy"),
                         nps, np.hanning, sampling_rate=10000)                # entra nella verosimiglianza
N = len(S)
M_of = PileupLikelihood(S_l, nps, w, dt_max=8e-4).statistic      # lo stimatore: src/pileup_likelihood.py

t0 = time.perf_counter(); D = {}
for seed in (1111, 2222):
    for pop, dtm in (("s", 0.0), ("p", 8e-4)):
        rng = np.random.default_rng(seed); Y, M, DT = [], [], []
        for k in range(0, NSIM, 500):
            fp, _, _, _, dd = sim.simulate_frequency_pulses(S, nps, 0.0, w, nsim=min(500, NSIM - k), seed=rng,
                                                            signal_scale=A, dt_max=dtm)
            x = np.fft.ifft(fp, axis=1).real
            M.append(M_of(x)); DT.append(dd)
            d = ds.NumpyDataset(x.astype(np.float32)); d.win_length = N
            Y.append(-np.asarray(an.get_PSD_interpole(d, K, f1, f2)[0]).ravel())    # segno: grande = pile-up
        D[(seed, pop)] = (np.concatenate(Y), np.concatenate(M), np.concatenate(DT))
print(f"ch{CH} wp{WP}: eventi e statistiche in {time.perf_counter()-t0:.0f} s")

def zmap(ref):
    r = np.sort(ref); n = len(r)
    return lambda v: norm.ppf(np.clip((np.searchsorted(r, v, side="right") - 0.5) / n, 0.5/n, 1 - 0.5/n))
zu, zm = zmap(D[(1111, "s")][0]), zmap(D[(1111, "s")][1])
def bi(Ds, Dp): return fn.K * np.mean(Dp <= np.percentile(Ds, 90))
OR = lambda k: (lambda y, m: np.maximum(zu(y), zm(m) + k))
grid = np.linspace(-6, 3, 181)
kk = grid[int(np.argmin([bi(OR(k)(*D[(1111, "s")][:2]), OR(k)(*D[(1111, "p")][:2])) for k in grid]))]
METHODS = [("Y (rapporto dei massimi, oggi)", lambda y, m: y), ("M (verosimiglianza mediata)", lambda y, m: m),
           ("Y oppure M", OR(kk))]
(Ys, Ms, _), (Yp, Mp, dtp) = D[(2222, "s")], D[(2222, "p")]
Sx = [f(Ys, Ms) for _, f in METHODS]; Px = [f(Yp, Mp) for _, f in METHODS]; BI = [bi(s, p) for s, p in zip(Sx, Px)]
rng = np.random.default_rng(0); n = len(Ys); boot = []
for _ in range(400):
    i = rng.integers(0, n, n); b0 = bi(Sx[0][i], Px[0][i]); boot.append([100*(bi(Sx[k][i], Px[k][i])/b0 - 1) for k in range(3)])
boot = np.array(boot)
print(f"\nch{CH} wp{WP}: scelta sul seme 1111, misura sul 2222, {n} eventi per popolazione")
for k, (name, _) in enumerate(METHODS):
    print(f"  {name:34s} BI_mc {BI[k]:.4e}" + (f"   {100*(BI[k]/BI[0]-1):+6.2f} +- {boot[:, k].std():.2f} % vs Y" if k else ""))
cs = [np.percentile(s, 90) for s in Sx]
print("  sopravvivenza dei pile-up per fascia di dt:     Y        M")
for lo, hi in ((0, 1e-4), (1e-4, 2e-4), (2e-4, 4e-4), (4e-4, 6e-4), (6e-4, 8e-4)):
    m = (dtp >= lo) & (dtp < hi)
    print(f"    dt {lo*1e3:.1f}-{hi*1e3:.1f} ms                          {100*np.mean(Px[0][m] <= cs[0]):5.1f}%   {100*np.mean(Px[1][m] <= cs[1]):5.1f}%")
