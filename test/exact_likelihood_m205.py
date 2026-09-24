"""
exact_likelihood_m205.py
========================
M prende, per ogni ipotesi, l'altezza e la posizione che fittano MEGLIO. Il test ottimo di
Neyman-Pearson invece le MEDIA. Qui si misura se mediarle migliora il BI, in due versioni:

  E_libera : altezza non nota (media su tutte le altezze positive), posizione mediata
             uniformemente sulla finestra di ricerca. Stessa informazione di M e di Y.
  E_nota   : altezza NOTA (quella della ROI, come nel MC dove e' sempre la stessa), posizione
             mediata. Usa un'informazione in piu': nella realta' viene dal canale di calore.

Per un modello m = a T e^{-iwt} il log del rapporto di verosimiglianza rispetto a "niente" e'
a c - a^2 n / 2, con c = <X, T e^{-iwt}> e n = ||T||^2 (singolo: c = y(t), n = R0; pile-up:
c = (1-r) y(t) + r y(t+dt), n = ||P||^2, gli stessi di src/pileup_likelihood.py).
  altezza nota a = A    : a c - a^2 n / 2
  altezza libera a >= 0 : log int_0^inf exp(a c - a^2 n/2) da
                          = c^2/(2n) - log(n)/2 + log Phi(c / sqrt n)   (+ costante che si cancella)
Poi logsumexp sui tempi (pesi uniformi) e, per il pile-up, su (dt, r) con i pesi del prior.

Protocollo: il campione di MISURA di test/likelihood_estimator_m205.py (seme 2222, stesso numero di
eventi), cosi' M e Y danno gli stessi numeri gia' noti; taglio al 90% dei singoli; errori da
bootstrap appaiato sugli eventi.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=2 python test/exact_likelihood_m205.py <ch> <wp> [n=30000]
"""
import os, sys, csv, glob, time, numpy as np
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, BASE)
import src.analysis as an, src.simulation as sim, src.dataset as ds, utility.functions as fn
from scipy.special import logsumexp, log_ndtr
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
lik = PileupLikelihood(S_l, nps, w, dt_max=8e-4)


def exact(x):
    """(E_libera, E_nota) per ogni evento: come lik.statistic, con le medie al posto dei massimi."""
    y = lik.correlation(x)
    y1 = y[:, :lik.n_t1]
    free = lambda c, n: c ** 2 / (2 * n) - 0.5 * np.log(n) + log_ndtr(c / np.sqrt(n))
    known = lambda c, n: A * c - A ** 2 * n / 2
    one = [logsumexp(f(y1, lik.R0), axis=1) for f in (free, known)]       # ipotesi: un impulso
    terms = ([], [])
    for j, d in enumerate(lik.d_steps):
        c = (1 - lik.r)[None, :, None] * y1[:, None, :] + lik.r[None, :, None] * y[:, None, d:d + lik.n_t1]
        n = lik.normP[j][None, :, None]
        for k, f in enumerate((free, known)):
            terms[k].append(lik.log_prior[None, :] + logsumexp(f(c, n), axis=2))
    return [logsumexp(np.concatenate(t, axis=1), axis=1) - o for t, o in zip(terms, one)]


t0 = time.perf_counter(); D = {}
for pop, dtm in (("s", 0.0), ("p", 8e-4)):
    rng = np.random.default_rng(2222); cols = [[] for _ in range(4)]
    for k in range(0, NSIM, 500):
        fp, *_ = sim.simulate_frequency_pulses(S, nps, 0.0, w, nsim=min(500, NSIM - k), seed=rng,
                                               signal_scale=A, dt_max=dtm)
        x = np.fft.ifft(fp, axis=1).real
        d = ds.NumpyDataset(x.astype(np.float32)); d.win_length = N
        cols[0].append(-np.asarray(an.get_PSD_interpole(d, K, f1, f2)[0]).ravel())   # grande = pile-up
        cols[1].append(lik.statistic(x))
        for c, v in zip(cols[2:], exact(x)):
            c.append(v)
    D[pop] = [np.concatenate(c) for c in cols]
print(f"ch{CH} wp{WP}: {NSIM} eventi per popolazione in {time.perf_counter()-t0:.0f} s")

NAMES = ["Y (rapporto dei massimi)", "M (fit migliore)", "E_libera (altezza mediata)", "E_nota (altezza nota)"]
def bi(s, p): return fn.K * np.mean(p <= np.percentile(s, 90))   # scartati: sopra il 90% dei singoli
bis = [bi(s, p) for s, p in zip(D["s"], D["p"])]
rng = np.random.default_rng(0); boot = []
for _ in range(400):                                   # bootstrap appaiato: stessi eventi per tutti
    i = rng.integers(0, NSIM, NSIM)
    boot.append([bi(s[i], p[i]) for s, p in zip(D["s"], D["p"])])
boot = np.array(boot)
print(f"\n{'':28s}{'BI':>11s}   vs Y            vs M")
for k, name in enumerate(NAMES):
    vy = 100 * (boot[:, k] / boot[:, 0] - 1); vm = 100 * (boot[:, k] / boot[:, 1] - 1)
    print(f"{name:28s}{bis[k]:11.4e}   {100*(bis[k]/bis[0]-1):+6.2f} +- {vy.std():.2f}%   "
          f"{100*(bis[k]/bis[1]-1):+6.2f} +- {vm.std():.2f}%")
