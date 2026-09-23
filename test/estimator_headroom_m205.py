"""
estimator_headroom_m205.py
==========================
QUANTO MARGINE C'E' PER UN ALTRO STIMATORE?  Prima di cercare un discriminante migliore del
rapporto dei massimi dei due filtri di banda, si misura quanto e' lontano dal MEGLIO POSSIBILE.

Per ogni (dt, r) della griglia del J (stessa griglia, stessa distribuzione di r):
  d_opt   distanza, in sigma di rumore, del pile-up P = A S [(1-r) + r e^{-i w dt}] dalla varieta'
          degli impulsi singoli {B S e^{-i w tau}} (minimo su ampiezza B e tempo tau). E' il limite
          di Neyman-Pearson per QUALSIASI test a 90% di accettanza con ampiezza e tempo ignoti,
          preso (dt, r) per (dt, r): nessuno stimatore sulla sola forma d'onda fa meglio.
  d_curv  media, in sigma, dello stimatore di LARGHEZZA  T = Re<X e^{i w t_OF}, g>/||g||,
          g = S (M2/M0 - w^2): la direzione di s2 = d2s/dt2 resa ortogonale a s (e alla derivata
          prima, per simmetria). Viene dallo sviluppo del pile-up attorno al baricentro temporale:
              p = s + (1/2) r(1-r) dt^2 s2 + (1/6) r(1-r)(2r-1) dt^3 s3 + ...
          il prim'ordine e' una traslazione (indistinguibile da un singolo), il primo termine che
          discrimina e' un ALLARGAMENTO simmetrico; l'asimmetria, l'unica cosa che una fase nei
          filtri potrebbe usare, arriva al terz'ordine e si annulla per r = 1/2.
BI = K * mean( Phi(z_90 - d) * rd ), come J. Metrica <a,b> = sum Re(a conj b)/nps: quella in cui
sigma_OF = 1/sqrt(sum|S|^2/nps), verificata contro il MC. Template = fit (la verita' del MC).

Misurato (2026-09-23), rispetto al BI Monte Carlo dei filtri Wwna _hist:
    ch  SNR_OF   Wwna sopra il limite   larghezza sopra il limite
    34   100        +1.7 %                   +1.1 %
    83    67        +2.6 %                   +1.6 %
    71    37        +1.8 %                   +1.5 %
    91    25        +4.4 %                   +6.1 %
    31    22        ~+5 %                    ~+13 %
-> il metodo addestrato e' gia' entro il 2-5 % dal limite di QUALSIASI stimatore sulla forma d'onda.
Con "--snr": lo stesso limite con SNR x sqrt(2) e x 2 (piu' INFORMAZIONE, non uno stimatore
migliore), rispetto al Wwna di oggi: ch31 wp7 -23 % / -37 %, ch34 wp15 -18 % / -32 %,
ch91 wp15 -22 % / -36 %, ch71 wp21 -18 % / -31 %. L'informazione vale 3-10 volte lo stimatore.

    KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/bin/python3.13 test/estimator_headroom_m205.py 31,34
    KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/bin/python3.13 test/estimator_headroom_m205.py --snr
"""
import sys, csv, glob, numpy as np
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
import src.analysis as an, utility.functions as fn
from scipy.stats import norm
from utility.double_beta_spectrum import pdf_ratio2b

Z = norm.ppf(0.9)
NG = 100
DT = np.linspace(0, 8e-4, NG); RR = np.linspace(0, 0.5, NG)
RD = pdf_ratio2b(RR); RD /= RD.mean()
TAU = np.arange(-1.2e-3, 1.2e-3 + 1e-9, 1e-6)          # griglia fine del tempo, 1 us

def rows(d):
    return {(int(r["channel"]), int(r["wp"])): r for r in csv.DictReader(open(glob.glob(f"{BASE}/{d}/BI_results_*.csv")[0]))}
def mc(d):
    return {(int(r["channel"]), int(r["wp"])): float(r["BI_mc"]) for r in csv.DictReader(open(f"{BASE}/{d}/BI_mc_error_m205.csv"))
            if r["gen"] == "fit" and r["seed"] == "1234"}
W_ = "m205_results_wiener_APsimfit10000led_npsclean_swna1_hist"; O_ = "m205_results_octopus_APsimfit10000led_npsclean_hist"
amp, mcW, mcO = rows(W_), mc(W_), mc(O_)

def point(ch, wp, scale=1.0):
    A = scale * float(amp[(ch, wp)]["signal_amp"])
    gen = np.load(f"{BASE}/residual_scan_bessel/fits_octopus/bestfit_ch{ch}_wp{wp}.npy")
    nps = np.load(f"{BASE}/m205_NPS_clean/ch{ch}/nps_ch{ch}_wp{wp}.npy")
    S, w, _ = an.compute_H(gen, nps, np.hanning, sampling_rate=10000)
    p = np.abs(S)**2 / nps
    M0, M2, M4 = p.sum(), (w**2 * p).sum(), (w**4 * p).sum()
    C = np.cos(np.outer(TAU, w))                         # (n_tau, n_bin)
    R = C @ p                                            # autocorrelazione nella metrica del rumore
    Q = C @ (p * (M2/M0 - w**2))                         # risposta della direzione di larghezza
    gnorm = np.sqrt(M4 - M2**2/M0)
    Ri = lambda t: np.interp(t, TAU, R); Qi = lambda t: np.interp(t, TAU, Q)
    dopt = np.zeros((NG, NG)); dcur = np.zeros((NG, NG))
    win = (TAU >= -1e-4) & (TAU <= 9e-4); tw = TAU[win]
    for j, dt in enumerate(DT):
        c = (1 - RR[:, None]) * Ri(tw)[None, :] + RR[:, None] * Ri(tw - dt)[None, :]   # (r, tau)
        k = c.argmax(axis=1); ts = tw[k]; cmax = c[np.arange(NG), k]
        P2 = ((1 - RR)**2 + RR**2) * M0 + 2 * RR * (1 - RR) * Ri(dt)
        dopt[:, j] = A * np.sqrt(np.clip(P2 - cmax**2 / M0, 0, None))
        dcur[:, j] = A * ((1 - RR) * Qi(ts) + RR * Qi(ts - dt)) / gnorm
    bi = lambda d: fn.K * np.mean(norm.cdf(Z - d) * RD[:, None])
    return dict(snr=A*np.sqrt(M0), bound=bi(dopt), curv=bi(dcur), dopt=dopt, dcur=dcur)

if sys.argv[1:] == ["--snr"]:
    print(f"{'punto':>10} | {'Wwna oggi (MC)':>14} {'limite oggi':>11} | {'limite SNR x1.41':>16} {'limite SNR x2':>13}")
    for ch, wp in ((31, 7), (34, 15), (91, 15), (71, 21)):
        b1, b2, b3 = (point(ch, wp, s)["bound"] for s in (1.0, 2**0.5, 2.0))
        m = mcW[(ch, wp)]
        print(f"ch{ch} wp{wp:<3} | {m:14.3e} {100*(b1/m-1):+10.1f}% | {100*(b2/m-1):+15.1f}% {100*(b3/m-1):+12.1f}%")
    sys.exit()
CH = [int(c) for c in sys.argv[1].split(",")]
print(f"{'ch':>3} {'wp':>3} {'SNR_OF':>7} | {'BI limite':>10} {'BI larghezza':>12} | {'BI_mc OF':>9} {'BI_mc Wwna':>10} | {'larghezza/limite':>16} {'Wwna/limite':>11} {'Wwna/larghezza':>14}")
for ch in CH:
    for wp in sorted(w for (c, w) in amp if c == ch):
        if (ch, wp) not in mcW or (ch, wp) not in mcO: continue
        r = point(ch, wp)
        print(f"{ch:>3} {wp:>3} {r['snr']:7.1f} | {r['bound']:10.3e} {r['curv']:12.3e} | {mcO[(ch,wp)]:9.3e} {mcW[(ch,wp)]:10.3e} | "
              f"{r['curv']/r['bound']:16.3f} {mcW[(ch,wp)]/r['bound']:11.3f} {mcW[(ch,wp)]/r['curv']:14.3f}", flush=True)
