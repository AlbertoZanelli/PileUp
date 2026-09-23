"""
check_family_reach_m205.py
==========================
La soluzione Wiener ADDESTRATA sta nella famiglia del filtro ottimo?

Il filtro applicato e' g = f_W * W. Siccome W/H e' reale positivo a ogni frequenza, lo stesso g
si scrive g = f_OF * H con f_OF = f_W * W/H >= 0: si mappano i filtri Wiener salvati nelle
coordinate dell'OF e si valuta la J con il codice DELL'OF (compute_J). Se coincide con la J
Wiener, quella configurazione e' raggiungibile dall'OF -- e se l'OF addestrato ha J peggiore,
non la trova per ragioni di OTTIMIZZAZIONE, non di famiglia.

Misurato 2026-09-23, ch31, cartelle _hist (500 passi entrambe):
    J_OF(f_W * W/H) = J_W(f_W) entro 6e-6 su 15/15 WP;
    OF addestrato peggiore di 0.4-1.9% (J analitica).

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 /opt/homebrew/bin/python3.13 \
        test/check_family_reach_m205.py 31
"""
import sys, csv, numpy as np, torch
BASE = "/Users/albertozanelli/Desktop/Tesi_Erasmus/PileUp"; sys.path.insert(0, BASE)
import src.analysis as an, utility.functions as fn
from scipy.stats import norm
from utility.double_beta_spectrum import pdf_ratio2b
OFD = f"{BASE}/m205_results_octopus_APsimfit10000led_npsclean_hist/trained_filters"
WD  = f"{BASE}/m205_results_wiener_APsimfit10000led_npsclean_swna1_hist/trained_filters"
full = lambda h: np.concatenate([h, np.conj(h[-2:0:-1]) if np.iscomplexobj(h) else h[-2:0:-1]])
amp = {(int(r["channel"]), int(r["wp"])): float(r["signal_amp"]) for r in
       csv.DictReader(open(f"{BASE}/m205_results_wiener_APsimfit10000led_npsclean_swna1_hist/BI_results_m205_wiener_APsimfit10000led_npsclean_swna1_hist.csv"))}
N = 100
r = np.linspace(0, .5, N); rd = pdf_ratio2b(r); rd /= rd.mean()
T = lambda a: torch.tensor(np.asarray(a), dtype=torch.cfloat)
t_t, r_t, rd_t, Ns = T(np.linspace(0, 8e-4, N)), T(r), T(rd), float(norm.ppf(.9))
CH = int(sys.argv[1])
print(f"ch{CH}:  J_W(f_W)   J_OF(f_W*W/H)   rel.diff   | J_OF(f_OF addestrati)   W vs OF addestrato")
for wp in range(1, 30, 2):
    tpl = np.load(f"{BASE}/m205_AP_sim/ch{CH}/simAP_APsimfit10000led_ch{CH}_wp{wp}.npy")
    nps = np.load(f"{BASE}/m205_NPS_clean/ch{CH}/nps_ch{CH}_wp{wp}.npy")
    S, w, H = an.compute_H(tpl, nps, np.hanning, sampling_rate=10000)
    Wk = full(np.load(f"{WD}/kernel_ch{CH}_wp{wp}.npy"))
    fw1, fw2 = (full(np.load(f"{WD}/f{i}_ch{CH}_wp{wp}.npy")) for i in (1, 2))
    fo1, fo2 = (full(np.load(f"{OFD}/f{i}_ch{CH}_wp{wp}.npy")) for i in (1, 2))
    ratio = Wk / H                           # reale positivo (verificato nell'handoff)
    assert np.abs(ratio.imag).max() < 1e-5 * np.abs(ratio).max()
    m1, m2 = fw1 * ratio.real, fw2 * ratio.real   # stessi filtri applicati, coordinate OF
    S_t, w_t, n_t, a = T(S), T(w), T(nps), torch.tensor(amp[(CH, wp)])
    Wt, Ht = T(Wk), T(H)
    Wd, WH, W2 = an.precompute_constants(S_t, Wt, w_t, t_t, n_t)
    Jw = float(an.compute_J_wiener(T(fw1), T(fw2), Wd, r_t, WH, W2, Wt, S_t, n_t, a, rd_t,
                                   N_sigma=Ns, use_interp=True))
    Sd, SH, S2 = an.precompute_constants(S_t, Ht, w_t, t_t, n_t)
    Jm = float(an.compute_J(T(m1), T(m2), Sd, r_t, SH, S2, a, rd_t, N_sigma=Ns, use_interp=True))
    Jo = float(an.compute_J(T(fo1), T(fo2), Sd, r_t, SH, S2, a, rd_t, N_sigma=Ns, use_interp=True))
    print(f"wp{wp:>2}: {Jw:.6f}   {Jm:.6f}     {Jm/Jw-1:+.1e}   | {Jo:.6f}              {100*(Jw/Jo-1):+.2f}%")
