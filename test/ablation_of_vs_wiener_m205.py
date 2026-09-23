"""
ablation_of_vs_wiener_m205.py
=============================
DA DOVE VIENE il guadagno del Wiener con lambda addestrabile sull'OF? Ablazione causale su UN
punto: da OF (come nella campagna) a Wiener-lambda + penalita' (come nella campagna), cambiando
UN fattore alla volta, e ogni braccio giudicato sul MONTE CARLO con eventi APPAIATI.

Perche' si puo' fare con un codice solo: optimize_filters_wiener con kernel H (il filtro ottimo)
E' l'OF -- la J coincide a 1e-6 (verificato su 45 coppie: il filtro Wiener addestrato, riscritto
come f_OF = f_W * W/H, ha fattore reale positivo e la stessa J). Quindi fra un braccio e l'altro
cambia solo quello che dice l'etichetta. Stessa inizializzazione dei filtri (seme 0) per tutti.

    A  OF, lr che decade a 1e-5          = la campagna OF (optimize_filters, eta_min 1e-5)
    B  OF, lr costante 1e-2              cambia: lo SCHEDULE
    C  Wiener lambda = 1 fissa           cambia: il KERNEL
    D  Wiener lambda addestrabile        cambia: LAMBDA
    E  + penalita' ("wna", 1.0)          cambia: la PENALITA'   = la campagna Wwna

Misurato (ch31 wp7, griglia 50x50, 500 passi, MC 30 000 eventi appaiati, errori da bootstrap
appaiato sugli eventi, 2026-09-23):
    A  BI_mc 6.739e-5                                 riferimento
    B  BI_mc 6.646e-5   -1.38 +- 0.35 %  vs A         lo schedule da solo
    C  BI_mc 6.794e-5   +2.24 +- 0.37 %  vs B         il kernel di Wiener a lambda=1 PEGGIORA
    D  BI_mc 6.684e-5   -1.63 +- 0.28 %  vs C         lambda addestrabile recupera
    E  BI_mc 6.618e-5   -0.98 +- 0.24 %  vs D         la penalita' aiuta
    A -> E: -1.79 +- 0.38 %, di cui -1.38 % e' lo schedule; il resto (E vs B, -0.41 %) e' entro
    gli errori. Gli estremi riproducono la campagna (analitico A->E -1.9 %, campagna -2.0 %).
Lambda finale: 6.1 senza penalita', 29.9 con: cresce verso il kernel dell'OF (W -> S*/(lam NPS)).

    F  Wwna con lr che decade a 1e-5    lo schedule dell'OF: il 2x2 {OF, Wwna} x {decade, costante}
Misurato 2x2 (bracci A, B, E, F; A, B, E riprodotti BIT PER BIT dal run precedente):
                                      ch31 wp7            ch34 wp15 (controllo)
    Wwna - OF, lr che decade  F vs A  +0.37 +- 0.27 %     +0.52 +- 0.33 %
    Wwna - OF, lr costante    E vs B  -0.41 +- 0.28 %     +0.17 +- 0.23 %
    schedule, OF              B vs A  -1.38 +- 0.35 %     -0.13 +- 0.30 %
    schedule, Wwna            E vs F  -2.15 +- 0.39 %     -0.48 +- 0.27 %
    interazione                       -0.78 +- 0.36 %     -0.35 +- 0.37 %
A parita' di schedule il Wwna NON e' migliore dell'OF (media pesata dei due punti: +0.43 +- 0.21 %
a lr che decade, -0.06 +- 0.18 % a lr costante). Lo schedule conta dove il training non e'
arrivato in fondo (ch31) e quasi niente dove lo e' (ch34). Con lr che decade lambda si ferma a 6.2
invece di 29.9: il decadimento ferma anche la deriva di lambda.
Uso:  ... ablation_of_vs_wiener_m205.py 31 7 50 A,B,E,F      (bracci a scelta, default tutti)

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 /opt/homebrew/bin/python3.13 \
        test/ablation_of_vs_wiener_m205.py <canale> <wp> <griglia>      (es. 31 7 50, ~30 min)
"""
import sys, csv, time, numpy as np, torch
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
import src.analysis as an, src.simulation as sim, src.dataset as ds, utility.functions as fn
from scipy.stats import norm
from utility.double_beta_spectrum import pdf_ratio2b

CH, WP, GRID, NT, NSIM = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), 500, 30000
PICK = sys.argv[4].split(",") if len(sys.argv) > 4 else None      # es. "A,B,E,F"; default tutti
amp = {(int(r["channel"]), int(r["wp"])): float(r["signal_amp"]) for r in csv.DictReader(open(
    f"{BASE}/m205_results_wiener_APsimfit10000led_npsclean_swna1/BI_results_m205_wiener_APsimfit10000led_npsclean_swna1.csv"))}[(CH, WP)]
tpl = np.load(f"{BASE}/m205_AP_sim/ch{CH}/simAP_APsimfit10000led_ch{CH}_wp{WP}.npy")
gen = np.load(f"{BASE}/residual_scan_bessel/fits_octopus/bestfit_ch{CH}_wp{WP}.npy")
nps = np.load(f"{BASE}/m205_NPS_clean/ch{CH}/nps_ch{CH}_wp{WP}.npy")
T = lambda a, dt=torch.cfloat: torch.tensor(np.asarray(a), dtype=dt)
S, w, H = an.compute_H(tpl, nps, np.hanning, sampling_rate=10000)
S_t, w_t, nps_t, H_t = T(S), T(w), T(nps), T(H)
r = np.linspace(0, 0.5, GRID); rd = pdf_ratio2b(r); rd /= rd.mean()
args = (T(np.linspace(0, 8e-4, GRID)), T(r))
amp_t = torch.tensor(amp, dtype=torch.float32); Ns = float(norm.ppf(0.9))
W1 = an.compute_W_torch(S_t, torch.tensor(nps, dtype=torch.float32), torch.tensor(1.0))
wna = lambda s1, s2: 1.0 * (s1**2 + s2**2)
n = len(S)//2 + 1
g = torch.Generator().manual_seed(0)
f1i, f2i = torch.abs(torch.rand(n, generator=g)), torch.abs(torch.rand(n, generator=g))

ARMS = [("A  OF, lr che decade a 1e-5   [= campagna OF]", "fix", H_t, 1e-5, None),
        ("B  OF, lr costante            [cambia: schedule]", "fix", H_t, 1e-2, None),
        ("C  Wiener lam=1 fissa         [cambia: kernel]", "fix", W1, 1e-2, None),
        ("D  Wiener lam addestrabile    [cambia: lambda]", "lam", None, 1e-2, None),
        ("E  + penalita' wna            [= campagna Wwna]", "lam", None, 1e-2, wna),
        ("F  Wwna con lr che decade a 1e-5 [schedule dell'OF]", "lam", None, 1e-5, wna)]
ARMS = [a for a in ARMS if PICK is None or a[0][0] in PICK]
res = []
for name, kind, K, eta, pen in ARMS:
    last = {}
    def grab(step, f1, f2, Wu, J, lam):          # filtri e kernel DELLO STESSO passo
        last.update(f1=f1.clone(), f2=f2.clone(), W=Wu.detach().clone(), J=J, lam=lam)
    t0 = time.perf_counter()
    common = dict(N_sigma=Ns, n_trials=NT, activation_fct=torch.abs, f1_init=f1i.clone(),
                  f2_init=f2i.clone(), use_interp=True, verbose=False,
                  validate=grab, val_every=NT, s_penalty=pen)
    if kind == "fix":
        _, _, J = an.optimize_filters_wiener(S_t, K, w_t, *args, nps_t, amp_t, T(rd), eta_min=eta, **common)
    else:
        out = an.optimize_filters_wiener_lambda(S_t, w_t, *args, nps_t, amp_t, T(rd), lambda_init=1.0,
                                                eta_min=eta, **common)
        J = out[4]
    v1, v2, _ = an.compute_vars_wiener(last["W"], S_t, nps_t, last["f1"], last["f2"])
    tail = 100 * (J[-min(101, len(J))] - J[-1]) / J[-1]
    res.append(dict(name=name, J=last["J"], lam=last["lam"], f1=last["f1"].numpy(), f2=last["f2"].numpy(),
                    W=last["W"].numpy(), s1=float(v1)**.5/amp, s2=float(v2)**.5/amp, tail=tail))
    print(f"{name}: J={last['J']:.6f}  discesa ultimi 100 passi {tail:.3f}%  lambda={last['lam']:.3g}  "
          f"s=({res[-1]['s1']:.3f},{res[-1]['s2']:.3f})  [{time.perf_counter()-t0:.0f} s]", flush=True)

# ── Monte Carlo appaiato: stessi eventi per tutti i bracci, a blocchi (memoria costante) ──
Sg, wg, _ = an.compute_H(gen, nps, np.hanning, sampling_rate=10000)
psd = {k: ([], []) for k in range(len(res))}
for pop, dt in ((0, 0.0), (1, 8e-4)):
    rng = np.random.default_rng(4242)
    for k in range(0, NSIM, 500):
        fp, *_ = sim.simulate_frequency_pulses(Sg, nps, 0.0, wg, nsim=500, seed=rng,
                                               signal_scale=amp, dt_max=dt, fold_ratio=False)
        p = np.fft.ifft(fp, axis=1).real.astype(np.float32)
        d = ds.NumpyDataset(p); d.win_length = p.shape[1]
        for i, R in enumerate(res):
            psd[i][pop].append(np.asarray(an.get_PSD_interpole(d, R["W"], R["f1"], R["f2"])[0]).ravel())
P = [(np.concatenate(psd[i][0]), np.concatenate(psd[i][1])) for i in range(len(res))]
def bi(ps, pp): return fn.K * (1 - np.mean(pp < np.percentile(ps, 10)))
BI = np.array([bi(*P[i]) for i in range(len(res))])
rng = np.random.default_rng(0); B = []
for _ in range(300):                               # bootstrap APPAIATO sugli eventi
    idx = rng.integers(0, NSIM, NSIM)
    B.append([bi(P[i][0][idx], P[i][1][idx]) for i in range(len(res))])
B = np.array(B)
print(f"\nch{CH} wp{WP}, griglia {GRID}x{GRID}, {NT} passi, stessa init, MC {NSIM} eventi appaiati\n")
print(f"{'braccio':52s} {'BI analitico':>12} {'BI MC':>10} {'MC/an':>6} | {'dBI_mc vs A':>12} {'vs braccio prec.':>17}")
for i, R in enumerate(res):
    ban = R["J"] * fn.K
    dA = 100 * (B[:, i] / B[:, 0] - 1); dP = 100 * (B[:, i] / B[:, i-1] - 1) if i else None
    sA = f"{100*(BI[i]/BI[0]-1):+6.2f}±{dA.std():.2f}%" if i else "—"
    sP = f"{100*(BI[i]/BI[i-1]-1):+6.2f}±{dP.std():.2f}%" if i else "—"
    print(f"{R['name']:52s} {ban:12.4e} {BI[i]:10.4e} {BI[i]/ban:6.3f} | {sA:>12} {sP:>17}")

# ── contrasti del 2x2 {OF, Wwna} x {lr che decade, lr costante}, se i bracci ci sono ────────
L = {R["name"][0]: i for i, R in enumerate(res)}
def contrast(a, b):                     # (b - a)/a in %, errore da bootstrap appaiato
    if a not in L or b not in L: return None
    d = 100 * (B[:, L[b]] / B[:, L[a]] - 1)
    return 100 * (BI[L[b]] / BI[L[a]] - 1), d.std()
print("\ncontrasti (errori da bootstrap appaiato sugli stessi eventi):")
for lab, a, b in (("modello, a lr che DECADE      (F vs A)", "A", "F"),
                  ("modello, a lr COSTANTE        (E vs B)", "B", "E"),
                  ("schedule, per l'OF            (B vs A)", "A", "B"),
                  ("schedule, per il Wwna         (E vs F)", "F", "E"),
                  ("campagne come erano           (E vs A)", "A", "E")):
    c = contrast(a, b)
    if c: print(f"  {lab}: {c[0]:+6.2f} +- {c[1]:.2f} %")
if all(k in L for k in "ABEF"):
    inter = 100 * ((B[:, L["E"]] / B[:, L["B"]]) - (B[:, L["F"]] / B[:, L["A"]]))
    val = 100 * (BI[L["E"]] / BI[L["B"]] - BI[L["F"]] / BI[L["A"]])
    print(f"  interazione modello x schedule: {val:+6.2f} +- {inter.std():.2f} %")
