"""
validate_lambda_curve_m205.py
=============================
CURVA DI VALIDAZIONE del training con lambda ADDESTRABILE: ogni VAL_EVERY passi i filtri
correnti (f1, f2 e il kernel W del lambda corrente) vengono applicati a EVENTI SIMULATI e si
misura il BI Monte Carlo. Accanto si guardano lambda, la J analitica e s1, s2.

PERCHE'. La loss addestrata e' il BI ANALITICO, che e' distorto in modo DIVERSO per filtri
diversi (misurato: MC/an 1.009 per l'OF, 1.019-1.027 per Wiener, fino a 6 con la fase libera).
Durante il training lambda sale di ordini di grandezza mentre J resta quasi piatta -- lambda e'
degenere, i filtri di banda riassorbono il kernel -- quindi dalla loss NON si vede se i punti a
lambda piccola (prima che il filtro tenda all'ottimo) valgano davvero di piu'. L'unico modo e'
applicarli ai dati simulati: e' questa curva.

COME LEGGERLA. BI_an e BI_mc sono nelle STESSE unita' (conteggi/(keV kg yr)), quindi il primo
pannello si legge direttamente: se BI_mc ha un minimo dove BI_an non ce l'ha, il training sta
ottimizzando la cosa sbagliata e il punto buono e' da fermare li'. Il secondo pannello e'
MC/an: se cresce, il modello analitico sta diventando piu' ottimista, cioe' si sta sovra-
addestrando. Il terzo e' lambda.

GLI EVENTI SONO GLI STESSI a ogni validazione (un banco generato UNA volta): l'errore statistico
del singolo punto e' ~2.3% con 5000 eventi (misurato su ch34 wp15: la frazione di sopravvivenza
e' 0.27, quindi sqrt(0.27*0.73/5000)/0.27), ma e' quasi tutto COMUNE ai punti della curva, quindi
la FORMA della curva e' molto piu' precisa del singolo punto -- e' lo stesso motivo per cui il
confronto appaiato di compare_templates_m205.py ha un errore 0.35 volte quello in quadratura.
Il banco e' generato dal FIT (la "verita'"), il training dall'AP simulato: nessuna
auto-consistenza, come nelle campagne. Il SEED (9001) sta FUORI da quelli del Monte Carlo
finale (1234...1283): con lo stesso seed il banco sarebbe esattamente i primi NSIM_VAL eventi
di quel run, e scegliere sulla curva per poi misurare su quegli eventi sarebbe leakage.
NB: le curve di ch34 e ch91 del 2026-09-21/22 furono fatte con seed = 1234; fra loro restano
appaiate, ma non si confrontano evento per evento con quelle nuove.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 /opt/homebrew/bin/python3.13 \
        validate_lambda_curve_m205.py [canale] [wp]
(OMP_NUM_THREADS=1 e' necessario: con piu' thread torch+scipy vanno in segfault su questo Mac)
"""
import os, sys, csv, time
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import src.analysis as an
import src.simulation as sim
import src.dataset as ds
import utility.functions as fn
from scipy.stats import norm
from utility.double_beta_spectrum import pdf_ratio2b

# ═════════════════════════════════════════════════════════════════════════════
# Config
# ═════════════════════════════════════════════════════════════════════════════
CHANNEL, WP = 34, 15          # sovrascrivibili da riga di comando
SIM_SOURCE  = "APsimfit10000led"   # template di TRAINING (come la campagna)
GEN_SOURCE  = "fit"                # template di GENERAZIONE degli eventi: la "verita'"
N_TRIALS    = 500
VAL_EVERY   = 25              # passi fra due validazioni
NSIM_VAL    = 5000            # eventi per popolazione nel banco di validazione.
                              # 5000 -> ~2.3% sul singolo punto, ma errore COMUNE alla curva
                              # (un punto costa ~9.4 s; il banco, una volta sola, ~10-20 s)
GRID        = 100             # N_T = N_R della griglia (r, dt): 100 e' quella delle campagne,
                              # 40 se serve un giro veloce (cambia un po' il training)
LAMBDA_INIT = 1.0
# Penalita' sulla condizione di validita' del modello analitico (vedi S_PENALTY nel programma
# di training). None = spenta; ("wna", W) = W*(s1^2+s2^2), senza soglia, tocca anche i punti
# sani; ("barrier", S_MAX, W) = W*sum relu(s/S_MAX - 1)^2, ESATTAMENTE NULLA per s <= S_MAX.
# Il banco di eventi non dipende da questa scelta, quindi run diversi sono APPAIATI: i BI_mc
# si confrontano direttamente, senza errore di generazione.
S_PENALTY   = ("wna", 1.0)    # None | ("wna", W) | ("barrier", S_MAX, W)

# Terzo argomento da riga di comando: sovrascrive S_PENALTY, cosi' la stessa coppia
# (canale, WP) si lancia con penalita' diverse senza toccare il file -- e i due run
# condividono lo stesso banco di eventi, quindi sono APPAIATI.
#     validate_lambda_curve_m205.py 91 15 none
if len(sys.argv) > 3:
    S_PENALTY = {"none": None, "wna": ("wna", 1.0),
                 "barrier": ("barrier", 0.15, 1.0)}[sys.argv[3].lower()]
ACCEPTANCE  = 0.9
T_MAX, R_MAX = 8e-4, 0.5
PEN_TAG     = ("nopen" if not S_PENALTY else
               ("swna%g" % S_PENALTY[1] if S_PENALTY[0] == "wna" else "sbar%g" % S_PENALTY[1]))
OUT_PNG     = os.path.join(BASE_DIR, "validation_curve_ch{ch}_wp{wp}_" + PEN_TAG + ".png")

VBIAS_LIST = np.array([0.6, 1.0, 1.4, 1.8, 2, 3, 4, 5, 6, 8, 10, 20, 26, 30, 40])


def load_inputs(ch, wp):
    """(template di training, template di generazione, nps, ampiezza di ROI)."""
    tpl = np.load(os.path.join(BASE_DIR, "m205_AP_sim", f"ch{ch}",
                               f"simAP_{SIM_SOURCE}_ch{ch}_wp{wp}.npy"))
    gen = np.load(os.path.join(BASE_DIR, "residual_scan_bessel", "fits_octopus",
                               f"bestfit_ch{ch}_wp{wp}.npy"))
    nps = np.load(os.path.join(BASE_DIR, "m205_NPS_clean", f"ch{ch}", f"nps_ch{ch}_wp{wp}.npy"))
    vb = round(float(VBIAS_LIST[wp // 2]), 3)
    amps = {(int(r["channel"]), round(float(r["vbias_V"]), 3)): float(r["amplitude_mV"]) * 1e-3
            for r in csv.DictReader(open(os.path.join(BASE_DIR, "amplitudes_m205.csv")))
            if (r["amplitude_mV"] or "").strip()}
    return tpl, gen, nps, amps[(ch, vb)]


def event_bank(gen, nps, signal_amp, seed=9001):
    """(singoli, pile-up) come dataset di impulsi nel tempo, generati UNA volta dal template
    `gen` piu' rumore con lo spettro nps. La generazione sta in src.simulation.build_event_bank,
    la stessa che usa la validazione dentro il training."""
    S, w, _ = an.compute_H(gen, nps, np.hanning, sampling_rate=10000)
    out = []
    for dt_max in (0.0, T_MAX):
        pulses = sim.build_event_bank(S, nps, w, signal_amp, NSIM_VAL, dt_max, seed=seed)
        d = ds.NumpyDataset(pulses)
        d.win_length = pulses.shape[1]
        out.append(d)
    return out


def main(ch, wp):
    tpl, gen, nps, signal_amp = load_inputs(ch, wp)
    print(f"ch{ch} wp{wp}  signal_amp = {signal_amp:.4g} V   griglia {GRID}x{GRID}   "
          f"{N_TRIALS} passi   validazione ogni {VAL_EVERY} su {NSIM_VAL} eventi/popolazione\n"
          f"penalita' {S_PENALTY}")
    t0 = time.perf_counter()
    singles, pileups = event_bank(gen, nps, signal_amp)
    print(f"banco di eventi generato in {time.perf_counter()-t0:.0f} s")

    S, w, _ = an.compute_H(tpl, nps, np.hanning, sampling_rate=10000)
    to = lambda a, d=torch.cfloat: torch.tensor(np.asarray(a), dtype=d)
    rr = np.linspace(0, R_MAX, GRID)
    rd = pdf_ratio2b(rr); rd /= rd.mean()
    S_t, w_t, nps_t = to(S), to(w), to(nps)
    amp_t = torch.tensor(signal_amp, dtype=torch.float32)
    if not S_PENALTY:
        penalty = None
    elif S_PENALTY[0] == "wna":
        penalty = lambda s1, s2: S_PENALTY[1] * (s1**2 + s2**2)
    else:
        _, s_max, wgt = S_PENALTY
        penalty = lambda s1, s2: wgt * (torch.relu(s1/s_max - 1)**2 + torch.relu(s2/s_max - 1)**2)

    rows = []
    def validate(step, f1, f2, W_unit, J, lam):
        k = W_unit.cpu().numpy()
        bi, _ = an.compute_BI(pileups, singles, ACCEPTANCE, k,
                              f1.cpu().numpy(), f2.cpu().numpy(), window_fct=np.ones)
        v1, v2, _ = an.compute_vars_wiener(W_unit, S_t, nps_t, f1, f2)
        rows.append(dict(step=step, J=J, lam=lam, BI_an=J * fn.K, BI_mc=bi,
                         s1=float(v1)**0.5 / signal_amp, s2=float(v2)**0.5 / signal_amp))
        r = rows[-1]
        print(f"  step {step:4d}  lambda={lam:9.3f}  BI_an={r['BI_an']:.4g}  "
              f"BI_mc={bi:.4g}  MC/an={bi/r['BI_an']:.3f}  s=({r['s1']:.3f},{r['s2']:.3f})")

    hist = {}
    an.optimize_filters_wiener_lambda(
        S_t, w_t, to(np.linspace(0, T_MAX, GRID)), to(rr), nps_t, amp_t, to(rd),
        N_sigma=float(norm.ppf(ACCEPTANCE)), activation_fct=torch.abs,
        lambda_init=LAMBDA_INIT, s_penalty=penalty, history=hist,
        validate=validate, val_every=VAL_EVERY,
        n_trials=N_TRIALS, use_interp=True, verbose=False)

    out_csv = OUT_PNG.format(ch=ch, wp=wp).replace(".png", ".csv")
    with open(out_csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    print(f"punti -> {out_csv}")

    best_mc = min(rows, key=lambda r: r["BI_mc"])
    last = rows[-1]
    print(f"\nmigliore sul MC: step {best_mc['step']} (lambda {best_mc['lam']:.3g}), "
          f"BI_mc {best_mc['BI_mc']:.4g}")
    print(f"fine training:   step {last['step']} (lambda {last['lam']:.3g}), "
          f"BI_mc {last['BI_mc']:.4g}  ->  fermarsi al migliore vale "
          f"{100*(best_mc['BI_mc']/last['BI_mc']-1):+.1f}%")
    plot(ch, wp, rows)


def plot(ch, wp, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    st = [r["step"] for r in rows]
    fig, ax = plt.subplots(3, 1, figsize=(7.5, 8), sharex=True)
    ax[0].plot(st, [r["BI_an"] for r in rows], "o-", label="analytic BI (training loss)")
    ax[0].plot(st, [r["BI_mc"] for r in rows], "s-", label=f"Monte-Carlo BI ({NSIM_VAL} events)")
    ax[0].set_ylabel("BI [counts/(keV kg yr)]"); ax[0].legend(); ax[0].set_yscale("log")
    ax[1].plot(st, [r["BI_mc"]/r["BI_an"] for r in rows], "o-", color="#0d2c6b")
    ax[1].axhline(1.0, color="k", lw=0.8, ls="--")
    ax[1].set_ylabel("MC / analytic")
    ax[2].plot(st, [r["lam"] for r in rows], "o-", color="C3")
    ax[2].set_ylabel(r"$\lambda$"); ax[2].set_yscale("log"); ax[2].set_xlabel("training step")
    for a in ax:
        a.grid(True, ls="--", alpha=0.4)
    ax[0].set_title(f"m205 Ch{ch} WP{wp} — validation during training "
                f"(trainable $\\lambda$, penalty {S_PENALTY})")
    fig.tight_layout()
    out = OUT_PNG.format(ch=ch, wp=wp)
    fig.savefig(out, dpi=130)
    print(f"figura -> {out}")


if __name__ == "__main__":
    ch = int(sys.argv[1]) if len(sys.argv) > 1 else CHANNEL
    wp = int(sys.argv[2]) if len(sys.argv) > 2 else WP
    main(ch, wp)
