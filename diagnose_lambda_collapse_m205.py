"""
diagnose_lambda_collapse_m205.py
================================
Perche' il lambda del filtro di Wiener COLLASSA verso la deconvoluzione pura, e perche'
quando collassa il BI analitico non e' piu' quello vero. Programma di diagnosi: non
produce risultati di fisica, produce i NUMERI e le FIGURE per spiegare il meccanismo.

────────────────────────────────────────────────────────────────────────────────
LE FORMULE (tutte verificate numericamente da questo programma)
────────────────────────────────────────────────────────────────────────────────
Kernel di Wiener con modulazione del rumore lambda (spettri normalizzati a pari potenza):

    W_lam(f) = S*(f) / ( |S(f)|^2 + lam * NPS(f) )
    lam -> inf : filtro ottimo  S*/NPS         (impulso filtrato LARGO)
    lam  = 1   : Wiener standard (CUORE norm_type=0)
    lam -> 0   : deconvoluzione pura 1/S        (impulso filtrato a DELTA, rumore amplificato)

Il parametro di forma e' il rapporto di due ampiezze filtrate, Y = A1/A2, con
g_i = f_i * W_lam. Il calcolo ANALITICO (src/analysis.py) assume uno stimatore LINEARE
(traccia filtrata letta a tempo FISSO) e propaga il rumore al prim'ordine:

    var_i  = sum_k |f_i|^2 |W|^2 NPS_k / ( sum_k |f_i| Re(W S)_k )^2      [compute_vars_wiener]
    s_i    = sqrt(var_i) / signal_amp                  risoluzione RELATIVA del filtro i
    rho    = cov_12 / sqrt(var_1 var_2)                correlazione fra A1 e A2
    sigma_Y = sqrt( s_1^2 + s_2^2 - 2 rho s_1 s_2 )  ~  s * sqrt( 2 (1 - rho) )   se s_1 ~ s_2

    (con la normalizzazione dei filtri di banda, mean|f_i W S| = 1, il picco dell'impulso
     filtrato vale ESATTAMENTE 1 perche' W S >= 0 reale: quindi E[A_i] = signal_amp e s_i
     e' davvero la risoluzione relativa. Il programma lo verifica.)

Da qui il BI: A(mu_Y, sigma_Y) e' l'accettanza, J la sua media sulla distribuzione
dei pile-up (r, dt), BI = K * J.

LE DUE ROTTURE quando lam -> 0
  (1) CONDIZIONAMENTO. sigma_Y dipende da rho solo attraverso 1-rho:

          d ln sigma_Y = - d rho / ( 2 (1 - rho) )   ==>   kappa = rho / (2 (1-rho))

      Nel regime collassato 1-rho ~ 1e-4...1e-6: un errore ASSOLUTO di 1e-3 su rho
      (quinta cifra decimale!) raddoppia sigma_Y. La metrica e' una differenza fra numeri
      quasi uguali: l'ottimizzatore ci cammina dentro perche' li' J scende.
  (2) STIMATORE SBAGLIATO. Il codice non legge la traccia filtrata a tempo fisso: cerca il
      MASSIMO su +-20 campioni (jitter del trigger) e interpola. Se l'impulso filtrato e'
      largo (filtro ottimo, FWHM ~ 35 campioni) le due cose coincidono; se e' una punta
      (lam piccolo, FWHM ~ 3) la ricerca del massimo aggancia i picchi del RUMORE:
      A_i viene sovrastimato (bias) e A1, A2 si DECORRELANO. Un rho che passa da 0.9994
      a 0.958 e' 1-rho x70, cioe' sigma_Y x8 -- e il MC lo vede, l'analitico no.

Il numero di massimi indipendenti nella finestra di ricerca e' ~ 2*jitter/FWHM: per
FWHM = 3 sono ~13, e il massimo di n gaussiane vale ~ sqrt(2 ln n) sigma. E' l'ordine di
grandezza del bias misurato.

────────────────────────────────────────────────────────────────────────────────
COSA CALCOLA
────────────────────────────────────────────────────────────────────────────────
  --diag   (veloce, ~10 s per coppia) Per OGNI (canale, WP) di una cartella di risultati
           gia' addestrata: prende f1, f2 e il lambda ADDESTRATI, calcola le quantita'
           analitiche (s, rho, sigma_Y, BI) e le stesse quantita' da Monte Carlo, con
           lo stimatore VERO (max search) e con quello del modello (tempo fisso).
           -> <OUT_DIR>/lambda_diag_<tag>.csv
  --scan   (lento) FETTA IN LAMBDA a (canale, WP) fissi: per ogni lambda della griglia
           ri-addestra f1, f2 (warm start dai filtri salvati, RETRAIN_STEPS passi) e
           rifa' tutti i conti. E' il "paesaggio" che l'ottimizzatore vede: BI analitico
           che scende verso lam -> 0 mentre il BI vero sale.
           -> <OUT_DIR>/lambda_scan_<tag>.csv     (SUBMIT_MODE = "qsub" -> un job per WP)
  --plots  Figure da quei CSV + lambda addestrato vs rho_t su TUTTI i canali.
  --selftest  Controlli runnabili: kernel ricostruito == kernel salvato, ampiezze ==
           get_PSD_interpole di produzione, picco dell'impulso filtrato == 1.

Esempi
    python diagnose_lambda_collapse_m205.py --diag
    python diagnose_lambda_collapse_m205.py --scan            # locale, oppure sottomette
    python diagnose_lambda_collapse_m205.py --scan --worker --channel 34 --wp 9
    python diagnose_lambda_collapse_m205.py --plots
"""

from __future__ import annotations

import os
import sys
import csv
import glob
import argparse
import subprocess
import tempfile

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════
# Cartella dei risultati Wiener gia' addestrati: da qui vengono f1, f2, il kernel e il
# lambda. Come in simulate_BI_error_m205.py, template di training e sorgente NPS sono
# DEDOTTI dal nome (e stampati); il kernel ricostruito viene confrontato con quello salvato.
#   m205_results_wiener                     -> root  + NPS octopus, 9 canali x 15 WP (il piu' ricco)
#   m205_results_wiener_sim_rootinj_npsclean-> sim   + NPS clean, ch34
RESULTS_NAME = "m205_results_wiener"

GEN_TEMPLATE = "root"        # template che GENERA gli eventi MC: "root" | "fit"
ONLY_CHANNELS = [31, 34, 71, 83, 91]   # None/[] = tutti quelli nel CSV
ONLY_WPS      = None                   # None/[] = tutti

# ── Fetta in lambda (--scan) ────────────────────────────────────────────────────────
SCAN_PAIRS    = [(34, 9), (34, 27), (91, 15)]      # (canale, WP) su cui fare la fetta
LAMBDA_GRID   = np.logspace(-4, 2, 13)
RETRAIN_STEPS = 200          # passi di ri-addestramento di f1, f2 a lambda FISSO, partendo
                             # dai filtri salvati (warm start). 0 = filtri congelati: piu'
                             # veloce ma non e' il minimo di J a quel lambda, quindi
                             # sovrastima il vantaggio dei lambda piccoli.

# ── Monte Carlo (stessa ricetta di simulate_BI_error_m205.py) ───────────────────────
NSIM  = 6_000        # eventi per popolazione (singoli e pile-up). Errore MC sul BI ~ 1.5%
CHUNK = 2_000        # simulate_frequency_pulses alloca 6 array (n, 10000) complessi
SEED  = 1234
ACCEPTANCE = 0.9
T_MAX = 8e-4         # ritardo massimo del pile-up [s]
JITTER_MAX = 20      # semi-finestra della ricerca del massimo [campioni] (= produzione)

# ── Griglia del calcolo analitico (= produzione) ────────────────────────────────────
N_T, N_R = 100, 100
T_MIN, T_MAX_AN = 0, 8e-4
R_MIN, R_MAX = 0.0, 0.5

WINDOW_SIZE   = 10_000
SAMPLING_RATE = 10_000
SAMPLING_TIME = WINDOW_SIZE / SAMPLING_RATE

OUT_DIR = os.path.join(BASE_DIR, "m205_lambda_collapse")

# ── Cluster (solo --scan con SUBMIT_MODE="qsub") ────────────────────────────────────
SUBMIT_MODE = "local"        # "local" = in sequenza qui ; "qsub" = un job per coppia
QUEUE, WALLTIME, RAM_GB = "cupid", "24:00:00", 6
JOB_NAME_PREFIX = "LAMD"
ENV_SETUP_LINES = ["source /home/zanelli/LoadOctopus.sh"]

VBIAS_LIST = np.array([0.6, 1.0, 1.4, 1.8, 2, 3, 4, 5, 6, 8, 10, 20, 26, 30, 40])
MEAS_NAME = "000205"
DATA_DIR = os.path.join(BASE_DIR, "Processed")
FIT_DIR = os.path.join(BASE_DIR, "residual_scan_bessel", "fits_octopus")
SIM_AP_DIR = os.path.join(BASE_DIR, "m205_AP_sim")
NPS_DIR = os.path.join(BASE_DIR, "m205_NPS_clean")


# ═════════════════════════════════════════════════════════════════════════════
# Configurazione dedotta dal nome della cartella (come simulate_BI_error_m205.py)
# ═════════════════════════════════════════════════════════════════════════════
def _parse_results_name(name):
    """('wiener'|'wiener_R', template di training, set sim, sorgente NPS) dal nome cartella."""
    if not name.startswith("m205_results_wiener"):
        raise SystemExit(f"[ERROR] {name}: serve una cartella Wiener (m205_results_wiener*)")
    tag = name[len("m205_results_wiener"):]
    nps = "clean" if tag.endswith("_npsclean") else "octopus"
    tag = tag[:-len("_npsclean")] if nps == "clean" else tag
    base = "wiener"
    if tag.endswith("_R"):
        base, tag = "wiener_R", tag[:-2]
    if tag in ("", "_root"):
        return base, "root", None, nps
    if tag == "_fit":
        return base, "fit", None, nps
    if tag.startswith("_sim_"):
        return base, "sim", tag[len("_sim_"):], nps
    raise SystemExit(f"[ERROR] non so dedurre il template da '{name}' (resto: '{tag}')")


FILTER_TYPE, TRAIN_TEMPLATE, SIM_AP_FROM, NPS_SOURCE = _parse_results_name(RESULTS_NAME)
RESULTS_DIR = os.path.join(BASE_DIR, RESULTS_NAME)
FILTERS_DIR = os.path.join(RESULTS_DIR, "trained_filters")
_TAG = RESULTS_NAME[len("m205_results_"):]
DIAG_CSV = os.path.join(OUT_DIR, f"lambda_diag_{_TAG}.csv")
SCAN_CSV = os.path.join(OUT_DIR, f"lambda_scan_{_TAG}.csv")

CSV_FIELDS = ["channel", "wp", "vbias", "lam", "retrain_steps", "signal_amp", "SNR",
              "BI_an", "BI_mc", "sigma_BI", "sigmaY_an", "sigmaY_mc_max", "sigmaY_mc_fix",
              "s1_an", "s2_an", "rho_an", "s1_mc", "s2_mc", "rho_mc", "rho_mc_fix",
              "bias_max", "fwhm", "cos_f1f2", "nsim"]


def wp_to_vbias(wp):
    return float(VBIAS_LIST[wp // 2])


def root_file(channel):
    for f in glob.glob(os.path.join(DATA_DIR, f"Processed_*_{MEAS_NAME}_*.root")):
        if os.path.basename(f).split("_")[-1].replace(".root", "") == str(channel):
            return f
    raise RuntimeError(f"file ROOT non trovato per il canale {channel}")


def flattop_power_factor(n):
    from scipy.signal.windows import flattop
    return float(n / np.sum(flattop(n) ** 2))


def template_pulse(channel, wp, source):
    """Template con picco 1. source: 'root' (medianAP), 'fit' (bestfit), 'sim' (AP simulato)."""
    import uproot
    if source == "root":
        with uproot.open(root_file(channel)) as f:
            p = np.asarray(f[f"averagepulse_ap_wp{wp}_medianAP"].values(), dtype=float)
    elif source == "fit":
        p = np.load(os.path.join(FIT_DIR, f"bestfit_ch{channel}_wp{wp}.npy"))
    elif source == "sim":
        p = np.load(os.path.join(SIM_AP_DIR, f"ch{channel}",
                                 f"simAP_{SIM_AP_FROM}_ch{channel}_wp{wp}.npy"))
    else:
        raise ValueError(source)
    p = np.asarray(p, dtype=float)
    return p / p.max()


def load_nps(channel, wp):
    """NPS nella convenzione del codice, dalla stessa sorgente usata nel training."""
    if NPS_SOURCE == "clean":
        return np.asarray(np.load(os.path.join(NPS_DIR, f"ch{channel}",
                                               f"nps_ch{channel}_wp{wp}.npy")), dtype=float)
    import uproot
    with uproot.open(root_file(channel)) as f:
        nps = np.asarray(f[f"averagepowerspectrum_noise_wp{wp}_medianpower"].values(), dtype=float)
    nps = np.concatenate([nps, nps[-2:0:-1]])
    return nps * flattop_power_factor(WINDOW_SIZE) * WINDOW_SIZE ** 2 / SAMPLING_TIME


def full_spectrum(half, complex_=False):
    return np.concatenate([half, np.conj(half[-2:0:-1])] if complex_ else [half, half[-2:0:-1]])


# ═════════════════════════════════════════════════════════════════════════════
# Ampiezze filtrate: stimatore VERO (max search) e stimatore del MODELLO (tempo fisso)
# ═════════════════════════════════════════════════════════════════════════════
def amplitudes(pulses, g, jitter_max=JITTER_MAX, interp_range=5, pulse_start_pos=-100):
    """(A_max, A_fix) per la traccia filtrata con il kernel totale g = f_i * W.

    A_max: massimo su +-jitter_max campioni + interpolazione cubica -- e' quello che usa
           get_PSD_interpole in produzione (e i dati veri, per il jitter del trigger).
    A_fix: valore a tempo FISSO nel centro della finestra -- e' lo stimatore LINEARE che
           il calcolo analitico modella. La differenza fra i due e' la seconda rottura.
    """
    from scipy.interpolate import interp1d
    n = pulses.shape[-1]
    tgt = n // 2
    pulses = pulses - pulses[:, :tgt + pulse_start_pos].mean(axis=-1, keepdims=True)
    filt = np.roll(np.fft.ifft(g * np.fft.fft(pulses, axis=-1), axis=-1).real, tgt, axis=-1)
    idx = np.argmax(filt[:, tgt - jitter_max:tgt + jitter_max], axis=1) + tgt - jitter_max
    offs = np.arange(-interp_range, interp_range + 1)
    y = filt[np.arange(len(filt))[:, None], idx[:, None] + offs]
    fine = interp1d(offs, y, kind="cubic", axis=1, bounds_error=False, fill_value="extrapolate")
    a_max = np.max(fine(np.arange(-interp_range, interp_range + 1, 0.05)), axis=1)
    return a_max, filt[:, tgt]


def simulate_population(S_gen, nps, w, g1, g2, signal_amp, dt_max, seed):
    """NSIM eventi (dt_max=0 -> singoli, >0 -> pile-up): rapporti Y con i due stimatori."""
    import src.simulation as sim
    y_max, y_fix, a1_max, a1_fix = [], [], [], []
    for k in range(0, NSIM, CHUNK):
        n = min(CHUNK, NSIM - k)
        fp, *_ = sim.simulate_frequency_pulses(S_gen, nps, 0.0, w, nsim=n, seed=seed + k // CHUNK,
                                               signal_scale=signal_amp, dt_max=dt_max)
        ev = np.fft.ifft(fp, axis=1).real
        del fp
        a1, a1f = amplitudes(ev, g1)
        a2, a2f = amplitudes(ev, g2)
        del ev
        y_max.append(a1 / a2)
        y_fix.append(a1f / a2f)
        a1_max.append(a1)
        a1_fix.append(a1f)
    return (np.concatenate(y_max), np.concatenate(y_fix),
            np.concatenate(a1_max), np.concatenate(a1_fix))


# ═════════════════════════════════════════════════════════════════════════════
# Un punto: (canale, WP, lambda, f1, f2) -> tutte le diagnostiche
# ═════════════════════════════════════════════════════════════════════════════
def analyse_point(channel, wp, lam, f1_half, f2_half, S, nps, w, S_gen, signal_amp,
                  shared, retrain_steps=0):
    import torch
    import src.analysis as an
    import utility.functions as fn

    St = torch.tensor(S, dtype=torch.cfloat)
    wt = torch.tensor(w, dtype=torch.cfloat)
    nt = torch.tensor(nps, dtype=torch.float32)
    W = an.compute_W_torch(St, nt, torch.tensor(float(lam)))

    if retrain_steps > 0:
        f1, f2, _ = an.optimize_filters_wiener(
            St, W, wt, shared["t"], shared["r"], nt, torch.tensor(signal_amp),
            shared["rd"], N_sigma=shared["N_sigma"], n_trials=retrain_steps,
            activation_fct=torch.abs,
            f1_init=torch.tensor(f1_half, dtype=torch.float),
            f2_init=torch.tensor(f2_half, dtype=torch.float),
            use_interp=True, verbose=False)
    else:
        f1 = torch.tensor(full_spectrum(f1_half), dtype=torch.cfloat)
        f2 = torch.tensor(full_spectrum(f2_half), dtype=torch.cfloat)
    # normalizzazione dei filtri di banda: mean|f W S| = 1 (come dentro l'ottimizzatore)
    f1 = f1 / torch.mean(torch.abs(f1 * W * St))
    f2 = f2 / torch.mean(torch.abs(f2 * W * St))

    # ── analitico ────────────────────────────────────────────────────────────
    S_H = St * W
    S_H_delayed = S_H[None, :] * torch.exp(-1j * wt[None, :] * shared["t"][:, None])
    J, _, muY, sigmaY = an.compute_J_wiener(f1, f2, S_H_delayed, shared["r"], S_H,
                                            (St.abs() ** 2) / nt, W, St, nt,
                                            torch.tensor(signal_amp), shared["rd"],
                                            N_sigma=shared["N_sigma"], use_interp=True,
                                            full_output=True)
    var1, var2, cov12 = an.compute_vars_wiener(W, St, nt, f1, f2)
    var1, var2, cov12 = float(var1), float(var2), float(cov12)
    s1, s2 = var1 ** 0.5 / signal_amp, var2 ** 0.5 / signal_amp
    rho_an = cov12 / (var1 * var2) ** 0.5

    g1 = (f1 * W).numpy()
    g2 = (f2 * W).numpy()
    pulse_f = np.fft.ifft(S * g1).real
    fwhm = int(np.sum(pulse_f >= pulse_f.max() / 2))
    a1n, a2n = np.abs(f1.numpy()), np.abs(f2.numpy())
    cos_f = float(np.sum(a1n * a2n) / np.sqrt(np.sum(a1n ** 2) * np.sum(a2n ** 2)))

    # ── Monte Carlo con lo stesso kernel ─────────────────────────────────────
    ys, ysf, a1_max, a1_fix = simulate_population(S_gen, nps, w, g1, g2, signal_amp, 0.0, SEED)
    yp, _, _, _ = simulate_population(S_gen, nps, w, g1, g2, signal_amp, T_MAX, SEED)
    cut = np.percentile(ys, 100 - ACCEPTANCE * 100)
    rp = float(np.mean(yp < cut))
    bi_mc = fn.K * (1.0 - rp)
    sigma_rp = np.sqrt(max(rp * (1 - rp), 1e-12) / len(yp))

    # stesse quantita' del calcolo analitico, ma misurate: s = std/mean, rho = corr
    a2_max = a1_max / ys
    a2_fix = a1_fix / ysf
    return {
        "channel": channel, "wp": wp, "vbias": wp_to_vbias(wp), "lam": float(lam),
        "retrain_steps": retrain_steps, "signal_amp": signal_amp,
        "SNR": signal_amp / float(an.compute_sigma_OF(S, nps)),
        "BI_an": float(J) * fn.K, "BI_mc": bi_mc, "sigma_BI": fn.K * sigma_rp,
        "sigmaY_an": float(sigmaY[0, 0]), "sigmaY_mc_max": float(ys.std()),
        "sigmaY_mc_fix": float(ysf.std()),
        "s1_an": s1, "s2_an": s2, "rho_an": rho_an,
        "s1_mc": float(a1_max.std() / a1_max.mean()), "s2_mc": float(a2_max.std() / a2_max.mean()),
        "rho_mc": float(np.corrcoef(a1_max, a2_max)[0, 1]),
        "rho_mc_fix": float(np.corrcoef(a1_fix, a2_fix)[0, 1]),
        "bias_max": float(a1_max.mean() / a1_fix.mean()),
        "fwhm": fwhm, "cos_f1f2": cos_f, "nsim": NSIM,
        # meta' indipendente dei filtri usati qui: serve al warm start del punto successivo
        # della fetta (non finisce nel CSV, write_rows tiene solo CSV_FIELDS).
        "f1_half": np.abs(f1.numpy()[:WINDOW_SIZE // 2 + 1]),
        "f2_half": np.abs(f2.numpy()[:WINDOW_SIZE // 2 + 1]),
    }


def build_shared():
    import torch
    from scipy.stats import norm
    from utility.double_beta_spectrum import pdf_ratio2b
    rd = pdf_ratio2b(np.linspace(R_MIN, R_MAX, N_R))
    rd = rd / np.mean(rd)
    return {"N_sigma": float(norm.ppf(ACCEPTANCE)),
            "t": torch.linspace(T_MIN, T_MAX_AN, N_T, dtype=torch.cfloat),
            "r": torch.linspace(R_MIN, R_MAX, N_R, dtype=torch.cfloat),
            "rd": torch.tensor(rd, dtype=torch.cfloat)}


def load_pair(channel, wp, row):
    """(S, nps, w, S_gen, f1_half, f2_half, signal_amp) e controllo del kernel salvato."""
    nps = load_nps(channel, wp)
    tpl = template_pulse(channel, wp, TRAIN_TEMPLATE)
    S = np.fft.fft(tpl * np.hanning(WINDOW_SIZE))
    w = 2 * np.pi * np.fft.fftfreq(WINDOW_SIZE, 1 / SAMPLING_RATE)
    S_gen = np.fft.fft(template_pulse(channel, wp, GEN_TEMPLATE) * np.hanning(WINDOW_SIZE))
    f1 = np.load(os.path.join(FILTERS_DIR, f"f1_ch{channel}_wp{wp}.npy"))
    f2 = np.load(os.path.join(FILTERS_DIR, f"f2_ch{channel}_wp{wp}.npy"))
    check_kernel(channel, wp, S, nps, float(row["lambda_wiener"]), row)
    return S, nps, w, S_gen, f1, f2, float(row["signal_amp"])


def check_kernel(channel, wp, S, nps, lam, row):
    """Il kernel ricostruito deve coincidere con quello salvato dal training: se non
    coincide, template/NPS/lambda dedotti dal nome della cartella sono sbagliati."""
    import torch
    import src.analysis as an
    path = os.path.join(FILTERS_DIR, f"kernel_ch{channel}_wp{wp}.npy")
    if not os.path.exists(path):
        return
    W = an.compute_W_torch(torch.tensor(S, dtype=torch.cfloat),
                           torch.tensor(nps, dtype=torch.float32),
                           torch.tensor(lam)).numpy()
    if FILTER_TYPE == "wiener_R":
        W = an.reliability_R(torch.tensor(S, dtype=torch.cfloat),
                             torch.tensor(nps, dtype=torch.float32),
                             int(float(row["n_events"])), float(row["beta_R"])).numpy() * W
    saved = full_spectrum(np.load(path), complex_=True)
    rel = np.abs(saved - W).max() / max(np.abs(W).max(), 1e-300)
    if rel > 1e-4:
        raise RuntimeError(f"ch{channel} wp{wp}: kernel ricostruito != salvato (scarto {rel:.1e}). "
                           f"Dedotti: template={TRAIN_TEMPLATE}, nps={NPS_SOURCE}, tipo={FILTER_TYPE}")


# ═════════════════════════════════════════════════════════════════════════════
# CSV
# ═════════════════════════════════════════════════════════════════════════════
def write_rows(path, rows, append=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = append and os.path.exists(path)
    with open(path, "a" if new else "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not new:
            wr.writeheader()
        for r in rows:
            wr.writerow({k: r[k] for k in CSV_FIELDS})
    print(f"[OK] {len(rows)} righe -> {path}")


def results_rows():
    """Righe del CSV di training filtrate da ONLY_CHANNELS / ONLY_WPS."""
    csvs = glob.glob(os.path.join(RESULTS_DIR, "BI_results_*.csv"))
    if not csvs:
        raise SystemExit(f"[ERROR] nessun CSV dei risultati in {RESULTS_DIR}")
    rows = list(csv.DictReader(open(csvs[0])))
    out = []
    for r in rows:
        ch, wp = int(r["channel"]), int(r["wp"])
        if ONLY_CHANNELS and ch not in ONLY_CHANNELS:
            continue
        if ONLY_WPS and wp not in ONLY_WPS:
            continue
        if not os.path.exists(os.path.join(FILTERS_DIR, f"f1_ch{ch}_wp{wp}.npy")):
            continue
        out.append(r)
    return sorted(out, key=lambda r: (int(r["channel"]), int(r["wp"])))


# ═════════════════════════════════════════════════════════════════════════════
# MODO --diag
# ═════════════════════════════════════════════════════════════════════════════
def run_diag():
    print(f"[CONFIG] {RESULTS_NAME}: filtro={FILTER_TYPE} template={TRAIN_TEMPLATE}"
          f"{'/' + SIM_AP_FROM if SIM_AP_FROM else ''} nps={NPS_SOURCE} | eventi da {GEN_TEMPLATE}")
    shared = build_shared()
    rows = results_rows()
    out = []
    for i, r in enumerate(rows, 1):
        ch, wp = int(r["channel"]), int(r["wp"])
        S, nps, w, S_gen, f1, f2, amp = load_pair(ch, wp, r)
        res = analyse_point(ch, wp, float(r["lambda_wiener"]), f1, f2, S, nps, w, S_gen,
                            amp, shared, retrain_steps=0)
        out.append(res)
        print(f"[{i}/{len(rows)}] ch{ch} wp{wp} V={res['vbias']:>4} lam={res['lam']:.2e} "
              f"| s={res['s1_an']:.3f} 1-rho={1 - res['rho_an']:.1e} "
              f"sY {res['sigmaY_an']:.4f} -> {res['sigmaY_mc_max']:.4f} "
              f"| BI {res['BI_an']:.3e} -> {res['BI_mc']:.3e} "
              f"({res['BI_mc'] / res['BI_an']:.2f}) bias={res['bias_max']:.2f}", flush=True)
    write_rows(DIAG_CSV, out)


# ═════════════════════════════════════════════════════════════════════════════
# MODO --scan  (fetta in lambda; opzionalmente un job per coppia)
# ═════════════════════════════════════════════════════════════════════════════
def scan_pair(channel, wp):
    shared = build_shared()
    rows = {(int(r["channel"]), int(r["wp"])): r for r in results_rows()}
    r = rows.get((channel, wp))
    if r is None:
        raise SystemExit(f"[ERROR] ch{channel} wp{wp} non e' in {RESULTS_DIR}")
    S, nps, w, S_gen, f1, f2, amp = load_pair(channel, wp, r)
    lam_trained = float(r["lambda_wiener"])
    grid = sorted(set(list(LAMBDA_GRID) + [lam_trained]))
    # CONTINUAZIONE: si parte dal lambda addestrato (dove i filtri salvati SONO l'ottimo) e
    # ci si allontana in salita e in discesa, ogni punto warm-startato dal precedente. Cosi'
    # RETRAIN_STEPS bastano anche a lambda molto diversi: partire ogni volta dai filtri
    # salvati lascerebbe i punti lontani non convergiuti, cioe' un BI analitico gonfiato.
    k = grid.index(lam_trained)
    order = [i for i in range(k, len(grid))] + [i for i in range(k - 1, -1, -1)]
    out = []
    for n, i in enumerate(order, 1):
        lam = grid[i]
        if n == len(grid) - k + 1:      # riparte in discesa: warm start dai filtri salvati
            f1, f2 = np.load(os.path.join(FILTERS_DIR, f"f1_ch{channel}_wp{wp}.npy")), \
                     np.load(os.path.join(FILTERS_DIR, f"f2_ch{channel}_wp{wp}.npy"))
        res = analyse_point(channel, wp, lam, f1, f2, S, nps, w, S_gen, amp, shared,
                            retrain_steps=RETRAIN_STEPS)
        f1, f2 = res["f1_half"], res["f2_half"]
        out.append(res)
        print(f"[{n}/{len(grid)}] ch{channel} wp{wp} lam={lam:.3e} "
              f"| s={res['s1_an']:.3f} 1-rho={1 - res['rho_an']:.1e} FWHM={res['fwhm']} "
              f"| BI_an={res['BI_an']:.3e} BI_mc={res['BI_mc']:.3e} "
              f"({res['BI_mc'] / res['BI_an']:.2f})", flush=True)
    write_rows(SCAN_CSV, out, append=True)


def submit_scan():
    os.makedirs(os.path.join(OUT_DIR, "logs"), exist_ok=True)
    for ch, wp in SCAN_PAIRS:
        lines = ["#!/bin/bash", f"#PBS -N {JOB_NAME_PREFIX}_{ch}_{wp}", f"#PBS -q {QUEUE}",
                 f"#PBS -l walltime={WALLTIME},mem={RAM_GB}gb", "#PBS -V",
                 f"#PBS -o {OUT_DIR}/logs/{ch}_{wp}.out", f"#PBS -e {OUT_DIR}/logs/{ch}_{wp}.err",
                 *ENV_SETUP_LINES, f"cd {BASE_DIR}",
                 f"python {os.path.abspath(__file__)} --scan --worker --channel {ch} --wp {wp}"]
        fh = tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False,
                                         dir=os.path.join(OUT_DIR, "logs"))
        fh.write("\n".join(lines) + "\n")
        fh.close()
        os.chmod(fh.name, 0o755)
        print(subprocess.run(["qsub", fh.name], capture_output=True, text=True).stdout.strip())


# ═════════════════════════════════════════════════════════════════════════════
# --selftest
# ═════════════════════════════════════════════════════════════════════════════
def selftest():
    import torch
    import src.analysis as an
    import src.dataset as ds
    r = results_rows()[0]
    ch, wp = int(r["channel"]), int(r["wp"])
    S, nps, w, S_gen, f1h, f2h, amp = load_pair(ch, wp, r)      # controlla anche il kernel
    St = torch.tensor(S, dtype=torch.cfloat)
    W = an.compute_W_torch(St, torch.tensor(nps, dtype=torch.float32),
                           torch.tensor(float(r["lambda_wiener"])))
    f1 = torch.tensor(full_spectrum(f1h), dtype=torch.cfloat)
    f1 = f1 / torch.mean(torch.abs(f1 * W * St))
    g1 = (f1 * W).numpy()

    peak = np.fft.ifft(S * g1).real.max()
    assert abs(peak - 1) < 1e-3, f"picco dell'impulso filtrato = {peak}, atteso 1"

    rng = np.random.default_rng(7)
    ev = np.fft.ifft(amp * S_gen[None, :] + (rng.normal(size=(64, WINDOW_SIZE))
                                             + 1j * rng.normal(size=(64, WINDOW_SIZE)))
                     * np.sqrt(nps)[None, :], axis=1).real
    a_max, a_fix = amplitudes(ev, g1)
    dset = ds.NumpyDataset(ev.astype(np.float32))
    dset.win_length = WINDOW_SIZE
    _, amp1, _ = an.get_PSD_interpole(dset, g1, np.ones(WINDOW_SIZE), np.ones(WINDOW_SIZE))
    assert np.allclose(a_max, np.asarray(amp1), rtol=1e-4), "amplitudes() != get_PSD_interpole"
    #   (il dataset di produzione e' float32: lo scarto misurato e' ~4e-6)
    assert abs(a_fix.mean() / (amp * peak) - 1) < 0.05, "stimatore a tempo fisso non calibrato"
    print(f"[selftest] ch{ch} wp{wp}: kernel OK, picco={peak:.4f}, "
          f"max-search == get_PSD_interpole, A_fix/atteso={a_fix.mean() / (amp * peak):.3f}")


# ═════════════════════════════════════════════════════════════════════════════
# --plots
# ═════════════════════════════════════════════════════════════════════════════
def _read(path):
    if not os.path.exists(path):
        return []
    out = []
    for r in csv.DictReader(open(path)):
        out.append({k: (int(v) if k in ("channel", "wp", "fwhm", "nsim", "retrain_steps")
                        else float(v)) for k, v in r.items()})
    return out


def run_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_lambda_vs_rhot(plt)
    diag = _read(DIAG_CSV)
    if diag:
        plot_diag(plt, diag)
    scan = _read(SCAN_CSV)
    for ch, wp in sorted({(r["channel"], r["wp"]) for r in scan}):
        plot_scan(plt, [r for r in scan if r["channel"] == ch and r["wp"] == wp], ch, wp)


def plot_diag(plt, rows):
    """Il collasso visto sui filtri gia' addestrati: dove lambda e' piccolo, l'analitico sbaglia."""
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
    chans = sorted({r["channel"] for r in rows})
    cmap = plt.get_cmap("tab10")
    col = {c: cmap(i % 10) for i, c in enumerate(chans)}
    for c in chans:
        s = sorted([r for r in rows if r["channel"] == c], key=lambda r: r["lam"])
        lam = [r["lam"] for r in s]
        kw = dict(color=col[c], label=f"ch {c}", marker="o", ms=4, lw=1)
        ax[0, 0].plot(lam, [r["BI_mc"] / r["BI_an"] for r in s], ls="none", **kw)
        ax[0, 1].plot(lam, [r["sigmaY_mc_max"] / r["sigmaY_an"] for r in s], ls="none", **kw)
        ax[1, 0].plot(lam, [1 - r["rho_an"] for r in s], ls="none", **kw)
        ax[1, 0].plot(lam, [1 - r["rho_mc"] for r in s], ls="none", marker="x", ms=5,
                      color=col[c], alpha=0.6)
        ax[1, 1].plot(lam, [r["bias_max"] for r in s], ls="none", **kw)
    for a in ax.ravel():
        a.set_xscale("log")
        a.set_xlabel(r"trained $\lambda$")
        a.grid(alpha=0.3)
        a.axvspan(1e-9, 0.1, color="red", alpha=0.06)
    ax[0, 0].set_yscale("log")
    ax[0, 0].axhline(1, color="k", lw=0.8)
    ax[0, 0].set_ylabel(r"BI$_{\rm MC}$ / BI$_{\rm analytic}$")
    ax[0, 0].set_title("The analytic BI is only trustworthy at large $\\lambda$")
    ax[0, 0].legend(fontsize=8, ncol=2)
    ax[0, 1].set_yscale("log")
    ax[0, 1].axhline(1, color="k", lw=0.8)
    ax[0, 1].set_ylabel(r"$\sigma_Y^{\rm MC}/\sigma_Y^{\rm analytic}$")
    ax[0, 1].set_title("Ratio resolution: model vs truth")
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_ylabel(r"$1-\rho$   ($\bullet$ analytic, $\times$ MC)")
    ax[1, 0].set_title(r"$\sigma_Y \simeq s\sqrt{2(1-\rho)}$: the metric lives on $1-\rho$")
    ax[1, 1].axhline(1, color="k", lw=0.8)
    ax[1, 1].set_ylabel(r"$\langle A^{\rm max\,search}\rangle / \langle A^{\rm fixed\,time}\rangle$")
    ax[1, 1].set_title("Max-search bias: the estimator the model ignores")
    fig.suptitle(f"$\\lambda$ collapse diagnostics — {RESULTS_NAME}  "
                 f"(shaded: $\\lambda<0.1$)", fontsize=11)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, f"lambda_diag_{_TAG}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[plot] {p}")


def plot_scan(plt, rows, ch, wp):
    """La fetta in lambda: cosa vede l'ottimizzatore (analitico) e cosa succede davvero (MC)."""
    rows = sorted(rows, key=lambda r: r["lam"])
    lam = np.array([r["lam"] for r in rows])
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
    g = lambda k: np.array([r[k] for r in rows])
    ax[0, 0].plot(lam, g("BI_an"), "o-", label="analytic (the loss J)")
    ax[0, 0].errorbar(lam, g("BI_mc"), yerr=g("sigma_BI"), fmt="s-", color="crimson",
                      label="Monte Carlo (true estimator)")
    ax[0, 0].set_ylabel("BI")
    ax[0, 0].set_title("What the optimizer minimizes vs what it gets")
    ax[0, 1].plot(lam, g("sigmaY_an"), "o-", label="analytic")
    ax[0, 1].plot(lam, g("sigmaY_mc_fix"), "^-", label="MC, fixed time (same estimator)")
    ax[0, 1].plot(lam, g("sigmaY_mc_max"), "s-", color="crimson", label="MC, max search (real)")
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_ylabel(r"$\sigma_Y$")
    ax[0, 1].set_title("Estimator mismatch")
    ax[1, 0].plot(lam, g("s1_an"), "o-", label=r"$s$ analytic")
    ax[1, 0].plot(lam, g("s1_mc"), "s-", color="crimson", label=r"$s$ MC")
    ax[1, 0].plot(lam, 1 - g("rho_an"), "o--", label=r"$1-\rho$ analytic")
    ax[1, 0].plot(lam, 1 - g("rho_mc"), "s--", color="crimson", label=r"$1-\rho$ MC")
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_ylabel(r"$s$,  $1-\rho$")
    ax[1, 0].set_title(r"Both factors of $\sigma_Y=s\sqrt{2(1-\rho)}$ degenerate")
    ax[1, 1].plot(lam, g("fwhm"), "o-", label="FWHM of filtered pulse [samples]")
    ax[1, 1].plot(lam, g("bias_max"), "s-", color="crimson", label="max-search bias")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_title("Why the max search fails: the pulse becomes a spike")
    for a in ax.ravel():
        a.set_xscale("log")
        a.set_xlabel(r"$\lambda$")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
        a.axvline(rows[0]["lam"] if False else _trained_lambda(ch, wp), color="k", ls=":",
                  lw=1.2)
    fig.suptitle(f"$\\lambda$ slice — ch {ch}, WP {wp} "
                 f"(V$_{{bias}}$ = {rows[0]['vbias']} V, retrain {rows[0]['retrain_steps']} steps; "
                 f"dotted: trained $\\lambda$)", fontsize=11)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, f"lambda_scan_{_TAG}_ch{ch}_wp{wp}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[plot] {p}")


def _trained_lambda(ch, wp):
    for r in results_rows():
        if int(r["channel"]) == ch and int(r["wp"]) == wp:
            return float(r["lambda_wiener"])
    return np.nan


def plot_lambda_vs_rhot(plt):
    """Perche' ch91 collassa a TUTTI i WP e ch34 solo sotto 4 V: il lambda addestrato segue
    rho_t = SNR * beta, la figura di merito temporale del pile-up. Legge il CSV di training
    con tutti i canali (m205_results_wiener)."""
    path = os.path.join(BASE_DIR, "m205_results_wiener", "BI_results_m205_wiener.csv")
    if not os.path.exists(path):
        return
    rows = list(csv.DictReader(open(path)))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    cmap = plt.get_cmap("tab10")
    chans = sorted({r["channel"] for r in rows}, key=int)
    for i, c in enumerate(chans):
        s = sorted([r for r in rows if r["channel"] == c], key=lambda r: float(r["vbias"]))
        v = [float(r["vbias"]) for r in s]
        lam = [float(r["lambda_wiener"]) for r in s]
        rt = [float(r["SNR"]) * float(r["beta_Hz"]) for r in s]
        good = c in ("31", "34", "71", "83", "91")
        kw = dict(color=cmap(i % 10), marker="o" if good else "x", ms=5,
                  alpha=1.0 if good else 0.35, lw=1 if good else 0)
        ax[0].plot(v, lam, label=f"ch {c}", **kw)
        ax[1].plot(rt, lam, ls="none", label=f"ch {c}", **kw)
    lg = [float(r["lambda_wiener"]) for r in rows if r["channel"] in ("31", "34", "71", "83", "91")]
    rg = [float(r["SNR"]) * float(r["beta_Hz"]) for r in rows
          if r["channel"] in ("31", "34", "71", "83", "91")]
    cc = np.corrcoef(np.log(lg), np.log(rg))[0, 1]
    for a, xl in zip(ax, ["$V_{bias}$ [V]", r"$\rho_t = \mathrm{SNR}\cdot\beta$ [Hz]"]):
        a.set_yscale("log")
        a.set_xlabel(xl)
        a.set_ylabel(r"trained $\lambda$")
        a.axhline(0.1, color="k", ls=":", lw=1)
        a.grid(alpha=0.3)
    ax[0].set_xscale("log")
    ax[1].set_xscale("log")
    ax[0].legend(fontsize=7, ncol=3)
    ax[0].set_title(r"Trained $\lambda$ (dotted: collapse threshold $\lambda\simeq0.1$)")
    ax[1].set_title(f"Good channels: corr(log $\\lambda$, log $\\rho_t$) = {cc:+.2f}")
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "lambda_vs_rhot.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[plot] {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--channel", type=int)
    ap.add_argument("--wp", type=int)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if a.selftest:
        selftest()
    elif a.plots:
        run_plots()
    elif a.scan:
        if a.worker:
            scan_pair(a.channel, a.wp)
        elif SUBMIT_MODE == "qsub":
            submit_scan()
        else:
            for ch, wp in SCAN_PAIRS:
                scan_pair(ch, wp)
    else:
        run_diag()


if __name__ == "__main__":
    main()
