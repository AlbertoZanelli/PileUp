"""
scan_residuals_bessel_m205.py
=============================
Fit degli average pulse MAX-ALIGNED di m205 col modello pole-zero + BESSEL (ordine 6 @ 2.5 kHz,
FISSI), per OGNI working point e per 4 MODELLI, parallelizzato sul cluster (un job per fit).
Alla fine produce, per ogni WP, una canva 2x2 coi quattro fit e i rispettivi residui.

Rispetto alla versione precedente di questo file (scan su tutte le combinazioni n_real x cc x
nzer, AP dal ROOT, errore = sigma di baseline costante) porta tutte le modifiche validate in
test/fit_one_pulse_m205.py:

  1. DATO: l'AP allineato al MASSIMO prodotto da build_medianAP_maxalign_m205.py
     (m205_AP_pulses/medianAP_maxalign_ch<ch>_wp<wp>.npy), non l'AP di Octopus dal ROOT.
  2. ERRORI per time-bin dai SINGOLI impulsi che formano l'AP (m205_AP_pulses/pulses_*.npy),
     riallineati al massimo, con incertezza della mediana stimata per BOOTSTRAP (nessuna
     ipotesi di gaussianita': la formula 1.2533*std/sqrt(N) sovrastima del ~25% sul fronte).
     Floor al livello della baseline (al picco tutti gli impulsi valgono 1 -> std = 0).
  3. FIT PESATO 1/err: la cost e' un chi^2 vero, quindi chi = RMS/err e' confrontabile con 1.
  4. PARAM_BOUND = 2*pi*5000 = NYQUIST (era 8000 = 1273 Hz: i modelli da 6 poli in su
     finivano con DUE POLI INCOLLATI AL BOUND).
  5. MULTI-START fitto (28 start): con 8 start il fit finiva in un minimo locale su 6 WP su 15.
  6. Diagnostica per ogni fit: RAILING (parametri sul bound), ERRORI dei parametri dalla
     covarianza (J^T J)^-1 e NUMERO DI CONDIZIONE (segnala i parametri non identificabili).
  7. NIENTE coppie complesse coniugate: solo poli reali (le CC non convergevano).

Config: CHANNELS (uno o piu' canali), AP_SOURCE ("root" = medianAP di Octopus, errori
dall'APdistro con floor sull'RMS di baseline; "maxalign" = AP allineato al massimo dai .npy,
errori per bootstrap dai singoli impulsi) e MODELS (lista di modelli da fittare).

Flusso (come gli altri programmi del progetto):
  1. Orchestratore (default): enumera (WP x modello) e sottomette un job per fit, poi termina.
  2. Worker (--worker --wp W --nreal N --cc 0 --nzer Z): fa UN fit e salva un .npz.
  3. --plot: a job finiti, costruisce le canve 2x2 (una per WP) dai .npz e salva il fit
     MIGLIORE di ogni WP come vettore .npy (bestfit_ch<ch>_wp<wp>.npy).
  4. --csv: tabella riassuntiva di tutti i fit.
  5. --psd: power spectrum dei fit migliori sovrapposto a quello del medianAP letto dal ROOT
     (stessa definizione di PSD del resto del progetto: compute_psd di plot_AP_spectra_m205).

Esempi:
    python scan_residuals_bessel_m205.py                                  # sottomette i job
    python scan_residuals_bessel_m205.py --worker --wp 15 --nreal 9 --cc 0 --nzer 4
    python scan_residuals_bessel_m205.py --plot
    python scan_residuals_bessel_m205.py --csv
"""

from __future__ import annotations

import os
import re
import sys
import glob
import time
import argparse

import numpy as np
from scipy.signal import besselap
from scipy.optimize import least_squares, minimize_scalar

import scan_residuals_m205 as sc   # riuso SOLO la meccanica del cluster (qsub, throttling, .sh)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE (tutto qui in testa)
# ═════════════════════════════════════════════════════════════════════════════
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
PULSE_DIR = os.path.join(BASE_DIR, "m205_AP_pulses")      # .npy di extract/build (AP + impulsi)
SCAN_DIR  = os.path.join(BASE_DIR, "residual_scan_bessel")

CHANNELS = [34, 83]        # uno o piu' canali: i file di output hanno il canale nel nome
DATA_DIR = os.path.join(BASE_DIR, "Processed")
MEAS_NAME = "000205"

# ── Da dove viene l'AP da fittare, e di conseguenza come si stimano gli errori:
# AP_SOURCE = "root"     -> medianAP di Octopus dal ROOT (disponibile per TUTTI i canali).
#   Errore per time-bin dalla TH2D averagepulse_ap_wp<wp>_APdistro, con FLOOR pari all'RMS di
#   baseline dell'AP. Il floor non e' un tappo arbitrario: l'APdistro ha bin di ampiezza 0.006,
#   sette volte piu' larghi del rumore, quindi nelle zone quiete tutti gli impulsi cadono nello
#   stesso bin e la sua dispersione e' ESATTAMENTE ZERO; l'RMS del pretrigger dell'AP e' invece
#   proprio l'errore della mediana li'. Verificato sul ch91 WP15 contro il bootstrap dai veri
#   impulsi: floor 1.73e-4 vs 1.88e-4 (-8%), fronte 1.28e-3 vs 1.40e-3 (-9%).
# AP_SOURCE = "octopus"  -> AP ricostruito come MEDIANA dei singoli impulsi COSI' COME SONO
#   salvati da extract_AP_pulses_m205.py, cioe' allineati alla MEZZA SALITA esattamente come fa
#   Octopus (triggerdelay.midsample -> pretrigger). L'AP che ne esce e' IDENTICO al medianAP del
#   ROOT (verificato: differenza 0.0 su tutti i WP), ma qui arrivano anche gli errori veri per
#   time-bin, dal BOOTSTRAP degli stessi impulsi. E' la modalita' da preferire: stesso dato del
#   ROOT, errori non binnati. Richiede i .npy degli impulsi, cioe' il file .bin del canale.
# AP_SOURCE = "maxalign" -> come "octopus" ma con gli impulsi RIALLINEATI AL MASSIMO prima della
#   mediana (build_medianAP_maxalign_m205.align_on_max): AP diverso da quello del ROOT.
AP_SOURCE = "octopus"

# Le cartelle di output PORTANO L'AP_SOURCE nel nome: fit fatti su template diversi non si
# sovrascrivono a vicenda (l'orchestratore azzera FITS_DIR a ogni lancio).
OUTDIR    = os.path.join(SCAN_DIR, "fits_" + AP_SOURCE)
FITS_DIR  = os.path.join(OUTDIR, "_fits")
LOG_DIR   = os.path.join(OUTDIR, "logs")
JOBS_DIR  = os.path.join(OUTDIR, "jobs")

AP_PATTERN    = "medianAP_maxalign_ch{ch}_wp{wp}.npy"     # AP allineato al massimo (maxalign)
PULSE_PATTERN = "pulses_ch{ch}_wp{wp}.npy"                # impulsi che lo formano (maxalign)
SAMPLING_RATE = 10_000                                    # Hz, per l'asse dei tempi

# ── Modelli da fittare: (n_poli_reali, n_zeri). Niente coppie CC.
MODELS = [(7, 3), (7, 4), (7, 5), (8, 5), (8, 6), (9, 4), (9, 5), (10, 4), (10, 5), (10, 6)]   # 4 modelli da fittare per ogni WP

BESSEL_ORDER = 6         # FISSO
FCUT         = 2500      # Hz, FISSO

PARAM_BOUND = 2 * np.pi * SAMPLING_RATE / 2   # 31416 rad/s = Nyquist: limite fisico
T0_WINDOW   = 0.01                            # t0 vincolato a +-10 ms attorno al picco

# ── Errori per time-bin dai singoli impulsi
ERR_METHOD = "bootstrap"   # "bootstrap" (nessuna ipotesi) | "gauss" (1.2533*std/sqrt(N))
BOOT_N     = 800           # ricampionamenti del bootstrap (seed fisso -> riproducibile)
WEIGHT_FIT = True          # fit pesato 1/err

# ── Multi-start: fattori (rise, decay) sui guess fisici. 28 start (vedi punto 5 in testa).
STARTS = [(rs, ds) for rs in (0.2, 0.3, 0.5, 1, 2, 3, 5) for ds in (0.5, 1, 2, 3)]
MAX_NFEV = 3000

RESET = True               # l'orchestratore riparte da una cartella di fit pulita

# ── Cluster (la meccanica e' quella di scan_residuals_m205)
sc.SUBMIT_MODE  = "qsub"   # "qsub" sul server, "local" per eseguire tutto in sequenza qui
sc.QUEUE        = "cupid"
sc.WALLTIME     = "24:00:00"
sc.RAM_GB       = 4
sc.MAX_PARALLEL_JOBS = 135
sc.JOB_NAME_PREFIX   = "BSCAN"
sc.ENV_SETUP_LINES   = ["source /home/zanelli/LoadOctopus.sh"]
sc.SCRIPT_PATH  = os.path.abspath(__file__)     # i job rilanciano QUESTO file
sc.BASE_DIR     = BASE_DIR
sc.LOG_DIR      = LOG_DIR
sc.JOBS_DIR     = JOBS_DIR


def label(n_real, nzer):
    return f"{n_real}p z{nzer}"


def pulse_path(ch, wp):
    """Percorso dei singoli impulsi. Si accetta sia PULSE_DIR/pulses_ch<ch>_wp<wp>.npy sia
    PULSE_DIR/ch<ch>/pulses_ch<ch>_wp<wp>.npy (i canali scaricati dal server arrivano in una
    sottocartella per canale)."""
    name = PULSE_PATTERN.format(ch=ch, wp=wp)
    flat = os.path.join(PULSE_DIR, name)
    return flat if os.path.exists(flat) else os.path.join(PULSE_DIR, f"ch{ch}", name)


def combo_npz(ch, wp, n_real, nzer):
    return os.path.join(FITS_DIR, f"scan_ch{ch}_wp{wp}_{n_real}p_z{nzer}.npz")


def root_file(ch):
    fs = glob.glob(os.path.join(DATA_DIR, f"Processed_*_{MEAS_NAME}_{ch}.root"))
    if not fs:
        raise RuntimeError(f"ROOT del canale {ch} non trovato in {DATA_DIR}")
    return fs[0]


def list_wps(ch):
    """WP dispari disponibili per il canale: dalle chiavi del ROOT ("root") oppure dai .npy
    dell'AP max-aligned ("maxalign")."""
    if AP_SOURCE == "root":
        import uproot
        with uproot.open(root_file(ch)) as f:
            keys = [m.group(1) for k in f.keys()
                    for m in [re.search(r"averagepulse_ap_wp(\d+)_medianAP", k)] if m]
        return sorted({int(w) for w in keys if int(w) % 2})
    pat = PULSE_PATTERN.format(ch=ch, wp="*")          # servono i singoli impulsi
    files = glob.glob(os.path.join(PULSE_DIR, pat)) + glob.glob(os.path.join(PULSE_DIR, f"ch{ch}", pat))
    return sorted({int(m.group(1)) for f in files
                   for m in [re.search(r"_wp(\d+)\.npy$", f)] if m})


# ═════════════════════════════════════════════════════════════════════════════
# DATO + ERRORI
# ═════════════════════════════════════════════════════════════════════════════
def load_pulses(ch, wp):
    """I singoli impulsi che formano l'AP (extract_AP_pulses_m205.py). Con AP_SOURCE="octopus"
    restano come sono, cioe' allineati alla mezza salita come fa Octopus; con "maxalign" vengono
    riallineati al massimo."""
    path = pulse_path(ch, wp)
    if not os.path.exists(path):
        raise RuntimeError(f"impulsi non trovati: {path}\n"
                           f"    servono per AP_SOURCE='{AP_SOURCE}': gira extract_AP_pulses_m205.py "
                           f"sul canale {ch} (richiede il file .bin del raw).")
    p = np.load(path)
    if AP_SOURCE == "maxalign":
        sys.path.insert(0, BASE_DIR)
        from build_medianAP_maxalign_m205 import align_on_max
        p = align_on_max(p)
    return p


def load_ap(ch, wp):
    """AP peak-normalizzato + asse dei tempi (centri dei bin), secondo AP_SOURCE:
    "root" = medianAP di Octopus dal file; "octopus"/"maxalign" = mediana dei singoli impulsi."""
    if AP_SOURCE == "root":
        import uproot
        with uproot.open(root_file(ch)) as f:
            v = np.asarray(f[f"averagepulse_ap_wp{wp}_medianAP"].values(), float)
    else:
        v = np.median(load_pulses(ch, wp), axis=0)
    t = (np.arange(len(v)) + 0.5) / SAMPLING_RATE
    return t, v / v.max()


def err_from_pulses(ch, wp):
    """Errore per TIME-BIN dai singoli impulsi, per BOOTSTRAP: la dispersione delle mediane
    ricampionate E' l'errore della mediana (contiene gia' il 1/sqrt(N)) e non assume nessuna
    forma di distribuzione. Floor al livello della baseline: con "maxalign" al picco tutti gli
    impulsi valgono 1 per costruzione (std = 0) e in coda c'e' lo zero-padding dello shift."""
    p = load_pulses(ch, wp)
    n = len(p)
    rng = np.random.default_rng(0)
    meds = np.empty((BOOT_N, p.shape[1]))
    for b in range(BOOT_N):
        meds[b] = np.median(p[rng.integers(0, n, n)], axis=0)
    err = meds.std(axis=0, ddof=1)
    n_base = int(0.40 * p.shape[1])
    return np.maximum(err, np.median(err[:n_base])), n


def err_from_apdistro(ch, wp, v):
    """Errore per TIME-BIN dalla TH2D APdistro (AP_SOURCE="root"): per ogni colonna la
    dispersione dei singoli impulsi / sqrt(N), col fattore 1.2533 = sqrt(pi/2) che porta
    dall'errore della media a quello della MEDIANA (vale per distribuzione gaussiana).

    FLOOR = RMS di baseline dell'AP: i bin dell'istogramma sono larghi 0.006, sette volte piu'
    del rumore, quindi dove il segnale e' quieto tutti gli impulsi cadono nello stesso bin e la
    dispersione misurata e' zero. L'RMS del pretrigger dell'AP e' proprio l'errore della mediana
    in quella regione, quindi il floor e' la stima corretta, non un tappo."""
    import uproot
    with uproot.open(root_file(ch)) as f:
        h = f[f"averagepulse_ap_wp{wp}_APdistro"]
        cnt = np.asarray(h.values(), float)
        yc = np.asarray(h.axis(1).centers(), float)
    N = cnt.sum(axis=1)
    n_ev = int(round(float(N.max())))
    good = N > 0
    mean = np.zeros(cnt.shape[0]); var = np.zeros(cnt.shape[0])
    mean[good] = (cnt[good] * yc).sum(1) / N[good]
    var[good] = (cnt[good] * (yc - mean[good][:, None]) ** 2).sum(1) / N[good]
    err = 1.2533 * np.sqrt(var) / np.sqrt(max(n_ev, 1))
    floor = float(v[:int(0.40 * len(v))].std())
    return np.maximum(err, floor), n_ev


def pulses_sigma(ch, wp, v):
    """Errore per time-bin secondo AP_SOURCE. Ritorna (err, n_impulsi)."""
    return (err_from_apdistro(ch, wp, v) if AP_SOURCE == "root"
            else err_from_pulses(ch, wp))


# ═════════════════════════════════════════════════════════════════════════════
# MODELLO: pole-zero (poli REALI) convoluto col Bessel analogico
# ═════════════════════════════════════════════════════════════════════════════
def make_pulse_bessel(bessel_order, fcut, zeros, poles):
    """Impulso pole-zero (N zeri, poli reali) convoluto con un Bessel analogico
    (ordine bessel_order, taglio fcut Hz). Normalizzato a picco 1 a t=0."""
    poles = np.asarray(poles, dtype=complex)
    zeros = np.asarray(zeros, dtype=complex)
    wc = 2 * np.pi * fcut
    diff = poles[:, None] - poles[None, :]
    denom = np.prod(np.where(np.eye(len(poles), dtype=bool), 1.0, diff), axis=1)
    num = np.ones(len(poles), dtype=complex)
    for z in zeros:
        num *= (poles - z)
    k = num / denom

    _, p_norm, g_norm = besselap(bessel_order)
    p_filt = wc * p_norm
    g_filt = g_norm * wc ** bessel_order
    diff_f = p_filt[:, None] - p_filt[None, :]
    denom_f = np.prod(np.where(np.eye(len(p_filt), dtype=bool), 1.0, diff_f), axis=1)
    A = g_filt / denom_f
    coef = k[:, None] * A[None, :] / (poles[:, None] - p_filt[None, :])
    B = np.sum(coef, axis=1)
    C = -np.sum(coef, axis=0)

    def f_raw(tt):
        tt = np.asarray(tt, float)
        out = np.zeros_like(tt, dtype=complex)
        m = tt >= 0
        x = tt[m]
        out[m] = (np.sum(B[:, None] * np.exp(poles[:, None] * x), axis=0)
                  + np.sum(C[:, None] * np.exp(p_filt[:, None] * x), axis=0))
        return out.real

    # Normalizzazione del picco con minimize_scalar (non su griglia): i poli sono tutti REALI,
    # quindi la risposta e' unimodale. Su griglia da 3000 punti in 0.15 s il picco sarebbe
    # localizzato a 50 us = mezzo campione, e il fit ne risente pesantemente (chi 1 -> 5).
    tpk = minimize_scalar(lambda x: -f_raw(np.array([x]))[0], bounds=(0, 0.1),
                          method="bounded").x
    pk = float(f_raw(np.array([tpk]))[0]) or 1.0
    return lambda x: f_raw(np.asarray(x, float) + tpk) / pk


def measure_timescales(t, v):
    """t_rise 10->90%, t_dec a 1/e, istante del picco: le scale da cui nascono i guess."""
    dt = t[1] - t[0]
    imax = int(np.argmax(v)); peak = float(v[imax])
    up = v[:imax]
    i10 = np.where(up > 0.10 * peak)[0]; i90 = np.where(up > 0.90 * peak)[0]
    t_rise = max((t[i90[0]] - t[i10[0]]) if (len(i10) and len(i90)) else 5 * dt, dt)
    be = np.where(v[imax:] < peak / np.e)[0]
    t_dec = (t[imax + be[0]] - t[imax]) if len(be) else 10 * dt
    return t_rise, t_dec, t[imax]


# ═════════════════════════════════════════════════════════════════════════════
# FIT di UN modello su UN WP
# ═════════════════════════════════════════════════════════════════════════════
def fit_model(t, v, err, n_real, nzer):
    """Fit pesato 1/err del modello (n_real poli reali, nzer zeri) + Bessel fisso.
    theta = [t0, zeri..., poli...]. Ritorna dict con fit, chi, rms, theta, errori dei
    parametri, numero di condizione e lista dei parametri in railing."""
    t_rise, t_dec, t_peak = measure_timescales(t, v)
    sig = err if WEIGHT_FIT else np.ones_like(err)

    B = PARAM_BOUND
    lo = np.array([t_peak - T0_WINDOW] + [-B] * nzer + [-B] * n_real)
    hi = np.array([t_peak + T0_WINDOW] + [0.0] * nzer + [0.0] * n_real)

    def unpack(th):
        return th[0], th[1:1 + nzer], th[1 + nzer:1 + nzer + n_real]

    def model(th):
        t0, zeros, poles = unpack(th)
        return make_pulse_bessel(BESSEL_ORDER, FCUT, zeros, poles)(t - t0)

    def resid(th):
        try:
            y = model(th)
        except Exception:
            return np.full(len(t), 1e3)
        return np.nan_to_num((y - v) / sig, nan=1e6, posinf=1e6, neginf=-1e6)

    def guess(rs, ds):
        """Guess FISICI: poli reali con tau geometricamente spaziate tra salita e coda
        (p0 = -1/t_rise ... p_last = -1/t_dec); zeri tra poli consecutivi."""
        taus = np.geomspace(t_rise * rs, t_dec * ds, n_real)
        poles = -1.0 / taus
        gaps = np.sqrt(taus[:-1] * taus[1:]) if n_real >= 2 else np.array([t_rise * rs])
        zeros = -1.0 / np.resize(gaps, nzer) if nzer else np.empty(0)
        return np.array([t_peak, *zeros, *poles], float)

    best = None
    for rs, ds in STARTS:
        th0 = np.clip(guess(rs, ds), lo + 1e-12, hi - 1e-12)
        try:
            r = least_squares(resid, th0, bounds=(lo, hi), method="trf", max_nfev=MAX_NFEV)
            y = model(r.x)
        except Exception:
            continue
        if not np.all(np.isfinite(y)):
            continue
        cost = float(np.mean(resid(r.x) ** 2))
        if best is None or cost < best[0]:
            best = (cost, r.x, y, r.jac)
    if best is None:
        raise RuntimeError("nessuno start convergente")
    _, theta, fit, jac = best

    rms = float(np.sqrt(np.mean((fit - v) ** 2)))
    chi = float(np.sqrt(np.mean(((fit - v) / err) ** 2)))

    # ── errori dei parametri: cov = (J^T J)^-1 * s^2, pseudo-inversa via SVD. Col fit pesato
    #    s^2 ~ 1. Il numero di condizione dice se i parametri sono identificabili (>1e8 = no).
    r_fin = resid(theta)
    s2 = float(np.sum(r_fin ** 2) / max(len(r_fin) - len(theta), 1))
    _, sv, vt = np.linalg.svd(jac, full_matrices=False)
    cond = float(sv.max() / sv.min()) if sv.min() > 0 else np.inf
    sv_inv = np.where(sv > sv.max() * 1e-14, 1.0 / np.maximum(sv, 1e-300), 0.0)
    perr = np.sqrt(np.abs(np.diag((vt.T * sv_inv ** 2) @ vt * s2)))

    # ── railing: t0 sull'intervallo; zeri/poli sul bound inferiore (|x| ~ PARAM_BOUND) o
    #    superiore (tau = 1/|x| piu' lungo della finestra -> il dato non lo vincola).
    T_win = float(t[-1] - t[0])
    names = ["t0"] + [f"zero{i+1}" for i in range(nzer)] + [f"pole{i+1}" for i in range(n_real)]
    rail = []
    for i, nm in enumerate(names):
        x = theta[i]
        if nm == "t0":
            bad = min(abs(x - lo[i]), abs(x - hi[i])) / (hi[i] - lo[i]) <= 1e-3
        else:
            bad = abs(x) >= 0.999 * PARAM_BOUND or (abs(x) and 1.0 / abs(x) >= T_win)
        if bad:
            rail.append(f"{nm}={x:.4g}")
    return {"fit": fit, "rms": rms, "chi": chi, "theta": theta, "perr": perr,
            "cond": cond, "rail": rail, "names": names}


# ═════════════════════════════════════════════════════════════════════════════
# WORKER: un fit -> un .npz
# ═════════════════════════════════════════════════════════════════════════════
def run_worker(ch, wp, n_real, nzer):
    try:
        t, v = load_ap(ch, wp)
        err, n_pulses = pulses_sigma(ch, wp, v)
        res = fit_model(t, v, err, n_real, nzer)
        os.makedirs(FITS_DIR, exist_ok=True)
        np.savez(combo_npz(ch, wp, n_real, nzer),
                 fit=res["fit"].astype(np.float32), err=err.astype(np.float32),
                 rms=res["rms"], chi=res["chi"], theta=res["theta"], perr=res["perr"],
                 cond=res["cond"], rail=np.array(res["rail"], dtype=object),
                 names=np.array(res["names"], dtype=object),
                 n_real=n_real, nzer=nzer, wp=wp, channel=ch, n_pulses=n_pulses,
                 ap_source=AP_SOURCE)
        rail = ("RAILING: " + ", ".join(res["rail"])) if res["rail"] else "no railing"
        print(f"[OK] ch{ch} wp{wp} {label(n_real, nzer)}  chi={res['chi']:.2f}  rms={res['rms']:.2e}  "
              f"cond={res['cond']:.1e}  {rail}")
    except Exception as e:
        print(f"[ERROR] ch{ch} wp{wp} {label(n_real, nzer)}: {e}")
        sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# PLOT: una canva 2x2 per WP (i 4 modelli, ciascuno con fit e residuo)
# ═════════════════════════════════════════════════════════════════════════════
def plot_wp(ch, wp):
    """Una canva per (canale, WP): un pannello per modello (griglia 2 colonne, o 1 se il
    modello e' uno solo), ciascuno con il fit sopra e il residuo in unita' di errore sotto."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, v = load_ap(ch, wp)
    t_rise, t_dec, t_peak = measure_timescales(t, v)
    ncol = 2 if len(MODELS) > 1 else 1
    nrow = int(np.ceil(len(MODELS) / ncol)) * 2          # ogni modello = riga fit + riga residuo
    fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 5 * nrow // 2), squeeze=False,
                             sharex="col", gridspec_kw={"height_ratios": [3, 1] * (nrow // 2)})
    n_ok = 0
    for k, (n_real, nzer) in enumerate(MODELS):
        row, col = (k // ncol) * 2, k % ncol
        ax, axr = axes[row][col], axes[row + 1][col]
        p = combo_npz(ch, wp, n_real, nzer)
        if not os.path.exists(p):
            ax.text(0.5, 0.5, f"{label(n_real, nzer)}: fit mancante", ha="center",
                    va="center", transform=ax.transAxes)
            continue
        d = np.load(p, allow_pickle=True)
        fit, err = np.asarray(d["fit"], float), np.asarray(d["err"], float)
        n_ok += 1
        ax.plot(t, v, "k.", ms=2, label=f"AP data ({AP_SOURCE})")
        ax.plot(t, fit, "r-", lw=1.4, label="fit")
        ax.fill_between(t, fit - 3 * err, fit + 3 * err, color="red", alpha=0.15,
                        label=r"fit $\pm\,3\cdot$err")
        ax.set_ylabel("pulse (peak-norm.)")
        rail = ", ".join(d["rail"].tolist()) if len(d["rail"]) else ""
        ax.set_title(f"{label(n_real, nzer)} — RMS={float(d['rms']):.2e}  χ={float(d['chi']):.2f}"
                     f"  cond={float(d['cond']):.1e}" + (f"  [RAILING: {rail}]" if rail else ""),
                     fontsize=9.5)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, ls="--", alpha=0.4)
        theta = np.asarray(d["theta"], float)
        txt = [f"t0 = {theta[0]:.5f} s"]
        txt += ["zeros: " + ", ".join(f"{z:.0f}" for z in theta[1:1 + nzer])] if nzer else []
        txt += ["poles: " + ", ".join(f"{q:.0f}" for q in theta[1 + nzer:])]
        ax.text(0.98, 0.62, "\n".join(txt), transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, family="monospace",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
        axr.plot(t, (fit - v) / err, "b-", lw=0.8)
        axr.axhline(0, color="k", lw=0.8)
        axr.axhline(+3, color="0.5", ls="--", lw=0.8)
        axr.axhline(-3, color="0.5", ls="--", lw=0.8)
        axr.set_ylabel("(fit-data)/err", fontsize=8)
        axr.grid(True, ls="--", alpha=0.4)
        axr.set_xlim(t_peak - 5 * t_rise, t_peak + 8 * t_dec)
    for col in range(ncol):
        axes[nrow - 1][col].set_xlabel("Time (s)")
    fig.suptitle(f"m205 Ch{ch} WP{wp} — pole-zero + Bessel({BESSEL_ORDER}@{FCUT}Hz), "
                 f"AP da '{AP_SOURCE}'", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUTDIR, f"fit_ch{ch}_wp{wp}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[{n_ok}/{len(MODELS)} fit] canva -> {out}")


def best_fit(ch, wp):
    """(n_real, nzer, dati del .npz) del modello col chi minore per questo (canale, WP);
    None se non c'e' nessun fit. Con un solo modello in MODELS e' semplicemente quello."""
    cands = []
    for n_real, nzer in MODELS:
        p = combo_npz(ch, wp, n_real, nzer)
        if os.path.exists(p):
            d = np.load(p, allow_pickle=True)
            cands.append((float(d["chi"]), n_real, nzer, d))
    if not cands:
        return None
    chi, n_real, nzer, d = min(cands, key=lambda x: x[0])
    return n_real, nzer, d


def save_best_fits():
    """Salva l'impulso fittato migliore di ogni (canale, WP) come vettore .npy, pronto da usare
    come template (stessa griglia dell'AP, picco normalizzato a 1).
    -> <OUTDIR>/bestfit_ch<ch>_wp<wp>.npy"""
    os.makedirs(OUTDIR, exist_ok=True)
    for ch in CHANNELS:
        for wp in list_wps(ch):
            b = best_fit(ch, wp)
            if b is None:
                continue
            n_real, nzer, d = b
            fit = np.asarray(d["fit"], float)
            out = os.path.join(OUTDIR, f"bestfit_ch{ch}_wp{wp}.npy")
            np.save(out, fit / fit.max())
            print(f"ch{ch} wp{wp:<3d} migliore: {label(n_real, nzer):8s} "
                  f"chi={float(d['chi']):.2f}  -> {os.path.basename(out)}")


def run_plot():
    os.makedirs(OUTDIR, exist_ok=True)
    for ch in CHANNELS:
        for wp in list_wps(ch):
            plot_wp(ch, wp)
    save_best_fits()


def run_psd():
    """Power spectrum del fit migliore di ogni WP, sovrapposto a quello del medianAP letto dal
    ROOT. Stessa definizione di PSD del resto del progetto: si riusa compute_psd di
    plot_AP_spectra_m205 (media nulla, finestra di Hann, |rfft|^2/(fs*N)). Una griglia per
    canale, un pannello per WP.  -> AP_fit_vs_root_psd_ch<ch>.png"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import uproot
    sys.path.insert(0, BASE_DIR)
    from plot_AP_spectra_m205 import compute_psd

    for ch in CHANNELS:
        wps = list_wps(ch)
        ncol = 4
        nrow = int(np.ceil(len(wps) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.1 * nrow),
                                 sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel()
        with uproot.open(root_file(ch)) as f:
            for ax, wp in zip(axes, wps):
                ap_root = np.asarray(f[f"averagepulse_ap_wp{wp}_medianAP"].values(), float)
                ap_root = ap_root / ap_root.max()
                fr, psd_root = compute_psd(ap_root, SAMPLING_RATE)
                ax.loglog(fr[1:], psd_root[1:], color="0.35", lw=1.0, label="medianAP (ROOT)")
                b = best_fit(ch, wp)
                if b is not None:
                    n_real, nzer, d = b
                    fit = np.asarray(d["fit"], float); fit = fit / fit.max()
                    ff, psd_fit = compute_psd(fit, SAMPLING_RATE)
                    ax.loglog(ff[1:], psd_fit[1:], color="tab:red", lw=1.2, alpha=0.9,
                              label=f"fit {label(n_real, nzer)} (χ={float(d['chi']):.2f})")
                ax.set_title(f"WP{wp}  ({sc.wp_to_vbias(wp):g} V)", fontsize=9)
                ax.grid(True, which="both", ls="--", alpha=0.35)
                ax.legend(fontsize=7, loc="lower left")
        for ax in axes[len(wps):]:
            ax.set_visible(False)
        for ax in axes[max(len(wps) - ncol, 0):len(wps)]:
            ax.set_xlabel("Frequency (Hz)")
        for k in range(0, len(wps), ncol):
            axes[k].set_ylabel("AP power spectrum")
        fig.suptitle(f"m205 Ch{ch} — power spectrum: fit vs medianAP di Octopus", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = os.path.join(OUTDIR, f"AP_fit_vs_root_psd_ch{ch}.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"PSD -> {out}")


# ═════════════════════════════════════════════════════════════════════════════
# CSV riassuntivo
# ═════════════════════════════════════════════════════════════════════════════
def run_csv():
    out = os.path.join(OUTDIR, "fit_params.csv")
    head = ("channel,wp,vbias,ap_source,model,n_real,nzer,bessel_order,fcut,n_pulses,rms,chi,"
            "cond,railing,t0,zeros,real_poles,param_errors")
    lines = [head]
    for ch in CHANNELS:
        for wp in list_wps(ch):
            for n_real, nzer in MODELS:
                p = combo_npz(ch, wp, n_real, nzer)
                if not os.path.exists(p):
                    continue
                d = np.load(p, allow_pickle=True)
                th = np.asarray(d["theta"], float); pe = np.asarray(d["perr"], float)
                js = lambda a: " ".join(f"{x:.6g}" for x in a)
                lines.append(",".join([
                    str(ch), str(wp), f"{sc.wp_to_vbias(wp):g}", AP_SOURCE, label(n_real, nzer),
                    str(n_real), str(nzer), str(BESSEL_ORDER), str(FCUT), str(int(d["n_pulses"])),
                    f"{float(d['rms']):.6e}", f"{float(d['chi']):.4f}", f"{float(d['cond']):.3e}",
                    f'"{"; ".join(d["rail"].tolist())}"',
                    f"{th[0]:.8f}", f'"{js(th[1:1+nzer])}"', f'"{js(th[1+nzer:])}"', f'"{js(pe)}"']))
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"CSV -> {out}")


# ═════════════════════════════════════════════════════════════════════════════
# ORCHESTRATORE: un job per (WP, modello)
# ═════════════════════════════════════════════════════════════════════════════
def make_job_lines(ch, wp, n_real, nzer):
    return ([f"cd {BASE_DIR}"] + list(sc.ENV_SETUP_LINES)
            + [f"{sys.executable} {sc.SCRIPT_PATH} --worker --channel {ch} --wp {wp} "
               f"--nreal {n_real} --cc 0 --nzer {nzer}"])


def run_orchestrator():
    for d in (OUTDIR, FITS_DIR, LOG_DIR, JOBS_DIR):
        os.makedirs(d, exist_ok=True)
    tasks = []
    for ch in CHANNELS:
        wps = list_wps(ch)
        if not wps:
            print(f"[WARN] canale {ch}: nessun WP disponibile con AP_SOURCE='{AP_SOURCE}' "
                  f"(mancano i {PULSE_PATTERN.format(ch=ch, wp='*')} in {PULSE_DIR}?) - SALTATO")
            continue
        tasks += [(ch, wp, nr, nz) for wp in wps for (nr, nz) in MODELS]
    if not tasks:
        sys.exit(f"[ERROR] nessun AP trovato (AP_SOURCE='{AP_SOURCE}', canali {CHANNELS})")
    print(f"Canali {CHANNELS} x WP x {len(MODELS)} modelli = {len(tasks)} fit "
          f"(AP da '{AP_SOURCE}').")

    if RESET:
        for old_npz in glob.glob(os.path.join(FITS_DIR, "*.npz")):
            os.remove(old_npz)
        print(f"Cartella fit azzerata: {FITS_DIR}")

    if sc.SUBMIT_MODE == "local":
        print("[INFO] SUBMIT_MODE='local': eseguo i fit in sequenza (no qsub).\n")
        for ch, wp, nr, nz in tasks:
            run_worker(ch, wp, nr, nz)
        run_plot()
        run_csv()
        run_psd()
        return

    submitted, failed = 0, []
    for ch, wp, nr, nz in tasks:
        sc.wait_for_slot()
        key = f"{ch}_{wp}_{nr}z{nz}"
        jobid = None
        for _ in range(3):
            jobid = sc.submit_task(key, sc.create_sh(make_job_lines(ch, wp, nr, nz)))
            if jobid is not None:
                break
            time.sleep(sc.SLEEP_INTERVAL)
        submitted += jobid is not None
        if jobid is None:
            failed.append(key)

    print("\n" + "=" * 65)
    print(f"  {submitted}/{len(tasks)} job sottomessi.")
    if failed:
        print(f"  {len(failed)} NON sottomessi: {failed}")
    print(f"  Ogni job salva il suo .npz in: {FITS_DIR}")
    print(f"  Log dei job in: {LOG_DIR}")
    print(f"  A job finiti:  python {os.path.basename(__file__)} --plot  (canve + bestfit .npy)")
    print(f"                 poi  --csv  e  --psd")
    print("=" * 65 + "\n")


def main():
    p = argparse.ArgumentParser(description="Fit AP max-aligned, 4 modelli pole-zero + Bessel (m205).")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--channel", type=int)
    p.add_argument("--wp", type=int)
    p.add_argument("--nreal", type=int)
    p.add_argument("--nzer", type=int)
    p.add_argument("--cc", type=int, choices=(0, 1), default=0,
                   help="ignorato: niente coppie complesse (accettato per compatibilita' dei job)")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--csv", action="store_true")
    p.add_argument("--psd", action="store_true",
                   help="power spectrum dei fit migliori sovrapposto a quello del medianAP (ROOT)")
    args = p.parse_args()

    if args.worker:
        if None in (args.channel, args.wp, args.nreal, args.nzer):
            sys.exit("[ERROR] --worker richiede --channel --wp --nreal --nzer")
        run_worker(args.channel, args.wp, args.nreal, args.nzer)
    elif args.plot:
        run_plot()
    elif args.csv:
        run_csv()
    elif args.psd:
        run_psd()
    else:
        run_orchestrator()


if __name__ == "__main__":
    main()
