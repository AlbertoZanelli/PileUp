"""
analyze_BI_m205.py
==================
Stima del BI per la misura m205 (load curves), parallelizzata sul cluster:
viene mandato un job indipendente per OGNI coppia (canale, WP).

Flusso:
  1. Orchestratore (default): leggero, da eseguire anche sul login node.
     Enumera le coppie (canale, WP) con ampiezza disponibile, sottomette un
     job qsub per ciascuna e TERMINA subito (non aspetta i job).
  2. Worker (--worker --channel C --wp W): eseguito dai job sui nodi. Calcola
     il BI di UNA coppia e APPENDE la riga dei risultati al CSV condiviso
     (lock esclusivo per la concorrenza).
  3. Il plot si fa a parte, con plot_BI_results.py, leggendo il CSV.

Esempi:
    python analyze_BI_m205.py                                  # sottomette i job e chiude
    python analyze_BI_m205.py --worker --channel 71 --wp 21    # eseguito dai job

Le ampiezze del segnale vengono lette dal CSV prodotto da plot_all_root.py
(LOAD_CURVE), per ogni coppia (canale, V_bias).
Due modalita', dalla config in testa:
  - TEMPLATE_SOURCE = "root" -> medianAP di Octopus (comportamento originale);
  - TEMPLATE_SOURCE = "fit"  -> impulso fittato migliore di scan_residuals_bessel_m205.py.
ONLY_CHANNELS limita l'analisi a un sottoinsieme di canali. Cartelle e CSV portano il tag della
modalita' (m205_results_octopus / m205_results_octopus_fit), quindi le due non si sovrascrivono.
"""

from __future__ import annotations

import os
import sys
import csv
import time
import glob
import re
import fcntl
import argparse
import tempfile
import subprocess
import shutil

import numpy as np   # leggero: serve per VBIAS_LIST / wp_to_vbias
import uproot        # serve all'orchestratore per elencare i WP

# NB: torch / scipy / matplotlib / src / utility NON vengono importati a livello
# di modulo: solo dentro le funzioni del worker, così l'orchestratore resta leggero.

# ═════════════════════════════════════════════════════════════════════════════
# Paths & experiment config
# ═════════════════════════════════════════════════════════════════════════════
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.abspath(__file__)

# La copia congelata da freeze_script() gira da un'altra cartella, quindi sys.path[0] non e'
# la radice del progetto e `import src.analysis` fallirebbe: si aggiunge BASE_DIR a mano.
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
DATA_DIR    = os.path.join(BASE_DIR, "Processed")

# ═════════════════════════════════════════════════════════════════════════════
# MODALITA': da dove viene il TEMPLATE, e quali canali elaborare
# ═════════════════════════════════════════════════════════════════════════════
# TEMPLATE_SOURCE = "root" -> medianAP di Octopus dal file ROOT (comportamento originale).
# TEMPLATE_SOURCE = "sim" -> AP SIMULATO da simulate_BI_error_m205.py --make-ap
#   (m205_AP_sim/ch<ch>/simAP_ch<ch>_wp<wp>.npy): stessa forma dell'AP vero ma con una
#   REALIZZAZIONE DIVERSA del rumore di template. Addestrando qui e valutando su eventi generati
#   dall'AP vero si rompe l'auto-consistenza (cfr. paper, sez. 4.5).
# TEMPLATE_SOURCE = "fit"  -> impulso FITTATO migliore di scan_residuals_bessel_m205.py
#   (residual_scan_bessel/fits_<ap_source>/bestfit_ch<ch>_wp<wp>.npy), cioe' un template
#   DENOISED: poche decine di parametri su 10000 punti mediano via il rumore finito-N.
# In entrambi i casi la NPS viene dal ROOT: cambia solo il template. Cartella di output, CSV e
# prefisso dei job dipendono dalla modalita', cosi' le due analisi non si sovrascrivono.
TEMPLATE_SOURCE = "sim"    # "root" | "fit" | "sim"

FIT_DIR     = os.path.join(BASE_DIR, "residual_scan_bessel", "fits_octopus")
FIT_PATTERN = "bestfit_ch{ch}_wp{wp}.npy"
SIM_DIR     = os.path.join(BASE_DIR, "m205_AP_sim")
SIM_PATTERN = os.path.join("ch{ch}", "simAP_{sim}_ch{ch}_wp{wp}.npy")
# Quale set di AP simulati usare, cioe' il tag nel nome del file
# (li produce build_simAP_injected_m205.py):
#   prefisso -> da quale TEMPLATE e' generato l'AP simulato:
#       "fit..."  dal bestfit, liscio: l'AP simulato ha solo rumore NUOVO e indipendente.
#                 E' quello che serve per rompere l'auto-consistenza.
#       "root..." dal medianAP, che e' gia' rumoroso: quel rumore e' identico in tutte le
#                 tracce, la mediana non lo riduce, e l'AP simulato viene ~1.5 volte piu'
#                 rumoroso del vero. NON e' una realizzazione indipendente: serve a misurare
#                 l'effetto del rumore di template, non ad addestrarci sopra.
#   suffisso -> da dove viene il RUMORE:
#       "...inj"  finestre VERE dal binario (NOISE_SOURCE="real"), consigliato: 1.08x il vero;
#       "...gen"  generato dalla NPS misurata (NOISE_SOURCE="clean_nps"): 1.17x, gaussiano.
SIM_SOURCE      = "APsimfit10000"      # tag dell'AP simulato: APsim<template><N>
                                     # (build_simAP_injected_m205.py, MODE="mc")
if TEMPLATE_SOURCE == "sim" and not SIM_SOURCE.startswith(("APsim", "APreal")):
    print(f"[WARN] SIM_SOURCE='{SIM_SOURCE}' e' un tag della vecchia nomenclatura "
          "(inj/gen). Si legge lo stesso, ma i tag nuovi sono APsim<template><N>.")

# ── Sorgente della NPS ──────────────────────────────────────────────────────────────
# "octopus": `averagepowerspectrum_noise_wp<wp>_medianpower` dal ROOT, come sempre. E' una
#   MEDIANA sugli eventi usata dove serve una MEDIA, dentro un array one-sided che poi viene
#   specchiato: misurato su ch91, vale 1.84 volte la potenza vera (2 / 1.084).
# "clean": NPS misurata dalle finestre di rumore vere del binario da build_NPS_clean_m205.py
#   (stessa selezione di Octopus + taglio RMS sulla finestra intera, media del periodogramma).
#   Riproduce la RMS di finestre indipendenti entro lo 0.4%. E' gia' nella convenzione giusta:
#   niente flattop, niente M^2/T. Con questa sigma scende del 18-30% e il BI cala.
NPS_SOURCE  = "clean"       # "octopus" | "clean"
NPS_DIR     = os.path.join(BASE_DIR, "m205_NPS_clean")
NPS_PATTERN = os.path.join("ch{ch}", "nps_ch{ch}_wp{wp}.npy")

# Canali da elaborare: lista, oppure None/[] per TUTTI quelli con ampiezza nel CSV.
ONLY_CHANNELS = [34]

def sim_folder_tag(tag):
    """Pezzo di nome della cartella dei risultati per un template simulato.
    I tag nuovi (APsim.../APreal...) si bastano; i vecchi (fitinj, rootgen, ...) tengono il
    prefisso "sim_" con cui furono creati, cosi' le cartelle esistenti restano quelle."""
    return tag if tag.startswith(("APsim", "APreal")) else "sim_" + tag


_TAG        = ({"root": "", "fit": "_fit",
                "sim": "_" + sim_folder_tag(SIM_SOURCE)}[TEMPLATE_SOURCE]
               + ("_npsclean" if NPS_SOURCE == "clean" else ""))
OUTPUT_DIR  = os.path.join(BASE_DIR, "m205_results_octopus" + _TAG)
LOG_DIR     = os.path.join(OUTPUT_DIR, "logs")     # stdout/stderr dei job
JOBS_DIR    = os.path.join(OUTPUT_DIR, "jobs")     # script .sh temporanei
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, f"BI_results_m205{_TAG}.csv")
# Cartella coi filtri di banda ADDESTRATI f1, f2 e il KERNEL applicato (qui il
# filtro ottimo H_unit), salvati come .npy (uno per coppia canale/WP, nomi distinti
# -> nessuna concorrenza). Il filtro totale applicato ai dati e' g_i = f_i * H_unit.
FILTERS_DIR = os.path.join(OUTPUT_DIR, "trained_filters")

MEAS_NAME = "000205"

# ── CSV delle ampiezze scritto da plot_all_root.py (LOAD_CURVE)
#    Atteso con colonne: channel, vbias_V, risetime_ms, amplitude_mV
# Il CSV delle ampiezze sta nella cartella base (come per analyse_BI_m205_wiener*), ma qui
# storicamente veniva cercato in Processed/: si accettano entrambe le posizioni.
AMP_CSV = next((p for p in (os.path.join(BASE_DIR, "amplitudes_m205.csv"),
                            os.path.join(DATA_DIR, "amplitudes_m205.csv")) if os.path.exists(p)),
               os.path.join(BASE_DIR, "amplitudes_m205.csv"))

# ═════════════════════════════════════════════════════════════════════════════
# Cluster / scheduler config  (ADATTA al tuo cluster)
# ═════════════════════════════════════════════════════════════════════════════
SUBMIT_MODE       = "qsub"   # "qsub" = un job per nodo ; "local" = esegui in sequenza (SOLO debug, pesante!)
QUEUE             = "cupid"
WALLTIME          = "24:00:00"
RAM_GB            = 4         # GB per job
MAX_PARALLEL_JOBS = 135
SLEEP_INTERVAL    = 20        # s tra un controllo di slot e l'altro
JOB_NAME_PREFIX   = "BI" + {"root": "", "fit": "F", "sim": "S"}[TEMPLATE_SOURCE]   # nome job / qstat
EXPORT_ENV        = True      # aggiunge "-V" al qsub: esporta l'ambiente corrente al job
RESET_CSV         = True      # se True l'orchestratore riparte da un CSV pulito (solo header)

# Righe di setup ambiente eseguite all'inizio di OGNI job (conda / venv / module ...).
# RIEMPILE in base al tuo ambiente: se i moduli (torch, uproot, src/...) non sono nel
# PATH del nodo, il worker fallirà. Con EXPORT_ENV=True spesso non serve, ma dipende dal cluster.
ENV_SETUP_LINES = [
    "source /home/zanelli/LoadOctopus.sh"
]

# ── V_bias look-up table (indexed by WP // 2)
VBIAS_LIST = np.array([0.6, 1.0, 1.4, 1.8, 2, 3, 4, 5, 6, 8, 10, 20, 26, 30, 40])

def wp_to_vbias(wp_idx: int) -> float:
    return VBIAS_LIST[wp_idx // 2]


_FLATTOP_FACTOR = {}
def flattop_power_factor(n: int) -> float:
    """Fattore di correzione di potenza della finestra FLATTOP usata per la NPS:
    N / sum(w^2) = 1/mean(w^2). Compensa l'attenuazione di potenza che la finestra
    introduce sul rumore. Per N=10000 vale ~5.7077, cioe' praticamente il vecchio
    valore hardcoded 5.708 (differenza ~0.004%). scipy importato pigramente (solo
    nel worker), cosi' l'orchestratore resta leggero; il risultato e' in cache."""
    if n not in _FLATTOP_FACTOR:
        from scipy.signal.windows import flattop
        w = flattop(n)
        _FLATTOP_FACTOR[n] = float(n / np.sum(w ** 2))
    return _FLATTOP_FACTOR[n]


def load_signal_amplitudes(csv_path: str) -> dict:
    """Legge la mappa (canale, V_bias) -> ampiezza dal CSV di plot_all_root.py.
    Il CSV salva l'ampiezza in mV: qui viene riconvertita in V.
    Le righe con cella 'amplitude_mV' vuota vengono saltate.
    Le V_bias sono arrotondate a 3 decimali per matchare wp_to_vbias().
    """
    amps = {}
    with open(csv_path, newline="") as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            amp_str = (row.get("amplitude_mV") or "").strip()
            if not amp_str:
                continue
            ch = int(row["channel"])
            vb = round(float(row["vbias_V"]), 3)
            amps[(ch, vb)] = float(amp_str) * 1e-3   # mV -> V
    return amps


def template_path(channel, wp) -> str:
    """Percorso del template esterno (TEMPLATE_SOURCE "fit" oppure "sim")."""
    pat, base = ((FIT_PATTERN, FIT_DIR) if TEMPLATE_SOURCE == "fit" else (SIM_PATTERN, SIM_DIR))
    return os.path.join(base, pat.format(ch=channel, wp=wp, sim=SIM_SOURCE))


def nps_path(channel, wp):
    """Percorso della NPS misurata (NPS_SOURCE = "clean")."""
    return os.path.join(NPS_DIR, NPS_PATTERN.format(ch=channel, wp=wp))


def load_nps(f, channel, wp):
    """NPS nella convenzione del codice: E|FFT(x)|^2 sull'intero spettro (10000 bin)."""
    if NPS_SOURCE == "clean":
        path = nps_path(channel, wp)
        if not os.path.exists(path):
            raise RuntimeError(f"NPS 'clean' non trovata: {path}")
        return np.asarray(np.load(path), dtype=float)   # gia' corretta per la finestra
    nps = np.asarray(f[f"averagepowerspectrum_noise_wp{wp}_medianpower"].values(), dtype=float)
    nps = np.concatenate([nps, nps[-2:0:-1]])
    return nps * flattop_power_factor(WINDOW_SIZE) * WINDOW_SIZE ** 2 / SAMPLING_TIME


def root_file_for_channel(channel) -> str | None:
    """Trova il file ROOT della misura corrispondente a un canale."""
    for f in glob.glob(os.path.join(DATA_DIR, f"Processed_*_{MEAS_NAME}_*.root")):
        if os.path.basename(f).split("_")[-1].replace(".root", "") == str(channel):
            return f
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Analysis parameters
# ═════════════════════════════════════════════════════════════════════════════
ACCEPTANCE = 0.9
WINDOW_SIZE = 10_000
SAMPLING_RATE = 10_000
SAMPLING_TIME = WINDOW_SIZE / SAMPLING_RATE
N_TRIALS = 300

T_MIN, T_MAX, N_T = 0, 8e-4, 100
R_MIN, R_MAX, N_R = 0.0, 0.5, 100

# Colonne del CSV dei risultati
#   beta_Hz = banda RMS pesata sul rumore del template (Hz, senza 2*pi)
#   rho_t   = SNR * beta  = figura di merito temporale per il pile-up (Hz)
CSV_FIELDNAMES = ["channel", "wp", "vbias", "signal_amp", "sigma_analytic", "SNR",
                  "beta_Hz", "rho_t", "template", "BI", "J_final"]


# ═════════════════════════════════════════════════════════════════════════════
# Scrittura CSV concorrenza-safe (ogni job appende una riga)
# ═════════════════════════════════════════════════════════════════════════════
def init_csv(path: str):
    """Crea il CSV (sovrascrivendolo) con la sola riga di header."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDNAMES).writeheader()


def append_row_to_csv(path: str, row: dict):
    """Appende una riga al CSV in modo sicuro tra processi concorrenti (flock)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)          # lock esclusivo
        try:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            if f.tell() == 0:                  # file vuoto -> scrivi prima l'header
                writer.writeheader()
            writer.writerow({k: row.get(k) for k in CSV_FIELDNAMES})
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _independent_half(vec):
    """Meta' indipendente di uno spettro hermitiano lungo N: i primi N//2+1 bin
    (DC..Nyquist). Il resto e' il mirror coniugato."""
    vec = np.asarray(vec).ravel()
    return vec[: len(vec) // 2 + 1]


def save_filters_npy(dirpath: str, channel, wp, f1, f2, kernel):
    """Salva come .npy i filtri di banda addestrati f1, f2 e il KERNEL applicato
    (qui il filtro ottimo H_unit). Nomi distinti per (canale, WP) -> nessun lock.
    Si salva solo la meta' indipendente dello spettro (N//2+1 bin); il vettore
    completo si ricostruisce con full = np.concatenate([half, half[-2:0:-1]]) per f1/f2 (REALI);
    per il KERNEL, che e' COMPLESSO, la meta' speculare va CONIUGATA:
    full = np.concatenate([half, np.conj(half[-2:0:-1])]).
    Il filtro TOTALE applicato ai dati e' g_i = f_i * kernel."""
    os.makedirs(dirpath, exist_ok=True)
    np.save(os.path.join(dirpath, f"f1_ch{channel}_wp{wp}.npy"), _independent_half(f1))
    np.save(os.path.join(dirpath, f"f2_ch{channel}_wp{wp}.npy"), _independent_half(f2))
    np.save(os.path.join(dirpath, f"kernel_ch{channel}_wp{wp}.npy"), _independent_half(kernel))


# ═════════════════════════════════════════════════════════════════════════════
# Quantità condivise + core BI estimator  (import pesanti SOLO qui dentro)
# ═════════════════════════════════════════════════════════════════════════════
def build_shared(device) -> dict:
    """Quantità condivise (poco costose): ricalcolate da ogni worker."""
    import torch
    from scipy.stats import norm
    from utility.double_beta_spectrum import pdf_ratio2b

    N_sigma = float(norm.ppf(1 - (1 - ACCEPTANCE)))
    ratio_distribution = pdf_ratio2b(np.linspace(R_MIN, R_MAX, N_R))
    ratio_distribution /= np.mean(ratio_distribution)
    return {
        "N_sigma": N_sigma,
        "t_torch": torch.linspace(T_MIN, T_MAX, N_T, dtype=torch.cfloat, device=device),
        "r_torch": torch.linspace(R_MIN, R_MAX, N_R, dtype=torch.cfloat, device=device),
        "ratio_distribution_torch": torch.tensor(
            ratio_distribution, dtype=torch.cfloat, device=device
        ),
    }


def estimate_BI_for_wp(channel, wp, vbias, meanpulse, nps, signal_amp,
                       samp_rate, shared, device) -> dict:
    import torch
    import src.analysis as an
    import utility.functions as fn

    # ── Optimal filter ────────────────────────────────────────────────────────
    S, w, H_unit = an.compute_H(meanpulse, nps, np.hanning, sampling_rate=samp_rate)
    sigma_analytic = an.compute_sigma_OF(S, nps)

    # ── Figura di merito temporale (Cramer-Rao sul tempo di arrivo) ────────────
    #   beta = banda RMS pesata sul rumore del template [Hz]:
    #     beta = sigma_OF / sigma_mod ,  sigma_mod = compute_sigma_OF(f*S, nps)
  
    sigma_mod = float(an.compute_sigma_OF(w * S, nps))
    beta_Hz = (float(sigma_analytic) / sigma_mod
               if np.isfinite(sigma_mod) and sigma_mod > 0 else float("nan"))
    SNR = float(signal_amp / sigma_analytic)
    rho_t = SNR * beta_Hz                                 # SNR * beta [Hz]

    # ── Torch tensors ─────────────────────────────────────────────────────────
    def to_t(arr, dtype=torch.cfloat):
        return torch.tensor(np.asarray(arr), dtype=dtype, device=device)

    S_torch = to_t(S)
    H_unit_torch = to_t(H_unit)
    w_torch = to_t(w)
    nps_torch = to_t(nps)
    signal_amp_torch = torch.tensor(signal_amp, dtype=torch.float32, device=device)

    # ── Optimise filters ──────────────────────────────────────────────────────
    f1_opt, f2_opt, J_values = an.optimize_filters(
        S_torch, H_unit_torch, w_torch,
        shared["t_torch"], shared["r_torch"], nps_torch,
        signal_amp_torch, shared["ratio_distribution_torch"],
        N_sigma = shared["N_sigma"],
        activation_fct = torch.abs,
        f1_init = None,
        f2_init = None,
        n_trials = N_TRIALS,
        use_interp = True,
        verbose = False,
    )

    BI_estimate = float(J_values[-1]) * fn.K

    return {
        "channel": channel,
        "wp": wp,
        "vbias": vbias,
        "signal_amp": float(signal_amp),
        "sigma_analytic": float(sigma_analytic),
        "SNR": SNR,
        "beta_Hz": beta_Hz,
        "rho_t": rho_t,
        "template": TEMPLATE_SOURCE,
        "BI": float(BI_estimate),
        "J_final": float(J_values[-1]),
        # Filtri di banda e kernel (qui il filtro ottimo H_unit), salvati a parte
        # come .npy in FILTERS_DIR; non entrano nel BI CSV (append_row_to_csv tiene
        # solo CSV_FIELDNAMES). Filtro totale applicato ai dati: g_i = f_i * H_unit.
        "f1": f1_opt.detach().cpu().numpy(),
        "f2": f2_opt.detach().cpu().numpy(),
        "kernel": np.asarray(H_unit),
    }


# ═════════════════════════════════════════════════════════════════════════════
# WORKER: calcola UNA coppia (canale, WP) e appende la riga al CSV
# ═════════════════════════════════════════════════════════════════════════════
def run_worker(channel: int, wp: int):
    import torch   # import pesante: solo nel job

    vbias = float(wp_to_vbias(wp))
    try:
        # 1. Ampiezza per questo (canale, V_bias) dal CSV
        amps = load_signal_amplitudes(AMP_CSV)
        signal_amp = amps.get((int(channel), round(vbias, 3)))
        if signal_amp is None:
            raise RuntimeError(f"Nessuna ampiezza nel CSV per (ch {channel}, V_bias {vbias:.3f})")

        # 2. File ROOT del canale
        filepath = root_file_for_channel(channel)
        if filepath is None:
            raise RuntimeError(f"File ROOT non trovato per il canale {channel}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        shared = build_shared(device)

        # 3. Estrazione Meanpulse + NPS
        #    Il template viene dal fit oppure dall'AP di Octopus (TEMPLATE_SOURCE);
        #    la NPS dal ROOT o dalle finestre vere (NPS_SOURCE).
        with uproot.open(filepath) as f:
            if TEMPLATE_SOURCE in ("fit", "sim"):
                tpath = template_path(channel, wp)
                if not os.path.exists(tpath):
                    raise RuntimeError(f"template '{TEMPLATE_SOURCE}' non trovato: {tpath}")
                meanpulse = np.asarray(np.load(tpath), dtype=float)
            else:
                meanpulse = np.asarray(f[f"averagepulse_ap_wp{wp}_medianAP"].values(), dtype=float)
            meanpulse = meanpulse / meanpulse.max()        # picco = 1 in entrambi i casi

            nps = load_nps(f, channel, wp)

        # 4. Stima BI: riga nel CSV + filtri e kernel come .npy
        res = estimate_BI_for_wp(str(channel), wp, vbias, meanpulse, nps,
                                 signal_amp, SAMPLING_RATE, shared, device)
        append_row_to_csv(OUTPUT_CSV, res)
        save_filters_npy(FILTERS_DIR, channel, wp, res["f1"], res["f2"], res["kernel"])
        print(f"[OK] ch {channel} wp {wp}: BI={res['BI']:.3e}  ->  {OUTPUT_CSV}")

    except Exception as e:
        # L'errore finisce nel file di stderr del job (LOG_DIR); nessuna riga nel CSV.
        print(f"[ERROR] ch {channel} wp {wp}: {e}")
        sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# Job submission helpers
# ═════════════════════════════════════════════════════════════════════════════
def create_sh(lines: list) -> str:
    """Crea uno script di shell temporaneo eseguibile."""
    os.makedirs(JOBS_DIR, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".sh", dir=JOBS_DIR)
    tmp.write("#!/bin/bash\n")
    tmp.write("\n".join(lines) + "\n")
    tmp.close()
    os.chmod(tmp.name, 0o755)
    return tmp.name


def freeze_script():
    """Copia questo file in OUTPUT_DIR e fa puntare i job alla COPIA.

    Serve perche' il job sottomesso a PBS contiene solo `python <file> --worker ...`: il file
    viene letto QUANDO IL JOB PARTE, non quando lo si sottomette. Senza la copia, modificare
    la config per lanciare una seconda campagna cambierebbe anche i job della prima ancora in
    coda. Effetto collaterale utile: la cartella dei risultati contiene il codice esatto che
    li ha prodotti."""
    global SCRIPT_PATH
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dst = os.path.join(OUTPUT_DIR, "_" + os.path.basename(SCRIPT_PATH))
    src = open(SCRIPT_PATH).read()
    # La copia sta in un'altra cartella, quindi os.path.dirname(__file__) darebbe la cartella
    # dei risultati e TUTTI i percorsi (Processed/, fit, AP simulati, NPS) punterebbero li'.
    # Si congela BASE_DIR al valore vero.
    src, n = re.subn(r"^BASE_DIR(\s*)= os\.path\.dirname\(os\.path\.abspath\(__file__\)\)",
                     lambda m: f'BASE_DIR{m.group(1)}= {BASE_DIR!r}'
                               '   # congelato da freeze_script(): la copia sta altrove',
                     src, count=1, flags=re.M)
    if n != 1:
        raise RuntimeError("freeze_script: non trovo la riga di BASE_DIR da congelare")
    with open(dst, "w") as fh:
        fh.write(src)
    SCRIPT_PATH = dst
    print(f"[INFO] config congelata: i job useranno {os.path.relpath(dst, BASE_DIR)}")


def make_job_lines(channel: int, wp: int) -> list:
    """Corpo dello script di job: rilancia questo stesso file in modalità worker."""
    lines = [f"cd {BASE_DIR}"]
    lines += ENV_SETUP_LINES
    lines.append(f"{sys.executable} {SCRIPT_PATH} --worker --channel {channel} --wp {wp}")
    return lines


def running_job_count() -> int:
    """Numero di nostri job attualmente in coda/esecuzione (per il throttling)."""
    user = os.environ.get("USER", "")
    cmd = f"qstat -u {user} | grep '{JOB_NAME_PREFIX}' | wc -l"
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return int(r.stdout.strip() or 0)
    except Exception as e:
        print("[WARN] qstat failed:", e)
        return 0


def wait_for_slot():
    while running_job_count() >= MAX_PARALLEL_JOBS:
        print(f"Max parallel jobs raggiunto ({MAX_PARALLEL_JOBS}). Attendo {SLEEP_INTERVAL}s...")
        time.sleep(SLEEP_INTERVAL)


def submit_task(task_key: str, sh_file: str) -> str | None:
    """Sottomette un job per la coppia descritta da task_key. Ritorna il job id o None."""
    job_name = f"{JOB_NAME_PREFIX}{task_key}"[:15]   # PBS limita la lunghezza del nome
    export = "-V " if EXPORT_ENV else ""
    cmd = (
        f"qsub -N {job_name} {export}-q {QUEUE} "
        f"-o localhost:{LOG_DIR}/ -e localhost:{LOG_DIR}/ "
        f"-l walltime={WALLTIME} -l mem={RAM_GB}G {sh_file}"
    )
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    except Exception as e:
        print(f"[ERROR] Eccezione su qsub per {task_key}: {e}")
        return None

    if r.returncode != 0:
        print(f"[ERROR] qsub fallito per {task_key}. stderr:\n{r.stderr}")
        return None

    full_jobid = r.stdout.strip()
    jobid = full_jobid.split(".")[0] if full_jobid else ""
    if not jobid.isdigit():
        print(f"[ERROR] job id non valido per {task_key}: '{full_jobid}'")
        return None

    print(f"[OK] Sottomesso {task_key} (job {jobid})")
    return jobid


# ═════════════════════════════════════════════════════════════════════════════
# ORCHESTRATORE: enumera le coppie (canale, WP), sottomette i job e CHIUDE
# ═════════════════════════════════════════════════════════════════════════════
def run_orchestrator():
    for d in (OUTPUT_DIR, LOG_DIR, JOBS_DIR):
        os.makedirs(d, exist_ok=True)

    if not os.path.exists(AMP_CSV):
        sys.exit(f"[ERROR] CSV delle ampiezze non trovato: {AMP_CSV}")
    amps = load_signal_amplitudes(AMP_CSV)
    if not amps:
        sys.exit(f"[ERROR] Nessuna ampiezza valida in {AMP_CSV}.")
    channels_with_amp = {ch for (ch, _vb) in amps}

    root_files = sorted(glob.glob(os.path.join(DATA_DIR, f"Processed_*_{MEAS_NAME}_*.root")))
    if not root_files:
        sys.exit(f"[ERROR] Nessun file ROOT trovato in {DATA_DIR} per {MEAS_NAME}.")

    # ── Costruzione della lista dei task (canale, WP) con ampiezza disponibile ─
    tasks = []
    for filepath in root_files:
        ch = int(os.path.basename(filepath).split("_")[-1].replace(".root", ""))
        if ch not in channels_with_amp:
            continue
        with uproot.open(filepath) as f:
            wp_indices = sorted(set(
                int(m.group(1)) for k in f.keys()
                for m in [re.search(r'averagepulse_ap_wp(\d+)_medianAP', k)]
                if m and (int(m.group(1)) % 2 != 0)
            ))
        for wp in wp_indices:
            if (ch, round(float(wp_to_vbias(wp)), 3)) in amps:
                tasks.append((ch, wp))

    if ONLY_CHANNELS:
        tasks = [(ch, wp) for (ch, wp) in tasks if ch in ONLY_CHANNELS]
        print(f"[INFO] ONLY_CHANNELS={ONLY_CHANNELS}: {len(tasks)} coppie (canale, WP) selezionate")

    # In modalita' "fit" servono i bestfit_ch<ch>_wp<wp>.npy dello scan: le coppie che non li
    # hanno vengono SALTATE invece di generare job destinati a fallire (lo scan puo' aver girato
    # solo su alcuni canali).
    if TEMPLATE_SOURCE in ("fit", "sim"):
        have = [(ch, wp) for (ch, wp) in tasks if os.path.exists(template_path(ch, wp))]
        if len(have) < len(tasks):
            missing = sorted({ch for (ch, wp) in tasks} - {ch for (ch, wp) in have})
            print(f"[INFO] TEMPLATE_SOURCE='{TEMPLATE_SOURCE}': saltate {len(tasks)-len(have)} coppie "
                  f"senza template (canali incompleti: {missing})")
        tasks = have

    print(f"Task totali (canale, WP) da elaborare: {len(tasks)}")
    if not tasks:
        sys.exit("[ERROR] Nessuna coppia (canale, WP) con ampiezza disponibile.")

    # I job devono usare una COPIA di questo file, non l'originale: vedi freeze_script().
    freeze_script()

    # ── CSV dei risultati: parte pulito (solo header) ──────────────────────────
    if RESET_CSV:
        init_csv(OUTPUT_CSV)
        # Azzera la cartella dei filtri (rimuove i .npy di run precedenti).
        os.makedirs(FILTERS_DIR, exist_ok=True)
        for old in glob.glob(os.path.join(FILTERS_DIR, "*.npy")):
            os.remove(old)
        print(f"CSV inizializzato (solo header): {OUTPUT_CSV}")
        print(f"Cartella filtri azzerata: {FILTERS_DIR}")

    # ── Modalità debug locale: esegue tutto in sequenza (PESANTE, no qsub) ─────
    if SUBMIT_MODE == "local":
        print("[INFO] SUBMIT_MODE='local': eseguo i task in sequenza (no qsub).\n")
        for ch, wp in tasks:
            run_worker(ch, wp)
        print(f"\nFatto. Risultati in {OUTPUT_CSV}")
        return

    # ── Sottomissione: un job per task, con throttling ─────────────────────────
    submitted, failed = 0, []
    for ch, wp in tasks:
        wait_for_slot()
        task_key = f"{ch}_{wp}"
        jobid = None
        for attempt in range(3):          # qualche retry per errori transitori di qsub
            sh_file = create_sh(make_job_lines(ch, wp))
            jobid = submit_task(task_key, sh_file)
            if jobid is not None:
                break
            time.sleep(SLEEP_INTERVAL)
        if jobid is not None:
            submitted += 1
        else:
            failed.append(task_key)
            print(f"[WARN] impossibile sottomettere {task_key} dopo 3 tentativi.")

    print("\n" + "=" * 65)
    print(f"  {submitted}/{len(tasks)} job sottomessi.")
    if failed:
        print(f"  {len(failed)} NON sottomessi: {failed}")
    print(f"  Ogni job scriverà la sua riga in: {OUTPUT_CSV}")
    print(f"  Log dei job in: {LOG_DIR}")
    print(f"  A job finiti, plottare con:  python plot_BI_results.py")
    print("=" * 65 + "\n")
    # L'orchestratore termina qui: i job girano in autonomia.


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Stima del BI (m205) parallelizzata su cluster.")
    parser.add_argument("--worker", action="store_true",
                        help="modalità worker: calcola una singola coppia (canale, WP)")
    parser.add_argument("--channel", type=int, help="canale (richiesto con --worker)")
    parser.add_argument("--wp", type=int, help="working point (richiesto con --worker)")
    args = parser.parse_args()

    if args.worker:
        if args.channel is None or args.wp is None:
            sys.exit("[ERROR] --worker richiede --channel e --wp")
        run_worker(args.channel, args.wp)
    else:
        run_orchestrator()


if __name__ == "__main__":
    main()