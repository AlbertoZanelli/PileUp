"""
analyse_BI_m205_wiener.py
=========================
Versione PARALLELA di analyse_BI_m205.py che, al posto del filtro ottimo, usa il
filtro di WIENER con fattore di modulazione del rumore lambda ADDESTRABILE
(optimize_filters_wiener_lambda in src/analysis.py). Il kernel di Wiener e'

    W_lambda = S* / ( |S|^2 + lambda * NPS )

e lambda (>0, parametrizzato come exp(log_lambda)) viene ottimizzato insieme ai
filtri di banda f1, f2 minimizzando la stessa metrica J del BI. lambda=1 riproduce
il Wiener standard (CUORE norm_type=0); lambda->0 tende alla deconvoluzione pura,
lambda->inf al filtro ottimo. Il lambda ottimizzato viene salvato nel CSV.

Tutto il resto (orchestratore/worker su cluster, CSV concorrenza-safe, beta/rho_t)
e' identico ad analyse_BI_m205.py; cambiano solo la stima del BI, le cartelle di
output (m205_results_wiener) e il prefisso dei job (BIW).

Oltre al CSV dei risultati (BI_results_m205_wiener.csv), ogni worker salva anche i
filtri di banda ADDESTRATI f1, f2 come vettori .npy nella cartella
m205_results_wiener/trained_filters/ (file f1_ch{ch}_wp{wp}.npy e
f2_ch{ch}_wp{wp}.npy). Si salva solo la META' INDIPENDENTE dello spettro (i primi
N//2+1 bin, da DC a Nyquist); il filtro completo si ricostruisce con
    full = np.concatenate([half, half[-2:0:-1]]).

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
    python analyse_BI_m205_wiener.py                                # sottomette i job e chiude
    python analyse_BI_m205_wiener.py --worker --channel 71 --wp 21  # eseguito dai job

Le ampiezze del segnale vengono lette dal CSV prodotto da plot_all_root.py
(LOAD_CURVE), per ogni coppia (canale, V_bias).
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

import numpy as np   # leggero: serve per VBIAS_LIST / wp_to_vbias
import uproot        # serve all'orchestratore per elencare i WP

# NB: torch / scipy / matplotlib / src / utility NON vengono importati a livello
# di modulo: solo dentro le funzioni del worker, così l'orchestratore resta leggero.

# ═════════════════════════════════════════════════════════════════════════════
# Paths & experiment config
# ═════════════════════════════════════════════════════════════════════════════
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.abspath(__file__)
DATA_DIR    = os.path.join(BASE_DIR, "Processed")
OUTPUT_DIR  = os.path.join(BASE_DIR, "m205_results_wiener")
LOG_DIR     = os.path.join(OUTPUT_DIR, "logs")     # stdout/stderr dei job
JOBS_DIR    = os.path.join(OUTPUT_DIR, "jobs")     # script .sh temporanei
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "BI_results_m205_wiener.csv")
# Cartella coi filtri di banda ADDESTRATI f1, f2, salvati come vettori .npy (uno
# per coppia canale/WP, nomi distinti -> nessun problema di concorrenza). Si salva
# solo la META' INDIPENDENTE dello spettro (i primi N//2+1 bin, 0..Nyquist); il
# resto e' il mirror hermitiano e si ricostruisce con np.concatenate([h, h[-2:0:-1]]).
FILTERS_DIR = os.path.join(OUTPUT_DIR, "trained_filters")

MEAS_NAME = "000205"

# ── CSV delle ampiezze scritto da plot_all_root.py (LOAD_CURVE)
#    Atteso con colonne: channel, vbias_V, risetime_ms, amplitude_mV
AMP_CSV = os.path.join(BASE_DIR, "amplitudes_m205.csv")

# ═════════════════════════════════════════════════════════════════════════════
# Cluster / scheduler config  (ADATTA al tuo cluster)
# ═════════════════════════════════════════════════════════════════════════════
SUBMIT_MODE       = "qsub"   # "qsub" = un job per nodo ; "local" = esegui in sequenza (SOLO debug, pesante!)
QUEUE             = "cupid"
WALLTIME          = "24:00:00"
RAM_GB            = 4         # GB per job
MAX_PARALLEL_JOBS = 135
SLEEP_INTERVAL    = 20        # s tra un controllo di slot e l'altro
JOB_NAME_PREFIX   = "BIW"     # usato per nominare i job e per il throttling via qstat
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
N_TRIALS = 800

T_MIN, T_MAX, N_T = 0, 8e-4, 100
R_MIN, R_MAX, N_R = 0.0, 0.5, 100

# Colonne del CSV dei risultati
#   beta_Hz       = banda RMS pesata sul rumore del template (Hz, senza 2*pi)
#   rho_t         = SNR * beta  = figura di merito temporale per il pile-up (Hz)
#   lambda_wiener = fattore di modulazione del rumore del Wiener, ottimizzato
CSV_FIELDNAMES = ["channel", "wp", "vbias", "signal_amp", "sigma_analytic", "SNR",
                  "beta_Hz", "rho_t", "lambda_wiener", "BI", "J_final"]


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
    """Restituisce la meta' indipendente di uno spettro hermitiano lungo N:
    i primi N//2+1 bin (da DC a Nyquist). Il resto e' il mirror coniugato."""
    vec = np.asarray(vec).ravel()
    return vec[: len(vec) // 2 + 1]


def save_filters_npy(dirpath: str, channel, wp, f1, f2):
    """Salva i filtri di banda addestrati f1, f2 come .npy nella cartella dei filtri.
    Ogni coppia (canale, WP) scrive due file con nomi distinti, quindi non serve
    alcun lock: i job non si pestano i piedi. Si salva solo la meta' indipendente
    dello spettro (N//2+1 bin); il filtro completo si ricostruisce con
        full = np.concatenate([half, half[-2:0:-1]]).
    """
    os.makedirs(dirpath, exist_ok=True)
    np.save(os.path.join(dirpath, f"f1_ch{channel}_wp{wp}.npy"), _independent_half(f1))
    np.save(os.path.join(dirpath, f"f2_ch{channel}_wp{wp}.npy"), _independent_half(f2))


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

    # ── Trasformata del template + riferimenti del filtro ottimo ───────────────
    #   S = FFT del pulse finestrato (Hanning); w = 2*pi*f. sigma_analytic, SNR e
    #   beta sono quantita' di RIFERIMENTO del filtro ottimo (proprieta' del
    #   template, indipendenti dal filtro di pile-up). Il kernel di Wiener NON
    #   viene precalcolato qui: optimize_filters_wiener_lambda lo ricostruisce
    #   internamente (compute_W_torch) in funzione del lambda addestrabile.
    S, w, _ = an.compute_H(meanpulse, nps, np.hanning, sampling_rate=samp_rate)
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
    w_torch = to_t(w)
    nps_torch = to_t(nps)
    signal_amp_torch = torch.tensor(signal_amp, dtype=torch.float32, device=device)

    # ── Optimise Wiener band-filters + trainable lambda ────────────────────────
    #   Il kernel W = S* / (|S|^2 + lambda*NPS) e' ricostruito ad ogni step in
    #   funzione di lambda; f1, f2 e lambda sono ottimizzati insieme minimizzando J.
    f1_opt, f2_opt, lam_opt, W_unit, J_values, lambda_values = \
        an.optimize_filters_wiener_lambda(
            S_torch, w_torch,
            shared["t_torch"], shared["r_torch"], nps_torch,
            signal_amp_torch, shared["ratio_distribution_torch"],
            N_sigma = shared["N_sigma"],
            activation_fct = torch.abs,
            f1_init = None,
            f2_init = None,
            lambda_init = 1.0,
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
        "lambda_wiener": float(lam_opt),
        "BI": float(BI_estimate),
        "J_final": float(J_values[-1]),
        # Filtri di banda addestrati (vettori), salvati a parte come .npy in
        # FILTERS_DIR; non entrano nel BI CSV perche' append_row_to_csv tiene solo
        # CSV_FIELDNAMES.
        "f1": f1_opt.detach().cpu().numpy(),
        "f2": f2_opt.detach().cpu().numpy(),
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
        with uproot.open(filepath) as f:
            hist_pulse = f[f"averagepulse_ap_wp{wp}_medianAP"]
            meanpulse = np.asarray(hist_pulse.values(), dtype=float)

            hist_nps = f[f"averagepowerspectrum_noise_wp{wp}_medianpower"]
            nps = np.asarray(hist_nps.values(), dtype=float)
            nps = np.concatenate([nps, nps[-2:0:-1]])
            
            nps *= 5.708
            nps *= WINDOW_SIZE**2
            nps *= (1 / SAMPLING_TIME)

        # 4. Stima BI: riga nel CSV + filtri addestrati come .npy
        res = estimate_BI_for_wp(str(channel), wp, vbias, meanpulse, nps,
                                 signal_amp, SAMPLING_RATE, shared, device)
        append_row_to_csv(OUTPUT_CSV, res)
        save_filters_npy(FILTERS_DIR, channel, wp, res["f1"], res["f2"])
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

    print(f"Task totali (canale, WP) da elaborare: {len(tasks)}")
    if not tasks:
        sys.exit("[ERROR] Nessuna coppia (canale, WP) con ampiezza disponibile.")

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