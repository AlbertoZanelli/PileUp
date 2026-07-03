"""
analyze_BI_singlerun.py
=======================
Stima del BI per una run SENZA curve di carico (un set di dati per canale),
parallelizzata sul cluster: un job indipendente per OGNI canale.

Supporta due modalità di lettura dei dati (variabile MODE):
  - "octopus"   : legge i file ROOT (uproot), istogrammi AP / NPS.
  - "argonauts" : legge i file .bin (np.fromfile), edmean / spec.
L'UNICA differenza tra le due modalità è come si ottengono e si processano
meanpulse (AP) e NPS: tutto isolato in load_ap_nps(). Job, qsub, CSV e stima
del BI sono identici.

Flusso:
  1. Orchestratore (default): leggerissimo, da eseguire anche sul login node.
     Enumera i canali con dati + ampiezza, sottomette un job qsub per ciascuno
     e TERMINA subito.
  2. Worker (--worker --channel C): eseguito dai job sui nodi. Carica AP/NPS,
     cerca l'ampiezza nel dizionario, calcola il BI e APPENDE la riga al CSV.
  3. Il plot si fa a parte, leggendo il CSV.

Esempi:
    python analyze_BI_singlerun.py                         # sottomette i job e chiude
    python analyze_BI_singlerun.py --worker --channel 71   # eseguito dai job
"""

from __future__ import annotations

import os
import re
import sys
import csv
import time
import glob
import fcntl
import argparse
import tempfile
import subprocess

import numpy as np   # leggero: usato dalle funzioni del worker

# NB: torch / scipy / matplotlib / uproot / src / utility sono importati SOLO
# dentro le funzioni del worker, cosi' l'orchestratore (login node) resta leggero.

# ═════════════════════════════════════════════════════════════════════════════
# MODE: sorgente dei dati
# ═════════════════════════════════════════════════════════════════════════════
MODE = "argonauts"   # "octopus" (legge ROOT) | "argonauts" (legge .bin)

# ═════════════════════════════════════════════════════════════════════════════
# Paths & experiment config
# ═════════════════════════════════════════════════════════════════════════════
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.abspath(__file__)

RUN_LABEL  = "m204"     # <-- etichetta usata nei file di output (cambiala per run)

# Output dipendente dalla MODE: cosi' octopus e argonauts non si sovrascrivono.
OUTPUT_DIR  = os.path.join(BASE_DIR, f"{RUN_LABEL}_results_{MODE}")
LOG_DIR     = os.path.join(OUTPUT_DIR, "logs")     # stdout/stderr dei job
JOBS_DIR    = os.path.join(OUTPUT_DIR, "jobs")     # script .sh temporanei
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, f"BI_results_{RUN_LABEL}_{MODE}.csv")

# ── OCTOPUS (ROOT) ────────────────────────────────────────────────────────────
OCTO_DATA_DIR  = os.path.join(BASE_DIR, "Processed")
OCTO_AP_DIR    = os.path.join(OCTO_DATA_DIR, "m204_AP")     # sottocartella con gli AP
OCTO_NPS_DIR   = os.path.join(OCTO_DATA_DIR, "m204_ANPS")   # sottocartella con gli ANPS
OCTO_MEAS_NAME = "000204"          # <-- numero della run (come nei nomi dei .root)
# Nomi esatti degli istogrammi (run senza load curve, niente wp)
HIST_PULSE_NAME = "averagepulse_ap_medianAP"
HIST_NPS_NAME   = "averagepowerspectrum_anps_medianpower"

# ── ARGONAUTS (.bin) ──────────────────────────────────────────────────────────
ARGO_DATA_DIR  = os.path.join(BASE_DIR, "NuoveAnalisiArgonauts_m204")
ARGO_MEAS_NAME = "000204"            # <-- prefisso della run (come nei nomi dei .bin)
ARGO_BIN_DTYPE = np.float64          # dtype di np.fromfile (default = float64)

# ── Ampiezza del segnale, DIZIONARIO {canale: ampiezza in mV} ─────────────────
SIGNAL_AMP_MV_BY_MODE = {
    # m204 argonauts
    "argonauts": {
        31: 7.526,
        34: 6.257,
        37: 9.223,
        40: 4.432,
        41: 5.564,
        71: 11.782,
        83: 9.799,
        91: 5.267,
        94: 3.223, 
    },
    # m204 octopus
    "octopus": {
        31: 7.66,
        34: 8.35,
        37: 8.47,
        40: 4.41,
        41: 2.09,
        71: 14.15,
        83: 10.56,
        91: 5.11,
        94: 4.29,
    },
}
SIGNAL_AMP_MV = SIGNAL_AMP_MV_BY_MODE[MODE]
AMP_UNIT_TO_V = 1e-3   # mV -> V. Metti 1.0 se SIGNAL_AMP_MV e' gia' in V.

# ═════════════════════════════════════════════════════════════════════════════
# Cluster / scheduler config  (ADATTA al tuo cluster)
# ═════════════════════════════════════════════════════════════════════════════
SUBMIT_MODE       = "qsub"   # "qsub" = un job per nodo ; "local" = esegui in sequenza
QUEUE             = "cupid"
WALLTIME          = "24:00:00"
RAM_GB            = 4         # GB per job
MAX_PARALLEL_JOBS = 135
SLEEP_INTERVAL    = 20        # s tra un controllo di slot e l'altro
JOB_NAME_PREFIX   = "BI"      # usato per nominare i job e per il throttling via qstat
EXPORT_ENV        = True      # aggiunge "-V" al qsub
RESET_CSV         = True      # riparte da un CSV pulito (solo header)

ENV_SETUP_LINES = [
    "source /home/zanelli/LoadOctopus.sh"
]


# ═════════════════════════════════════════════════════════════════════════════
# Localizzazione dei dati + enumerazione canali (dipende da MODE)
# ═════════════════════════════════════════════════════════════════════════════

def argo_bin_files_for_channel(channel) -> tuple[str | None, str | None]:
    pat = os.path.join(ARGO_DATA_DIR, f"{ARGO_MEAS_NAME}_*_{int(channel):03d}_*.bin_edmean.bin")
    matches = sorted(glob.glob(pat))
    if not matches:
        return None, None
    if len(matches) > 1:
        print(f"[WARN] piu' file edmean per ch {channel}: {matches} -> uso il primo")
    mean_file = matches[0]
    spec_file = mean_file.replace(".bin_edmean.bin", ".bin_spec.bin")
    if not os.path.exists(spec_file):
        raise RuntimeError(f"File spec mancante per ch {channel}: {spec_file}")
    return mean_file, spec_file

def octo_root_file_for_channel(channel, directory) -> str | None:
    # Cerca il pattern: Processed_..._000204_31.root oppure Processed_..._000204_31_new.root
    rx = re.compile(rf"Processed_.*_{OCTO_MEAS_NAME}_{channel}(?:_[a-zA-Z0-9_]+)?\.root$")
    for f in glob.glob(os.path.join(directory, f"Processed_*_{OCTO_MEAS_NAME}_*.root")):
        if rx.search(os.path.basename(f)):
            return f
    return None

def available_channels() -> list[int]:
    chans = []
    if MODE == "octopus":
        # Cattura il numero del canale (\d+) subito dopo la MEAS_NAME, ignorando eventuali suffissi come _new
        rx = re.compile(rf"Processed_.*_{OCTO_MEAS_NAME}_(\d+)(?:_[a-zA-Z0-9_]+)?\.root$")
        for f in glob.glob(os.path.join(OCTO_AP_DIR, f"Processed_*_{OCTO_MEAS_NAME}_*.root")):
            m = rx.search(os.path.basename(f))
            if m:
                chans.append(int(m.group(1)))
    elif MODE == "argonauts":
        # Usa un gruppo di cattura (\d{3}) per trovare qualsiasi numero di canale a 3 cifre
        rx = re.compile(rf"{ARGO_MEAS_NAME}_.*_(\d{{3}})_.*\.bin_edmean\.bin$")
        for f in glob.glob(os.path.join(ARGO_DATA_DIR, f"{ARGO_MEAS_NAME}_*_*_*.bin_edmean.bin")):
            m = rx.search(os.path.basename(f))
            if m:
                chans.append(int(m.group(1)))
    else:
        raise RuntimeError(f"MODE sconosciuta: {MODE!r}")
    
    return sorted(set(chans))

def build_amp_map() -> dict:
    """Incrocia i canali di cui abbiamo i dati con le ampiezze definite nel dizionario."""
    chans = available_channels()
    valid_map = {}
    for ch in chans:
        if ch in SIGNAL_AMP_MV:
            valid_map[ch] = float(SIGNAL_AMP_MV[ch])
        else:
            print(f"[WARN] Il canale {ch} ha dei dati disponibili, ma NON ha un'ampiezza in SIGNAL_AMP_MV. Verrà ignorato.")
    return valid_map

# ═════════════════════════════════════════════════════════════════════════════
# Caricamento AP / NPS
# ═════════════════════════════════════════════════════════════════════════════
def load_ap_nps(channel) -> tuple[np.ndarray, np.ndarray]:
    if MODE == "octopus":
        import uproot
        ap_file  = octo_root_file_for_channel(channel, OCTO_AP_DIR)
        nps_file = octo_root_file_for_channel(channel, OCTO_NPS_DIR)
        if ap_file is None:
            raise RuntimeError(f"File ROOT AP non trovato per il canale {channel}")
        if nps_file is None:
            raise RuntimeError(f"File ROOT ANPS non trovato per il canale {channel}")

        with uproot.open(ap_file) as f:
            meanpulse = np.asarray(f[HIST_PULSE_NAME].values(), dtype=float)
        with uproot.open(nps_file) as f:
            nps = np.asarray(f[HIST_NPS_NAME].values(), dtype=float)

        nps = np.concatenate([nps, nps[-2:0:-1]])
        nps *= 5.708
        nps *= WINDOW_SIZE**2
        nps *= (1 / SAMPLING_TIME)
        return meanpulse, nps

    elif MODE == "argonauts":
        mean_file, spec_file = argo_bin_files_for_channel(channel)
        if mean_file is None:
            raise RuntimeError(f"File .bin (edmean) non trovato per il canale {channel}")
        meanpulse = np.fromfile(mean_file, dtype=ARGO_BIN_DTYPE)
        nps = np.fromfile(spec_file, dtype=ARGO_BIN_DTYPE)
        nps *= 1e-12
        nps *= (8. / 3.)
        return meanpulse, nps
    else:
        raise RuntimeError(f"MODE sconosciuta: {MODE!r}")

# ═════════════════════════════════════════════════════════════════════════════
# Analysis parameters
# ═════════════════════════════════════════════════════════════════════════════
ACCEPTANCE = 0.9
WINDOW_SIZE = 800
SAMPLING_RATE = 2_000
SAMPLING_TIME = WINDOW_SIZE / SAMPLING_RATE
N_TRIALS = 300

T_MIN, T_MAX, N_T = 0, 8e-4, 100
R_MIN, R_MAX, N_R = 0.0, 0.5, 100

CSV_FIELDNAMES = ["channel", "signal_amp", "sigma_analytic", "SNR", "BI", "J_final"]

# ═════════════════════════════════════════════════════════════════════════════
# Scrittura CSV
# ═════════════════════════════════════════════════════════════════════════════
def init_csv(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDNAMES).writeheader()

def append_row_to_csv(path: str, row: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow({k: row.get(k) for k in CSV_FIELDNAMES})
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

# ═════════════════════════════════════════════════════════════════════════════
# Core BI estimator
# ═════════════════════════════════════════════════════════════════════════════
def build_shared(device) -> dict:
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

def estimate_BI_for_channel(channel, meanpulse, nps, signal_amp,
                            samp_rate, shared, device) -> dict:
    import torch
    import src.analysis as an
    import utility.functions as fn

    S, w, H_unit = an.compute_H(meanpulse, nps, np.hanning, sampling_rate=samp_rate)
    sigma_analytic = an.compute_sigma_OF(S, nps)

    def to_t(arr, dtype=torch.cfloat):
        return torch.tensor(np.asarray(arr), dtype=dtype, device=device)

    S_torch = to_t(S)
    H_unit_torch = to_t(H_unit)
    w_torch = to_t(w)
    nps_torch = to_t(nps)
    signal_amp_torch = torch.tensor(signal_amp, dtype=torch.float32, device=device)

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
        "signal_amp": float(signal_amp),
        "sigma_analytic": float(sigma_analytic),
        "SNR": float(signal_amp / sigma_analytic),
        "BI": float(BI_estimate),
        "J_final": float(J_values[-1]),
    }


# ═════════════════════════════════════════════════════════════════════════════
# WORKER: calcola UN canale (Cerca l'ampiezza per chiave dal dizionario)
# ═════════════════════════════════════════════════════════════════════════════
def run_worker(channel: int):
    import torch    # import pesante: solo nel job

    try:
        # 1. Ampiezza per questo canale (interroga direttamente il dizionario)
        amp_mv = SIGNAL_AMP_MV.get(int(channel))
        if amp_mv is None:
            raise RuntimeError(f"Nessuna ampiezza definita nel dizionario per il canale {channel}")
        signal_amp = float(amp_mv) * AMP_UNIT_TO_V

        # 2. AP + NPS 
        meanpulse, nps = load_ap_nps(channel)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        shared = build_shared(device)

        # 3. Stima BI e append al CSV
        res = estimate_BI_for_channel(str(channel), meanpulse, nps,
                                      signal_amp, SAMPLING_RATE, shared, device)
        append_row_to_csv(OUTPUT_CSV, res)
        print(f"[OK] ch {channel} [{MODE}]: BI={res['BI']:.3e}  ->  {OUTPUT_CSV}")

    except Exception as e:
        print(f"[ERROR] ch {channel}: {e}")
        sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# Job submission helpers
# ═════════════════════════════════════════════════════════════════════════════
def create_sh(lines: list) -> str:
    os.makedirs(JOBS_DIR, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".sh", dir=JOBS_DIR)
    tmp.write("#!/bin/bash\n")
    tmp.write("\n".join(lines) + "\n")
    tmp.close()
    os.chmod(tmp.name, 0o755)
    return tmp.name

def make_job_lines(channel: int) -> list:
    lines = [f"cd {BASE_DIR}"]
    lines += ENV_SETUP_LINES
    lines.append(f"{sys.executable} {SCRIPT_PATH} --worker --channel {channel}")
    return lines

def running_job_count() -> int:
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
    job_name = f"{JOB_NAME_PREFIX}{task_key}"[:15]   
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
# ORCHESTRATORE
# ═════════════════════════════════════════════════════════════════════════════
def run_orchestrator():
    for d in (OUTPUT_DIR, LOG_DIR, JOBS_DIR):
        os.makedirs(d, exist_ok=True)

    if MODE not in ("octopus", "argonauts"):
        sys.exit(f"[ERROR] MODE non valida: {MODE!r}")
    if not SIGNAL_AMP_MV:
        sys.exit("[ERROR] SIGNAL_AMP_MV e' vuoto: definisci le ampiezze per canale.")

    amp_map = build_amp_map()
    tasks = sorted(amp_map.keys())

    print(f"MODE = {MODE}")
    print(f"Canali confermati (dati + ampiezza note): {len(tasks)}")
    for ch in tasks:
        print(f"   ch {ch}: amp = {amp_map[ch]} (x{AMP_UNIT_TO_V} -> V)")
    if not tasks:
        sys.exit("[ERROR] Nessun canale disponibile per cui è definita un'ampiezza.")

    if RESET_CSV:
        init_csv(OUTPUT_CSV)
        print(f"CSV inizializzato (solo header): {OUTPUT_CSV}")

    if SUBMIT_MODE == "local":
        print("[INFO] SUBMIT_MODE='local': eseguo i task in sequenza (no qsub).\n")
        for ch in tasks:
            run_worker(ch)
        print(f"\nFatto. Risultati in {OUTPUT_CSV}")
        return

    submitted, failed = 0, []
    for ch in tasks:
        wait_for_slot()
        task_key = f"{ch}"
        jobid = None
        for attempt in range(3):          
            sh_file = create_sh(make_job_lines(ch))
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
    print("=" * 65 + "\n")

# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Stima del BI (run senza load curve) parallelizzata su cluster.")
    parser.add_argument("--worker", action="store_true",
                        help="modalità worker: calcola un singolo canale")
    parser.add_argument("--channel", type=int, help="canale (richiesto con --worker)")
    args = parser.parse_args()

    if args.worker:
        if args.channel is None:
            sys.exit("[ERROR] --worker richiede --channel")
        run_worker(args.channel)
    else:
        run_orchestrator()

if __name__ == "__main__":
    main()