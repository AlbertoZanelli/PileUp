"""
scan_residuals_m205.py
======================
Diagnostica della FORMA DEI RESIDUI del fit dell'average pulse (m205), per OGNI
combinazione del modello a poli/zeri, PARALLELIZZATA sul cluster come
analyse_BI_m205.py: viene sottomesso un job indipendente per OGNI singolo fit
(coppia WP + combinazione).

Combinazioni (regola: n_zeri <= n_poli_reali - 1):
  - poli reali n_real da NREAL_MIN a NREAL_MAX (default 3..6)
  - zeri nzer da 0 a n_real - 1
  - con e senza coppia complessa coniugata (CC); nel modello CC la coppia aggiunge
    2 poli, quindi npol_totale = n_real + 2.

Flusso (come analyse_BI_m205.py):
  1. Orchestratore (default): leggero. Enumera i task (WP, combinazione), sottomette
     un job qsub per ciascuno e TERMINA subito. Con SUBMIT_MODE='local' esegue tutto
     in sequenza e produce direttamente le canve.
  2. Worker (--worker --wp W --nreal R --cc {0,1} --nzer Z): fa UN fit e salva il
     risultato (curva fittata + RMS) in un .npz con nome distinto (nessun lock).
  3. Plot (--plot): raccoglie i .npz e produce UNA canva per WP (un pannello per
     combinazione, col residuo e l'RMS; titolo verde = RMS minimo). Da lanciare a
     job finiti.

Le canve finiscono nella cartella OUTDIR: resid_scan_ch<ch>_wp<wp>.png

Il fit e' quello scipy least_squares di plot_AP_spectra_m205.fit_average_pulse
(residui pesati per 1/sigma). Nei pannelli il residuo e' mostrato DIVISO PER SIGMA
(sigma = RMS del baseline), cioe' (dato - fit)/sigma, e in titolo l'RMS in unita' di
sigma. Serve un python con numpy/scipy/uproot (es. /opt/homebrew/bin/python3.13).

Esempi:
    KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/bin/python3.13 scan_residuals_m205.py            # sottomette / (local) esegue
    KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/bin/python3.13 scan_residuals_m205.py --worker --wp 1 --nreal 4 --cc 1 --nzer 2
    KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/bin/python3.13 scan_residuals_m205.py --plot     # a job finiti: fa le canve
"""

from __future__ import annotations

import os
import re
import sys
import glob
import math
import time
import argparse
import tempfile
import subprocess

import numpy as np
import uproot
# NB: plot_AP_spectra_m205 (scipy/matplotlib) importato PIGRAMENTE dentro worker/plot,
# così l'orchestratore resta leggero (come analyse_BI_m205.py).

# ═════════════════════════════════════════════════════════════════════════════
# Config diagnostica
# ═════════════════════════════════════════════════════════════════════════════
CHANNEL     = 91
N_FIRST_WPS = 3               # primi N working point (odd wp), per adesso
NREAL_MIN, NREAL_MAX = 3, 6
# Valori di cc da provare nello scan. (False,) = solo poli reali (modelli semplici, la CC
# serve solo con oscillazioni vere); (False, True) = include anche la coppia complessa.
CC_OPTIONS = (False,)
ZOOM_MS     = (-3.0, 120.0)   # finestra di zoom del residuo (ms, rispetto al picco)
ONSET_PRE_MS = 1.0           # (modalità --onset) ms di baseline prima del 5% da mostrare
# (--onset-resid) finestra ADATTIVA attorno alla salita, definita sul singolo AP (non ms fissi):
#   - FINE (destra) = dove l'AP raggiunge RESID_END_FRAC del picco in salita (frazione di ampiezza);
#   - INIZIO (sinistra) = RESID_PRE_FRAC * (tempo del 5% al picco) PRIMA del 5% del picco.
#     Il 5% è un riferimento FISSO per l'inizio: cambiare RESID_END_FRAC non tocca il margine sinistro.
RESID_END_FRAC = 1.05
RESID_PRE_FRAC = 0.3
# Banda di ONSET esclusa dal fit: dallo stacco fino a MASK_HI_FRAC del picco (0 = niente
# maschera). Segnata in rosso nei plot (griglie fit/residui e onset_resid).
MASK_HI_FRAC = 0.10

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.abspath(__file__)
DATA_DIR    = os.path.join(BASE_DIR, "Processed")
OUTDIR      = os.path.join(BASE_DIR, "m205_residual_scan")
FITS_DIR    = os.path.join(OUTDIR, "_fits")        # .npz per (wp, combinazione)
LOG_DIR     = os.path.join(OUTDIR, "logs")         # stdout/stderr dei job
JOBS_DIR    = os.path.join(OUTDIR, "jobs")         # script .sh temporanei
MEAS_NAME   = "000205"
HIST_TMPL   = "averagepulse_ap_wp{wp}_medianAP"

VBIAS_LIST = np.array([0.6, 1.0, 1.4, 1.8, 2, 3, 4, 5, 6, 8, 10, 20, 26, 30, 40])


def wp_to_vbias(wp_idx: int) -> float:
    return float(VBIAS_LIST[wp_idx // 2])


# ═════════════════════════════════════════════════════════════════════════════
# Cluster / scheduler config  (come analyse_BI_m205.py — ADATTA al tuo cluster)
# ═════════════════════════════════════════════════════════════════════════════
SUBMIT_MODE       = "local"    # "qsub" = un job per fit ; "local" = sequenziale (debug/no cluster)
QUEUE             = "cupid"
WALLTIME          = "24:00:00"
RAM_GB            = 4
MAX_PARALLEL_JOBS = 135
SLEEP_INTERVAL    = 20
JOB_NAME_PREFIX   = "RSCAN"
EXPORT_ENV        = True
RESET             = True       # se True svuota FITS_DIR prima di sottomettere
ENV_SETUP_LINES   = ["source /home/zanelli/LoadOctopus.sh"]


# ═════════════════════════════════════════════════════════════════════════════
# Combinazioni del modello
# ═════════════════════════════════════════════════════════════════════════════
def combos():
    """Lista (n_real, cc, nzer) secondo la regola nzer <= n_real - 1."""
    return [(n_real, cc, nzer)
            for n_real in range(NREAL_MIN, NREAL_MAX + 1)
            for cc in CC_OPTIONS
            for nzer in range(0, n_real)]


def label(n_real, cc, nzer):
    """Etichetta per poli REALI, es. '4p+cc z2' = 4 reali + coppia CC + 2 zeri."""
    return f"{n_real}p{'+cc' if cc else ''} z{nzer}"


def combo_npz(wp, n_real, cc, nzer):
    return os.path.join(FITS_DIR,
                        f"scan_ch{CHANNEL}_wp{wp}_{n_real}p_cc{int(cc)}_z{nzer}.npz")


def _load_pulse(f, wp):
    """AP peak-normalizzato + asse tempi da un file ROOT aperto."""
    h = f[HIST_TMPL.format(wp=wp)]
    v = np.asarray(h.values(), dtype=float)
    t = np.asarray(h.axis().centers(), dtype=float)
    return t, v / v.max()


def list_wps(f):
    """Primi N_FIRST_WPS working point (odd wp) presenti nel file ROOT."""
    return sorted(set(
        int(mo.group(1)) for k in f.keys()
        for mo in [re.search(r"averagepulse_ap_wp(\d+)_medianAP", k)]
        if mo and (int(mo.group(1)) % 2 != 0)
    ))[:N_FIRST_WPS]


def root_file_for_channel() -> str | None:
    roots = glob.glob(os.path.join(DATA_DIR, f"Processed_*_{MEAS_NAME}_{CHANNEL}.root"))
    return roots[0] if roots else None


# ═════════════════════════════════════════════════════════════════════════════
# WORKER: un singolo fit -> .npz (nome distinto, nessun lock)
# ═════════════════════════════════════════════════════════════════════════════
def run_worker(wp, n_real, cc, nzer):
    import plot_AP_spectra_m205 as m   # motore scipy least_squares (import pesante: solo nel job)

    try:
        fp = root_file_for_channel()
        if fp is None:
            raise RuntimeError(f"ROOT del canale {CHANNEL} non trovato in {DATA_DIR}")
        with uproot.open(fp) as f:
            t, v = _load_pulse(f, wp)

        npol = n_real + (2 if cc else 0)      # CC: la coppia aggiunge 2 poli
        res = m.fit_average_pulse(t, v, nzer=nzer, npol=npol, cc=cc, mask_hi_frac=MASK_HI_FRAC)

        os.makedirs(FITS_DIR, exist_ok=True)
        np.savez(combo_npz(wp, n_real, cc, nzer),
                 fit=res["fit"].astype(np.float32), rms=float(res["rms"]),
                 sigma=float(res["sigma"]),
                 n_real=n_real, cc=int(cc), nzer=nzer, wp=wp)
        print(f"[OK] wp{wp} {label(n_real, cc, nzer)}  rms={res['rms']:.2e}")
    except Exception as e:
        print(f"[ERROR] wp{wp} {label(n_real, cc, nzer)}: {e}")
        sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# Job submission helpers (come analyse_BI_m205.py)
# ═════════════════════════════════════════════════════════════════════════════
def create_sh(lines: list) -> str:
    os.makedirs(JOBS_DIR, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".sh", dir=JOBS_DIR)
    tmp.write("#!/bin/bash\n" + "\n".join(lines) + "\n")
    tmp.close()
    os.chmod(tmp.name, 0o755)
    return tmp.name


def make_job_lines(wp, n_real, cc, nzer) -> list:
    lines = [f"cd {BASE_DIR}"] + ENV_SETUP_LINES
    lines.append(f"{sys.executable} {SCRIPT_PATH} --worker "
                 f"--wp {wp} --nreal {n_real} --cc {int(cc)} --nzer {nzer}")
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
        print(f"Max parallel jobs ({MAX_PARALLEL_JOBS}) raggiunto. Attendo {SLEEP_INTERVAL}s...")
        time.sleep(SLEEP_INTERVAL)


def submit_task(task_key: str, sh_file: str) -> str | None:
    job_name = f"{JOB_NAME_PREFIX}{task_key}"[:15]
    export = "-V " if EXPORT_ENV else ""
    cmd = (f"qsub -N {job_name} {export}-q {QUEUE} "
           f"-o localhost:{LOG_DIR}/ -e localhost:{LOG_DIR}/ "
           f"-l walltime={WALLTIME} -l mem={RAM_GB}G {sh_file}")
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    except Exception as e:
        print(f"[ERROR] eccezione su qsub per {task_key}: {e}")
        return None
    if r.returncode != 0:
        print(f"[ERROR] qsub fallito per {task_key}. stderr:\n{r.stderr}")
        return None
    jobid = (r.stdout.strip().split(".")[0] if r.stdout.strip() else "")
    if not jobid.isdigit():
        print(f"[ERROR] job id non valido per {task_key}: '{r.stdout.strip()}'")
        return None
    print(f"[OK] sottomesso {task_key} (job {jobid})")
    return jobid


# ═════════════════════════════════════════════════════════════════════════════
# PLOT: raccoglie i .npz e fa una canva per WP
# ═════════════════════════════════════════════════════════════════════════════
def plot_wp(t, v, wp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_AP_spectra_m205 as m           # per rise_mask_band (stessa banda del fit)

    cs = combos()
    loaded = []
    for n_real, cc, nzer in cs:
        path = combo_npz(wp, n_real, cc, nzer)
        if os.path.exists(path):
            d = np.load(path)
            sigma = float(d["sigma"]) if "sigma" in d else 1.0
            loaded.append((n_real, cc, nzer, float(d["rms"]), np.asarray(d["fit"], float), sigma))
        else:
            loaded.append((n_real, cc, nzer, np.inf, None, 1.0))

    rmss = [r[3] for r in loaded]
    best_i = int(np.argmin(rmss)) if np.isfinite(min(rmss)) else -1
    t_peak = t[int(np.argmax(v))]
    sel = (t > t_peak + ZOOM_MS[0] / 1e3) & (t < t_peak + ZOOM_MS[1] / 1e3)
    tm = (t - t_peak) * 1e3
    band = m.rise_mask_band(t, v, MASK_HI_FRAC)                     # banda di onset esclusa dal fit
    band_ms = ((t[band[0]] - t_peak) * 1e3, (t[band[1]] - t_peak) * 1e3) if band else None

    n = len(loaded)
    ncols = 6
    nrows = math.ceil(n / ncols)

    def _grid(kind):
        """kind='resid' -> residuo/σ con banda ±3σ ; kind='fit' -> AP + fit."""
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.9 * ncols, 1.9 * nrows), squeeze=False)
        axf = axes.ravel()
        for i, (ax, (n_real, cc, nzer, rms, fit, sigma)) in enumerate(zip(axf, loaded)):
            if fit is not None:
                if band_ms is not None:                            # banda di onset esclusa dal fit (rossa)
                    ax.axvspan(band_ms[0], band_ms[1], color="#c1121f", alpha=0.18, lw=0)
                if kind == "resid":
                    ax.axhspan(-3, 3, color="#4a90d9", alpha=0.25, lw=0)   # banda ±3σ (residui in unità di σ)
                    ax.plot(tm, (v - fit) / sigma, lw=0.8, color="#c1121f")
                    ax.axhline(0.0, color="gray", ls=":", lw=0.7)
                    ax.set_xlim(left=-5, right=15)
                else:
                    ax.plot(tm, v, ".", ms=2, color="k")
                    ax.plot(tm, fit, "-", lw=1.1, color="#c1121f")
                    ax.set_xlim(left=-5, right=15)
            else:
                ax.text(0.5, 0.5, "FAIL / missing", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=8)
            best = (i == best_i)
            rms_sig = rms / sigma if np.isfinite(rms) else rms          # RMS in unità di sigma
            ax.set_title(f"{label(n_real, cc, nzer)}   rms/σ={rms_sig:.1f}", fontsize=8,
                         color=("green" if best else "black"),
                         fontweight=("bold" if best else "normal"))
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=6)
        for ax in axf[n:]:
            ax.axis("off")
        if kind == "resid":
            head = "Residual shape"; extra = "residual = (data - fit) / σ, blue = ±3σ band"
            ylab = "residual / σ"; tag = "resid_scan"
        else:
            head = "AP + fit"; extra = "black = AP data, red = fit"
            ylab = "AP amplitude (peak-normalized)"; tag = "fit_scan"
        fig.suptitle(f"{head} per model combination — Ch {CHANNEL} · WP {wp} "
                     f"({wp_to_vbias(wp):g} V) · Measurement 000205\n"
                     f"green = lowest RMS   ({extra}, zoom on the pulse)",
                     fontsize=13, fontweight="bold")
        fig.supxlabel("t - t_peak [ms]", fontsize=11)
        fig.supylabel(ylab, fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = os.path.join(OUTDIR, f"{tag}_ch{CHANNEL}_wp{wp}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    _grid("resid")
    _grid("fit")
    got = sum(1 for r in loaded if r[4] is not None)
    b = loaded[best_i] if best_i >= 0 else None
    best_txt = f"best {label(b[0], b[1], b[2])} (rms={b[3]:.2e})" if b else "no fit"
    print(f"Ch {CHANNEL} WP {wp}: {got}/{n} combos  ->  resid_scan/fit_scan_ch{CHANNEL}_wp{wp}.png   {best_txt}")


def run_onset():
    """UNA canva (un pannello per WP) con la sola PARTE DI ONSET dell'average pulse:
    da poco prima della salita (ONSET_PRE_MS ms prima) fino al 5% del picco. Nessun
    fit, solo il dato. -> onset_ch<ch>_m205.png"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUTDIR, exist_ok=True)
    fp = root_file_for_channel()
    if fp is None:
        sys.exit(f"[ERROR] ROOT del canale {CHANNEL} non trovato in {DATA_DIR}")
    with uproot.open(fp) as f:
        wps = sorted(set(                             # TUTTI i WP (odd), non solo i primi N
            int(mo.group(1)) for k in f.keys()
            for mo in [re.search(r"averagepulse_ap_wp(\d+)_medianAP", k)]
            if mo and (int(mo.group(1)) % 2 != 0)))
        pulses = [(wp, *_load_pulse(f, wp)) for wp in wps]

    ncols = 5
    nrows = math.ceil(len(wps) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.2 * nrows), squeeze=False)
    axf = axes.ravel()
    for ax, (wp, t, v) in zip(axf, pulses):
        peak = float(v.max()); imax = int(np.argmax(v))
        up = v[:imax]
        cross = np.where(up >= 0.05 * peak)[0]        # primo campione al 5% del picco (in salita)
        i5 = cross[0] if len(cross) else imax
        t5 = t[i5]
        sel = (t >= t5 - ONSET_PRE_MS / 1e3) & (t <= t5)
        ax.plot((t[sel] - t5) * 1e3, v[sel], ".-", ms=3, lw=0.8, color="k")
        ax.axhline(0.05 * peak, color="crimson", ls="--", lw=0.8, label="5% del picco")
        ax.axhline(0.0, color="gray", ls=":", lw=0.7)
        ax.set_title(f"WP {wp} · {wp_to_vbias(wp):g} V", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
    axf[0].legend(fontsize=7, loc="upper left")
    for ax in axf[len(wps):]:
        ax.axis("off")
    fig.suptitle(f"Average pulse — onset region (poco prima della salita .. 5% del picco) — "
                 f"Ch {CHANNEL} · Measurement 000205", fontsize=14, fontweight="bold")
    fig.supxlabel("t - t(5%) [ms]", fontsize=11)
    fig.supylabel("AP amplitude (peak-normalized)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUTDIR, f"onset_ch{CHANNEL}_m205.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Ch {CHANNEL}: onset di {len(wps)} WP  ->  {os.path.basename(out)}")


def _best_combo_for_wp(wp):
    """(n_real, cc, nzer, rms, fit_array, sigma) del combo a RMS minimo per il WP,
    letto dagli .npz già prodotti dallo scan. None se non ce n'è nessuno su disco."""
    best = None
    for n_real, cc, nzer in combos():
        path = combo_npz(wp, n_real, cc, nzer)
        if not os.path.exists(path):
            continue
        d = np.load(path)
        rms = float(d["rms"])
        sigma = float(d["sigma"]) if "sigma" in d else 1.0
        if best is None or rms < best[3]:
            best = (n_real, cc, nzer, rms, np.asarray(d["fit"], float), sigma)
    return best


def run_onset_resid():
    """UNA canva (un pannello per WP con fit disponibile) come onset ma con SOVRAPPOSTI
    i residui del fit MIGLIORE (RMS minimo tra i combo dello scan). Asse sinistro = AP
    (dato + fit), asse destro = residuo/σ; finestra RESID_WIN_MS attorno al picco per
    localizzare lo spike (che punti dell'impulso lo generano). -> onset_resid_ch<ch>_m205.png"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_AP_spectra_m205 as m           # per rise_mask_band (stessa banda del fit)

    os.makedirs(OUTDIR, exist_ok=True)
    fp = root_file_for_channel()
    if fp is None:
        sys.exit(f"[ERROR] ROOT del canale {CHANNEL} non trovato in {DATA_DIR}")
    with uproot.open(fp) as f:
        all_wps = sorted(set(
            int(mo.group(1)) for k in f.keys()
            for mo in [re.search(r"averagepulse_ap_wp(\d+)_medianAP", k)]
            if mo and (int(mo.group(1)) % 2 != 0)))
        items = []                                    # solo i WP che hanno almeno un fit su disco
        for wp in all_wps:
            best = _best_combo_for_wp(wp)
            if best is None:
                continue
            t, v = _load_pulse(f, wp)
            items.append((wp, t, v, best))
    if not items:
        sys.exit(f"[ERROR] nessun .npz di fit per il canale {CHANNEL} in {FITS_DIR}. "
                 f"Lancia prima lo scan.")

    ncols = min(3, len(items))
    nrows = math.ceil(len(items) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.2 * nrows), squeeze=False)
    axf = axes.ravel()
    C_DATA, C_FIT, C_RESID = "k", "#c1121f", "#1f5fd9"
    for ax, (wp, t, v, best) in zip(axf, items):
        n_real, cc, nzer, rms, fit, sigma = best      # sigma = RMS baseline (unità del residuo)
        peak = float(v.max()); imax = int(np.argmax(v)); t_peak = t[imax]
        up = v[:imax + 1]
        # riferimento FISSO: istante in cui l'AP raggiunge il 5% del picco (in salita)
        c5 = np.where(up >= 0.05 * peak)[0]
        t5 = t[c5[0]] if len(c5) else t_peak
        # FINE: dove l'AP raggiunge RESID_END_FRAC del picco (in salita)
        ce = np.where(up >= RESID_END_FRAC * peak)[0]
        t_end = t[ce[0]] if len(ce) else t_peak
        # INIZIO: RESID_PRE_FRAC volte il tempo (5%→picco), PRIMA del 5%
        t_start = t5 - RESID_PRE_FRAC * (t_peak - t5)
        sel = (t >= t_start) & (t <= t_end)
        tm = (t[sel] - t5) * 1e3                       # asse allineato sul 5% del picco

        band = m.rise_mask_band(t, v, MASK_HI_FRAC)    # banda di onset esclusa dal fit (rossa)
        if band is not None:
            ax.axvspan((t[band[0]] - t5) * 1e3, (t[band[1]] - t5) * 1e3,
                       color=C_FIT, alpha=0.18, lw=0, zorder=0)

        ln_d, = ax.plot(tm, v[sel], ".-", ms=4, lw=0.8, color=C_DATA, label="AP data", zorder=3)
        ln_f, = ax.plot(tm, fit[sel], "-", lw=1.4, color=C_FIT, label="best fit", zorder=4)
        ax.set_ylabel("AP amplitude (peak-norm.)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

        axr = ax.twinx()                              # residuo/σ sull'asse destro
        axr.axhspan(-3, 3, color=C_RESID, alpha=0.10, lw=0)   # banda ±3σ
        axr.axhline(0.0, color=C_RESID, ls=":", lw=0.7, alpha=0.6)
        r = (v[sel] - fit[sel]) / sigma
        ln_r, = axr.plot(tm, r, "-", lw=1.0, color=C_RESID, alpha=0.9, label="residual / σ", zorder=2)
        # scala dell'asse sui residui FUORI dalla banda mascherata (dentro schizzano e la
        # schiacciano): la banda non è fittata, quindi non deve dettare la scala.
        idx = np.where(sel)[0]
        out = np.ones(len(idx), bool) if band is None else ~((idx >= band[0]) & (idx <= band[1]))
        if out.any():
            lo_r, hi_r = float(r[out].min()), float(r[out].max())
            pad = 0.15 * (hi_r - lo_r + 1e-9)
            axr.set_ylim(min(lo_r - pad, -3.5), max(hi_r + pad, 3.5))
        axr.set_ylabel("residual / σ", fontsize=9, color=C_RESID)
        axr.tick_params(labelsize=8, colors=C_RESID)

        ax.set_title(f"WP {wp} · {wp_to_vbias(wp):g} V   —   best: {label(n_real, cc, nzer)}"
                     f"   rms/σ={rms / sigma:.1f}", fontsize=10)
        if ax is axf[0]:
            ax.legend([ln_d, ln_f, ln_r], [l.get_label() for l in (ln_d, ln_f, ln_r)],
                      fontsize=8, loc="upper right")
    for ax in axf[len(items):]:
        ax.axis("off")
    fig.suptitle(f"Average pulse + best-fit residual on the rise — "
                 f"Ch {CHANNEL} · Measurement 000205\n"
                 f"black = AP data, red = best fit (min RMS combo), blue = residual / σ  "
                 f"(adaptive window: end at {RESID_END_FRAC:.0%} of peak)",
                 fontsize=13, fontweight="bold")
    fig.supxlabel("t - t(5%) [ms]", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUTDIR, f"onset_resid_ch{CHANNEL}_m205.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Ch {CHANNEL}: onset+residuo di {len(items)} WP  ->  {os.path.basename(out)}")


def run_plot():
    os.makedirs(OUTDIR, exist_ok=True)
    fp = root_file_for_channel()
    if fp is None:
        sys.exit(f"[ERROR] ROOT del canale {CHANNEL} non trovato in {DATA_DIR}")
    with uproot.open(fp) as f:
        for wp in list_wps(f):
            t, v = _load_pulse(f, wp)
            plot_wp(t, v, wp)
    print("\nFatto (plot).")


# ═════════════════════════════════════════════════════════════════════════════
# ORCHESTRATORE
# ═════════════════════════════════════════════════════════════════════════════
def run_orchestrator():
    for d in (OUTDIR, FITS_DIR, LOG_DIR, JOBS_DIR):
        os.makedirs(d, exist_ok=True)
    fp = root_file_for_channel()
    if fp is None:
        sys.exit(f"[ERROR] ROOT del canale {CHANNEL} non trovato in {DATA_DIR}")
    with uproot.open(fp) as f:
        wps = list_wps(f)

    tasks = [(wp, n_real, cc, nzer) for wp in wps for (n_real, cc, nzer) in combos()]
    print(f"Canale {CHANNEL}: WP {wps} × {len(combos())} combinazioni = {len(tasks)} fit (task).")

    if RESET:
        for old in glob.glob(os.path.join(FITS_DIR, "*.npz")):
            os.remove(old)
        print(f"Cartella fit azzerata: {FITS_DIR}")

    # ── Modalità locale: esegue tutti i fit in sequenza e fa subito le canve ────
    if SUBMIT_MODE == "local":
        print("[INFO] SUBMIT_MODE='local': eseguo i fit in sequenza (no qsub).\n")
        for wp, n_real, cc, nzer in tasks:
            run_worker(wp, n_real, cc, nzer)
        run_plot()
        return

    # ── Sottomissione: un job per fit, con throttling ──────────────────────────
    submitted, failed = 0, []
    for wp, n_real, cc, nzer in tasks:
        wait_for_slot()
        task_key = f"{wp}_{n_real}{'c' if cc else ''}z{nzer}"
        jobid = None
        for _ in range(3):
            sh_file = create_sh(make_job_lines(wp, n_real, cc, nzer))
            jobid = submit_task(task_key, sh_file)
            if jobid is not None:
                break
            time.sleep(SLEEP_INTERVAL)
        if jobid is not None:
            submitted += 1
        else:
            failed.append(task_key)

    print("\n" + "=" * 65)
    print(f"  {submitted}/{len(tasks)} job sottomessi.")
    if failed:
        print(f"  {len(failed)} NON sottomessi: {failed}")
    print(f"  Ogni job salva il suo .npz in: {FITS_DIR}")
    print(f"  Log dei job in: {LOG_DIR}")
    print(f"  A job finiti, fai le canve con:  python scan_residuals_m205.py --plot")
    print("=" * 65 + "\n")


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="Scan dei residui per combinazione (m205), parallelizzato.")
    p.add_argument("--worker", action="store_true", help="modalità worker: un singolo fit")
    p.add_argument("--wp", type=int)
    p.add_argument("--nreal", type=int)
    p.add_argument("--cc", type=int, choices=(0, 1))
    p.add_argument("--nzer", type=int)
    p.add_argument("--plot", action="store_true", help="raccoglie i .npz e fa le canve")
    p.add_argument("--onset", action="store_true",
                   help="canva della sola regione di onset (dato, no fit) per tutti i WP")
    p.add_argument("--onset-resid", dest="onset_resid", action="store_true",
                   help="canva onset con sovrapposti i residui del fit migliore (usa gli .npz)", default=False)
    args = p.parse_args()

    if args.worker:
        if None in (args.wp, args.nreal, args.cc, args.nzer):
            sys.exit("[ERROR] --worker richiede --wp --nreal --cc --nzer")
        run_worker(args.wp, args.nreal, bool(args.cc), args.nzer)
    elif args.plot:
        run_plot()
    elif args.onset:
        run_onset()
    elif args.onset_resid:
        run_onset_resid()
    else:
        run_orchestrator()


if __name__ == "__main__":
    main()
