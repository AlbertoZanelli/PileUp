"""
ulysse_pileup_BI.py
===================
Pile-up rejection on the Ulysse RUN128 light detectors (0 V NTL, PU-M1/M2), with the muon AP and the NPS of
ulysse_run128/ (written by MisureUlysse_pileup/export_pileup_inputs.py; points.csv has fs, ROI amplitude at LY 0.3,
rise time, photons). Same algorithms and event generation as the m205 v2 campaign:
  OF = optimum filter + two band filters trained on the analytic BI, ratio of the maxima (analyse_BI_m205.py);
  M  = likelihood ratio (src/pileup_likelihood.py), t_window = dt_max + 0.1 ms;
  events: simulate_frequency_pulses_v2 (photon statistics, sum energy in the ROI, sub-sample shift), baseline on
  the pretrigger; dt_max = 10-90% rise time of the AP; cut at 90% of the singles; BI = K (1 - rejected pile-up).
Template of generation = template of training = muon AP: its noise is 4e-4 of the peak, negligible next to the
ROI pulse (SNR 4-14), so no fitted template is needed.

Two steps on the cluster, each submits its PBS jobs and exits:
  python ulysse_pileup_BI.py train     one job per LD: band filters (N_TRIALS steps) and analytic BI
                                       -> OUT_DIR/BI_results.csv, OUT_DIR/trained_filters/
  python ulysse_pileup_BI.py mc        when the training is done: one job per LD running the N_SEEDS seeds, OF and M
                                       on the same NSIM singles + NSIM pile-ups -> OUT_DIR/BI_mc_seeds<N_SEEDS>.csv
                                       (~2.5 min per seed on the Mac, so ~1 h per job)
  python ulysse_pileup_BI.py summary   median BI over the seeds, M vs OF seed by seed
The jobs run a frozen copy of this file (OUT_DIR/_ulysse_pileup_BI.py): editing the config does not touch
queued jobs. SUBMIT_MODE = "local" runs everything here, in sequence (debug only).
"""
import csv
import fcntl
import os
import shutil
import subprocess
import sys
import time

import numpy as np

BASE_DIR = os.environ.get("PILEUP_DIR", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# ── CONFIGURATION ─────────────────────────────────────────────────────────────────────────────
IN_DIR   = os.path.join(BASE_DIR, "ulysse_run128")
OUT_DIR  = os.path.join(BASE_DIR, "ulysse_results_muonAP_dtrise")
CHANNELS = [2, 5, 1, 3, 6]      # Octo_m6, m8 (PU-M1), m11, m15, m10 (PU-M2)
N_TRIALS = 500                  # training steps, as the m205 campaign
NSIM     = 50_000               # events per population and seed
CHUNK    = 500                  # events generated at a time (changes the events: keep it fixed)
SEED     = 1234
N_SEEDS  = 20                   # seeds SEED .. SEED + N_SEEDS - 1, one row per seed and algorithm
DT_BINS  = 20                   # rejection vs delay: pile-up and rejected pile-up in DT_BINS bins of [0, dt_max]
ACCEPTANCE = 0.9

SUBMIT_MODE       = "qsub"      # "qsub" | "local"
QUEUE             = "cupid"
WALLTIME          = "24:00:00"
RAM_GB            = 4
MAX_PARALLEL_JOBS = 100         # queue cupid: max_user_run 80, max_user_queuable 160
JOB_PREFIX        = "UL"
ENV_SETUP_LINES   = ["source /home/zanelli/LoadOctopus.sh"]

FILTERS_DIR = os.path.join(OUT_DIR, "trained_filters")
TRAIN_CSV   = os.path.join(OUT_DIR, "BI_results.csv")
MC_CSV      = os.path.join(OUT_DIR, f"BI_mc_seeds{N_SEEDS}.csv")
LOG_DIR     = os.path.join(OUT_DIR, "logs")


def point(ch):
    """Inputs of one LD: row of points.csv, AP (peak 1), NPS (E|FFT|^2), dt_max = rise time [s]."""
    row = next(r for r in csv.DictReader(open(os.path.join(IN_DIR, "points.csv"))) if int(r["channel"]) == ch)
    ap = np.load(os.path.join(IN_DIR, f"ap_ch{ch}.npy"))
    nps = np.load(os.path.join(IN_DIR, f"nps_ch{ch}.npy"))
    return row, ap / ap.max(), nps, float(row["risetime_s"])


def append_row(path, row):
    """One row in a CSV shared by parallel jobs (exclusive lock, header if the file is empty)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0, os.SEEK_END)
        w = csv.DictWriter(f, fieldnames=list(row))
        if f.tell() == 0:
            w.writeheader()
        w.writerow(row)
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


# ── WORKERS ───────────────────────────────────────────────────────────────────────────────────
def train(ch):
    """Band filters of the OF for one LD (analyse_BI_m205.estimate_BI_for_wp, wp = 0) and its analytic BI."""
    import torch
    import analyse_BI_m205 as tr
    row, ap, nps, dt_max = point(ch)
    tr.N_TRIALS = N_TRIALS
    device = torch.device("cpu")
    res = tr.estimate_BI_for_wp(ch, 0, 0.0, ap, nps, float(row["signal_amp_V"]), float(row["fs_Hz"]),
                                tr.build_shared(device, dt_max), device)
    tr.save_filters_npy(FILTERS_DIR, ch, 0, res["f1"], res["f2"], res["kernel"])
    np.save(os.path.join(FILTERS_DIR, f"J_ch{ch}.npy"), res["J_hist"])
    append_row(TRAIN_CSV, {"channel": ch, "ld": row["ld"], "signal_amp": res["signal_amp"], "dt_max": dt_max,
                           "sigma_analytic": res["sigma_analytic"], "SNR": res["SNR"], "beta_Hz": res["beta_Hz"],
                           "BI": res["BI"], "J_final": res["J_final"], "n_trials": res["n_trials"],
                           "train_s": res["train_s"]})
    print(f"[OK] train ch {ch} ({row['ld']}): BI analytic {res['BI']:.3e}, SNR {res['SNR']:.1f}, {res['train_s']:.0f} s")


def full(half):
    """Whole spectrum from the saved half (DC..Nyquist), conjugate mirror for the complex kernel."""
    m = half[-2:0:-1]
    return np.concatenate([half, np.conj(m) if np.iscomplexobj(half) else m])


def mc(ch, seed):
    """One LD, one seed: NSIM singles and NSIM pile-ups (same noise, amplitude, r, energy and shift: only the delay
    changes), discriminant of OF (ratio of the maxima) and M (-M) on the same events, BI of each."""
    import src.analysis as an
    import src.dataset as ds
    import src.simulation as sim
    import utility.functions as fn
    from src.pileup_likelihood import PileupLikelihood
    row, ap, nps, dt_max = point(ch)
    fs, amp = float(row["fs_Hz"]), float(row["signal_amp_V"])
    S, w, H = an.compute_H(ap, nps, np.hanning, sampling_rate=fs)
    rel = np.abs(full(np.load(os.path.join(FILTERS_DIR, f"kernel_ch{ch}_wp0.npy"))) - H).max() / np.abs(H).max()
    if rel > 1e-5:
        raise SystemExit(f"[ERROR] ch {ch}: trained kernel does not match the AP/NPS of {IN_DIR} ({rel:.1e})")
    f1, f2 = (full(np.load(os.path.join(FILTERS_DIR, f"{k}_ch{ch}_wp0.npy"))) for k in ("f1", "f2"))
    lik = PileupLikelihood(S, nps, w, dt_max=dt_max, t_window=dt_max + 1e-4)
    jitter = max(20, int(np.ceil(dt_max * fs)) + 10)          # max search must reach the second pulse
    K = fn.K_for(dt_max)

    def run(dt):
        rng, rng_x = np.random.default_rng(seed), np.random.default_rng([seed, 1])
        of, m, dts = [], [], []
        for k in range(0, NSIM, CHUNK):
            fp, info = sim.simulate_frequency_pulses_v2(S, nps, w, min(CHUNK, NSIM - k), rng, rng_x, signal_scale=amp,
                                                        dt_max=dt, n_photons=float(row["n_photons"]),
                                                        e_roi=fn.E_ROI_MEV, q_value=fn.Q_BB_MEV, shift_max=1 / fs)
            pulses = np.fft.ifft(fp, axis=1).real.astype(np.float32)
            data = ds.NumpyDataset(pulses)
            data.win_length = pulses.shape[1]
            of.append(np.ravel(an.get_PSD_interpole(data, H, f1, f2, jitter_max=jitter, pre_baseline=True)[0]))
            m.append(-lik.statistic(pulses))                    # pile-up in the low tail, as the ratio of the maxima
            dts.append(info["dt"])
        return np.concatenate(of), np.concatenate(m), np.concatenate(dts)

    t0 = time.perf_counter()
    of_s, m_s, _ = run(0.0)
    of_p, m_p, dt = run(dt_max)
    idx = np.clip((dt / dt_max * DT_BINS).astype(int), 0, DT_BINS - 1)
    for name, s, p in (("OF", of_s, of_p), ("M", m_s, m_p)):
        cut = np.percentile(s, 100 * (1 - ACCEPTANCE))
        rej = p < cut
        rp = float(rej.mean())
        sigma_rp, sigma_bi = an.compute_BI_uncertainty(s, p, ACCEPTANCE, rp, K=K)
        append_row(MC_CSV, {"channel": ch, "ld": row["ld"], "seed": seed, "filter": name, "BI_mc": K * (1 - rp),
                            "sigma_BI": sigma_bi, "rp": rp, "sigma_rp": sigma_rp, "nsim": NSIM, "K": K, "dt_max": dt_max,
                            "signal_amp": amp, "n_dt": " ".join(map(str, np.bincount(idx, minlength=DT_BINS))),
                            "n_dt_rej": " ".join(map(str, np.bincount(idx[rej], minlength=DT_BINS)))})
        print(f"[OK] mc ch {ch} ({row['ld']}) seed {seed} {name}: BI {K * (1 - rp):.4e}, rejected {rp:.1%}")
    print(f"     {time.perf_counter() - t0:.0f} s")


# ── SUMMARY ───────────────────────────────────────────────────────────────────────────────────
def summary():
    """Per LD: median BI over the seeds and half width (P84 - P16)/2 for OF and M, analytic BI of OF, M vs OF seed by
    seed (same events)."""
    an_bi = {r["channel"]: float(r["BI"]) for r in csv.DictReader(open(TRAIN_CSV))}
    rows = list(csv.DictReader(open(MC_CSV)))
    for ch in map(str, CHANNELS):
        bi = {f: {r["seed"]: float(r["BI_mc"]) for r in rows if r["channel"] == ch and r["filter"] == f} for f in ("OF", "M")}
        if not bi["OF"]:
            continue
        ld = next(r["ld"] for r in rows if r["channel"] == ch)
        line = f"{ld:9s}"
        for f, v in bi.items():
            x = np.array(list(v.values()))
            line += f" | {f}: {np.median(x):.3e} +- {np.diff(np.percentile(x, [16, 84]))[0] / 2:.1e} ({len(x)} seeds)"
        common = sorted(set(bi["OF"]) & set(bi["M"]))
        d = np.array([bi["M"][s] / bi["OF"][s] - 1 for s in common])
        print(line + f" | OF analytic {an_bi.get(ch, np.nan):.3e} | M vs OF {np.median(d):+.2%}")


# ── SUBMISSION ────────────────────────────────────────────────────────────────────────────────
def submit(tasks):
    """One PBS job per task (worker arguments), running the frozen copy of this file."""
    os.makedirs(LOG_DIR, exist_ok=True)
    frozen = os.path.join(OUT_DIR, "_" + os.path.basename(__file__))
    shutil.copy(os.path.abspath(__file__), frozen)
    count = f"qstat -u {os.environ.get('USER', '')} | grep ' {JOB_PREFIX}' | wc -l"
    for args in tasks:
        while int(subprocess.run(count, shell=True, capture_output=True, text=True).stdout.strip() or 0) >= MAX_PARALLEL_JOBS:
            time.sleep(20)
        sh = os.path.join(LOG_DIR, f"job_{'_'.join(args)}.sh")
        with open(sh, "w") as f:
            f.write("\n".join(["#!/bin/bash", f"cd {BASE_DIR}"] + ENV_SETUP_LINES +
                              ["export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
                               f"PILEUP_DIR={BASE_DIR} {sys.executable} {frozen} {' '.join(args)}"]) + "\n")
        name = f"{JOB_PREFIX}{args[0][0].upper()}{args[1]}"
        r = subprocess.run(f"qsub -N {name} -V -q {QUEUE} -o localhost:{LOG_DIR}/ -e localhost:{LOG_DIR}/ "
                           f"-l walltime={WALLTIME} -l mem={RAM_GB}G {sh}", shell=True, capture_output=True, text=True)
        print(f"[{'OK' if r.returncode == 0 else 'ERROR'}] {' '.join(args)}: {r.stdout.strip()}{r.stderr.strip()}")


def main():
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step == "train-worker":
        return train(int(sys.argv[2]))
    if step == "mc-worker":
        for i in range(N_SEEDS):
            mc(int(sys.argv[2]), SEED + i)
        return
    if step == "summary":
        return summary()
    if step == "train":
        tasks = [("train-worker", str(ch)) for ch in CHANNELS]
    elif step == "mc":
        missing = [ch for ch in CHANNELS if not os.path.exists(os.path.join(FILTERS_DIR, f"f1_ch{ch}_wp0.npy"))]
        if missing:
            raise SystemExit(f"[ERROR] no trained filters for channels {missing}: run the train step first")
        tasks = [("mc-worker", str(ch)) for ch in CHANNELS]
    else:
        raise SystemExit(__doc__)
    os.makedirs(OUT_DIR, exist_ok=True)
    if SUBMIT_MODE == "local":
        for args in tasks:
            subprocess.run([sys.executable, os.path.abspath(__file__), *args], check=True)
    else:
        submit(tasks)


if __name__ == "__main__":
    main()
