#!/usr/bin/env python3
"""
analyse_RUN9.py
===============
Lo stimatore M (src/pileup_likelihood.py) e il discriminante Y di Pageot sugli STESSI eventi del suo paper
(arXiv:2606.17948): CROSS RUN9 a LSC, misura 000813_20230628T161508, 8 rivelatori di luce.

Tutto come nel suo test/tutorial.ipynb, con il suo codice:
  - eventi: CachedBinaryDataset_withgenerated (src/dataset.py) = rumore vero del .bin del canale + il suo
    average_pulse iniettato a posizioni, ampiezze, smearing e dt dei suoi .pos (pup_n1-d0 singoli, pup_n1-d8
    pile-up, dt in [0, 0.8 ms]), finestra 4096, i primi 15000 (i suoi rp sono multipli esatti di 1/15000);
  - Y: compute_BI_torch (src/analysis.py, identica alla sua) con i suoi filtri addestrati functions_eff90,
    H dell'OF da compute_H(meanpulse_mean, NPS x 8/3, Hann). Se il suo rp torna, i dati sono gli stessi;
  - M: stesso template e stessa NPS, dt_max = 0.8 ms; il primo impulso si cerca entro +-0.9 ms perche' nella
    sua iniezione il 2o impulso arriva PRIMA del 1o (fino a dt_max), piu' lo smearing di +-0.05 ms.
Accettanza 90%, BI = fn.K (1 - rp) con K della finestra di 0.8 ms (stessi parametri di Pageot).

Input:  Pileup_data_RUN9/ (AP/, NPS/, filters/, .pos, BI_results.csv); il .bin si cerca sotto BIN_DIR.
Output: RUN9_results/BI_RUN9.csv (rp e BI di M, di Y ricalcolato e di Y dal suo CSV) e
        RUN9_results/discriminants_ch{ch}.npz (M e Y di ogni evento, dt e r dei pile-up) per i plot.
Controllo nel CSV: nps_ratio = NPS misurata sul rumore delle finestre / NPS di Pageot x 8/3 (mediana 1-500 Hz),
deve stare vicino a 1.

Sul server, dalla cartella del progetto (un job, pochi minuti):
    echo 'cd $PBS_O_WORKDIR && source /home/zanelli/LoadOctopus.sh && python analyse_RUN9.py' | qsub -q cupid -N RUN9 -j oe -l walltime=02:00:00 -l mem=6G
Controllo locale su un .bin sintetico: python test/run9/check_run9.py
"""
import os
import sys
import csv
import glob
import argparse

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import src.analysis as an
import src.dataset as ds
import utility.functions as fn
from src.pileup_likelihood import PileupLikelihood

BIN_DIR    = "/data2/LSC/DATA/RUN9"        # sul server; il .bin si cerca anche nelle sottocartelle
DATA_DIR   = os.path.join(BASE_DIR, "Pileup_data_RUN9")
OUT_DIR    = os.path.join(BASE_DIR, "RUN9_results")
MEAS       = "000813_20230628T161508"
CHANNELS   = [3, 5, 9, 11, 13, 15, 17, 19]
WINDOW     = 4096
FS         = 10_000
N_EVENTS   = 15_000
DT_MAX     = 8e-4
T_WINDOW   = DT_MAX + 1e-4
ACCEPTANCE = 0.9
BATCH      = 1000        # eventi per chiamata di M (memoria)


def pos_path(name, ch):
    return os.path.join(DATA_DIR, f"{name}_{MEAS}_{ch:03}_000_stdcut.pos")


def pageot_reference():
    """{canale: (channel_ID, rp, BI)} dal suo BI_results.csv, all'accettanza ACCEPTANCE."""
    with open(os.path.join(DATA_DIR, "BI_results.csv")) as f:
        return {int(r["channel"]): (int(r["channel_ID"]), float(r["rp"]), float(r["BI"]))
                for r in csv.DictReader(f) if abs(float(r["acceptance"]) - ACCEPTANCE) < 1e-9}


def run_channel(ch, bin_dir, n, out_dir):
    found = sorted(glob.glob(os.path.join(bin_dir, "**", f"{MEAS}_{ch:03}_000.bin"), recursive=True))
    if not found:
        raise SystemExit(f"manca {MEAS}_{ch:03}_000.bin sotto {bin_dir}")
    tpl = np.load(os.path.join(DATA_DIR, "AP", f"channel{ch}_meanpulse_mean.npy"))
    pulse = np.load(os.path.join(DATA_DIR, "AP", f"average_pulse_channel{ch}.npy"))
    nps = np.load(os.path.join(DATA_DIR, "NPS", f"NPS_channel{ch}_win{WINDOW}_{MEAS}.npy")) * 8 / 3
    f1, f2 = np.load(os.path.join(DATA_DIR, "filters", f"functions_eff90_channel{ch}.npy"))
    S, w, H = an.compute_H(tpl, nps, np.hanning, sampling_rate=FS)

    data = {k: ds.CachedBinaryDataset_withgenerated(found[0], pos_path(name, ch), WINDOW, pulse=pulse, n_windows=n)
            for k, name in (("single", "pup_n1-d0"), ("pileup", "pup_n1-d8"))}

    # NPS misurata sul solo rumore delle finestre dei singoli, contro la sua
    noise = ds.CachedBinaryDataset(found[0], WINDOW, (np.loadtxt(pos_path("pup_n1-d0", ch))[:n, 0] // 4).astype(int)).data.numpy()
    han = np.hanning(WINDOW)
    meas = np.mean(np.abs(np.fft.fft(noise * han, axis=1)) ** 2, axis=0) * WINDOW / np.sum(han ** 2)
    band = slice(int(np.ceil(WINDOW / FS)), int(500 * WINDOW / FS) + 1)                    # 1-500 Hz
    nps_ratio = float(np.median(meas[band] / nps[band]))
    del noise

    # M: alto = pile-up; il taglio accetta ACCEPTANCE dei singoli. Gli impulsi si preparano come per Y
    # (get_PSD_interpole_torch): via la baseline media del pre-trigger, poi Hann (S e' gia' finestrato).
    # Senza, l'offset in DC del .bin vero domina y(t) e M ~ 1e5 per tutti gli eventi.
    pre = WINDOW // 2 - 100
    prep = lambda x: (x - x[:, :pre].mean(axis=1, keepdims=True)) * han
    lik = PileupLikelihood(S, nps, w, dt_max=DT_MAX, t_window=T_WINDOW)
    M = {k: np.concatenate([lik.statistic(prep(d.data[i:i + BATCH].numpy())) for i in range(0, len(d), BATCH)])
         for k, d in data.items()}
    cut = np.percentile(M["single"], ACCEPTANCE * 100)
    rp_M = float(np.mean(M["pileup"] > cut))
    s_M, _ = an.compute_BI_uncertainty(-M["single"], -M["pileup"], ACCEPTANCE, rp_M)

    # Y di Pageot, chiamata come nel suo tutorial: basso = pile-up
    cf = lambda a: torch.tensor(a, dtype=torch.cfloat)
    _, rp_Y, Y_p, Y_s, *_ = an.compute_BI_torch(data["pileup"], data["single"], ACCEPTANCE, cf(H), cf(f1), cf(f2),
                                                window_fct=np.hanning, batch_size=2048, use_loader=False,
                                                full_output=True)
    Y = {"single": Y_s.numpy(), "pileup": Y_p.numpy()}
    rp_Y = float(rp_Y)
    s_Y, _ = an.compute_BI_uncertainty(Y["single"], Y["pileup"], ACCEPTANCE, rp_Y)

    pos = np.loadtxt(pos_path("pup_n1-d8", ch))[:n]
    np.savez(os.path.join(out_dir, f"discriminants_ch{ch}.npz"), M_single=M["single"], M_pileup=M["pileup"],
             Y_single=Y["single"], Y_pileup=Y["pileup"], dt=pos[:, -1] / FS,
             r=np.minimum(pos[:, 1], pos[:, 2]) / (pos[:, 1] + pos[:, 2]))
    return dict(nps_ratio=nps_ratio, n=len(M["single"]), rp_M=rp_M, sigma_rp_M=s_M, BI_M=fn.K * (1 - rp_M),
                rp_Y=rp_Y, sigma_rp_Y=s_Y, BI_Y=fn.K * (1 - rp_Y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=int, nargs="+", default=CHANNELS)
    ap.add_argument("--n", type=int, default=N_EVENTS, help="eventi per popolazione")
    ap.add_argument("--bin-dir", default=BIN_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    ref = pageot_reference()
    rows = []
    for ch in a.channels:
        res = run_channel(ch, a.bin_dir, a.n, a.out_dir)
        cid, rp_p, bi_p = ref.get(ch, (-1, np.nan, np.nan))
        rows.append(dict(channel=ch, channel_ID=cid, **res, rp_Y_pageot_csv=rp_p, BI_Y_pageot_csv=bi_p))
        print(f"ch{ch:2d}: rp  M {res['rp_M']:.4f}  Y ricalcolato {res['rp_Y']:.4f}  Y dal suo CSV {rp_p:.4f}"
              f"   (+- {res['sigma_rp_M']:.4f})   nps_ratio {res['nps_ratio']:.3f}", flush=True)
    path = os.path.join(a.out_dir, "BI_RUN9.csv")
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
