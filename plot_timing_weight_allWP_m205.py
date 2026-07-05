"""
plot_timing_weight_allWP_m205.py
================================
Same spectral weights as plot_timing_weight_m205.py, but ONE canvas per channel
laid out as a table: a panel for every working point (WP). Each panel overlays
  - amplitude weight |S|^2/N          (blue,  what SNR / sigma_OF use)
  - timing weight (2*pi*f)^2 |S|^2/N   (red,   Cramer-Rao / rho_t)
with vertical markers at 1/(2*pi*tau_rise) and 1/tau_rise.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/bin/python3.13 plot_timing_weight_allWP_m205.py
"""

import os
import re
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import uproot

BASE    = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(BASE, "Processed")
PATTERN = "Processed_*_000205_*.root"
AMP_CSV = os.path.join(BASE, "m205_results_octopus", "amplitudes_m205.csv")
OUTDIR  = os.path.join(BASE, "m205_results_octopus")

CHANNELS = [31, 34, 71, 83, 91]
VBIAS_LIST = np.array([0.6, 1.0, 1.4, 1.8, 2, 3, 4, 5, 6, 8, 10, 20, 26, 30, 40])
SAMPLING_RATE = 10_000.0
AP_HIST  = "averagepulse_ap_wp{wp}_medianAP"
NPS_HIST = "averagepowerspectrum_noise_wp{wp}_medianpower"


def wp_to_vbias(wp):
    return float(VBIAS_LIST[wp // 2])


def weights(mp, nps):
    N = len(mp)
    S = np.fft.rfft(mp)   # no window
    freq = np.fft.rfftfreq(N, d=1.0 / SAMPLING_RATE)
    n = min(len(S), len(nps))
    S, nps, freq = S[:n], nps[:n], freq[:n]
    dens_amp = (np.abs(S) ** 2) / nps
    dens_time = (2 * np.pi * freq) ** 2 * dens_amp
    da = dens_amp / np.nanmax(dens_amp[1:])
    dt = dens_time / np.nanmax(dens_time[1:])
    return freq, da, dt


def main():
    amp = pd.read_csv(AMP_CSV)
    files = {}
    for fp in glob.glob(os.path.join(DATADIR, PATTERN)):
        try:
            ch = int(os.path.basename(fp).split("_")[-1].replace(".root", ""))
        except ValueError:
            continue
        if ch in CHANNELS:
            files[ch] = fp

    legend_handles = [
        Line2D([0], [0], color="steelblue", lw=1.6, label=r"amplitude $|S|^2/N$"),
        Line2D([0], [0], color="crimson", lw=1.6, label=r"timing $(2\pi f)^2|S|^2/N$"),
        Line2D([0], [0], color="green", ls="--", lw=1.1, label=r"$1/(2\pi\tau_r)$"),
        Line2D([0], [0], color="darkgreen", ls=":", lw=1.1, label=r"$1/\tau_r$"),
    ]

    for ch in CHANNELS:
        fp = files.get(ch)
        if not fp:
            print(f"[warn] no file for ch {ch}")
            continue
        with uproot.open(fp) as f:
            wps = sorted(set(int(m.group(1)) for k in f.keys()
                             for m in [re.search(r"averagepulse_ap_wp(\d+)_medianAP", k)]
                             if m and int(m.group(1)) % 2 != 0))
            data = []
            for wp in wps:
                try:
                    mp = np.asarray(f[AP_HIST.format(wp=wp)].values(), dtype=float)
                    nps = np.asarray(f[NPS_HIST.format(wp=wp)].values(), dtype=float)
                except Exception:
                    continue
                data.append((wp_to_vbias(wp), weights(mp, nps)))

        nplt = len(data)
        ncols = 5
        nrows = int(np.ceil(nplt / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows),
                                 squeeze=False)
        axf = axes.ravel()
        fig.suptitle(f"Ch {ch} — timing (red) vs amplitude (blue) spectral weight, all WPs — m205",
                     fontsize=15, fontweight="bold")
        for ax, (vb, (freq, da, dt)) in zip(axf, data):
            ax.plot(freq[1:], da[1:], color="steelblue", lw=1.2)
            ax.plot(freq[1:], dt[1:], color="crimson", lw=1.2)
            row = amp[(amp["channel"] == ch) & (np.round(amp["vbias_V"], 3) == round(vb, 3))]
            if not row.empty and pd.notna(row["risetime_ms"].iloc[0]):
                tau = row["risetime_ms"].iloc[0] * 1e-3
                ax.axvline(1.0 / (2 * np.pi * tau), color="green", ls="--", lw=1.0)
                ax.axvline(1.0 / tau, color="darkgreen", ls=":", lw=1.0)
            ax.set_xscale("log")
            ax.set_title(f"{vb:.1f} V", fontsize=11)
            ax.set_xlabel("freq [Hz]", fontsize=9)
            ax.set_ylabel("norm. weight", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(True, which="both", alpha=0.3)
        for ax in axf[nplt:]:
            ax.axis("off")
        fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=11, frameon=False)
        fig.tight_layout(rect=[0, 0.04, 1, 0.95])
        out = os.path.join(OUTDIR, f"timing_weight_ch{ch}_allWP_m205.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  -> {os.path.basename(out)}  ({nplt} WPs)")


if __name__ == "__main__":
    main()
