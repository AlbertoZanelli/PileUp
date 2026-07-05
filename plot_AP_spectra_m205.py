"""
plot_AP_spectra_m205.py
=======================
Average-pulse power spectra for the m205 load curves: one panel (canvas) per
channel, overlaying the AP power spectrum of every working point (WP), colored
by V_bias. Same PSD definition used in the m204 study (peak-normalized AP,
Hann window). The dashed line marks the HF cutoff (500 Hz) used for HF-power.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/bin/python3.13 plot_AP_spectra_m205.py
"""

import os
import re
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
import uproot

BASE    = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(BASE, "Processed")
PATTERN = "Processed_*_000205_*.root"
OUTDIR  = os.path.join(BASE, "m205_results_octopus")
OUT_PNG = os.path.join(OUTDIR, "AP_power_spectra_m205.png")

CHANNELS = [31, 34, 71, 83, 91]
VBIAS_LIST = np.array([0.6, 1.0, 1.4, 1.8, 2, 3, 4, 5, 6, 8, 10, 20, 26, 30, 40])
HF_CUT_HZ = 500.0
HIST_TMPL = "averagepulse_ap_wp{wp}_medianAP"


def wp_to_vbias(wp: int) -> float:
    return float(VBIAS_LIST[wp // 2])


def compute_psd(signal, sampling_rate, window_fct=np.hanning):
    signal = np.asarray(signal, dtype=float)
    signal = signal - np.mean(signal)
    xw = signal * window_fct(len(signal))
    fft_vals = np.fft.rfft(xw)
    psd = (np.abs(fft_vals) ** 2) / (sampling_rate * len(signal))
    freq = np.fft.rfftfreq(len(signal), d=1.0 / sampling_rate)
    return freq, psd


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    files = {}
    for fp in glob.glob(os.path.join(DATADIR, PATTERN)):
        try:
            ch = int(os.path.basename(fp).split("_")[-1].replace(".root", ""))
        except ValueError:
            continue
        if ch in CHANNELS:
            files[ch] = fp

    norm = LogNorm(vmin=float(VBIAS_LIST.min()), vmax=float(VBIAS_LIST.max()))
    cmap = plt.get_cmap("viridis")

    ncols = 3
    nrows = int(np.ceil(len(CHANNELS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.6 * nrows), squeeze=False)
    axf = axes.ravel()
    fig.suptitle("Average-pulse power spectra vs working point — Measurement 000205",
                 fontsize=16, fontweight="bold")

    for ax, ch in zip(axf, CHANNELS):
        fp = files.get(ch)
        if fp is None:
            ax.set_title(f"Ch {ch}  (file not found)")
            continue
        with uproot.open(fp) as f:
            wps = sorted(set(
                int(m.group(1)) for k in f.keys()
                for m in [re.search(r"averagepulse_ap_wp(\d+)_medianAP", k)]
                if m and (int(m.group(1)) % 2 != 0)
            ))
            n_ok = 0
            for wp in wps:
                try:
                    h = f[HIST_TMPL.format(wp=wp)]
                    pulse = np.asarray(h.values(), dtype=float)
                    t_s = np.asarray(h.axis().centers(), dtype=float)
                except Exception:
                    continue
                if pulse.max() <= 0 or len(pulse) < 2:
                    continue
                pulse = pulse / pulse.max()
                sr = 1.0 / (t_s[1] - t_s[0])
                fr, ps = compute_psd(pulse, sr)
                ax.loglog(fr[1:], ps[1:], color=cmap(norm(wp_to_vbias(wp))),
                          lw=0.9, alpha=0.85)
                n_ok += 1
        ax.axvline(HF_CUT_HZ, color="crimson", ls="--", lw=1.0, alpha=0.7,
                   label="HF cut (500 Hz)")
        ax.set_title(f"Ch {ch}   ({n_ok} WPs)", fontsize=13)
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel("PSD [a.u.]")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9, loc="lower left")
        print(f"Ch {ch}: {n_ok} WP spectra")

    for ax in axf[len(CHANNELS):]:
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 0.94, 0.96])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cax = fig.add_axes([0.95, 0.12, 0.012, 0.76])
    fig.colorbar(sm, cax=cax, label=r"$V_{bias}$ (V)")
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)
    print(f"\n  -> {OUT_PNG}")


if __name__ == "__main__":
    main()
