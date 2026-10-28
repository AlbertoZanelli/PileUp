"""
plot_APPS_templates_m205.py
===========================
Average-pulse power spectrum (APPS) |S|^2 of the three templates, overlaid, one panel per WP, one image
per channel: Octopus AP (medianAP of ~38 LED pulses), simulated AP APsimfit10000led, fit.
S = FFT(peak-normalised template x Hann), exactly as compute_H does in training. Diagnostic, not thesis.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 /opt/homebrew/bin/python3.13 test/fit/plot_APPS_templates_m205.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # project root (this script is in test/fit/)
import simulate_BI_error_m205 as mc

CHANNELS = [31, 34, 71, 83, 91]
WPS = range(1, 30, 2)
VBIAS = [0.6, 1.0, 1.4, 1.8, 2, 3, 4, 5, 6, 8, 10, 20, 26, 30, 40]
TEMPLATES = [("Octopus AP", ("root",), "0.55", 1.0),
             ("AP sim 10000 LED", ("sim", "APsimfit10000led"), "#eb6834", 0.8),
             ("fit", ("fit",), "#2a78d6", 0.8)]
OUTDIR = os.path.dirname(os.path.abspath(__file__))


def apps(src, ch, wp):
    p = mc.template_pulse(ch, wp, *src)
    return np.abs(np.fft.rfft(p * np.hanning(len(p)))) ** 2


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    f = np.fft.rfftfreq(mc.WINDOW_SIZE, 1 / mc.SAMPLING_RATE)
    for ch in CHANNELS:
        fig, axes = plt.subplots(5, 3, figsize=(14, 17), sharex=True)
        for ax, wp in zip(axes.ravel(), WPS):
            for lab, src, c, lw in TEMPLATES:
                ax.loglog(f[1:], apps(src, ch, wp)[1:], color=c, lw=lw, label=lab)
            ax.set_title(f"WP{wp}  ({VBIAS[wp // 2]:g} V)", fontsize=10)
            ax.set_xlim(1, mc.SAMPLING_RATE / 2)
            ax.grid(True, which="both", ls=":", alpha=0.3)
        for ax in axes[-1]:
            ax.set_xlabel("frequency [Hz]")
        for ax in axes[:, 0]:
            ax.set_ylabel(r"$|S|^2$ (peak-normalised pulse, Hann)")
        axes[0, 0].legend(fontsize=9, loc="lower left")
        fig.suptitle(f"m205 ch{ch}: average-pulse power spectrum of the three templates", y=0.995)
        fig.tight_layout()
        out = os.path.join(OUTDIR, f"APPS_templates_ch{ch}.png")
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"  -> {os.path.relpath(out, mc.BASE_DIR)}")


if __name__ == "__main__":
    main()
