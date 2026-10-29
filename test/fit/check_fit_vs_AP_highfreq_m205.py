#!/usr/bin/env python3
"""
check_fit_vs_AP_highfreq_m205.py
================================
Il fit (residual_scan_bessel/fits_octopus) puo' avere piu' potenza ad alta frequenza dell'AP di
Octopus su cui e' stato fatto: nel tempo la differenza sta entro i residui, ma la risoluzione
temporale (beta, banda RMS pesata sulla NPS) sale e il BI del modello e del Monte Carlo, che usano
tutti e due il fit, scende senza che i dati lo giustifichino.

Figura 1 (fit_vs_AP_highfreq_overview_m205.png), tutti i punti:
  - beta(fit) / beta(AP di Octopus), con la NPS pulita del punto;
  - BI della tesi (0.8 ms) addestrato sull'AP di Octopus diviso quello addestrato sul fit.
Figura 2 (fit_vs_AP_highfreq_ch<CH>_m205.png): fronte di salita, differenza e spettro di potenza
del fit e dell'AP per i WP di DETAIL_WPS.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=. python3 test/fit/check_fit_vs_AP_highfreq_m205.py
"""
import os, sys, csv, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import src.analysis as an
import simulate_BI_error_m205 as mc

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHANNELS = [31, 34, 71, 83, 91]
WPS = list(range(1, 30, 2))
VBIAS = [0.6, 1.0, 1.4, 1.8, 2, 3, 4, 5, 6, 8, 10, 20, 26, 30, 40]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]
DETAIL_CH, DETAIL_WPS = 71, (1, 3, 5)
FS = 10_000
OUT = os.path.join(BASE, "comparisons")


def spectrum(tpl, nps):
    """S = FFT(template * Hann), come nell'addestramento (compute_H), e beta [Hz]."""
    S, w, _ = an.compute_H(tpl, nps, np.hanning, sampling_rate=FS)
    return S, float(an.compute_sigma_OF(S, nps)) / float(an.compute_sigma_OF(w * S, nps)) / (2 * np.pi)


def thesis_bi(folder):
    p = glob.glob(os.path.join(BASE, folder, "BI_results_*.csv"))[0]
    return {(int(r["channel"]), int(r["wp"])): float(r["BI"]) for r in csv.DictReader(open(p))}


def style(a):
    a.grid(True, color="0.9", lw=0.6)
    a.spines[["top", "right"]].set_visible(False)


def overview():
    bi_ap, bi_fit = thesis_bi("m205_results_octopus_npsclean"), thesis_bi("m205_results_octopus_fit_npsclean")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for ch, c, m in zip(CHANNELS, COLORS, MARKERS):
        ratio = []
        for wp in WPS:
            nps = mc.load_noise(ch, wp)
            ratio.append(spectrum(mc.template_pulse(ch, wp, "fit"), nps)[1] / spectrum(mc.template_pulse(ch, wp, "root"), nps)[1])
        ax[0].plot(VBIAS, ratio, color=c, marker=m, ms=6, lw=2, label=f"Ch {ch}")
        ax[1].plot(VBIAS, [bi_ap[(ch, wp)] / bi_fit[(ch, wp)] for wp in WPS], color=c, marker=m, ms=6, lw=2, label=f"Ch {ch}")
        print(f"ch{ch} beta fit/AP: " + " ".join(f"{r:.2f}" for r in ratio))
    for a in ax:
        a.axhspan(0.9, 1.1, color="0.92", zorder=0)
        a.axhline(1, color="0.5", lw=1)
        a.set(xscale="log", xlabel="V_bias [V]")
        style(a)
    ax[0].set(ylabel="beta(fit) / beta(Octopus AP)", title="time resolution of the template: fit vs measured AP")
    ax[1].set(ylabel="BI(trained on Octopus AP) / BI(trained on fit)", title="thesis campaign (0.8 ms), analytic BI")
    ax[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fit_vs_AP_highfreq_overview_m205.png"), dpi=150)


def detail():
    t_ms = (np.arange(10_000) - 5_000) / FS * 1e3
    f = np.fft.rfftfreq(10_000, 1 / FS)
    fig, ax = plt.subplots(3, len(DETAIL_WPS), figsize=(13, 8.6), sharex="row", sharey="row")
    for j, wp in enumerate(DETAIL_WPS):
        fit, ap, nps = mc.template_pulse(DETAIL_CH, wp, "fit"), mc.template_pulse(DETAIL_CH, wp, "root"), mc.load_noise(DETAIL_CH, wp)
        (Sf, bf), (Sa, ba) = spectrum(fit, nps), spectrum(ap, nps)
        rms = float(ap[:4000].std())
        ax[0, j].plot(t_ms, ap, "o", color="0.35", ms=4, label="Octopus AP (measured)")
        ax[0, j].plot(t_ms, fit, color=COLORS[0], lw=2, label="fit")
        ax[0, j].set(xlim=(-1, 3), title=f"Ch {DETAIL_CH} WP{wp} ({VBIAS[wp // 2]} V):  beta fit {bf:.0f} Hz, AP {ba:.0f} Hz")
        ax[1, j].axhspan(-rms * 1e3, rms * 1e3, color="0.9", label="AP noise (pretrigger RMS)")
        ax[1, j].plot(t_ms, (fit - ap) * 1e3, color=COLORS[0], lw=1.5, label="fit - AP")
        ax[1, j].set(xlim=(-1, 3), xlabel="time [ms]")
        ax[2, j].loglog(f[1:], np.abs(Sa[1:len(f)]) ** 2, color="0.35", lw=1, label="Octopus AP")
        ax[2, j].loglog(f[1:], np.abs(Sf[1:len(f)]) ** 2, color=COLORS[0], lw=1.5, label="fit")
        ax[2, j].set(xlim=(30, 5000), ylim=(1e-9, 2e4), xlabel="frequency [Hz]")
        for a in ax[:, j]:
            style(a)
    ax[0, 0].set_ylabel("normalised")
    ax[1, 0].set_ylabel("fit - AP  [x 1e-3]")
    ax[2, 0].set_ylabel("|S|^2  (Hann window)")
    for r in range(3):
        ax[r, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fit_vs_AP_highfreq_ch{DETAIL_CH}_m205.png"), dpi=150)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    overview()
    detail()
    print(f"-> {os.path.relpath(OUT, BASE)}/fit_vs_AP_highfreq_*.png")
