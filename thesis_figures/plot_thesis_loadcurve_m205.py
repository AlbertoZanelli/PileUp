r"""
plot_thesis_loadcurve_m205.py
=============================
Thesis figures for Chapter 4, "Load-Curve Data and Algorithm Inputs": the 5 channels used for
the BI (31, 34, 71, 83, 91), 15 WPs each. Nothing is recomputed, every number comes from the
files the analysis already uses:

  AP_ch<ch>          Octopus average pulse (medianAP in the Processed ROOT), all WPs, peak-
                     normalised, aligned at 50% of the rise; inset on the rising edge
  NPS_ch<ch>         clean NPS (m205_NPS_clean, the one used for training and MC), all WPs,
                     converted from E|FFT|^2 to a one-sided PSD in V^2/Hz
  load_curves        4-axis load curves vs I_bol: V_bol (Excel, load_curve_data_m205.txt),
                     amplitude at Q_bb, sigma_OF, SNR_OF (BI CSV); rows 2-2-1, full page
  vs_vbias_<var>     rise time, decay time (amplitudes_m205.csv, from the Octopus AP), amplitude
                     at Q_bb, sigma_OF, SNR_OF (BI CSV of the OF campaign, clean NPS)

PDF in OUTDIR (PNG would need dvipng, not installed). Sizes: SMALL for \includegraphics[width=0.48\textwidth]
(or 0.6 at 1:1); load_curves is a full-page \textwidth figure. Text 10 pt like the thesis body, LaTeX fonts.

Run:  KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/bin/python3.13 thesis_figures/plot_thesis_loadcurve_m205.py
"""

import os
import csv
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter, NullFormatter, LogLocator
import uproot

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project root (this script is in thesis_figures/)
OUTDIR   = os.path.join(BASE, "thesis_figures", "load_curve")
CHANNELS = [31, 34, 71, 83, 91]
WPS      = range(1, 30, 2)
VBIAS    = [0.6, 1.0, 1.4, 1.8, 2, 3, 4, 5, 6, 8, 10, 20, 26, 30, 40]   # indexed by WP // 2
FS       = 10_000.0                                                    # Hz
NPS_FILE = os.path.join(BASE, "m205_NPS_clean", "ch{ch}", "nps_ch{ch}_wp{wp}.npy")
BI_CSV   = os.path.join(BASE, "m205_results_octopus_APsimfit10000led_npsclean",
                        "BI_results_m205_APsimfit10000led_npsclean.csv")
AMP_CSV  = os.path.join(BASE, "amplitudes_m205.csv")
LC_TXT   = os.path.join(BASE, "m205_load_curves", "load_curve_data_m205.txt")
# Template for sigma_OF = 1/sqrt(sum |S|^2/NPS) and SNR_OF = A/sigma_OF in the figures:
#   "octopus"  = Octopus average pulse (the AP shown in AP_ch*), with the clean NPS
#   "campaign" = values of BI_CSV (simulated AP of the OF campaign, = fit to 1e-4)
SIGMA_TEMPLATE = "octopus"

AP_XLIM, AP_ZOOM = (-5, 80), (-1.5, 4)   # ms: main panel, inset on the rise
NPS_XLIM = (1, 5000)                     # Hz
SMALL = (3.4, 2.6)     # inches; thesis \textwidth = 5.71 in

# Categorical palette, slots 1-5 (1-3 = series1-3 of main.tex), plus a marker per channel so
# identity never rests on colour alone (print, colour blindness).
CH_COLOR = dict(zip(CHANNELS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
CH_MARK  = dict(zip(CHANNELS, "osD^v"))
# Sequential one-hue ramp for V_bias: light = low bias, dark = high bias.
VB_CMAP = LinearSegmentedColormap.from_list("vbias", ["#86b6ef", "#3987e5", "#256abf",
                                                       "#104281", "#0d366b"])
VB_NORM = LogNorm(min(VBIAS), max(VBIAS))
VB_TICKS = [0.6, 1, 2, 5, 10, 20, 40]

plt.rcParams.update({
    "text.usetex": True, "font.family": "serif", "font.size": 10,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def vbias(wp):
    return VBIAS[wp // 2]


def root_file(ch):
    return glob.glob(os.path.join(BASE, "Processed", f"Processed_*_000205_{ch}.root"))[0]


def save(fig, name):
    fig.savefig(os.path.join(OUTDIR, f"{name}.pdf"), dpi=300)   # dpi only for rasterized lines
    plt.close(fig)
    print(f"  -> {os.path.relpath(os.path.join(OUTDIR, name), BASE)}.pdf")


def vbias_colorbar(fig, ax):
    cb = fig.colorbar(ScalarMappable(VB_NORM, VB_CMAP), ax=ax, pad=0.02, aspect=25)
    cb.minorticks_off()
    cb.set_ticks(VB_TICKS)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    cb.ax.tick_params(direction="out", labelsize=8)
    cb.set_label(r"$V_{\rm bias}$ [V]")
    cb.outline.set_linewidth(0.6)


def channel_legend(ax):
    """One row above the axes: never on top of the data."""
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(CHANNELS),
              handlelength=1.3, handletextpad=0.4, columnspacing=0.9, borderaxespad=0.3)


def log_vbias_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(VB_TICKS)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"$V_{\rm bias}$ [V]")


def plot_ap(ch):
    fig, ax = plt.subplots(figsize=SMALL)
    ins = ax.inset_axes([0.47, 0.40, 0.50, 0.55])
    with uproot.open(root_file(ch)) as f:
        for wp in WPS:
            h = f[f"averagepulse_ap_wp{wp}_medianAP"]
            p = np.asarray(h.values(), dtype=float)
            t = np.asarray(h.axis().centers(), dtype=float)
            p /= p.max()
            t_ms = (t - t[np.argmax(p >= 0.5)]) * 1e3
            for a, (lo, hi) in ((ax, AP_XLIM), (ins, AP_ZOOM)):
                m = (t_ms >= lo) & (t_ms <= hi)
                a.plot(t_ms[m], p[m], color=VB_CMAP(VB_NORM(vbias(wp))), lw=0.8)
    ax.set_xlim(AP_XLIM)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel(r"Time [ms]")
    ax.set_ylabel("Normalized amplitude")
    ins.set_xlim(AP_ZOOM)
    ins.set_ylim(-0.03, 1.05)
    ins.tick_params(labelsize=7)
    ins.set_xlabel("Time [ms]", fontsize=7, labelpad=1)
    vbias_colorbar(fig, ax)
    save(fig, f"AP_ch{ch}")


def plot_nps(ch):
    fig, ax = plt.subplots(figsize=SMALL)
    for wp in WPS:
        nps = np.load(NPS_FILE.format(ch=ch, wp=wp))   # E|FFT(x)|^2, two-sided, x in V
        n = len(nps)
        k = np.arange(1, n // 2)
        psd = 2 * nps[k] / (n * FS)                    # one-sided PSD, V^2/Hz
        # the code's convention is var(x) = sum(nps)/n^2: the PSD must integrate to it
        assert abs(psd.sum() * FS / n / (nps[1:].sum() / n**2) - 1) < 0.01
        ax.loglog(k * FS / n, psd, color=VB_CMAP(VB_NORM(vbias(wp))), lw=0.5,
                  rasterized=True)
    ax.set_xlim(NPS_XLIM)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"NPS [V$^2$/Hz]")
    vbias_colorbar(fig, ax)
    save(fig, f"NPS_ch{ch}")


# 4-axis load-curve panel (style of plot_load_curve_full_m205.py): (key, label, factor, marker, colour)
LC_QTY = [("V_bol_V",        r"$V_{\rm bol}$ [mV]",             1e3, "o", "#2a78d6"),
          ("signal_amp",     r"$A(Q_{\beta\beta})$ [mV]",        1e3, "s", "#eb6834"),
          ("sigma_analytic", r"$\sigma_{\rm OF}$ [$\mu$V]",       1e6, "v", "#4a3aa7"),
          ("SNR",            r"SNR$_{\rm OF}$",                  1,   "D", "#1baf7a")]
LC_OFFSET = 0.42   # in, distance between the stacked right spines
LC_XTICKS = [0.3, 1, 3, 10, 20]   # nA; log axis: the bias points are log-spaced


def plot_load_curves(tab):
    """One panel per channel, x = I_bol (Excel, log), 4 y axes: V_bol, amplitude at Q_bb,
    sigma_OF, SNR_OF. Rows of 2, 2, 1 (the fifth centred), legend in one row on top."""
    left, w, h, col, row = 0.48, 1.55, 1.6, 3.3, 2.4   # in; ~6.6 in wide, scale to \\textwidth
    W, top = 2 * col, 0.45
    H = top + 2 * row + h + 0.45
    fig = plt.figure(figsize=(W, H))
    for n, ch in enumerate(CHANNELS):
        r, c = divmod(n, 2)
        x0 = left + (c * col if n < 4 else col / 2)
        host = fig.add_axes([x0 / W, (H - top - h - r * row) / H, w / W, h / H])
        v = [round(vbias(wp), 3) for wp in WPS]
        x = [float(tab[(ch, b)]["I_bol_A"]) * 1e9 for b in v]
        for j, (key, label, k, mk, color) in enumerate(LC_QTY):
            ax = host if j == 0 else host.twinx()
            if j > 1:
                ax.spines["right"].set_position(("outward", 72 * LC_OFFSET * (j - 1)))
            ax.plot(x, [float(tab[(ch, b)][key]) * k for b in v], "-" + mk, color=color,
                    lw=0.9, ms=3.2, mec="white", mew=0.4, label=label if n == 0 else None)
            ax.set_ylabel(label, color=color, fontsize=9, labelpad=1)
            ax.tick_params(axis="y", which="both", colors=color, labelsize=8, direction="out",
                           pad=1, right=j > 0, left=j == 0)
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
            ax.spines["right" if j else "left"].set_color(color)
            if j:
                for side in ("left", "top", "bottom"):   # keep only the host's frame
                    ax.spines[side].set_visible(False)
        host.set_xscale("log")
        host.set_xticks(LC_XTICKS)
        host.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        host.xaxis.set_minor_formatter(NullFormatter())
        host.tick_params(axis="x", which="both", labelsize=8, top=False)
        host.set_xlabel(r"$I_{\rm bol}$ [nA]", fontsize=9, labelpad=1)
        host.set_title(f"Ch {ch}", fontsize=10, loc="left", pad=3)
    fig.legend(loc="upper center", ncol=len(LC_QTY), fontsize=9, handlelength=2,
               columnspacing=1.5, bbox_to_anchor=(0.5, 1.0))
    save(fig, "load_curves")


def read_table():
    """(channel, V_bias) -> row of amplitudes_m205.csv + BI CSV + Excel load-curve data."""
    tab = {}
    for path, vcol in ((AMP_CSV, "vbias_V"), (BI_CSV, "vbias"), (LC_TXT, "bias_V")):
        for r in csv.DictReader(l for l in open(path) if not l.startswith("#")):
            tab.setdefault((int(r["channel"]), round(float(r[vcol]), 3)), {}).update(r)
    if SIGMA_TEMPLATE == "octopus":
        for ch in CHANNELS:
            with uproot.open(root_file(ch)) as f:
                for wp in WPS:
                    t = np.asarray(f[f"averagepulse_ap_wp{wp}_medianAP"].values(), float)
                    S = np.fft.fft(t / t.max() * np.hanning(len(t)))   # as src.analysis.compute_H
                    nps = np.load(NPS_FILE.format(ch=ch, wp=wp))
                    row = tab[(ch, round(vbias(wp), 3))]
                    row["sigma_analytic"] = 1 / np.sqrt(np.sum(np.abs(S) ** 2 / nps))
                    row["SNR"] = float(row["signal_amp"]) / row["sigma_analytic"]
    return tab


# (file suffix, y label, column, factor to the plotted unit, log y)
VARS = [
    ("risetime",  r"Rise time (10--90\%) [ms]",            "risetime_ms",    1,   False),
    ("decaytime", r"Decay time (90--10\%) [ms]",           "decaytime_ms",   1,   True),
    ("amplitude", r"Amplitude at $Q_{\beta\beta}$ [mV]",   "signal_amp",     1e3, True),
    ("sigma_OF",  r"$\sigma_{\rm OF}$ [$\mu$V]",           "sigma_analytic", 1e6, True),
    ("SNR_OF",    r"SNR$_{\rm OF} = A/\sigma_{\rm OF}$",   "SNR",            1,   True),
]


def plot_vs_vbias(tab):
    for name, ylabel, col, k, logy in VARS:
        fig, ax = plt.subplots(figsize=SMALL)
        for ch in CHANNELS:
            v = [round(vbias(wp), 3) for wp in WPS]
            ax.plot(v, [float(tab[(ch, x)][col]) * k for x in v], "-" + CH_MARK[ch],
                    color=CH_COLOR[ch], lw=1, ms=4, mec="white", mew=0.5, label=f"Ch {ch}")
        log_vbias_axis(ax)
        if logy:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel(ylabel)
        channel_legend(ax)
        save(fig, f"vs_vbias_{name}")


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    for ch in CHANNELS:
        plot_ap(ch)
        plot_nps(ch)
    tab = read_table()
    plot_load_curves(tab)
    plot_vs_vbias(tab)
