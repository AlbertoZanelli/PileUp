"""
analyze_BI_octo_vs_argo.py
==========================
Investigates WHY the Background Index (BI) estimated with the 'octopus' and
'argonauts' pipelines differ on run m204, and plots BI vs the other parameters.

Two questions are addressed:

  (1) What drives the BI difference between the two pipelines?
      Hypothesis: it is NOT the difference in the measured signal amplitude, but
      the residual NOISE contained in the octopus average pulse (AP). The AP is
      the template that builds S = FFT(meanpulse); noise leaking into it biases
      the optimal filter and therefore the BI.
      -> We quantify the residual noise of each AP as the RMS of the (peak-
         normalized) pre-trigger baseline, and correlate |Delta BI %| against
         this noise vs against the amplitude difference.

  (2) How does the BI depend on the other estimated parameters on run m204?
      -> BI vs SNR / amplitude / analytic sigma, both pipelines overlaid
         (single working point per channel, so one point per channel).

Inputs
------
  - BI CSVs produced by analyze_BI_singlerun.py:
        m204_results_octopus/BI_results_m204_octopus.csv
        m204_results_argonauts/BI_results_m204_argonauts.csv
    (columns: channel, signal_amp, sigma_analytic, SNR, BI, J_final)
  - Average pulses:
        octopus  : Processed/m204_AP/*_000204_<ch>_new.root  (hist medianAP)
        argonauts: NuoveAnalisiArgonauts_m204/*_<ch:03d>_000.bin_edmean.bin

Outputs (in --outdir, default m204_comparison/)
  - BI_diff_analysis_m204.csv       full per-channel table
  - BI_diff_drivers_m204.png        |Delta BI %| vs candidate drivers (+Pearson r)
  - AP_noise_overlay_m204.png       pre-trigger baseline of the noisiest channels
  - BI_vs_params_m204.png           BI vs SNR / amplitude / sigma (both pipelines)

Run:
    KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/bin/python3.13 analyze_BI_octo_vs_argo.py
"""

from __future__ import annotations

import os
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot


# ── Configuration ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OCTO_CSV = os.path.join(BASE_DIR, "m204_results_octopus",   "BI_results_m204_octopus.csv")
ARGO_CSV = os.path.join(BASE_DIR, "m204_results_argonauts", "BI_results_m204_argonauts.csv")

# Per-channel pulse-shape variables (risetime, freq_sigma) from
# risetime_and_amplitude_.py. Optional: merged in if present.
OCTO_RT_CSV = os.path.join(BASE_DIR, "Processed", "m204_AP",           "risetime_m204_octopus.csv")
ARGO_RT_CSV = os.path.join(BASE_DIR, "NuoveAnalisiArgonauts_m204",     "risetime_m204_argonauts.csv")

OCTO_AP_DIR = os.path.join(BASE_DIR, "Processed", "m204_AP")
ARGO_AP_DIR = os.path.join(BASE_DIR, "NuoveAnalisiArgonauts_m204")

OCTO_MEAS = "000204"
HIST_AP_NAME = "averagepulse_ap_medianAP"
ARGO_BIN_DTYPE = np.float64

# Channels excluded from the whole analysis (drivers, plots, CSV).
EXCLUDE_CHANNELS = [41]

# m204 acquisition sampling rate (window 800 -> 0.4 s, dt = 0.5 ms).
SAMPLING_RATE_HZ = 2000.0

# High-frequency cutoff (Hz) above which the average-pulse spectrum is pure noise:
# the pulse is low-pass, so past ~Nyquist/2 the signal PSD is negligible and any
# residual power there is noise leaked into the template. Integrating the AP PSD
# above this cutoff gives a single number for that noise. The choice f=500 Hz
# (= Nyquist/2) sits in the plateau where the |dBI| correlation is maximal.
F_CUT_HF = 500.0

# Pre-trigger fraction of the window: the peak sits at ~50% (pretrigger/windowlen
# = 0.2/0.4). We measure the noise on the flat baseline strictly BEFORE the rise,
# using a safety margin so the rising edge is never included.
PRETRIG_FRAC = 0.50
BASELINE_MARGIN = 0.75   # use the first 75% of the pre-trigger region


# ── AP loading (peak-normalized) ─────────────────────────────────────────────
def load_octo_ap(channel: int) -> np.ndarray:
    hits = glob.glob(os.path.join(OCTO_AP_DIR, f"*{OCTO_MEAS}_{channel}_new.root"))
    if not hits:
        raise FileNotFoundError(f"octopus AP not found for channel {channel}")
    with uproot.open(hits[0]) as f:
        ap = np.asarray(f[HIST_AP_NAME].values(), dtype=float)
    return ap


def load_argo_ap(channel: int) -> np.ndarray:
    hits = glob.glob(os.path.join(ARGO_AP_DIR, f"*_{channel:03d}_000.bin_edmean.bin"))
    if not hits:
        raise FileNotFoundError(f"argonauts AP not found for channel {channel}")
    return np.fromfile(hits[0], dtype=ARGO_BIN_DTYPE)


def compute_psd(signal, sampling_rate, window_fct=np.hanning):
    """Power spectral density of a waveform (windowed rFFT), user-provided form."""
    signal = np.asarray(signal, dtype=float)
    signal = signal - np.mean(signal)
    win = window_fct(len(signal))
    xw = signal * win
    fft_vals = np.fft.rfft(xw)
    psd = (np.abs(fft_vals) ** 2) / (sampling_rate * len(signal))
    freq = np.fft.rfftfreq(len(signal), d=1 / sampling_rate)
    return freq, psd


def baseline_rms(ap: np.ndarray) -> float:
    """
    Residual-noise proxy: RMS of the peak-normalized AP over the pre-trigger
    baseline (the flat region before the pulse rise). A clean template has an
    almost-zero baseline; leftover noise shows up as a non-negligible RMS.
    """
    ap = np.asarray(ap, dtype=float)
    peak = ap.max()
    if peak <= 0:
        return np.nan
    ap = ap / peak
    n_base = int(len(ap) * PRETRIG_FRAC * BASELINE_MARGIN)
    base = ap[:n_base]
    return float(base.std())


def hf_power(ap: np.ndarray, sampling_rate: float = SAMPLING_RATE_HZ,
             f_cut: float = F_CUT_HF) -> float:
    """
    High-frequency power of the average pulse: the AP power spectrum integrated
    above ``f_cut``. There the pulse (low-pass) has no signal left, so this is a
    direct, single-number measure of the noise leaked into the template. The AP
    is peak-normalized first, so the value is comparable across channels/pipelines.
    """
    ap = np.asarray(ap, dtype=float)
    peak = ap.max()
    if peak <= 0:
        return np.nan
    freq, psd = compute_psd(ap / peak, sampling_rate)
    return float(psd[freq >= f_cut].sum())


# ── Analysis ─────────────────────────────────────────────────────────────────
def build_table() -> pd.DataFrame:
    octo = pd.read_csv(OCTO_CSV).add_suffix("_octo").rename(columns={"channel_octo": "channel"})
    argo = pd.read_csv(ARGO_CSV).add_suffix("_argo").rename(columns={"channel_argo": "channel"})
    df = pd.merge(octo, argo, on="channel", how="inner").sort_values("channel").reset_index(drop=True)

    # Drop excluded channels up front, so nothing downstream loads or plots them.
    if EXCLUDE_CHANNELS:
        df = df[~df["channel"].isin(EXCLUDE_CHANNELS)].reset_index(drop=True)

    # Optional pulse-shape variables (risetime_ms, freq_sigma_Hz), one per pipeline.
    for path, suff in [(OCTO_RT_CSV, "octo"), (ARGO_RT_CSV, "argo")]:
        if os.path.exists(path):
            rt = pd.read_csv(path).add_suffix(f"_{suff}").rename(columns={f"channel_{suff}": "channel"})
            df = pd.merge(df, rt, on="channel", how="left")

    # Residual noise of each AP template: broadband time-domain RMS on the flat
    # baseline, and the frequency-domain high-frequency power (>F_CUT_HF).
    noise_o, noise_a, hf_o, hf_a = [], [], [], []
    for ch in df["channel"].astype(int):
        apo, apa = load_octo_ap(ch), load_argo_ap(ch)
        noise_o.append(baseline_rms(apo)); noise_a.append(baseline_rms(apa))
        hf_o.append(hf_power(apo));        hf_a.append(hf_power(apa))
    df["ap_noise_octo"] = noise_o
    df["ap_noise_argo"] = noise_a
    df["ap_noise_ratio"] = df["ap_noise_octo"] / df["ap_noise_argo"]
    df["ap_hfpow_octo"] = hf_o
    df["ap_hfpow_argo"] = hf_a

    # Relative differences (Argonauts vs Octopus, in %). Pulse-shape vars are
    # included only if their CSVs were merged in above.
    diff_vars = ["BI", "signal_amp", "sigma_analytic", "SNR"]
    for v in ["risetime_ms"]:
        if f"{v}_octo" in df.columns and f"{v}_argo" in df.columns:
            diff_vars.append(v)
    for var in diff_vars:
        df[f"{var}_diff_pct"] = (df[f"{var}_argo"] - df[f"{var}_octo"]) / df[f"{var}_octo"] * 100.0
    df["BI_absdiff_pct"] = df["BI_diff_pct"].abs()
    df["BI_ratio"] = df["BI_argo"] / df["BI_octo"]

    # Symmetric percentage difference of the AP residual noise. Unlike the raw
    # ratio (which explodes to >2e4% when octopus is far noisier), this is bounded
    # to [-200, +200] %, so the linear trend against BI_diff_pct stays visible.
    # Sign convention: positive => OCTOPUS is the noisier template.
    o, a = df["ap_noise_octo"], df["ap_noise_argo"]
    df["ap_noise_diff_pct"] = (o - a) / ((o + a) / 2.0) * 100.0
    ho, ha = df["ap_hfpow_octo"], df["ap_hfpow_argo"]
    df["ap_hfpow_diff_pct"] = (ho - ha) / ((ho + ha) / 2.0) * 100.0
    return df


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


# ── Plots ────────────────────────────────────────────────────────────────────
def _fit_line(ax, x, y):
    """Overlay a least-squares line and return its slope (linear panels only)."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return
    slope, intercept = np.polyfit(x[m], y[m], 1)
    xs = np.linspace(x[m].min(), x[m].max(), 100)
    ax.plot(xs, slope * xs + intercept, "k--", lw=1.2, alpha=0.7, zorder=2)


def plot_drivers(df: pd.DataFrame, path: str):
    """
    Test which pipeline difference explains the BI difference.

    Every panel compares the *relative change* of BI (BI_diff_pct) against the
    *relative change* of a candidate driver — always as a percentage difference,
    on a LINEAR scale, with a least-squares line and the Pearson r. The last panel
    ranks all drivers by |r| so the dominant one is obvious at a glance.
    """
    # Candidate drivers, all as signed percentage differences (Argo - Octo).
    candidates = [
        ("ap_hfpow_diff_pct",       r"$\Delta$ AP HF-power [%] (sym, >500 Hz)"),
        ("ap_noise_diff_pct",       r"$\Delta$ AP-noise RMS [%] (sym)"),
        ("sigma_analytic_diff_pct", r"$\Delta$ $\sigma_{OF}$ [%]"),
        ("risetime_ms_diff_pct",    r"$\Delta$ risetime [%]"),
        ("signal_amp_diff_pct",     r"$\Delta$ amplitude [%]"),
        ("SNR_diff_pct",            r"$\Delta$ SNR [%]"),
    ]
    panels = [(c, l) for c, l in candidates if c in df.columns]

    # Dynamic grid: one cell per driver + one ranking cell.
    n = len(panels) + 1
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    fig.suptitle("What drives the Octopus–Argonauts BI difference? (run m204)",
                 fontsize=16, fontweight="bold")
    axf = np.atleast_1d(axes).ravel()

    y = df["BI_diff_pct"].values
    rvals = []
    for ax, (col, label) in zip(axf, panels):
        x = df[col].values
        r = pearson(x, y)
        rvals.append((label, r))
        ax.scatter(x, y, s=80, c=df["channel"], cmap="tab10", zorder=3)
        for xi, yi, ch in zip(x, y, df["channel"].astype(int)):
            ax.annotate(str(ch), (xi, yi), xytext=(5, 4),
                        textcoords="offset points", fontsize=9)
        _fit_line(ax, x, y)
        ax.axhline(0, color="grey", lw=0.6)
        ax.axvline(0, color="grey", lw=0.6)
        ax.set_xlabel(label)
        ax.set_ylabel(r"$\Delta$ BI [%]  (Argo$-$Octo)")
        ax.set_title(f"Pearson r = {r:.2f}")
        ax.grid(True, alpha=0.3)

    # Ranking panel: |r| of every driver, sorted, in the remaining cell.
    ax = axf[len(panels)]
    rvals_sorted = sorted(rvals, key=lambda t: abs(t[1]))
    labels = [l.replace(r"$\Delta$ ", "").replace("[%]", "").strip() for l, _ in rvals_sorted]
    rs = [r for _, r in rvals_sorted]
    colors = ["seagreen" if r >= 0 else "indianred" for r in rs]
    ax.barh(labels, [abs(r) for r in rs], color=colors)
    for i, r in enumerate(rs):
        ax.annotate(f"{r:+.2f}", (abs(r), i), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"$|$Pearson r$|$ vs $\Delta$ BI [%]")
    ax.set_title("Driver ranking (green +, red −)")
    ax.grid(True, axis="x", alpha=0.3)

    # Hide any leftover empty axes.
    for ax in axf[len(panels) + 1:]:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[OK] {path}")


def plot_ap_overlay(df: pd.DataFrame, path: str, n_worst: int = 2):
    """Pre-trigger baseline of the AP for the channels with the largest BI diff."""
    worst = df.sort_values("BI_absdiff_pct", ascending=False).head(n_worst)
    fig, axes = plt.subplots(1, n_worst, figsize=(6 * n_worst, 4.5), squeeze=False)
    fig.suptitle("Residual noise in the average pulse (pre-trigger baseline, peak = 1)",
                 fontsize=14, fontweight="bold")
    for ax, (_, row) in zip(axes[0], worst.iterrows()):
        ch = int(row["channel"])
        apo = load_octo_ap(ch); apo = apo / apo.max()
        apa = load_argo_ap(ch); apa = apa / apa.max()
        n_base = int(len(apo) * PRETRIG_FRAC * BASELINE_MARGIN)
        ax.plot(apo[:n_base], color="steelblue", lw=1.0,
                label=f"Octopus (RMS={row['ap_noise_octo']:.1e})")
        ax.plot(apa[:n_base], color="darkorange", lw=1.0,
                label=f"Argonauts (RMS={row['ap_noise_argo']:.1e})")
        ax.axhline(0, color="k", lw=0.6, alpha=0.5)
        ax.set_title(f"Ch {ch}   |$\\Delta$BI| = {row['BI_absdiff_pct']:.0f}%")
        ax.set_xlabel("sample (pre-trigger)")
        ax.set_ylabel("normalized amplitude")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[OK] {path}")


def plot_bi_vs_params(df: pd.DataFrame, path: str):
    """BI vs SNR / amplitude / analytic sigma, both pipelines overlaid (run m204)."""
    params = [
        ("SNR", "SNR", "log"),
        ("signal_amp", "signal amplitude [V]", "log"),
        ("sigma_analytic", r"analytic $\sigma_{OF}$ [V]", "log"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("BI vs parameters — run m204 (one working point per channel)",
                 fontsize=15, fontweight="bold")
    for ax, (col, label, xscale) in zip(axes, params):
        for suff, name, color in [("octo", "Octopus", "steelblue"),
                                   ("argo", "Argonauts", "darkorange")]:
            xs = df[f"{col}_{suff}"].values
            ys = df[f"BI_{suff}"].values
            ax.scatter(xs, ys, s=70, color=color, label=name, zorder=3)
            for xi, yi, ch in zip(xs, ys, df["channel"].astype(int)):
                ax.annotate(str(ch), (xi, yi), xytext=(4, 3),
                            textcoords="offset points", fontsize=8, color=color)
        ax.set_xscale(xscale)
        ax.set_yscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("Background Index (BI)")
        ax.set_title(f"BI vs {label.split('[')[0].strip()}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[OK] {path}")


def _grid(n, ncols=3, panel=(5, 3.5)):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel[0] * ncols, panel[1] * nrows))
    return fig, np.atleast_1d(axes).ravel()


def plot_ap_all(df: pd.DataFrame, path: str):
    """Average pulse (time domain, peak-normalized) — Octopus vs Argonauts, all channels."""
    chans = df["channel"].astype(int).tolist()
    fig, axf = _grid(len(chans))
    fig.suptitle("Average pulse overlay — Octopus vs Argonauts (run m204, peak = 1)",
                 fontsize=15, fontweight="bold")
    for ax, ch in zip(axf, chans):
        apo = load_octo_ap(ch); apo = apo / apo.max()
        apa = load_argo_ap(ch); apa = apa / apa.max()
        t_o = np.arange(len(apo)) / SAMPLING_RATE_HZ * 1e3   # ms
        t_a = np.arange(len(apa)) / SAMPLING_RATE_HZ * 1e3
        ax.plot(t_o, apo, color="steelblue", lw=1.0, label="Octopus")
        ax.plot(t_a, apa, color="darkorange", lw=1.0, label="Argonauts")
        ax.set_title(f"Ch {ch}")
        ax.set_xlabel("time [ms]"); ax.set_ylabel("norm. amplitude")
        ax.grid(True, alpha=0.3)
    axf[0].legend(fontsize=9)
    for ax in axf[len(chans):]:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[OK] {path}")


def plot_ap_psd_all(df: pd.DataFrame, path: str):
    """Power spectrum of the average pulse (compute_psd) — Octopus vs Argonauts, all channels."""
    chans = df["channel"].astype(int).tolist()
    fig, axf = _grid(len(chans))
    fig.suptitle("Average-pulse power spectrum — Octopus vs Argonauts (run m204)",
                 fontsize=15, fontweight="bold")
    row_by_ch = {int(r["channel"]): r for _, r in df.iterrows()}
    for ax, ch in zip(axf, chans):
        apo = load_octo_ap(ch); apo = apo / apo.max()
        apa = load_argo_ap(ch); apa = apa / apa.max()
        f_o, psd_o = compute_psd(apo, SAMPLING_RATE_HZ)
        f_a, psd_a = compute_psd(apa, SAMPLING_RATE_HZ)
        ax.loglog(f_o[1:], psd_o[1:], color="steelblue", lw=1.0, label="Octopus")
        ax.loglog(f_a[1:], psd_a[1:], color="darkorange", lw=1.0, label="Argonauts")
        # Shade the pure-noise band and print its Octo-vs-Argo percentage excess.
        ax.axvspan(F_CUT_HF, SAMPLING_RATE_HZ / 2, color="grey", alpha=0.12)
        hf_pct = row_by_ch[ch].get("ap_hfpow_diff_pct", np.nan)
        ax.set_title(f"Ch {ch}   HF $\\Delta$={hf_pct:+.0f}%")
        ax.set_xlabel("frequency [Hz]"); ax.set_ylabel("PSD [a.u.]")
        ax.grid(True, which="both", alpha=0.3)
    axf[0].legend(fontsize=9)
    for ax in axf[len(chans):]:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[OK] {path}")


def main():
    p = argparse.ArgumentParser(description="Octopus vs Argonauts BI difference study (m204).")
    p.add_argument("--outdir", default=os.path.join(BASE_DIR, "m204_comparison"))
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = build_table()

    # Console summary.
    cols = ["channel", "BI_octo", "BI_argo", "BI_diff_pct",
            "ap_noise_octo", "ap_noise_argo", "signal_amp_diff_pct"]
    print("\n" + "=" * 92)
    print(df[cols].to_string(index=False,
          float_format=lambda v: f"{v:.3e}" if abs(v) < 1e-2 else f"{v:.2f}"))
    print("=" * 92)
    print("\nPearson r of Delta BI % (signed) vs percentage-difference drivers:")
    candidates = [("ap_hfpow_diff_pct", "AP HF-power diff [%] (>500Hz)"),
                  ("ap_noise_diff_pct", "AP-noise RMS diff [%] (sym)"),
                  ("sigma_analytic_diff_pct", "sigma_OF diff [%]"),
                  ("risetime_ms_diff_pct", "risetime diff [%]"),
                  ("signal_amp_diff_pct", "amplitude diff [%]"),
                  ("SNR_diff_pct", "SNR diff [%]")]
    ranking = [(lbl, pearson(df[col], df["BI_diff_pct"]))
               for col, lbl in candidates if col in df.columns]
    for lbl, r in sorted(ranking, key=lambda t: abs(t[1]), reverse=True):
        print(f"  {lbl:<30}: r = {r:+.3f}")
    print()

    df.to_csv(os.path.join(args.outdir, "BI_diff_analysis_m204.csv"), index=False)
    print(f"[OK] {os.path.join(args.outdir, 'BI_diff_analysis_m204.csv')}")

    plot_drivers(df, os.path.join(args.outdir, "BI_diff_drivers_m204.png"))
    plot_ap_overlay(df, os.path.join(args.outdir, "AP_noise_overlay_m204.png"))
    plot_bi_vs_params(df, os.path.join(args.outdir, "BI_vs_params_m204.png"))
    plot_ap_all(df, os.path.join(args.outdir, "AP_overlay_all_m204.png"))
    plot_ap_psd_all(df, os.path.join(args.outdir, "AP_PSD_overlay_all_m204.png"))


if __name__ == "__main__":
    main()
