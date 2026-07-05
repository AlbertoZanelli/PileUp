"""
analyze_timing_SNR_m205.py
==========================
Tests a pile-up figure of merit better than the amplitude SNR, on m205.

Motivation. The amplitude SNR = A/sigma_OF weights the noise by |S|^2/NPS but NOT
by frequency, so two working points with the same SNR but noise sitting at
different frequencies (relative to the pulse band ~1/risetime) are indistinguishable
to SNR — yet they reject pile-up differently. The Cramer-Rao bound for pulse
ARRIVAL TIME instead weights by (2*pi*f)^2:

    1/sigma_t^2 = A^2 * INT (2*pi*f)^2 |S~(f)|^2 / NPS(f) df

which factorizes as   1/sigma_t = SNR * beta ,  with the noise-weighted RMS
bandwidth (pulse "characteristic frequency")

    beta^2 = INT f^2 |S|^2/NPS df / INT |S|^2/NPS df   [Hz^2]   (NPS scale cancels).

Timing figure of merit (large => better pile-up rejection):
    rho_t = SNR * beta   [Hz]        sigma_t = 1/(2*pi*SNR*beta)   [s]

We compute beta per (channel, WP) from the average pulse and the noise spectrum
in the ROOT files, take SNR and BI from BI_results_m205.csv, and check whether
BI is a cleaner (monotonic) function of rho_t than of SNR — in particular whether
it straightens out the channels that were folded/non-monotonic in BI vs SNR.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/bin/python3.13 analyze_timing_SNR_m205.py
"""

import os
import re
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

BASE    = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import src.analysis as an   # compute_sigma_OF (the pipeline's own function)

DATADIR = os.path.join(BASE, "Processed")
PATTERN = "Processed_*_000205_*.root"
BI_CSV  = os.path.join(BASE, "m205_results_octopus", "BI_results_m205.csv")
OUTDIR  = os.path.join(BASE, "m205_results_octopus")

CHANNELS = [31, 34, 71, 83, 91]
VBIAS_LIST = np.array([0.6, 1.0, 1.4, 1.8, 2, 3, 4, 5, 6, 8, 10, 20, 26, 30, 40])
SAMPLING_RATE = 10_000.0   # m205: 10 kHz (window 10000 -> 1 s)
WINDOW_SIZE   = 10_000
SAMPLING_TIME = WINDOW_SIZE / SAMPLING_RATE
AP_HIST  = "averagepulse_ap_wp{wp}_medianAP"
NPS_HIST = "averagepowerspectrum_noise_wp{wp}_medianpower"


def wp_to_vbias(wp: int) -> float:
    return float(VBIAS_LIST[wp // 2])


def sigma_and_bandwidth(meanpulse, nps_hist_vals, sampling_rate=SAMPLING_RATE):
    """
    Returns (sigma_OF, sigma_mod, beta_Hz) using the pipeline function
    compute_sigma_OF from analysis.py, with NO window applied to the average
    pulse (the template is already ~zero at the window edges, so windowing only
    broadens the spectrum).

      sigma_OF  = compute_sigma_OF(S,      nps) = 1/sqrt( sum |S|^2 / N )
      sigma_mod = compute_sigma_OF(f * S,  nps) = 1/sqrt( sum f^2 |S|^2 / N )   (freq-weighted)
      beta_Hz   = sigma_OF / sigma_mod          = sqrt( sum f^2|S|^2/N / sum |S|^2/N )  [Hz]

    beta is the noise-weighted RMS bandwidth in Hz (using f, no 2*pi). It is a
    ratio, so the overall NPS/|S| scale cancels; only the spectral SHAPE matters.
    """
    N = len(meanpulse)
    nps = np.concatenate([nps_hist_vals, nps_hist_vals[-2:0:-1]])   # -> two-sided, len N
    nps = nps * 5.708 * (WINDOW_SIZE ** 2) * (1.0 / SAMPLING_TIME)  # pipeline normalization
    S = np.fft.fft(meanpulse)                                       # NO window
    freq = np.fft.fftfreq(N, d=1.0 / sampling_rate)                # Hz (two-sided)
    sigma_OF  = float(an.compute_sigma_OF(S, nps))                 # standard OF resolution
    sigma_mod = float(an.compute_sigma_OF(freq * S, nps))         # frequency-weighted resolution
    if not np.isfinite(sigma_mod) or sigma_mod <= 0:
        return sigma_OF, sigma_mod, np.nan
    return sigma_OF, sigma_mod, sigma_OF / sigma_mod               # beta [Hz]


def build_bandwidths() -> pd.DataFrame:
    rows = []
    files = {}
    for fp in glob.glob(os.path.join(DATADIR, PATTERN)):
        try:
            ch = int(os.path.basename(fp).split("_")[-1].replace(".root", ""))
        except ValueError:
            continue
        if ch in CHANNELS:
            files[ch] = fp
    import uproot
    for ch in CHANNELS:
        fp = files.get(ch)
        if not fp:
            print(f"[warn] no ROOT file for channel {ch}")
            continue
        with uproot.open(fp) as f:
            wps = sorted(set(int(m.group(1)) for k in f.keys()
                             for m in [re.search(r"averagepulse_ap_wp(\d+)_medianAP", k)]
                             if m and int(m.group(1)) % 2 != 0))
            for wp in wps:
                try:
                    mp = np.asarray(f[AP_HIST.format(wp=wp)].values(), dtype=float)
                    npv = np.asarray(f[NPS_HIST.format(wp=wp)].values(), dtype=float)
                except Exception:
                    continue
                sig_of, sig_mod, beta = sigma_and_bandwidth(mp, npv)
                rows.append({"channel": ch, "wp": wp, "vbias": round(wp_to_vbias(wp), 3),
                             "sigma_OF_nowin": sig_of, "sigma_mod_nowin": sig_mod,
                             "beta_Hz": beta})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    bw = build_bandwidths()
    bi = pd.read_csv(BI_CSV)
    bi = bi[bi["channel"].isin(CHANNELS)].copy()
    df = pd.merge(bi, bw[["channel", "wp", "beta_Hz"]], on=["channel", "wp"], how="inner")
    df = df.dropna(subset=["beta_Hz", "SNR", "BI"])

    # Timing figure of merit and time resolution.
    df["rho_t"] = df["SNR"] * df["beta_Hz"]                       # Hz, large => better
    df["sigma_t_ms"] = 1e3 / (2 * np.pi * df["rho_t"])           # ms
    df = df.sort_values(["channel", "vbias"]).reset_index(drop=True)

    df.to_csv(os.path.join(OUTDIR, "timing_SNR_m205.csv"), index=False)
    print(f"[OK] timing_SNR_m205.csv  ({len(df)} rows)\n")

    # ── monotonicity: |Spearman(BI, x)| per channel, for SNR vs rho_t ──────────
    print("Monotonicity of BI vs predictor  (|Spearman|, per channel; 1.0 = perfectly monotonic)")
    print(f"{'ch':>4} {'BI-vs-SNR':>11} {'BI-vs-beta':>11} {'BI-vs-rho_t':>12}")
    agg = {"SNR": [], "beta_Hz": [], "rho_t": []}
    for ch in CHANNELS:
        d = df[df["channel"] == ch]
        vals = {}
        for k in agg:
            r = abs(spearmanr(d["BI"], d[k]).statistic)
            agg[k].append(r); vals[k] = r
        print(f"{ch:>4} {vals['SNR']:>11.3f} {vals['beta_Hz']:>11.3f} {vals['rho_t']:>12.3f}")
    print(f"{'mean':>4} {np.mean(agg['SNR']):>11.3f} {np.mean(agg['beta_Hz']):>11.3f} "
          f"{np.mean(agg['rho_t']):>12.3f}")
    print("\n  If BI-vs-rho_t is ~1 for all channels while BI-vs-SNR is not, the")
    print("  frequency-weighted figure of merit removes the SNR non-monotonicity.\n")

    _plot(df, os.path.join(OUTDIR, "BI_vs_timing_SNR_m205.png"))
    print("Done.")


def _plot(df, out_png):
    cmap = plt.get_cmap("tab10")
    chs = sorted(df["channel"].unique())
    colors = {ch: cmap(i % 10) for i, ch in enumerate(chs)}
    panels = [
        ("SNR",     "SNR",                         False),
        ("beta_Hz", r"RMS bandwidth  $\beta$ (Hz)", False),
        ("rho_t",   r"timing FoM  $\rho_t=$SNR$\cdot\beta$ (Hz)", True),
        ("sigma_t_ms", r"time resolution  $\sigma_t$ (ms)", True),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2))
    fig.suptitle("A pile-up figure of merit beyond SNR — Measurement 000205",
                 fontsize=16, fontweight="bold")
    for ax, (col, xlabel, logx) in zip(axes, panels):
        for ch in chs:
            d = df[df["channel"] == ch].sort_values("vbias")
            ax.plot(d[col], d["BI"], marker="o", ms=4, lw=1.2,
                    color=colors[ch], label=f"Ch {ch}")
        if logx:
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Background Index (BI)", fontsize=11)
        ax.set_title(f"BI vs {xlabel.split('(')[0].split('$')[0].strip() or col}", fontsize=12)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"  -> {os.path.basename(out_png)}")


if __name__ == "__main__":
    main()
