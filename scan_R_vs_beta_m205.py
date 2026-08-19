#!/usr/bin/env python3
"""R(f) al variare di beta, per (canale, WP) di m205.
R = Sabove/(Sabove + beta*ANPS/N + eps),  Sabove = max(APPS - beta*ANPS/N, 0).

  KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/bin/python3.13 scan_R_vs_beta_m205.py
"""
import os, glob, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import uproot

BASE = os.path.dirname(os.path.abspath(__file__))   # path relativi allo script, non alla cwd
WP, N, FS = 15, 38, 10_000.0
CHANNEL = 91
EPS_FRAC = 1e-6
VBIAS = np.array([0.6, 1.0, 1.4, 1.8, 2, 3, 4, 5, 6, 8, 10, 20, 26, 30, 40])


def R(apps, tnoise, beta, eps):
    thr = beta * tnoise
    s = np.maximum(apps - thr, 0.0)
    return s / (s + thr + eps)


def amp_V(ch, vb):
    for r in csv.DictReader(open(os.path.join(BASE, "amplitudes_m205.csv"))):
        if int(float(r["channel"])) == ch and abs(float(r["vbias_V"]) - vb) < 0.15:
            return float(r["amplitude_mV"]) * 1e-3
    raise SystemExit(f"no amplitude for ch{ch} @ {vb} V")


def apps_anps(ch):
    f = uproot.open(glob.glob(os.path.join(BASE, f"Processed/Processed_*_000205_{ch}.root"))[0])
    ap = np.asarray(f[f"averagepulse_ap_wp{WP}_medianAP"].values(), float)
    nps = np.asarray(f[f"averagepowerspectrum_noise_wp{WP}_medianpower"].values(), float) * 5.708
    x = ap * amp_V(ch, VBIAS[WP // 2]); n = len(x)
    apps = (np.abs(np.fft.rfft(x)) ** 2) / (n * n) / (FS / n)
    apps[1:] *= 2
    if n % 2 == 0:
        apps[-1] /= 2
    return np.fft.rfftfreq(n, 1 / FS), apps, nps          # freq, APPS, ANPS (stessa griglia)


assert abs(R(np.array([2.0]), np.array([1.0]), 1.0, 0.0)[0] - 0.5) < 1e-9  # SNR=2β -> R=0.5

BETAS = np.linspace(0.01, 10, 12)                        # 12 pannelli -> griglia 3x4
freq, apps, nps = apps_anps(CHANNEL)
tnoise = nps / N
eps = EPS_FRAC * float(np.nanmax(apps))

fig, axes = plt.subplots(3, 4, figsize=(18, 11), sharex=True)
ncol = axes.shape[1]
for i, (ax, b) in enumerate(zip(axes.ravel(), BETAS)):
    col = i % ncol
    ax.loglog(freq[1:], apps[1:], color="tab:blue", lw=1.2, alpha=0.7, label="APPS")
    ax.loglog(freq[1:], nps[1:], color="0.4", lw=1.0, alpha=0.5, label="ANPS")
    ax.set_xlim(1, FS / 2); ax.grid(True, which="both", ls="--", alpha=0.3)
    axr = ax.twinx()                                     # R su asse lineare 0..1
    axr.semilogx(freq[1:], R(apps, tnoise, b, eps)[1:], color="tab:green", lw=1.4, alpha=0.9)
    axr.set_ylim(-0.03, 1.05)
    ax.set_title(fr"$\beta$ = {b:.2f}")
    if col != 0:                                         # PSD (sx) solo sulla prima colonna
        ax.tick_params(labelleft=False)
    else:
        ax.set_ylabel(r"PSD (V$^2$/Hz)")
    if col != ncol - 1:                                  # R (dx) solo sull'ultima colonna
        axr.tick_params(labelright=False)
    else:
        axr.set_ylabel("R(f)", color="tab:green")
for ax in axes[-1]:
    ax.set_xlabel("Frequency (Hz)")
axes[0, 0].legend(fontsize=8, loc="lower left")
fig.suptitle(f"R(f) (green) vs $\\beta$, con APPS e ANPS  —  m205 Ch{CHANNEL} WP{WP}, N={N}",
             fontweight="bold")
out = os.path.join(BASE, "R_vs_beta_m205.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("✓", out)
