#!/usr/bin/env python3
"""
compare_BI_vs_hfpower_m205.py
-----------------------------
Scatter of the per-channel relative differences between Argonauts and Octopus:
  y = Delta BI        [%]  =  100 * (BI_argo   - BI_octo)   / BI_octo
  x = Delta AP HF-power [%] (symmetric) = 100 * (HF_argo - HF_octo) / ((HF_argo+HF_octo)/2)
one point per channel, labelled, with a least-squares dashed line and the
Pearson correlation coefficient in the title (same style as the m204 study).

BI values are hard-coded below (pasted table). HF-power (>500 Hz) comes from the
two CSVs, matched per channel at the SAME V_bias (Argonauts working point).

NB (m205): channels 71/83/91/94 have only the Argonauts noise spectrum
(*_spec.bin), NOT the average pulse (*_edmean.bin), so their AP HF-power is not
available; channel 41 has HF-power but no BI entry. The plot therefore uses the
channels present in BOTH sources (31, 34, 37, 40).

Run with either python (no ROOT needed):
  KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/bin/python3.13 compare_BI_vs_hfpower_m205.py
"""

import os, csv
import numpy as np
import matplotlib.pyplot as plt

# ── User settings ──────────────────────────────────────────────────────────────
BASE          = os.path.dirname(os.path.abspath(__file__))
ARGONAUTS_CSV = os.path.join(BASE, "AnalisiArgonauts_m205", "risetime_m205_argonauts_wp.csv")
OCTOPUS_CSV   = os.path.join(BASE, "amplitudes_m205.csv")
OUT_PNG       = os.path.join(BASE, "AnalisiArgonauts_m205", "BI_vs_hfpower_m205.png")

VBIAS_TOL = 0.15   # V — max |V_bias| difference to match Argonauts <-> Octopus

# BI (baseline index) per channel: {channel: (BI_argonauts, BI_octopus)}  [pasted table]
BI_DATA = {
    31: (6.806000e-05, 7.292000e-05),
    34: (8.736000e-05, 9.101000e-05),
    37: (7.652000e-05, 8.305000e-05),
    40: (7.520000e-05, 7.865000e-05),
    71: (6.269000e-05, 6.675000e-05),
    83: (7.490000e-05, 7.928000e-05),
    91: (1.470000e-04, 1.641000e-04),
    94: (7.168000e-05, 7.600000e-05),
}


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


# ── Argonauts vs Octopus HF-power per channel (matched at same V_bias) ─────────
argo = read_csv(ARGONAUTS_CSV)
octo = read_csv(OCTOPUS_CSV)

hf_argo, hf_octo = {}, {}   # channel -> HF-power
for a in argo:
    ch = int(a["channel"]); vb = to_float(a["vbias_V"])
    cand = [o for o in octo if int(o["channel"]) == ch]
    if not cand:
        continue
    o = min(cand, key=lambda r: abs(to_float(r["vbias_V"]) - vb))
    if abs(to_float(o["vbias_V"]) - vb) > VBIAS_TOL:
        continue
    hf_argo[ch] = to_float(a["hf_power"])
    hf_octo[ch] = to_float(o["hf_power"])

# ── Build the (Delta HF-power, Delta BI) points for the common channels ────────
rows = []   # (ch, dHF_sym, dBI_rel)
for ch in sorted(BI_DATA):
    if ch not in hf_argo:
        print(f"  [skip] ch{ch}: no Argonauts AP HF-power (no *_edmean.bin)")
        continue
    bi_a, bi_o = BI_DATA[ch]
    ha, ho = hf_argo[ch], hf_octo[ch]
    if np.isnan(ha) or np.isnan(ho) or (ha + ho) == 0:
        print(f"  [skip] ch{ch}: HF-power unavailable")
        continue
    dHF = 100.0 * (ha - ho) / ((ha + ho) / 2.0)   # symmetric %
    dBI = 100.0 * (bi_a - bi_o) / bi_o            # relative to Octopus %
    rows.append((ch, dHF, dBI))

if len(rows) < 2:
    raise SystemExit("Need at least 2 channels with both BI and HF-power to plot.")

# Console table
print("\n  ch |  Delta HF-power (sym) [%] |  Delta BI (Argo-Octo) [%]")
print("  ----+---------------------------+--------------------------")
for ch, dHF, dBI in rows:
    print(f"  {ch:<3} | {dHF:>+23.2f}   | {dBI:>+22.2f}")

chs = [r[0] for r in rows]
x   = np.array([r[1] for r in rows])
y   = np.array([r[2] for r in rows])

# ── Plot ─────────────────────────────────────────────────────────────────────
r_pearson = np.corrcoef(x, y)[0, 1]
cmap = plt.get_cmap("tab10")

fig, ax = plt.subplots(figsize=(6.4, 6.0))
for i, (ch, xi, yi) in enumerate(zip(chs, x, y)):
    ax.scatter(xi, yi, s=70, color=cmap(i % 10), zorder=3)
    ax.annotate(str(ch), (xi, yi), textcoords="offset points", xytext=(6, 4), fontsize=9)

# least-squares dashed line across the x-range
m, q = np.polyfit(x, y, 1)
xl = np.array([x.min(), x.max()])
pad = 0.15 * (xl[1] - xl[0] if xl[1] > xl[0] else 1.0)
xline = np.array([xl[0] - pad, xl[1] + pad])
ax.plot(xline, m * xline + q, ls="--", color="gray", lw=1, zorder=1)

ax.axhline(0, color="0.7", lw=0.8, zorder=0)
ax.axvline(0, color="0.7", lw=0.8, zorder=0)
ax.set_xlabel("$\\Delta$ AP HF-power [%] (sym, >500 Hz)")
ax.set_ylabel("$\\Delta$ BI [%] (Argo$-$Octo)")
ax.set_title(f"Pearson r = {r_pearson:.2f}")
ax.grid(True, ls=":", alpha=0.5)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150)
print(f"\n✓ Figure saved to: {OUT_PNG}")
