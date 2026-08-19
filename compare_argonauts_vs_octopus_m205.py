#!/usr/bin/env python3
"""
compare_argonauts_vs_octopus_m205.py
------------------------------------
Scatter comparison of the m205 average-pulse observables (Risetime, Decaytime,
HF-power) extracted with **Argonauts** vs the ones extracted with **Octopus**.

Argonauts gives ONE average pulse per channel, at a given working point (WP ->
V_bias); Octopus gives the full load curve. We therefore match, per channel, the
Argonauts point to the Octopus value at the SAME V_bias, and scatter Octopus (x)
against Argonauts (y). The dashed y=x line is perfect agreement.

Run with either python (no ROOT needed):
  KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/bin/python3.13 compare_argonauts_vs_octopus_m205.py
"""

import os, csv
import numpy as np
import matplotlib.pyplot as plt

# ── User settings ──────────────────────────────────────────────────────────────
BASE          = os.path.dirname(os.path.abspath(__file__))
ARGONAUTS_CSV = os.path.join(BASE, "AnalisiArgonauts_m205", "risetime_m205_argonauts_wp.csv")
OCTOPUS_CSV   = os.path.join(BASE, "amplitudes_m205.csv")
OUT_PNG       = os.path.join(BASE, "AnalisiArgonauts_m205", "argonauts_vs_octopus_m205.png")

VBIAS_TOL = 0.15   # V — max |V_bias| difference to consider two points "the same WP"

# Metrics to compare: (csv_column, short legend label)
METRICS = [
    ("risetime_ms",  "Risetime"),
    ("decaytime_ms", "Decaytime"),
    ("hf_power",     "HF-power"),
]


def read_csv(path):
    """Read a CSV into a list of dict rows (values kept as strings)."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


# ── Load data ────────────────────────────────────────────────────────────────
argo = read_csv(ARGONAUTS_CSV)
octo = read_csv(OCTOPUS_CSV)

# For each Argonauts point (channel @ V_bias), find the Octopus row of the same
# channel with the closest V_bias (within VBIAS_TOL).
matches = []   # (channel, vbias, {metric: (octo_val, argo_val)})
for a in argo:
    ch = int(a["channel"])
    vb = to_float(a["vbias_V"])
    cand = [o for o in octo if int(o["channel"]) == ch]
    if not cand:
        print(f"  [warn] channel {ch}: not found in Octopus CSV — skipped")
        continue
    o = min(cand, key=lambda r: abs(to_float(r["vbias_V"]) - vb))
    if abs(to_float(o["vbias_V"]) - vb) > VBIAS_TOL:
        print(f"  [warn] channel {ch} @ {vb} V: no Octopus point within {VBIAS_TOL} V — skipped")
        continue
    vals = {col: (to_float(o[col]), to_float(a[col])) for col, _ in METRICS}
    matches.append((ch, vb, vals))
    print(f"✓ Ch {ch:<3} @ {vb:>5} V  matched to Octopus @ {to_float(o['vbias_V']):>5} V")

if not matches:
    raise SystemExit("No channel could be matched between the two CSVs.")

# ── Relative percentage difference  100·(Argonauts − Octopus)/Octopus ──────────
matches.sort(key=lambda m: m[0])
channels = [ch for ch, _, _ in matches]
# pct[col] = list of % differences, aligned with `channels` (NaN if unavailable)
pct = {col: [] for col, _ in METRICS}
for ch, vb, vals in matches:
    for col, _ in METRICS:
        ox, ay = vals[col]
        pct[col].append(100.0 * (ay - ox) / ox if (ox not in (0.0,) and not np.isnan(ox) and not np.isnan(ay)) else np.nan)

# Console table
print("\nRelative difference  100·(Argonauts − Octopus)/Octopus  [%]")
head = "  ch   V_bias" + "".join(f"{lbl:>12}" for _, lbl in METRICS)
print(head); print("  " + "-" * (len(head) - 2))
for (ch, vb, _), i in zip(matches, range(len(matches))):
    row = f"  {ch:<4} {vb:>5} V "
    row += "".join(f"{pct[col][i]:>+11.2f}%" for col, _ in METRICS)
    print(row)

# ── Plot: grouped bars, one bar group per channel, one color per metric ───────
fig, ax = plt.subplots(figsize=(1.6 * len(channels) + 4, 5.2))
x = np.arange(len(channels))
n = len(METRICS)
width = 0.8 / n
colors = ["tab:blue", "tab:orange", "tab:green"]

for j, (col, lbl) in enumerate(METRICS):
    off = (j - (n - 1) / 2) * width
    vals = np.array(pct[col], dtype=float)
    bars = ax.bar(x + off, np.nan_to_num(vals), width, label=lbl,
                  color=colors[j % len(colors)], zorder=3)
    for xi, v in zip(x + off, vals):
        if np.isnan(v):
            continue
        ax.annotate(f"{v:+.1f}", (xi, v), textcoords="offset points",
                    xytext=(0, 3 if v >= 0 else -11), ha="center", fontsize=8)

ax.axhline(0, color="black", lw=1, zorder=2)
ax.set_xticks(x)
ax.set_xticklabels([f"ch{ch}\n{vb:g}V" for ch, vb, _ in matches])
ax.set_ylabel("Relative difference  (Argonauts − Octopus)/Octopus  [%]")
ax.set_title("m205 average pulse — Argonauts vs Octopus (matched per channel @ same V_bias)")
ax.grid(True, axis="y", ls=":", alpha=0.5)
ax.legend(loc="best", fontsize=9)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150)
print(f"\n✓ Figure saved to: {OUT_PNG}")
