#!/usr/bin/env python3
"""
plot_BI_vs_V_m205.py
====================
BI (modello analitico del training) contro V_bias per ogni canale: template dai fit time+freq
(APsimfit10000ledv2tf) contro quelli dai vecchi fit (APsimfit10000ledv2), filtro ottimo e Wiener + penalita' su s.

    KMP_DUPLICATE_LIB_OK=TRUE python3 plot_BI_vs_V_m205.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_K_rejection_m205 import load, BASE_DIR, BI_TARGET, CHANNELS

BLUE, ORANGE = "#2a78d6", "#eb6834"
OF, W = "m205_results_octopus_APsimfit10000{}_npsclean_dtrise", "m205_results_wiener_APsimfit10000{}_npsclean_swna1_dtrise"
SETS = [("optimum filter, old fits", OF.format("ledv2"), BLUE, ":", 1.3, 0),
        ("Wiener + s-penalty, old fits", W.format("ledv2"), ORANGE, ":", 1.3, 0),
        ("optimum filter, new fits", OF.format("ledv2tf"), BLUE, "-", 2, 5),
        ("Wiener + s-penalty, new fits", W.format("ledv2tf"), ORANGE, "-", 2, 5)]
OUT = os.path.join(BASE_DIR, "comparisons", "BI_vs_V_APsimfit10000ledv2tf-dtrise.png")


def main():
    data = {s[0]: load(s[1]) for s in SETS}
    fig, ax = plt.subplots(1, len(CHANNELS), figsize=(16, 4.6), sharex=True, sharey=True, gridspec_kw=dict(wspace=0.08))
    for a, ch in zip(ax, CHANNELS):
        for name, _, c, ls, lw, ms in SETS:
            vb, _, bi, _ = data[name][ch]
            a.plot(vb, bi, color=c, ls=ls, lw=lw, marker="o" if ms else None, ms=ms, label=name)
        a.axhline(BI_TARGET, color="0.45", ls="--", lw=1, label="CUPID pile-up target")
        a.set(xscale="log", yscale="log", xlabel="V_bias [V]", title=f"Ch {ch}")
        a.grid(True, which="major", color="0.9", lw=0.6)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("BI  [counts / (keV kg yr)]")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Pile-up BI vs bias: dt_max = rise time of the point, analytic model", y=1.08, fontsize=12)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"-> {os.path.relpath(OUT, BASE_DIR)}\n")
    print("  ch  V_bias   OF old    OF new   new/old |  W old     W new   new/old | W/OF new  [BI x 1e-5]")
    all_of, all_w, wof = [], [], []
    for ch in CHANNELS:
        vb, *_ = data[SETS[0][0]][ch]
        oo, wo, on, wn = (data[s[0]][ch][2] for s in SETS)
        all_of += list(on / oo); all_w += list(wn / wo); wof += list(wn / on)
        for i, v in enumerate(vb):
            print(f"  {ch} {v:6.1f}  {oo[i]*1e5:7.2f}  {on[i]*1e5:7.2f}  {on[i]/oo[i]:7.2f}  | {wo[i]*1e5:7.2f}  {wn[i]*1e5:7.2f}  "
                  f"{wn[i]/wo[i]:7.2f}  | {wn[i]/on[i]:6.3f}")
    print(f"\nnew/old OF: mediana {np.median(all_of):.3f}, min {min(all_of):.3f}, max {max(all_of):.3f}")
    print(f"new/old W : mediana {np.median(all_w):.3f}, min {min(all_w):.3f}, max {max(all_w):.3f}")
    print(f"W/OF nuovi: mediana {np.median(wof):.3f}, min {min(wof):.3f}, max {max(wof):.3f}")
    print("sotto il target: " + ", ".join(f"{s[0]}: {sum(int((data[s[0]][ch][2] < BI_TARGET).sum()) for ch in CHANNELS)}/75" for s in SETS))


if __name__ == "__main__":
    main()
