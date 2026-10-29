#!/usr/bin/env python3
"""
plot_K_rejection_m205.py
========================
Per ogni canale, al variare di V_bias, dai risultati del training (BI_results_*.csv, BI = J_final * K):
  - K: la frequenza dei pile-up con ritardo in [0, dt_max], cioe' il BI senza nessun rigetto.
    Con dt_max = rise time del punto (cartelle "_dtrise") segue il rise time: K = K(0.8 ms) * dt_max / 0.8 ms;
  - BI = K (1 - R);
  - R = 1 - J_final: la probabilita' di rigetto dei pile-up (media su ritardo e ripartizione dell'energia).
Sopra K e BI sullo stesso asse logaritmico (la distanza fra le due curve e' -log(1 - R)), poi R, in
fondo il rise time 10-90% del punto (= dt_max, da amplitudes_m205.csv).

    KMP_DUPLICATE_LIB_OK=TRUE python3 plot_K_rejection_m205.py
"""
import os, csv, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SETS = {"optimum filter": ("m205_results_octopus_APsimfit10000ledv2_npsclean_dtrise", "#2a78d6", "-"),
        "Wiener + s-penalty": ("m205_results_wiener_APsimfit10000ledv2_npsclean_swna1_dtrise", "#eb6834", "--")}
CHANNELS = [31, 34, 71, 83, 91]
BI_TARGET = 5e-5             # target CUPID sul pile-up
K_COLOR = "#5f5e5a"
OUT = os.path.join(BASE_DIR, "comparisons", "K_rejection_APsimfit10000ledv2-dtrise.png")


def load(folder):
    """{canale: (V_bias, K, BI, R)} ordinati per V_bias, dal CSV del training della cartella."""
    rows = list(csv.DictReader(open(glob.glob(os.path.join(BASE_DIR, folder, "BI_results_*.csv"))[0])))
    out = {}
    for ch in CHANNELS:
        r = sorted((float(x["vbias"]), float(x["BI"]), float(x["J_final"])) for x in rows if int(x["channel"]) == ch)
        vb, bi, j = map(np.array, zip(*r))
        out[ch] = (vb, bi / j, bi, 1 - j)
    return out


def risetimes():
    """{canale: (V_bias, rise time [ms])} da amplitudes_m205.csv: e' il dt_max dei punti."""
    rows = [r for r in csv.DictReader(open(os.path.join(BASE_DIR, "amplitudes_m205.csv"))) if r["risetime_ms"]]
    return {ch: tuple(map(np.array, zip(*sorted((float(r["vbias_V"]), float(r["risetime_ms"]))
                                                  for r in rows if int(r["channel"]) == ch)))) for ch in CHANNELS}


def main():
    data = {name: load(folder) for name, (folder, _, _) in SETS.items()}
    rt = risetimes()
    fig, ax = plt.subplots(3, len(CHANNELS), figsize=(16, 8.4), sharex=True, sharey="row",
                           gridspec_kw=dict(height_ratios=[1.3, 1, 0.8], hspace=0.08, wspace=0.08))
    for j, ch in enumerate(CHANNELS):
        top, bot, low = ax[0, j], ax[1, j], ax[2, j]
        low.plot(*rt[ch], color=K_COLOR, lw=2, marker="s", ms=5, label="rise time 10-90% = dt_max")
        low.axhline(0.8, color="0.55", ls=":", lw=1.2, label="0.8 ms (thesis)")
        low.set(xscale="log", xlabel="V_bias [V]")
        vb, K, _, _ = next(iter(data.values()))[ch]
        top.plot(vb, K, color=K_COLOR, lw=2, marker="s", ms=5, label="K  (BI with no rejection)")
        for name, (_, c, ls) in SETS.items():
            vb, _, bi, R = data[name][ch]
            top.plot(vb, bi, color=c, lw=2, ls=ls, marker="o", ms=4, label=f"BI = K (1 - R), {name}")
            bot.plot(vb, 100 * R, color=c, lw=2, ls=ls, marker="o", ms=4, label=f"R, {name}")
        top.axhline(BI_TARGET, color="0.55", ls=":", lw=1.2, label="CUPID pile-up target")
        top.set(yscale="log", title=f"Ch {ch}")
        for a in (top, bot, low):
            a.grid(True, which="major", color="0.9", lw=0.6)
            a.spines[["top", "right"]].set_visible(False)
    ax[0, 0].yaxis.set_major_locator(matplotlib.ticker.FixedLocator([5e-5, 1e-4, 2e-4, 4e-4, 8e-4]))
    ax[0, 0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 1e4:g}"))
    ax[0, 0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0, 0].set_ylabel("counts / (keV kg yr)   [x 1e-4]")
    ax[1, 0].set_ylabel("pile-up rejection R  [%]")
    ax[2, 0].set_ylabel("rise time [ms]")
    h, l = zip(*[hl for a in ax[:, 0] for hl in zip(*a.get_legend_handles_labels())])
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("K, BI and pile-up rejection vs bias  —  dt_max = rise time of the point, analytic model, "
                 "template APsimfit10000ledv2", y=1.06, fontsize=12)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"-> {os.path.relpath(OUT, BASE_DIR)}")
    for name in SETS:
        print(f"\n{name}:\n  ch   K [1e-4]          R [%]            BI [1e-5]")
        for ch in CHANNELS:
            _, K, bi, R = data[name][ch]
            print(f"  {ch}  {K.min() * 1e4:4.2f} - {K.max() * 1e4:4.2f}    {100 * R.min():4.1f} - {100 * R.max():4.1f}    "
                  f"{bi.min() * 1e5:5.2f} - {bi.max() * 1e5:5.2f}")


if __name__ == "__main__":
    main()
