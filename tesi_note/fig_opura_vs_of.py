"""
fig_opura_vs_of.py
==================
Figura e tabella della sezione Risultati di stimatore_verosimiglianza.tex: OPURA contro l'algoritmo
con i filtri (OF, addestrato), sugli STESSI eventi del Monte Carlo a 50 seed.

Per ogni punto con tutti i seed: Delta BI = mediana sui seed di (BI_OPURA / BI_OF - 1), con errore
la LARGHEZZA (P84 - P16)/2 della distribuzione dei Delta BI sui seed, cioe' l'incertezza di una
simulazione da 50 000 eventi (non divisa per sqrt(n)). Scrive fig_opura_vs_of.pdf e stampa le righe LaTeX della tabella.

    python tesi_note/fig_opura_vs_of.py
"""
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
OF = "m205_results_octopus_APsimfit10000led_npsclean"
OPURA = "m205_results_likelihood_APsimfit10000led_npsclean"
MC_CSV, GEN, N_SEEDS = "BI_mc_error_m205_seeds50.csv", "fit", 50
COLORS = {31: "#2a78d6", 34: "#eb6834", 71: "#1baf7a", 83: "#eda100", 91: "#e87ba4"}


def load(folder):
    d = defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(BASE, folder, MC_CSV))):
        if r["gen"] == GEN:
            d[(int(r["channel"]), int(r["wp"]))][int(r["seed"])] = float(r["BI_mc"])
    return d


def main():
    of, op = load(OF), load(OPURA)
    train = {(int(r["channel"]), int(r["wp"])): r
             for r in csv.DictReader(open(os.path.join(BASE, OF, f"BI_results_m205_{OF[len('m205_results_octopus_'):]}.csv")))}
    rows = []
    for p in sorted(of):
        seeds = sorted(of[p].keys() & op[p].keys())
        if len(seeds) < N_SEEDS:
            continue                                  # punto non ancora completo
        a = np.array([of[p][s] for s in seeds]); b = np.array([op[p][s] for s in seeds])
        d = 100 * (b / a - 1)
        w = (np.percentile(d, 84.13) - np.percentile(d, 15.87)) / 2
        rows.append(dict(ch=p[0], wp=p[1], vb=float(train[p]["vbias"]), snr=float(train[p]["SNR"]),
                         of=np.median(a), op=np.median(b), d=np.median(d),
                         e=np.sqrt(np.pi / 2) * w / np.sqrt(len(seeds)), w=w))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw=dict(wspace=0.25))
    for ch in sorted({r["ch"] for r in rows}):
        R = [r for r in rows if r["ch"] == ch]
        for a, x in ((ax[0], "vb"), (ax[1], "snr")):
            a.errorbar([r[x] for r in R], [r["d"] for r in R], yerr=[r["w"] for r in R], fmt="o-" if x == "vb" else "o",
                       color=COLORS[ch], ms=6, lw=1.5, capsize=2, label=f"ch {ch}")
    for a, lab in ((ax[0], "bias voltage [V]"), (ax[1], "optimum-filter SNR")):
        a.axhline(0, color="0.4", lw=1)
        a.set_xscale("log")
        a.set(xlabel=lab, ylabel=r"$\Delta$BI  OPURA vs OF  [%]")
        a.grid(alpha=0.25)
    ax[1].set_xticks([20, 30, 50, 100, 150]); ax[1].set_xticklabels(["20", "30", "50", "100", "150"])
    ax[1].minorticks_off()
    ax[0].set_title("(a) per working point")
    ax[1].set_title("(b) against signal-to-noise ratio")
    ax[0].legend(frameon=False)
    out = os.path.join(HERE, "fig_opura_vs_of.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"[OK] {out}  ({len(rows)} punti completi)\n")

    for r in rows:
        print(f"{r['ch']} & {r['wp']} & {r['vb']:g} & {r['snr']:.0f} & {r['of']*1e5:.2f} & {r['op']*1e5:.2f} & "
              f"${r['d']:+.2f} \\pm {r['w']:.2f}$ \\\\")
    D = np.array([r["d"] for r in rows]); S = np.array([r["snr"] for r in rows])
    print(f"\nmediana {np.median(D):+.2f}%, OPURA meglio in {np.sum(D < 0)}/{len(rows)}")
    for ch in sorted({r["ch"] for r in rows}):
        R = [r for r in rows if r["ch"] == ch]; d = np.array([r["d"] for r in R])
        print(f"ch{ch}: {len(R)} punti, mediana {np.median(d):+.2f}%, da {d.min():+.2f} a {d.max():+.2f}, "
              f"meglio {np.sum(d < 0)}/{len(R)}, meglio oltre 2 larghezze {sum(r['d'] < -2 * r['w'] for r in R)}, "
              f"oltre 1 {sum(r['d'] < -r['w'] for r in R)}; peggio oltre 1 larghezza {sum(r['d'] > r['w'] for r in R)}")
    for lo, hi in ((0, 30), (30, 60), (60, 1e9)):
        k = (S >= lo) & (S < hi)
        print(f"SNR {lo:g}-{hi:g}: {k.sum()} punti, mediana {np.median(D[k]):+.2f}%")


if __name__ == "__main__":
    main()
