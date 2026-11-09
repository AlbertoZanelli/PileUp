#!/usr/bin/env python3
"""
plot_rejection_vs_dt_m205.py
============================
Percentuale di pile-up rigettati in funzione del ritardo dt, per ogni WP: una griglia per canale (15 WP,
ordinati per V_bias). Barre = stimatore M, linee = le cartelle di training (OF, Wiener) sugli STESSI
eventi. Legge i DT_NAME scritti da simulate_BI_error_m205.py con DT_BINS > 0 (cartelle e seed della sua
configurazione attuale) e somma i conteggi di tutti i seed. Linea rossa: dt_max = rise time del punto,
la finestra del BI; oltre (barre chiare, fino a (1 + DT_EXTRA) dt_max) i pile-up simulati a parte, che non
entrano nel BI. L'R nel titolo e' quello in [0, dt_max]: coincide con l'R del Monte Carlo.

    KMP_DUPLICATE_LIB_OK=TRUE python3 plot_rejection_vs_dt_m205.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import simulate_BI_error_m205 as s

LABELS = {"likelihood": "M (likelihood ratio)", "optimum": "optimum filter", "wiener": "Wiener + s-penalty"}
LINES = {"optimum": ("#2a78d6", "-"), "wiener": ("#eb6834", "--")}


def load(folder_dir):
    """{(ch, wp): (vbias, dt_max [ms], dt_hi [ms], n, n_rej, n. seed)} sommati sui seed, o {} se il CSV non c'e'."""
    path = os.path.join(folder_dir, s.DT_NAME)
    if not os.path.exists(path):
        print(f"[WARN] manca {os.path.relpath(path, s.BASE_DIR)}")
        return {}
    df = pd.read_csv(path)
    df = df[df["gen"] == s.GEN_TEMPLATE]
    out = {}
    for (ch, wp), g in df.groupby(["channel", "wp"]):
        n = np.sum([np.array(x.split(), int) for x in g["n"]], axis=0)
        k = np.sum([np.array(x.split(), int) for x in g["n_rej"]], axis=0)
        out[(ch, wp)] = (float(g["vbias"].iloc[0]), 1e3 * float(g["dt_max"].iloc[0]), 1e3 * float(g["dt_hi"].iloc[0]),
                         n, k, len(g))
    return out


def main():
    M = load(s.likelihood_dir())
    others = {f["filter"]: load(f["dir"]) for f in s.FOLDERS}
    out_dir = s.likelihood_dir()
    for ch in sorted({c for c, _ in M}):
        wps = sorted((wp for c, wp in M if c == ch), key=lambda wp: M[(ch, wp)][0])
        fig, axes = plt.subplots(3, 5, figsize=(19, 10.5), sharey=True)
        for ax in axes.ravel()[len(wps):]:
            ax.set_axis_off()
        for ax, wp in zip(axes.ravel(), wps):
            vb, rise, hi, n, k, nseed = M[(ch, wp)]
            edges = np.linspace(0, hi, len(n) + 1)
            mid = 0.5 * (edges[1:] + edges[:-1])
            win = mid < rise                                    # bin dentro la finestra del BI
            frac = 100 * k / n
            ax.bar(edges[:-1], frac, width=np.diff(edges), align="edge", color=np.where(win, "#7fb0e3", "#cfe1f5"),
                   edgecolor="white", lw=0.5, label=LABELS["likelihood"])
            ax.errorbar(mid, frac, yerr=100 * np.sqrt(frac / 100 * (1 - frac / 100) / n), fmt="none", ecolor="0.35", lw=0.8)
            title = [f"M {100 * k[win].sum() / n[win].sum():.1f}%"]
            for filt, data in others.items():
                if (ch, wp) in data:
                    *_, n2, k2, _ = data[(ch, wp)]
                    c, ls = LINES.get(filt, ("0.3", ":"))
                    ax.stairs(100 * k2 / n2, edges, color=c, ls=ls, lw=1.6, label=LABELS.get(filt, filt))
                    title.append(f"{'OF' if filt == 'optimum' else 'W'} {100 * k2[win].sum() / n2[win].sum():.1f}%")
            ax.axvline(rise, color="#d62728", lw=2.2)
            ax.text(rise, 102, f" rise time\n {rise:.2f} ms", color="#d62728", fontsize=9, ha="left", va="bottom")
            ax.set_title(f"WP {wp}, V_bias = {vb:g} V\nR in [0, rise]: " + ", ".join(title), fontsize=10)
            ax.set(xlim=(0, hi), ylim=(0, 118))
            ax.spines[["top", "right"]].set_visible(False)
        for ax in axes[:, 0]:
            ax.set_ylabel("pile-ups rejected [%]")
        for ax in axes[-1]:
            ax.set_xlabel("delay dt [ms]")
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.01, 0.975), ncol=3, fontsize=10, frameon=False)
        fig.suptitle(f"Ch {ch}: pile-up rejection vs delay, 90% singles acceptance  "
                     f"(MC, {nseed} seed{'s' if nseed > 1 else ''}; dark: inside the BI window [0, rise time], "
                     f"light: pile-ups simulated beyond it, not in the BI)",
                     fontsize=12, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out = os.path.join(out_dir, f"rejection_vs_dt_ch{ch}_m205.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"-> {os.path.relpath(out, s.BASE_DIR)}")


if __name__ == "__main__":
    main()
