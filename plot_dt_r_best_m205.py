#!/usr/bin/env python3
"""
plot_dt_r_best_m205.py
======================
Dt e r veri dei pile-up simulati contro il (dt, r) migliore trovato da M (il termine piu' grande della somma,
src/pileup_likelihood.py, statistic(best=True)). Una griglia per canale e per grandezza, 15 WP ordinati per
V_bias, come plot_rejection_vs_dt_m205.py. Legge i file events_*.npz che simulate_BI_error_m205.py scrive con
SAVE_EVENTS = True (cartella e seed della sua configurazione attuale). Solo i pile-up che M etichetta come
pile-up (M > taglio al 90% di accettanza dei singoli, la frazione R nel titolo): per gli altri M preferisce
un singolo e il (dt, r) migliore non descrive niente (con dt = 0, r non conta e r = 0.5 viene dal prior).
Sotto ogni pannello, la frazione di TUTTI i pile-up simulati che M etichetta, per ogni bin del valore vero (le
stesse colonne del pannello sopra): il pannello dice dove cade la stima, la curva quanti eventi ci sono dietro.
Copie *_untagged_*: le stesse griglie con i soli pile-up che M NON etichetta (M <= taglio), curva = 100% - etichettati.

Ogni colonna dell'istogramma e' normalizzata a 1: il colore e' la distribuzione del valore stimato a valore
vero fissato. Linea scura: mediana della stima. dt: anche i pile-up simulati oltre dt_max (fino a
(1 + DT_EXTRA) dt_max, fuori dal BI): M li cerca solo fino a dt_max, quindi la stima satura li'.
r = frazione di energia del secondo impulso (quello in ritardo di dt), stessa convenzione del generatore.

    KMP_DUPLICATE_LIB_OK=TRUE python3 plot_dt_r_best_m205.py
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import simulate_BI_error_m205 as s

GRID = 1e-5             # passo in dt di M (PileupLikelihood, step) [s]
N_BINS = 30             # bin in dt fra 0 e dt_max, in ogni WP (multipli interi di GRID: con dt_max corto, 1 passo)
MIN_COL = 50            # colonne con meno eventi veri restano bianche
VMAX = {"dt": 0.4, "r": 0.8}   # scala dei colori, uguale in tutti i pannelli (oltre: colore pieno)
INK, MUTED, RED = "#222222", "#8a8a8a", "#d62728"


def load():
    """{canale: [eventi di ogni WP]}, con i WP ordinati per V_bias."""
    out = {}
    for f in sorted(glob.glob(os.path.join(s.likelihood_dir(), "events", f"events_ch*_seed*{s._V2}.npz"))):
        z = dict(np.load(f))
        for pop in ("pileup", "extra"):                   # pile-up etichettati da M
            if f"M_{pop}" in z:
                z[f"tag_{pop}"] = z[f"M_{pop}"] > z["M_cut"]
        z["n"], z["rp"] = len(z["M_pileup"]), z["tag_pileup"].mean()
        out.setdefault(int(z["channel"]), []).append(z)
    return {ch: sorted(v, key=lambda z: float(z["vbias"])) for ch, v in out.items()}


def medians(x, y, xe):
    """Mediana di y in ogni bin di x (nan sotto MIN_COL eventi)."""
    i = np.digitize(x, xe) - 1
    return np.array([np.median(y[i == k]) if np.sum(i == k) >= MIN_COL else np.nan for k in range(len(xe) - 1)])


def panel(ax, x, y, xe, ye, scale, vmax):
    """Istogramma 2D normalizzato per colonna, mediana di y per colonna, diagonale."""
    h, _, _ = np.histogram2d(x, y, (xe, ye))
    n = h.sum(axis=1, keepdims=True)
    h = np.where(n >= MIN_COL, h / np.maximum(n, 1), np.nan)
    im = ax.pcolormesh(xe * scale, ye * scale, h.T, cmap="Blues", vmin=0, vmax=vmax, rasterized=True)
    ax.plot(0.5 * (xe[1:] + xe[:-1]) * scale, medians(x, y, xe) * scale, color=INK, lw=1.6, label="median estimate")
    lim = max(xe[-1], ye[-1]) * scale
    ax.plot([0, lim], [0, lim], color=MUTED, ls="--", lw=1, label="estimate = true")
    ax.spines[["top", "right"]].set_visible(False)
    return im


def frac_curve(ax, x, sel, xe, scale):
    """Frazione dei pile-up selezionati (sel) in ogni bin del valore vero x [%]."""
    n, k = np.histogram(x, xe)[0], np.histogram(x[sel], xe)[0]
    f = np.where(n >= MIN_COL, 100 * k / np.maximum(n, 1), np.nan)
    ax.plot(0.5 * (xe[1:] + xe[:-1]) * scale, f, color=INK, lw=1.5)
    ax.set(ylim=(0, 105), yticks=[0, 50, 100])
    ax.grid(axis="y", color=MUTED, lw=0.5, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def grid(ch, evs, what, tagged=True):
    """tagged=True: i pile-up etichettati da M; False: quelli che M non etichetta."""
    word = "tagged" if tagged else "untagged"
    fig, axs = plt.subplots(6, 5, figsize=(19, 14), layout="constrained", height_ratios=[1, 0.28] * 3)
    axes, strips = axs[0::2], axs[1::2]                         # pannello e, sotto, la frazione di etichettati
    for a in (*axes.ravel()[len(evs):], *strips.ravel()[len(evs):]):
        a.set_axis_off()
    for ax, st, z in zip(axes.ravel(), strips.ravel(), evs):
        st.sharex(ax)
        dm = float(z["dt_max"])
        if what == "dt":
            x = np.concatenate([z["dt_pileup"], z.get("dt_extra", [])])
            y = np.concatenate([z["dt_hat_pileup"], z.get("dt_hat_extra", [])])
            tag = np.concatenate([z["tag_pileup"], z.get("tag_extra", np.zeros(0, bool))])
            hi, scale = x.max(), 1e3
            bw = max(1, round(dm / GRID / N_BINS)) * GRID              # ogni bin contiene lo stesso numero di valori di griglia
            xe = np.arange(0, hi + bw, bw)
            ye = np.arange(-0.5 * GRID, dm + bw, bw)
            sel = tag == tagged
            im = panel(ax, x[sel], y[sel], xe, ye, scale, VMAX[what])
            for a in (ax, st):
                a.axvline(dm * scale, color=RED, lw=1.8)
            ax.set(xlim=(0, hi * scale), ylim=(0, hi * scale))
            t = z["tag_pileup"] == tagged
            err = f"median |error| = {1e6 * np.median(np.abs(z['dt_hat_pileup'] - z['dt_pileup'])[t]):.0f} µs"
        else:
            x, y, tag, scale = z["r_pileup"], z["r_hat_pileup"], z["tag_pileup"], 1
            xe = np.linspace(0, 1, 21)
            sel = tag == tagged
            im = panel(ax, x[sel], y[sel], xe, np.linspace(0.025, 0.975, 20), scale, VMAX[what])
            ax.set(xlim=(0, 1), ylim=(0, 1))
            err = f"median |error| = {np.median(np.abs(y - x)[sel]):.2f}"
        frac_curve(st, x, sel, xe, scale)
        ax.tick_params(labelbottom=False)
        ax.set_title(f"WP {int(z['wp'])}, V_bias = {float(z['vbias']):g} V,  R = {100 * z['rp']:.1f}%\n{err}", fontsize=10)
    fig.colorbar(im, ax=axs, shrink=0.5, pad=0.01, label=f"fraction of the column ({word} pile-ups only)", extend="max")
    if what == "dt":
        lab = ("true delay Δt [ms]", "best Δt of M [ms]")
        note = (f"red line: dt_max = rise time, the BI window; beyond it the pile-ups simulated outside the BI, "
                f"where M saturates at dt_max. |error| in [0, dt_max]")
    else:
        lab = ("true r (energy fraction of the delayed pulse)", "best r of M")
        note = "pile-ups in [0, dt_max]; M's r grid: 0.05 ... 0.95 in steps of 0.05"
    for st in strips[-1]:
        st.set_xlabel(lab[0])
    for ax, st in zip(axes[:, 0], strips[:, 0]):
        ax.set_ylabel(lab[1])
        st.set_ylabel(f"{word} [%]")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=len(h), fontsize=10, frameon=False)
    fig.suptitle(f"Ch {ch}: true vs best {'Δt' if what == 'dt' else 'r'} of M, only the pile-ups {'' if tagged else 'NOT '}tagged by M "
                 f"(cut at 90% singles acceptance; R = tagged fraction of the {evs[0]['n']} per WP)\n{note}; "
                 f"lower panels: fraction of all simulated pile-ups {word} by M at that true value", fontsize=12, x=0.01, ha="left")
    out = os.path.join(s.likelihood_dir(), f"{what}_true_vs_best{'' if tagged else '_untagged'}_ch{ch}_m205.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {os.path.relpath(out, s.BASE_DIR)}")


def main():
    for ch, evs in load().items():
        for what in ("dt", "r"):
            for tagged in (True, False):
                grid(ch, evs, what, tagged)


if __name__ == "__main__":
    main()
