"""
compare_wiener_vs_optimum_m205.py
=================================
Confronta DUE set di risultati BI (m205) qualsiasi, punto per punto (canale,
V_bias). Generico: "A" e' il riferimento (baseline), "B" e' il risultato che si
vuole valutare. Casi d'uso tipici:
  - filtro OTTIMO (A) vs filtro di WIENER a lambda scalare (B)   [default]
  - Wiener lambda scalare (A) vs Wiener lambda(f) freq-dipendente (B)
  - due qualsiasi CSV di risultati BI con le colonne channel, vbias, BI (+ rho_t).

Il BI e' da MINIMIZZARE: il "miglioramento" di B su A e' la DIMINUZIONE percentuale
del BI, punto per punto:

    improvement_% = 100 * (BI_A - BI_B) / BI_A

  > 0  -> B abbassa il BI rispetto ad A (meglio)
  < 0  -> B peggiora il BI

Output (nella cartella --outdir, default: accanto al CSV B), con <tag> = --tag:
  - BI_improvement_vs_Vbias_<tag>_m205.png    : miglioramento % vs V_bias, per canale
  - BI_improvement_per_channel_<tag>_m205.png : miglioramento medio per canale (bar)
  - BI_vs_Vbias_<tag>_m205.png                : griglia per canale, curve BI vs V_bias (A e B)
  - BI_vs_SNRbeta_<tag>_m205.png              : idem ma BI vs SNR*beta (rho_t condiviso)
  - BI_improvement_<tag>_m205.csv             : tabella punto-per-punto + riepilogo
A video: tabella dei miglioramenti medi per canale e miglioramento medio globale.

Uso:
    # default: filtro ottimo (A) vs Wiener lambda scalare (B)
    python compare_wiener_vs_optimum_m205.py

    # Wiener lambda scalare (A) vs Wiener lambda(f) (B)
    python compare_wiener_vs_optimum_m205.py \\
        --csv-a m205_results_wiener/BI_results_m205_wiener.csv \\
        --csv-b m205_results_wiener_freq/BI_results_m205_wiener_freq.csv \\
        --label-a "Wiener lambda" --label-b "Wiener lambda(f)" --tag WF_vs_WFfreq

    python compare_wiener_vs_optimum_m205.py --exclude 91
"""

import os
import csv
import math
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DEFAULT_A_CSV  = os.path.join(BASE_DIR, "m205_results_octopus", "BI_results_m205.csv")
DEFAULT_B_CSV  = os.path.join(BASE_DIR, "m205_results_wiener", "BI_results_m205_wiener.csv")
DEFAULT_LABEL_A = "Optimum filter"
DEFAULT_LABEL_B = "Wiener filter"
DEFAULT_TAG     = "OF_vs_WF"
MEAS_NAME      = "000205"

# Canali esclusi di default (in aggiunta a quelli passati con --exclude)
EXCLUDE_CHANNELS = [37, 40, 41, 94]

# Colori fissi per i due set (indipendenti dal canale): navy = A (baseline),
# ambra = B (valutato).
A_COLOR = "#1f4e79"
B_COLOR = "#e8871e"


# ═════════════════════════════════════════════════════════════════════════════
# Lettura dati
# ═════════════════════════════════════════════════════════════════════════════
def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def read_bi_map(path: str) -> dict:
    """Legge un CSV dei risultati BI -> mappa (canale, V_bias arrotondato) ->
    {"BI":..., "rho_t":...}. rho_t (=SNR*beta) e' opzionale (puo' mancare in alcuni
    CSV). Tiene solo righe con BI valido e > 0."""
    bmap = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ch = _to_float(row.get("channel"))
            vb = _to_float(row.get("vbias"))
            bi = _to_float(row.get("BI"))
            if ch is None or vb is None or bi is None or bi <= 0:
                continue
            bmap[(int(ch), round(vb, 3))] = {"BI": bi, "rho_t": _to_float(row.get("rho_t"))}
    return bmap


def channel_colors(channels: list) -> dict:
    """Una tonalità distinta e stabile per canale (coerente con plot_BI_results.py)."""
    cmap = plt.get_cmap("tab10")
    chs = sorted(set(channels))
    return {ch: cmap(i % 10) for i, ch in enumerate(chs)}


# ═════════════════════════════════════════════════════════════════════════════
# Confronto
# ═════════════════════════════════════════════════════════════════════════════
def build_comparison(map_a: dict, map_b: dict, exclude: set) -> list:
    """Unisce le due mappe sulle chiavi comuni (canale, V_bias) e calcola il
    miglioramento percentuale del BI di B rispetto ad A."""
    rows = []
    for key in sorted(set(map_a) & set(map_b)):
        ch, vb = key
        if ch in exclude:
            continue
        bi_a = map_a[key]["BI"]
        bi_b = map_b[key]["BI"]
        # rho_t = SNR*beta e' una proprieta' del template (stesso S del filtro
        # ottimo), tipicamente identica nei due set: uso quella di B e, se assente,
        # ricado su quella di A. Serve da asse x condiviso.
        rho_t = map_b[key]["rho_t"]
        if rho_t is None:
            rho_t = map_a[key]["rho_t"]
        rows.append({
            "channel": ch,
            "vbias": vb,
            "BI_a": bi_a,
            "BI_b": bi_b,
            "rho_t": rho_t,
            "improvement_pct": 100.0 * (bi_a - bi_b) / bi_a,
        })
    rows.sort(key=lambda r: (r["channel"], r["vbias"]))
    return rows


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


# ═════════════════════════════════════════════════════════════════════════════
# Plot
# ═════════════════════════════════════════════════════════════════════════════
def plot_improvement_vs_vbias(rows, colors, out_png, label_a, label_b):
    """Miglioramento % del BI vs V_bias, una curva per canale (scala lineare)."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for ch in sorted(set(r["channel"] for r in rows)):
        pts = sorted([(r["vbias"], r["improvement_pct"]) for r in rows if r["channel"] == ch])
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                marker="o", ms=5, lw=1.4, color=colors[ch], label=f"Ch {ch}")
    ax.axhline(0.0, color="k", lw=1.0, ls="--", alpha=0.7)   # baseline = A
    ax.set_xlabel(r"$V_{bias}$ (V)", fontsize=12)
    ax.set_ylabel("BI improvement (%)", fontsize=12)
    ax.set_title(f"{label_b} vs {label_a} — BI improvement vs Bias Voltage\n"
                 f"Measurement {MEAS_NAME}  (>0 = {label_b} lowers BI)", fontsize=13)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"  → {os.path.basename(out_png)}")


def plot_improvement_per_channel(per_ch, global_mean, colors, out_png, label_a, label_b):
    """Bar chart del miglioramento medio per canale + linea della media globale."""
    chs = sorted(per_ch.keys())
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar([str(ch) for ch in chs], [per_ch[ch] for ch in chs],
           color=[colors[ch] for ch in chs])
    ax.axhline(0.0, color="k", lw=1.0, ls="-", alpha=0.6)
    if global_mean is not None:
        ax.axhline(global_mean, color="crimson", lw=1.8, ls="--",
                   label=f"Global mean = {global_mean:+.1f}%")
        ax.legend(fontsize=11)
    ax.set_xlabel("Channel", fontsize=12)
    ax.set_ylabel("Mean BI improvement (%)", fontsize=12)
    ax.set_title(f"{label_b} vs {label_a} — Mean BI improvement per channel\n"
                 f"Measurement {MEAS_NAME}", fontsize=13)
    ax.grid(True, axis="y", which="both", linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"  → {os.path.basename(out_png)}")


def plot_bi_compare_grid(rows, out_png, xkey, xlabel, title, label_a, label_b, logx=False):
    """Griglia con un pannello per canale: BI vs xkey con DUE curve sovrapposte,
    A (navy, tratteggiata) e B (ambra, piena). L'asse x e' condiviso (stessa V_bias
    / stesso SNR*beta); i punti sono connessi lungo lo sweep in V_bias. y log."""
    from matplotlib.lines import Line2D
    channels = sorted(set(r["channel"] for r in rows))
    if not any(r.get(xkey) is not None for r in rows):
        print(f"[WARN] nessun dato '{xkey}' disponibile: salto {os.path.basename(out_png)}.")
        return
    n = len(channels)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.7 * nrows), squeeze=False)
    axf = axes.ravel()
    for ax, ch in zip(axf, channels):
        d = sorted([r for r in rows if r["channel"] == ch and r.get(xkey) is not None],
                   key=lambda r: r["vbias"])
        if not d:
            ax.axis("off")
            continue
        xs = [r[xkey] for r in d]
        ax.plot(xs, [r["BI_a"] for r in d], "o--", ms=5, lw=1.2, color=A_COLOR, label=label_a)
        ax.plot(xs, [r["BI_b"] for r in d], "o-", ms=5, lw=1.5, color=B_COLOR, label=label_b)
        ax.set_yscale("log")
        if logx:
            ax.set_xscale("log")
        ax.set_title(f"Ch {ch}", fontsize=12, fontweight="bold")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
    for ax in axf[n:]:
        ax.axis("off")
    handles = [Line2D([0], [0], marker="o", ls="--", color=A_COLOR, label=label_a),
               Line2D([0], [0], marker="o", ls="-", color=B_COLOR, label=label_b)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.055),
               ncol=2, fontsize=12, frameon=False)
    fig.suptitle(f"{title} — Measurement {MEAS_NAME}", fontsize=15, fontweight="bold")
    fig.supxlabel(xlabel, fontsize=12, y=0.01)
    fig.supylabel("Background Index (BI)", fontsize=12)
    fig.tight_layout(rect=[0, 0.10, 1, 0.96])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"  → {os.path.basename(out_png)}")


# ═════════════════════════════════════════════════════════════════════════════
# Output CSV
# ═════════════════════════════════════════════════════════════════════════════
def write_csv(rows, per_ch, global_mean, out_csv, label_a, label_b):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# A = {label_a} (baseline),  B = {label_b},  "
                    "improvement_pct = 100*(BI_A - BI_B)/BI_A"])
        w.writerow(["channel", "vbias_V", "BI_A", "BI_B", "improvement_pct"])
        for r in rows:
            w.writerow([r["channel"], f"{r['vbias']:.3f}", f"{r['BI_a']:.6e}",
                        f"{r['BI_b']:.6e}", f"{r['improvement_pct']:.4f}"])
        w.writerow([])
        w.writerow(["# mean improvement per channel"])
        w.writerow(["channel", "mean_improvement_pct", "n_points"])
        for ch in sorted(per_ch):
            n = sum(1 for r in rows if r["channel"] == ch)
            w.writerow([ch, f"{per_ch[ch]:.4f}", n])
        w.writerow([])
        w.writerow(["# global mean improvement (all points)", f"{global_mean:.4f}"])
    print(f"  → {os.path.basename(out_csv)}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Confronto generico di due set di risultati BI (m205): A (baseline) vs B.")
    parser.add_argument("--csv-a", "--optimum-csv", dest="csv_a", default=DEFAULT_A_CSV,
                        help="CSV del set A (baseline). Alias storico: --optimum-csv")
    parser.add_argument("--csv-b", "--wiener-csv", dest="csv_b", default=DEFAULT_B_CSV,
                        help="CSV del set B (valutato). Alias storico: --wiener-csv")
    parser.add_argument("--label-a", default=DEFAULT_LABEL_A, help="etichetta del set A")
    parser.add_argument("--label-b", default=DEFAULT_LABEL_B, help="etichetta del set B")
    parser.add_argument("--tag", default=DEFAULT_TAG,
                        help="suffisso nei nomi dei file di output (default OF_vs_WF)")
    parser.add_argument("--outdir", default=None,
                        help="cartella di output (default: accanto al CSV B)")
    parser.add_argument("--exclude", nargs="*", type=int, default=None,
                        help="canali da escludere, es. --exclude 31 94")
    args = parser.parse_args()

    for tag, path in ((args.label_a, args.csv_a), (args.label_b, args.csv_b)):
        if not os.path.exists(path):
            raise SystemExit(f"[ERROR] CSV '{tag}' non trovato: {path}")

    map_a = read_bi_map(args.csv_a)
    map_b = read_bi_map(args.csv_b)
    if not map_a or not map_b:
        raise SystemExit("[ERROR] Uno dei due CSV non contiene BI validi.")

    exclude = set(EXCLUDE_CHANNELS) | set(args.exclude or [])
    rows = build_comparison(map_a, map_b, exclude)
    if not rows:
        raise SystemExit("[ERROR] Nessuna coppia (canale, V_bias) in comune tra i due CSV.")

    # ── Miglioramento medio per canale e globale ───────────────────────────────
    channels = sorted(set(r["channel"] for r in rows))
    per_ch = {ch: mean([r["improvement_pct"] for r in rows if r["channel"] == ch])
              for ch in channels}
    global_mean = mean([r["improvement_pct"] for r in rows])            # media su tutti i punti
    per_ch_mean = mean(list(per_ch.values()))                          # media delle medie per canale

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.csv_b))
    os.makedirs(outdir, exist_ok=True)
    colors = channel_colors(channels)

    n_common = len(rows)
    n_a_only = len(set(map_a) - set(map_b))
    n_b_only = len(set(map_b) - set(map_a))
    print(f"Confronto:  A = {args.label_a}   vs   B = {args.label_b}")
    print(f"Punti in comune: {n_common} su {len(channels)} canali "
          f"(solo-A: {n_a_only}, solo-B: {n_b_only}).")
    if exclude:
        print(f"Canali esclusi: {sorted(exclude)}")
    print(f"Genero l'output in {outdir}:")

    # ── Tabella a video ────────────────────────────────────────────────────────
    print(f"\n  Miglioramento % del BI (B={args.label_b} vs A={args.label_a})  "
          f"[>0 = B abbassa il BI]")
    print(f"  {'Ch':>4} {'mean %':>9} {'min %':>9} {'max %':>9} {'n':>4}")
    for ch in channels:
        imps = [r["improvement_pct"] for r in rows if r["channel"] == ch]
        print(f"  {ch:>4} {per_ch[ch]:>+9.2f} {min(imps):>+9.2f} {max(imps):>+9.2f} {len(imps):>4}")
    print(f"  {'-'*40}")
    print(f"  Media per canale (media delle medie): {per_ch_mean:+.2f} %")
    print(f"  Media globale (su tutti i punti):     {global_mean:+.2f} %")

    # ── Plot + CSV ─────────────────────────────────────────────────────────────
    tag = f"_{args.tag}" if args.tag else ""

    def p(name):
        return os.path.join(outdir, name)

    print()
    plot_improvement_vs_vbias(rows, colors, p(f"BI_improvement_vs_Vbias{tag}_m205.png"),
                              args.label_a, args.label_b)
    plot_improvement_per_channel(per_ch, global_mean, colors,
                                 p(f"BI_improvement_per_channel{tag}_m205.png"),
                                 args.label_a, args.label_b)
    # Confronto diretto delle due curve BI (A vs B), un pannello per canale.
    plot_bi_compare_grid(rows, p(f"BI_vs_Vbias{tag}_m205.png"), "vbias",
                         r"$V_{bias}$ (V)", f"BI vs Bias Voltage — {args.label_b} vs {args.label_a}",
                         args.label_a, args.label_b, logx=False)
    plot_bi_compare_grid(rows, p(f"BI_vs_SNRbeta{tag}_m205.png"), "rho_t",
                         r"SNR·$\beta$ (Hz)", f"BI vs SNR·$\\beta$ — {args.label_b} vs {args.label_a}",
                         args.label_a, args.label_b, logx=True)
    write_csv(rows, per_ch, global_mean, p(f"BI_improvement{tag}_m205.csv"),
              args.label_a, args.label_b)

    print("\nFatto.")


if __name__ == "__main__":
    main()
