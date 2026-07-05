"""
compare_wiener_vs_optimum_m205.py
=================================
Confronta la stima del BI ottenuta con il filtro di WIENER a lambda addestrabile
(analyse_BI_m205_wiener.py) con quella del filtro OTTIMO (analyse_BI_m205.py) sulla
misura m205.

Il BI e' una quantita' da MINIMIZZARE: il "miglioramento" del Wiener e' la
DIMINUZIONE percentuale del BI rispetto al filtro ottimo, calcolata punto per punto
(canale, V_bias):

    improvement_% = 100 * (BI_optimum - BI_wiener) / BI_optimum

  > 0  -> il Wiener abbassa il BI (meglio)
  < 0  -> il Wiener peggiora il BI

Output (nella cartella --outdir, default m205_results_wiener):
  - BI_improvement_vs_Vbias_m205.png : miglioramento % vs V_bias, una curva per canale
  - BI_improvement_per_channel_m205.png : miglioramento medio per canale (bar chart)
                                          con la media globale come linea di riferimento
  - BI_improvement_m205.csv : tabella punto-per-punto (canale, V_bias, BI_opt,
                              BI_wiener, improvement_%) + riepilogo per canale
A video: tabella dei miglioramenti medi per canale e miglioramento medio globale.

Uso:
    python compare_wiener_vs_optimum_m205.py
    python compare_wiener_vs_optimum_m205.py --exclude 31 94
    python compare_wiener_vs_optimum_m205.py --optimum-csv path/BI_opt.csv \
                                             --wiener-csv path/BI_wiener.csv --outdir path/
"""

import os
import csv
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OPTIMUM_CSV = os.path.join(BASE_DIR, "m205_results_octopus", "BI_results_m205.csv")
DEFAULT_WIENER_CSV  = os.path.join(BASE_DIR, "m205_results_wiener", "BI_results_m205_wiener.csv")
MEAS_NAME          = "000205"

# Canali esclusi di default (in aggiunta a quelli passati con --exclude)
EXCLUDE_CHANNELS = [37, 40, 41, 94]


# ═════════════════════════════════════════════════════════════════════════════
# Lettura dati
# ═════════════════════════════════════════════════════════════════════════════
def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def read_bi_map(path: str) -> dict:
    """Legge un CSV dei risultati BI -> mappa (canale, V_bias arrotondato) -> BI.
    Tiene solo le righe con channel/vbias/BI validi e BI > 0."""
    bmap = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ch = _to_float(row.get("channel"))
            vb = _to_float(row.get("vbias"))
            bi = _to_float(row.get("BI"))
            if ch is None or vb is None or bi is None or bi <= 0:
                continue
            bmap[(int(ch), round(vb, 3))] = bi
    return bmap


def channel_colors(channels: list) -> dict:
    """Una tonalità distinta e stabile per canale (coerente con plot_BI_results.py)."""
    cmap = plt.get_cmap("tab10")
    chs = sorted(set(channels))
    return {ch: cmap(i % 10) for i, ch in enumerate(chs)}


# ═════════════════════════════════════════════════════════════════════════════
# Confronto
# ═════════════════════════════════════════════════════════════════════════════
def build_comparison(opt_map: dict, wie_map: dict, exclude: set) -> list:
    """Unisce le due mappe sulle chiavi comuni (canale, V_bias) e calcola il
    miglioramento percentuale del BI del Wiener rispetto al filtro ottimo."""
    rows = []
    for key in sorted(set(opt_map) & set(wie_map)):
        ch, vb = key
        if ch in exclude:
            continue
        bi_opt = opt_map[key]
        bi_wie = wie_map[key]
        rows.append({
            "channel": ch,
            "vbias": vb,
            "BI_optimum": bi_opt,
            "BI_wiener": bi_wie,
            "improvement_pct": 100.0 * (bi_opt - bi_wie) / bi_opt,
        })
    rows.sort(key=lambda r: (r["channel"], r["vbias"]))
    return rows


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


# ═════════════════════════════════════════════════════════════════════════════
# Plot
# ═════════════════════════════════════════════════════════════════════════════
def plot_improvement_vs_vbias(rows, colors, out_png):
    """Miglioramento % del BI vs V_bias, una curva per canale (scala lineare)."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for ch in sorted(set(r["channel"] for r in rows)):
        pts = sorted([(r["vbias"], r["improvement_pct"]) for r in rows if r["channel"] == ch])
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                marker="o", ms=5, lw=1.4, color=colors[ch], label=f"Ch {ch}")
    ax.axhline(0.0, color="k", lw=1.0, ls="--", alpha=0.7)   # baseline = filtro ottimo
    ax.set_xlabel(r"$V_{bias}$ (V)", fontsize=12)
    ax.set_ylabel("BI improvement (%)", fontsize=12)
    ax.set_title(f"Wiener vs Optimum filter — BI improvement vs Bias Voltage\n"
                 f"Measurement {MEAS_NAME}  (>0 = Wiener lowers BI)", fontsize=13)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"  → {os.path.basename(out_png)}")


def plot_improvement_per_channel(per_ch, global_mean, colors, out_png):
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
    ax.set_title(f"Wiener vs Optimum filter — Mean BI improvement per channel\n"
                 f"Measurement {MEAS_NAME}", fontsize=13)
    ax.grid(True, axis="y", which="both", linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"  → {os.path.basename(out_png)}")


# ═════════════════════════════════════════════════════════════════════════════
# Output CSV
# ═════════════════════════════════════════════════════════════════════════════
def write_csv(rows, per_ch, global_mean, out_csv):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel", "vbias_V", "BI_optimum", "BI_wiener", "improvement_pct"])
        for r in rows:
            w.writerow([r["channel"], f"{r['vbias']:.3f}", f"{r['BI_optimum']:.6e}",
                        f"{r['BI_wiener']:.6e}", f"{r['improvement_pct']:.4f}"])
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
        description="Confronto BI: filtro di Wiener (lambda addestrabile) vs filtro ottimo (m205).")
    parser.add_argument("--optimum-csv", default=DEFAULT_OPTIMUM_CSV,
                        help="CSV dei risultati BI col filtro ottimo")
    parser.add_argument("--wiener-csv", default=DEFAULT_WIENER_CSV,
                        help="CSV dei risultati BI col filtro di Wiener")
    parser.add_argument("--outdir", default=None,
                        help="cartella di output (default: accanto al CSV Wiener)")
    parser.add_argument("--exclude", nargs="*", type=int, default=None,
                        help="canali da escludere, es. --exclude 31 94")
    args = parser.parse_args()

    for label, path in (("ottimo", args.optimum_csv), ("Wiener", args.wiener_csv)):
        if not os.path.exists(path):
            raise SystemExit(f"[ERROR] CSV {label} non trovato: {path}")

    opt_map = read_bi_map(args.optimum_csv)
    wie_map = read_bi_map(args.wiener_csv)
    if not opt_map or not wie_map:
        raise SystemExit("[ERROR] Uno dei due CSV non contiene BI validi.")

    exclude = set(EXCLUDE_CHANNELS) | set(args.exclude or [])
    rows = build_comparison(opt_map, wie_map, exclude)
    if not rows:
        raise SystemExit("[ERROR] Nessuna coppia (canale, V_bias) in comune tra i due CSV.")

    # ── Miglioramento medio per canale e globale ───────────────────────────────
    channels = sorted(set(r["channel"] for r in rows))
    per_ch = {ch: mean([r["improvement_pct"] for r in rows if r["channel"] == ch])
              for ch in channels}
    global_mean = mean([r["improvement_pct"] for r in rows])            # media su tutti i punti
    per_ch_mean = mean(list(per_ch.values()))                          # media delle medie per canale

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.wiener_csv))
    os.makedirs(outdir, exist_ok=True)
    colors = channel_colors(channels)

    n_common = len(rows)
    n_opt_only = len(set(opt_map) - set(wie_map))
    n_wie_only = len(set(wie_map) - set(opt_map))
    print(f"Punti in comune: {n_common} su {len(channels)} canali "
          f"(solo-ottimo: {n_opt_only}, solo-Wiener: {n_wie_only}).")
    if exclude:
        print(f"Canali esclusi: {sorted(exclude)}")
    print(f"Genero l'output in {outdir}:")

    # ── Tabella a video ────────────────────────────────────────────────────────
    print("\n  Miglioramento % del BI (Wiener vs ottimo)  [>0 = Wiener abbassa il BI]")
    print(f"  {'Ch':>4} {'mean %':>9} {'min %':>9} {'max %':>9} {'n':>4}")
    for ch in channels:
        imps = [r["improvement_pct"] for r in rows if r["channel"] == ch]
        print(f"  {ch:>4} {per_ch[ch]:>+9.2f} {min(imps):>+9.2f} {max(imps):>+9.2f} {len(imps):>4}")
    print(f"  {'-'*40}")
    print(f"  Media per canale (media delle medie): {per_ch_mean:+.2f} %")
    print(f"  Media globale (su tutti i punti):     {global_mean:+.2f} %")

    # ── Plot + CSV ─────────────────────────────────────────────────────────────
    def p(name):
        return os.path.join(outdir, name)

    print()
    plot_improvement_vs_vbias(rows, colors, p("BI_improvement_vs_Vbias_m205.png"))
    plot_improvement_per_channel(per_ch, global_mean, colors, p("BI_improvement_per_channel_m205.png"))
    write_csv(rows, per_ch, global_mean, p("BI_improvement_m205.csv"))

    print("\nFatto.")


if __name__ == "__main__":
    main()
