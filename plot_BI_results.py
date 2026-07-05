"""
plot_BI_results.py
==================
Analisi e plot dei risultati di stima del BI (Background Index) prodotti dai job
di analyse_BI_m205.py.

Funziona sia con i risultati del filtro OTTIMO (analyse_BI_m205.py) sia con quelli
del filtro di WIENER a lambda addestrabile (analyse_BI_m205_wiener.py): basta
passare il relativo CSV con --bi-csv. Se il CSV contiene lambda_wiener, viene
prodotto anche il grafico lambda vs V_bias.

Sorgenti dati:
  - BI_results_m205.csv  (dai job): channel, wp, vbias, signal_amp, sigma_analytic,
                                     SNR, BI, J_final; opzionali beta_Hz, rho_t
                                     (=SNR*beta) e lambda_wiener (solo Wiener)
  - amplitudes_m205.csv  (da risetime_and_amplitude_.py): channel, vbias_V,
                                     risetime_ms, decaytime_ms, amplitude_mV, hf_power
Risetime, decaytime e HF-power vengono uniti ai risultati BI per (canale, V_bias).
rho_t viene letto inline dal BI CSV; --timing-csv resta come fallback per i CSV
vecchi che non hanno quella colonna.

Plot prodotti (PNG nella cartella dei risultati) — accorpati in poche canvas:
  - BI_vs_parameters_m205.png   (3x3): BI vs V_bias / SNR / risetime / decaytime /
                                       HF-power / amplitude / sigma / SNR-over-risetime /
                                       SNR-over-(decay/rise)
  - params_vs_Vbias_m205.png    (2x2): signal_amp / sigma / SNR / HF-power vs V_bias
  - BI_vs_SNR_risetime_3D_m205.html   scatter 3D interattivo BI vs SNR vs risetime
  - BI_min_per_channel_m205.png       bar chart + BI_summary_m205.csv (tabella riepilogo)
  - lambda_vs_Vbias_m205.png          lambda del Wiener vs V_bias (solo CSV Wiener)

Lo spectral spread (freq_sigma) e' stato sostituito da HF-power (potenza dello
spettro dell'AP sopra ~500 Hz), la metrica di rumore adottata nello studio m204.

E' possibile escludere canali (costante EXCLUDE_CHANNELS oppure --exclude).
Il BI e' una quantita' da MINIMIZZARE: il riepilogo individua, per ciascun
canale, il punto di lavoro (V_bias) a BI minimo.

Uso:
    python plot_BI_results.py
    python plot_BI_results.py --exclude 31 94
    python plot_BI_results.py --bi-csv path/BI.csv --amp-csv path/amp.csv --outdir path/
    # modalità Wiener: percorsi in m205_results_wiener e output con suffisso _wiener
    # (grafico lambda vs V_bias incluso):
    python plot_BI_results.py --wiener
"""

import os
import csv
import math
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
<<<<<<< HEAD
DEFAULT_BI_CSV  = os.path.join(BASE_DIR, "m205_results_octopus", "BI_results_m205.csv")
DEFAULT_AMP_CSV = os.path.join(BASE_DIR, "m205_results_octopus", "amplitudes_m205.csv")
DEFAULT_TIMING_CSV = os.path.join(BASE_DIR, "m205_results_octopus", "timing_SNR_m205.csv")
# Modalità Wiener (--wiener): risultati di analyse_BI_m205_wiener.py. Le ampiezze
# sono le stesse (proprietà del template), quindi cambia solo il CSV dei BI.
DEFAULT_BI_CSV_WIENER = os.path.join(BASE_DIR, "m205_results_wiener", "BI_results_m205_wiener.csv")
=======
DEFAULT_BI_CSV  = os.path.join(BASE_DIR, "m205_results_wiener", "BI_results_m205.csv")
DEFAULT_AMP_CSV = os.path.join(BASE_DIR, "m205_results_wiener", "amplitudes_m205.csv")
DEFAULT_TIMING_CSV = os.path.join(BASE_DIR, "m205_results_wiener", "timing_SNR_m205.csv")
>>>>>>> d6ab10d910466cca972e08fd8aff2d8e309ee362
MEAS_NAME       = "000205"

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


def read_bi_results(path: str) -> list:
    """Legge il CSV dei risultati BI. Tiene solo le righe con channel/vbias/BI validi."""
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ch = _to_float(row.get("channel"))
            vb = _to_float(row.get("vbias"))
            bi = _to_float(row.get("BI"))
            if ch is None or vb is None or bi is None:
                continue
            rows.append({
                "channel": int(ch),
                "vbias": vb,
                "BI": bi,
                "SNR": _to_float(row.get("SNR")),
                "signal_amp": _to_float(row.get("signal_amp")),
                "sigma_analytic": _to_float(row.get("sigma_analytic")),
                "wp": _to_float(row.get("wp")),
                # Colonne opzionali gia' presenti nei CSV arricchiti (OF e Wiener):
                # beta_Hz e rho_t (=SNR*beta), e lambda_wiener nella versione Wiener.
                "beta_Hz": _to_float(row.get("beta_Hz")),
                "rho_t": _to_float(row.get("rho_t")),
                "lambda_wiener": _to_float(row.get("lambda_wiener")),
            })
    return rows


def read_amplitudes(path: str) -> dict:
    """Mappa (canale, V_bias arrotondato) -> {risetime_ms, decaytime_ms, amplitude_mV, hf_power}."""
    amap = {}
    if not os.path.exists(path):
        print(f"[WARN] amplitudes non trovato ({path}): i plot col timing/HF-power saranno saltati.")
        return amap
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ch = _to_float(row.get("channel"))
            vb = _to_float(row.get("vbias_V"))
            if ch is None or vb is None:
                continue
            amap[(int(ch), round(vb, 3))] = {
                "risetime_ms": _to_float(row.get("risetime_ms")),
                "decaytime_ms": _to_float(row.get("decaytime_ms")),
                "amplitude_mV": _to_float(row.get("amplitude_mV")),
                "hf_power": _to_float(row.get("hf_power")),
            }
    return amap


def read_timing(path: str) -> dict:
    """Map (channel, V_bias) -> rho_t = SNR*beta (timing figure of merit), if present."""
    tmap = {}
    if not os.path.exists(path):
        print(f"[WARN] timing CSV non trovato ({path}): il pannello SNR·beta sara' saltato.")
        return tmap
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ch = _to_float(row.get("channel"))
            vb = _to_float(row.get("vbias"))
            if ch is None or vb is None:
                continue
            tmap[(int(ch), round(vb, 3))] = _to_float(row.get("rho_t"))
    return tmap


def merge_timing(rows: list, amap: dict):
    """Aggiunge timing, amplitude e spectral spread a ogni riga BI, unendo per (canale, V_bias)."""
    for r in rows:
        info = amap.get((r["channel"], round(r["vbias"], 3)))
        r["risetime_ms"]   = info["risetime_ms"]   if info else None
        r["decaytime_ms"]  = info["decaytime_ms"]  if info else None
        r["amplitude_mV"]  = info["amplitude_mV"]  if info else None
        r["hf_power"]      = info["hf_power"]       if info else None
        
        # Calcolo derivato: SNR / risetime_ms
        if r.get("SNR") is not None and r.get("risetime_ms") not in (None, 0):
            r["SNR_over_risetime"] = r["SNR"] / r["risetime_ms"]
        else:
            r["SNR_over_risetime"] = None

        # Calcolo derivato: SNR / (decaytime_ms / risetime_ms)
        if r.get("SNR") is not None and r.get("decaytime_ms") not in (None, 0) and r.get("risetime_ms") not in (None, 0):
            r["SNR_over_timing_ratio"] = r["SNR"] / (r["decaytime_ms"] / r["risetime_ms"])
        else:
            r["SNR_over_timing_ratio"] = None


# ═════════════════════════════════════════════════════════════════════════════
# Helper di plot
# ═════════════════════════════════════════════════════════════════════════════
def channel_colors(channels: list) -> dict:
    """Una tonalità distinta e stabile per canale, coerente fra tutti i grafici.
    tab10 dà hue ben separate (necessario per la sfumatura per-canale by V_bias)."""
    cmap = plt.get_cmap("tab10")
    chs = sorted(set(channels))
    return {ch: cmap(i % 10) for i, ch in enumerate(chs)}


def shade_by(base, t: float):
    """Sfumatura della tonalità `base`: t in [0,1], chiaro (t=0) -> scuro (t=1),
    mantenendo la hue del canale. Usata per codificare V_bias entro il canale."""
    import colorsys
    from matplotlib.colors import to_rgb
    h, _, _ = colorsys.rgb_to_hsv(*to_rgb(base))
    v = 0.95 - 0.60 * t          # luminosità: alta (chiaro) -> bassa (scuro)
    s = 0.30 + 0.60 * t          # saturazione: bassa (chiaro) -> alta (scuro)
    return colorsys.hsv_to_rgb(h, s, v)


def _plot_on_ax(ax, rows, xkey, ykey, xlabel, ylabel, title, colors,
                sort_x=True, logx=False, logy=True):
    """Disegna ykey vs xkey (una curva per canale) su un asse dato. Ritorna n. punti."""
    channels = sorted(set(r["channel"] for r in rows))
    n_pts = 0
    for ch in channels:
        # Keep V_bias with each point: the connecting line is ordered along the
        # bias sweep, NOT by x. Amplitude and SNR are non-monotonic in V_bias
        # (they peak at intermediate bias), so ordering by x would fold the curve
        # over itself and draw spurious zig-zag spikes; ordering by V_bias traces
        # the physical sweep. For monotonic x (risetime, HF-power, ...) it is
        # identical to ordering by x.
        pts = [(r[xkey], r[ykey], r.get("vbias")) for r in rows
               if r["channel"] == ch and r.get(xkey) is not None and r.get(ykey) is not None]
        if not pts:
            continue
        if sort_x:
            pts.sort(key=lambda p: (p[2] if p[2] is not None else p[0]))
        n_pts += len(pts)
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                marker="o", ms=4, lw=1.2, color=colors[ch], label=f"Ch {ch}")
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    return n_pts


def plot_grid(rows, specs, ncols, out_png, colors, suptitle):
    """Combina più pannelli (una `spec` per pannello) in un'unica canva, con
    un'unica legenda condivisa dei canali. Riduce il numero di immagini prodotte."""
    from matplotlib.lines import Line2D
    n = len(specs)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 4.1 * nrows), squeeze=False)
    axf = axes.ravel()
    total = 0
    for ax, spec in zip(axf, specs):
        total += _plot_on_ax(ax, rows, colors=colors, **spec)
    for ax in axf[n:]:
        ax.axis("off")
    if total == 0:
        print(f"[WARN] nessun dato per '{suptitle}', salto.")
        plt.close(fig)
        return
    chs = sorted(set(r["channel"] for r in rows))
    handles = [Line2D([0], [0], marker="o", ms=6, lw=1.5, color=colors[ch], label=f"Ch {ch}") for ch in chs]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(chs), 9),
               fontsize=10, frameon=False)
    fig.suptitle(f"{suptitle} — Measurement {MEAS_NAME}", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"  → {os.path.basename(out_png)}  ({total} punti, {n} pannelli)")


def plot_grid_vbias(rows, specs, ncols, out_png, colors, suptitle):
    """Come plot_grid, ma ogni canale usa la propria TONALITÀ e i marker sono
    SFUMATI per V_bias (chiaro = bias basso, scuro = bias alto). Così si legge sia
    il canale (hue) sia il punto dello sweep (luminosità), utile sui pannelli
    ripiegati (BI vs ampiezza / SNR). Una colorbar neutra spiega la sfumatura."""
    from matplotlib.lines import Line2D
    from matplotlib.colors import LogNorm
    from matplotlib.cm import ScalarMappable
    vbs_all = [r["vbias"] for r in rows if r.get("vbias")]
    norm = LogNorm(vmin=min(vbs_all), vmax=max(vbs_all))
    n = len(specs)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 4.1 * nrows), squeeze=False)
    axf = axes.ravel()
    for ax, spec in zip(axf, specs):
        for ch in sorted(set(r["channel"] for r in rows)):
            pts = [(r[spec["xkey"]], r[spec["ykey"]], r["vbias"]) for r in rows
                   if r["channel"] == ch and r.get(spec["xkey"]) is not None
                   and r.get(spec["ykey"]) is not None and r.get("vbias") is not None]
            if not pts:
                continue
            pts.sort(key=lambda p: p[2])
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; vv = [p[2] for p in pts]
            cols = [shade_by(colors[ch], norm(v)) for v in vv]
            ax.plot(xs, ys, "-", color=colors[ch], lw=0.8, alpha=0.35, zorder=1)
            ax.scatter(xs, ys, c=cols, s=26, zorder=3, edgecolors="k", linewidths=0.2)
        if spec.get("logx", False):
            ax.set_xscale("log")
        if spec.get("logy", True):
            ax.set_yscale("log")
        ax.set_xlabel(spec["xlabel"], fontsize=10)
        ax.set_ylabel(spec["ylabel"], fontsize=10)
        ax.set_title(spec["title"], fontsize=12)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
    for ax in axf[n:]:
        ax.axis("off")
    chs = sorted(set(r["channel"] for r in rows))
    handles = [Line2D([0], [0], marker="o", ms=7, lw=2.5, color=colors[ch], label=f"Ch {ch}") for ch in chs]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(chs), 9), fontsize=10, frameon=False)
    fig.suptitle(f"{suptitle} — Measurement {MEAS_NAME}", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 0.93, 0.96])
    # Colorbar neutra: spiega solo la direzione/scala della sfumatura (bias).
    sm = ScalarMappable(norm=norm, cmap=plt.get_cmap("Greys"))
    cax = fig.add_axes([0.945, 0.12, 0.012, 0.76])
    fig.colorbar(sm, cax=cax, label=r"$V_{bias}$ (V):  light $\to$ dark")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"  → {os.path.basename(out_png)}  (tonalità per canale, sfumatura per V_bias)")


def plot_bi_3d_html(rows, out_html, colors):
    """Scatter 3D interattivo (plotly) in HTML: BI vs SNR vs risetime, per canale.
    L'asse z mostra log10(BI); il valore reale di BI è nel tooltip."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[WARN] plotly non disponibile: scatter 3D HTML saltato (pip install plotly).")
        return
    from matplotlib.colors import to_hex

    pts = [r for r in rows
           if r.get("SNR") is not None and r.get("risetime_ms") is not None
           and r.get("BI") is not None and r["BI"] > 0]
    if not pts:
        print("[WARN] nessun punto con SNR+risetime+BI per lo scatter 3D, salto.")
        return

    fig = go.Figure()
    channels = sorted(set(r["channel"] for r in pts))
    for ch in channels:
        sub = [r for r in pts if r["channel"] == ch]
        fig.add_trace(go.Scatter3d(
            x=[r["SNR"] for r in sub],
            y=[r["risetime_ms"] for r in sub],
            z=[math.log10(r["BI"]) for r in sub],
            mode="markers",
            name=f"Ch {ch}",
            marker=dict(size=5, color=to_hex(colors[ch])),
            customdata=[[r["BI"], r["vbias"]] for r in sub],
            hovertemplate=(f"Ch {ch}"
                           "<br>SNR = %{x:.3f}"
                           "<br>risetime = %{y:.4f} ms"
                           "<br>BI = %{customdata[0]:.3e}"
                           "<br>V_bias = %{customdata[1]:.3f} V"
                           "<extra></extra>"),
        ))

    fig.update_layout(
        title=f"BI vs SNR vs Risetime — Measurement {MEAS_NAME}",
        scene=dict(
            xaxis_title="SNR",
            yaxis_title="Risetime (ms)",
            zaxis_title="log10(BI)",
        ),
        legend_title="Canale",
        template="plotly_white",
        height=750,
    )
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"  → {os.path.basename(out_html)}  ({len(pts)} punti)")


# ═════════════════════════════════════════════════════════════════════════════
# Riepilogo: punto di lavoro ottimo (BI minimo) per canale
# ═════════════════════════════════════════════════════════════════════════════
def summarize_best(rows, out_csv, out_png, colors):
    """Per ogni canale trova il BI minimo e il relativo punto di lavoro."""
    best = {}
    for r in rows:
        ch = r["channel"]
        if ch not in best or r["BI"] < best[ch]["BI"]:
            best[ch] = r

    if not best:
        print("[WARN] nessun risultato per il riepilogo.")
        return

    chs = sorted(best.keys())

    # ── CSV riepilogo ─────────────────────────────────────────────────────────
    fields = ["channel", "best_vbias_V", "min_BI", "SNR_at_best",
              "risetime_ms_at_best", "decaytime_ms_at_best", "amplitude_mV_at_best", "hf_power_at_best"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for ch in chs:
            b = best[ch]
            w.writerow({
                "channel": ch,
                "best_vbias_V": f"{b['vbias']:.3f}",
                "min_BI": f"{b['BI']:.6e}",
                "SNR_at_best": f"{b['SNR']:.4f}" if b.get("SNR") is not None else "",
                "risetime_ms_at_best": f"{b['risetime_ms']:.6f}" if b.get("risetime_ms") is not None else "",
                "decaytime_ms_at_best": f"{b['decaytime_ms']:.6f}" if b.get("decaytime_ms") is not None else "",
                "amplitude_mV_at_best": f"{b['amplitude_mV']:.6f}" if b.get("amplitude_mV") is not None else "",
                "hf_power_at_best": f"{b['hf_power']:.6e}" if b.get("hf_power") is not None else "",
            })
    print(f"  → {os.path.basename(out_csv)}")

    # ── Tabella a video ───────────────────────────────────────────────────────
    print("\n  Punto di lavoro ottimo (BI minimo) per canale:")
    print(f"  {'Ch':>4} {'V_bias':>8} {'BI_min':>12} {'SNR':>8} {'risetime_ms':>12} {'decaytime_ms':>12}")
    for ch in chs:
        b = best[ch]
        snr = f"{b['SNR']:.3f}" if b.get("SNR") is not None else "  -"
        rt  = f"{b['risetime_ms']:.4f}" if b.get("risetime_ms") is not None else "   -"
        dt  = f"{b['decaytime_ms']:.4f}" if b.get("decaytime_ms") is not None else "   -"
        print(f"  {ch:>4} {b['vbias']:>8.3f} {b['BI']:>12.4e} {snr:>8} {rt:>12} {dt:>12}")

    # ── Bar chart del BI minimo per canale ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    bars_x = [str(ch) for ch in chs]
    bars_y = [best[ch]["BI"] for ch in chs]
    ax.bar(bars_x, bars_y, color=[colors[ch] for ch in chs])
    ax.set_yscale("log")
    ax.set_xlabel("Channel", fontsize=12)
    ax.set_ylabel("Minimum BI", fontsize=12)
    ax.set_title(f"Minimum BI per channel\nMeasurement {MEAS_NAME}", fontsize=14)
    ax.grid(True, axis="y", which="both", linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"  → {os.path.basename(out_png)}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Analisi e plot dei risultati BI (m205).")
    parser.add_argument("--wiener", action="store_true",
                        help="modalità Wiener: usa i risultati di analyse_BI_m205_wiener.py "
                             "(m205_results_wiener) e aggiunge il suffisso _wiener a TUTTI i "
                             "file di output. --bi-csv/--outdir espliciti hanno la precedenza.")
    parser.add_argument("--bi-csv", default=None,
                        help="CSV dei risultati BI (default: dipende da --wiener)")
    parser.add_argument("--amp-csv", default=DEFAULT_AMP_CSV, help="CSV ampiezze/timing/spettro")
    parser.add_argument("--timing-csv", default=DEFAULT_TIMING_CSV,
                        help="CSV timing (fallback per rho_t=SNR*beta se non gia' nel BI CSV)")
    parser.add_argument("--outdir", default=None, help="cartella di output (default: accanto al BI CSV)")
    parser.add_argument("--exclude", nargs="*", type=int, default=None,
                        help="canali da escludere, es. --exclude 31 94")
    args = parser.parse_args()

    # ── Risoluzione della modalità → percorsi e suffisso dei file di output ─────
    #   In modalità Wiener: BI CSV di default in m205_results_wiener e suffisso
    #   "_wiener" su ogni output; un --bi-csv esplicito resta comunque rispettato.
    suffix = "_wiener" if args.wiener else ""
    if args.bi_csv is None:
        args.bi_csv = DEFAULT_BI_CSV_WIENER if args.wiener else DEFAULT_BI_CSV

    if not os.path.exists(args.bi_csv):
        raise SystemExit(f"[ERROR] CSV BI non trovato: {args.bi_csv}")

    rows = read_bi_results(args.bi_csv)
    if not rows:
        raise SystemExit(f"[ERROR] Nessun dato leggibile in {args.bi_csv}")

    amap = read_amplitudes(args.amp_csv)
    merge_timing(rows, amap)

    # Timing figure of merit rho_t = SNR*beta. I CSV arricchiti (OF e Wiener) lo
    # contengono gia' inline: il CSV timing serve solo da fallback per le righe
    # che non ce l'hanno (es. vecchi BI_results senza la colonna rho_t).
    if any(r.get("rho_t") is None for r in rows):
        tmap = read_timing(args.timing_csv)
        for r in rows:
            if r.get("rho_t") is None:
                r["rho_t"] = tmap.get((r["channel"], round(r["vbias"], 3)))

    # ── Esclusione canali ──────────────────────────────────────────────────────
    exclude = set(EXCLUDE_CHANNELS) | set(args.exclude or [])
    if exclude:
        before = len(rows)
        rows = [r for r in rows if r["channel"] not in exclude]
        print(f"Canali esclusi {sorted(exclude)}: rimosse {before - len(rows)} righe.")
        if not rows:
            raise SystemExit("[ERROR] Tutti i dati sono stati esclusi.")

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.bi_csv))
    os.makedirs(outdir, exist_ok=True)

    colors = channel_colors([r["channel"] for r in rows])

    def p(name):
        """Percorso di output: inserisce il suffisso di modalità (es. _wiener)
        prima dell'estensione, così tutti i file cambiano nome coerentemente."""
        root, ext = os.path.splitext(name)
        return os.path.join(outdir, f"{root}{suffix}{ext}")

    n_ch = len(set(r["channel"] for r in rows))
    print(f"Dati: {len(rows)} punti su {n_ch} canali.\nGenero i plot in {outdir}:")

    # ── Canva 1: BI vs tutti i parametri (una sola immagine, 3x3) ──────────────
    bi_specs = [
        # row 1: V_bias, risetime, decaytime
        dict(xkey="vbias",         ykey="BI", xlabel=r"$V_{bias}$ (V)", ylabel="BI", title="BI vs $V_{bias}$"),
        dict(xkey="risetime_ms",   ykey="BI", xlabel="Risetime (ms)",   ylabel="BI", title="BI vs Risetime"),
        dict(xkey="decaytime_ms",  ykey="BI", xlabel="Decay Time (ms)", ylabel="BI", title="BI vs Decay Time"),
        # row 2: SNR, amplitude, sigma
        dict(xkey="SNR",           ykey="BI", xlabel="SNR",             ylabel="BI", title="BI vs SNR"),
        dict(xkey="signal_amp",    ykey="BI", xlabel="Amplitude (V)",   ylabel="BI", title="BI vs Amplitude"),
        dict(xkey="sigma_analytic",ykey="BI", xlabel=r"$\sigma$ (V)",   ylabel="BI", title="BI vs $\\sigma$ (noise)", logx=True),
        # row 3: SNR/risetime, SNR*beta (timing FoM), HF-power
        dict(xkey="SNR_over_risetime", ykey="BI", xlabel="SNR / Risetime",       ylabel="BI", title="BI vs SNR/Risetime"),
        dict(xkey="rho_t",         ykey="BI", xlabel=r"SNR·$\beta$ (Hz)",        ylabel="BI", title=r"BI vs SNR·$\beta$ (timing FoM)"),
        dict(xkey="hf_power",      ykey="BI", xlabel="AP HF-power ( >500 Hz )",  ylabel="BI", title="BI vs HF-power", logx=True),
    ]
    plot_grid(rows, bi_specs, ncols=3, out_png=p("BI_vs_parameters_m205.png"),
              colors=colors, suptitle="BI vs parameters")

    # Variante con marker colorati per V_bias (per confronto di leggibilità)
    plot_grid_vbias(rows, bi_specs, ncols=3, out_png=p("BI_vs_parameters_vbiascolor_m205.png"),
                    colors=colors, suptitle="BI vs parameters (hue = channel, shade = V_bias)")

    # ── Canva 2: andamento delle grandezze vs V_bias (2x2) ─────────────────────
    vb_specs = [
        dict(xkey="vbias", ykey="signal_amp",     xlabel=r"$V_{bias}$ (V)", ylabel="Signal Amplitude (V)",  title="Amplitude vs $V_{bias}$", logy=True),
        dict(xkey="vbias", ykey="sigma_analytic", xlabel=r"$V_{bias}$ (V)", ylabel=r"$\sigma$ analytic (V)", title="$\\sigma$ vs $V_{bias}$", logy=True),
        dict(xkey="vbias", ykey="SNR",            xlabel=r"$V_{bias}$ (V)", ylabel="SNR",                    title="SNR vs $V_{bias}$", logy=False),
        dict(xkey="vbias", ykey="hf_power",       xlabel=r"$V_{bias}$ (V)", ylabel="AP HF-power ( >500 Hz )", title="HF-power vs $V_{bias}$", logy=True),
    ]
    plot_grid(rows, vb_specs, ncols=2, out_png=p("params_vs_Vbias_m205.png"),
              colors=colors, suptitle="Quantities vs Bias Voltage")

    # ── Standalone BI vs V_bias (dedicated figure for slides) ──────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    _plot_on_ax(ax, rows, "vbias", "BI", r"$V_{bias}$ (V)", "Background Index (BI)",
                "BI vs Bias Voltage", colors, logy=True)
    ax.set_title(f"BI vs Bias Voltage — Measurement {MEAS_NAME}", fontsize=14)
    ax.legend(fontsize=10)
    fig.tight_layout()
    out_bi_vbias = p("BI_vs_Vbias_m205.png")
    fig.savefig(out_bi_vbias, dpi=200)
    plt.close(fig)
    print(f"  → {os.path.basename(out_bi_vbias)}")

    # ── lambda del Wiener vs V_bias (solo se il CSV ha lambda_wiener) ───────────
    #    Presente nei risultati di analyse_BI_m205_wiener.py; assente nel filtro
    #    ottimo, nel qual caso il pannello viene saltato.
    if any(r.get("lambda_wiener") is not None for r in rows):
        fig, ax = plt.subplots(figsize=(9, 6))
        n = _plot_on_ax(ax, rows, "vbias", "lambda_wiener",
                        r"$V_{bias}$ (V)", r"Wiener $\lambda$",
                        r"Trained Wiener $\lambda$ vs Bias Voltage", colors, logy=True)
        if n:
            ax.axhline(1.0, color="gray", ls=":", lw=1,
                       label=r"$\lambda=1$ (standard Wiener)")
            ax.set_title(rf"Trained Wiener $\lambda$ vs Bias Voltage — Measurement {MEAS_NAME}",
                         fontsize=14)
            ax.legend(fontsize=9)
            fig.tight_layout()
            out_lambda = p("lambda_vs_Vbias_m205.png")
            fig.savefig(out_lambda, dpi=200)
            print(f"  → {os.path.basename(out_lambda)}")
        plt.close(fig)

    # ── Scatter 3D interattivo BI vs SNR vs risetime (HTML) ─────────────────────
    plot_bi_3d_html(rows, p("BI_vs_SNR_risetime_3D_m205.html"), colors)

    # ── Riepilogo punto ottimo ─────────────────────────────────────────────────
    summarize_best(rows, p("BI_summary_m205.csv"), p("BI_min_per_channel_m205.png"), colors)

    print("\nFatto.")

if __name__ == "__main__":
    main()