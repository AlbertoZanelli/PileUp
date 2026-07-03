"""
plot_BI_results.py
==================
Analisi e plot dei risultati di stima del BI (Background Index) prodotti dai job
di analyse_BI_m205.py.

Sorgenti dati:
  - BI_results_m205.csv  (dai job): channel, wp, vbias, signal_amp,
                                     sigma_analytic, SNR, BI, J_final
  - amplitudes_m205.csv  (da plot_all_root.py): channel, vbias_V,
                                     risetime_ms, decaytime_ms, amplitude_mV, freq_sigma_Hz
Il risetime, decaytime e lo spectral spread vengono uniti ai risultati BI per (canale, V_bias).

Plot prodotti (PNG nella cartella dei risultati):
  - BI vs V_bias                (curva di lavoro del BI, per canale)
  - BI vs SNR
  - BI vs risetime
  - BI vs decay time
  - BI vs ampiezza del segnale
  - BI vs sigma (risoluzione/rumore)
  - BI vs SNR/risetime
  - BI vs SNR/(decaytime/risetime)
  - BI vs spectral spread (freq_sigma_Hz)
  - SNR vs V_bias
  - signal_amp vs V_bias
  - sigma_analytic vs V_bias
  - spectral spread vs V_bias
  - scatter 3D interattivo (HTML): BI vs SNR vs risetime
  - BI minimo per canale        (bar chart) + tabella riepilogo (CSV)

E' possibile escludere canali (costante EXCLUDE_CHANNELS oppure --exclude).
Il BI e' una quantita' da MINIMIZZARE: il riepilogo individua, per ciascun
canale, il punto di lavoro (V_bias) a BI minimo.

Uso:
    python plot_BI_results.py
    python plot_BI_results.py --exclude 31 94
    python plot_BI_results.py --bi-csv path/BI.csv --amp-csv path/amp.csv --outdir path/
"""

import os
import csv
import math
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BI_CSV  = os.path.join(BASE_DIR, "m205_results_octopus", "BI_results_m205.csv")
DEFAULT_AMP_CSV = os.path.join(BASE_DIR, "Processed", "amplitudes_m205.csv")
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
            })
    return rows


def read_amplitudes(path: str) -> dict:
    """Mappa (canale, V_bias arrotondato) -> {risetime_ms, decaytime_ms, amplitude_mV, freq_sigma_Hz}."""
    amap = {}
    if not os.path.exists(path):
        print(f"[WARN] amplitudes non trovato ({path}): i plot col timing/spettro saranno saltati.")
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
                "freq_sigma_Hz": _to_float(row.get("freq_sigma_Hz")),
            }
    return amap


def merge_timing(rows: list, amap: dict):
    """Aggiunge timing, amplitude e spectral spread a ogni riga BI, unendo per (canale, V_bias)."""
    for r in rows:
        info = amap.get((r["channel"], round(r["vbias"], 3)))
        r["risetime_ms"]   = info["risetime_ms"]   if info else None
        r["decaytime_ms"]  = info["decaytime_ms"]  if info else None
        r["amplitude_mV"]  = info["amplitude_mV"]  if info else None
        r["freq_sigma_Hz"] = info["freq_sigma_Hz"] if info else None
        
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
    """Un colore stabile per canale, coerente fra tutti i grafici."""
    cmap = plt.get_cmap("tab20")
    chs = sorted(set(channels))
    return {ch: cmap(i % 20) for i, ch in enumerate(chs)}


def plot_xy(rows, xkey, ykey, xlabel, ylabel, title, out_png, colors,
            sort_x=True, logx=False, logy=True):
    """Grafico ykey vs xkey, una curva per canale. Salta i punti con valori mancanti."""
    fig, ax = plt.subplots(figsize=(10, 6))
    channels = sorted(set(r["channel"] for r in rows))
    n_pts = 0
    for ch in channels:
        pts = [(r[xkey], r[ykey]) for r in rows
               if r["channel"] == ch and r.get(xkey) is not None and r.get(ykey) is not None]
        if not pts:
            continue
        if sort_x:
            pts.sort(key=lambda p: p[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        n_pts += len(pts)
        ax.plot(xs, ys, marker="o", linestyle="-", color=colors[ch], label=f"Ch {ch}")

    if n_pts == 0:
        print(f"[WARN] nessun dato per '{title}', salto.")
        plt.close(fig)
        return

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{title}\nMisura {MEAS_NAME}", fontsize=14)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"  → {os.path.basename(out_png)}  ({n_pts} punti)")


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
        title=f"BI vs SNR vs Risetime — Misura {MEAS_NAME}",
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
              "risetime_ms_at_best", "decaytime_ms_at_best", "amplitude_mV_at_best", "freq_sigma_Hz_at_best"]
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
                "freq_sigma_Hz_at_best": f"{b['freq_sigma_Hz']:.3f}" if b.get("freq_sigma_Hz") is not None else "",
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
    ax.set_ylabel("BI minimo", fontsize=12)
    ax.set_title(f"BI minimo per canale\nMisura {MEAS_NAME}", fontsize=14)
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
    parser.add_argument("--bi-csv", default=DEFAULT_BI_CSV, help="CSV dei risultati BI")
    parser.add_argument("--amp-csv", default=DEFAULT_AMP_CSV, help="CSV ampiezze/timing/spettro")
    parser.add_argument("--outdir", default=None, help="cartella di output (default: accanto al BI CSV)")
    parser.add_argument("--exclude", nargs="*", type=int, default=None,
                        help="canali da escludere, es. --exclude 31 94")
    args = parser.parse_args()

    if not os.path.exists(args.bi_csv):
        raise SystemExit(f"[ERROR] CSV BI non trovato: {args.bi_csv}")

    rows = read_bi_results(args.bi_csv)
    if not rows:
        raise SystemExit(f"[ERROR] Nessun dato leggibile in {args.bi_csv}")

    amap = read_amplitudes(args.amp_csv)
    merge_timing(rows, amap)

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
        return os.path.join(outdir, name)

    n_ch = len(set(r["channel"] for r in rows))
    print(f"Dati: {len(rows)} punti su {n_ch} canali.\nGenero i plot in {outdir}:")

    # ── Curva di lavoro principale ─────────────────────────────────────────────
    plot_xy(rows, "vbias", "BI",
            r"$V_{bias}$ (V)", "Background Index (BI)",
            "BI vs Bias Voltage", p("BI_vs_Vbias_m205.png"), colors,
            sort_x=True, logy=True)

    # ── Correlazioni BI vs Parametri ───────────────────────────────────────────
    plot_xy(rows, "SNR", "BI",
            "SNR", "Background Index (BI)",
            "BI vs SNR", p("BI_vs_SNR_m205.png"), colors,
            sort_x=True, logy=True)
    
    plot_xy(rows, "SNR_over_risetime", "BI",
            "SNR / Risetime", "Background Index (BI)",
            "BI vs SNR_over_risetime", p("BI_vs_SNR_risetime_m205.png"), colors,
            sort_x=True, logy=True)

    plot_xy(rows, "SNR_over_timing_ratio", "BI",
            "SNR / (Decay Time / Risetime)", "Background Index (BI)",
            "BI vs SNR_over_timing_ratio", p("BI_vs_SNR_timing_ratio_m205.png"), colors,
            sort_x=True, logy=True)

    plot_xy(rows, "risetime_ms", "BI",
            "Risetime (ms)", "Background Index (BI)",
            "BI vs Risetime", p("BI_vs_risetime_m205.png"), colors,
            sort_x=True, logy=True)

    plot_xy(rows, "decaytime_ms", "BI",
            "Decay Time (ms)", "Background Index (BI)",
            "BI vs Decay Time", p("BI_vs_decaytime_m205.png"), colors,
            sort_x=True, logy=True)

    plot_xy(rows, "freq_sigma_Hz", "BI",
            r"Spectral Spread $\sigma_f$ (Hz)", "Background Index (BI)",
            "BI vs Spectral Spread", p("BI_vs_freq_sigma_m205.png"), colors,
            sort_x=True, logy=True)

    plot_xy(rows, "signal_amp", "BI",
            "Amplitude (V)", "Background Index (BI)",
            "BI vs Amplitude", p("BI_vs_amplitude_m205.png"), colors,
            sort_x=True, logy=True)

    plot_xy(rows, "sigma_analytic", "BI",
            r"$\sigma$ analytic (V)", "Background Index (BI)",
            "BI vs Sigma (noise)", p("BI_vs_sigma_m205.png"), colors,
            sort_x=True, logx=True, logy=True)

    # ── Andamento delle grandezze vs V_bias ────────────────────────────────────
    plot_xy(rows, "vbias", "signal_amp",
            r"$V_{bias}$ (V)", "Signal Amplitude (V)",
            "Signal Amplitude vs Bias Voltage", p("signal_amp_vs_Vbias_m205.png"), colors,
            sort_x=True, logy=True)

    plot_xy(rows, "vbias", "sigma_analytic",
            r"$V_{bias}$ (V)", r"$\sigma$ analytic (V)",
            "Sigma vs Bias Voltage", p("sigma_vs_Vbias_m205.png"), colors,
            sort_x=True, logy=True)

    plot_xy(rows, "vbias", "SNR",
            r"$V_{bias}$ (V)", "SNR",
            "SNR vs Bias Voltage", p("SNR_vs_Vbias_m205.png"), colors,
            sort_x=True, logy=False)

    plot_xy(rows, "vbias", "freq_sigma_Hz",
            r"$V_{bias}$ (V)", r"Spectral Spread $\sigma_f$ (Hz)",
            "Spectral Spread vs Bias Voltage", p("freq_sigma_vs_Vbias_m205.png"), colors,
            sort_x=True, logy=False)

    # ── Scatter 3D interattivo BI vs SNR vs risetime (HTML) ─────────────────────
    plot_bi_3d_html(rows, p("BI_vs_SNR_risetime_3D_m205.html"), colors)

    # ── Riepilogo punto ottimo ─────────────────────────────────────────────────
    summarize_best(rows, p("BI_summary_m205.csv"), p("BI_min_per_channel_m205.png"), colors)

    print("\nFatto.")

if __name__ == "__main__":
    main()