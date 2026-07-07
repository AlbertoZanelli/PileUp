"""
plot_load_curves_m205.py
========================
Curve di carico "classiche" per la misura m205, una per canale, nello stile del
plot di riferimento: piu' grandezze sovrapposte in funzione della corrente del
bolometro I_bol, ciascuna col suo asse y colorato.

Grandezze (dal CSV dei risultati BI):
  - V_bias   (V)   -> blu, cerchi pieni          (left axis)
  - AP Amplitude (mV) -> rosso, quadrati pieni    (right axis 1)
  - OF RMS   (mV)  -> viola, triangoli giu'       (right axis 2)
  - OF SNR         -> verde, quadrati vuoti       (right axis 3)

Asse x: corrente del bolometro
    I_bol = V_bias / R          (R = 2 GOhm per tutti i canali)
espressa in nA (I[nA] = V_bias / 2, essendo R = 2e9 Ohm). Si assume I = V_bias/R
(caduta sul bolometro trascurata rispetto alla resistenza di carico).

Produce UNA immagine (load_curves_m205.png) con un pannello per canale (impilati).

Uso:
    python plot_load_curves_m205.py
    python plot_load_curves_m205.py --bi-csv path/BI.csv --outdir path/ --exclude 91
    python plot_load_curves_m205.py --resistance 2e9
"""

import os
import csv
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BI_CSV = os.path.join(BASE_DIR, "m205_results_octopus", "BI_results_m205.csv")
MEAS_NAME      = "000205"
R_DEFAULT      = 2.0e9          # Ohm: resistenza di carico (uguale per tutti i canali)

# Canali esclusi di default (in aggiunta a quelli passati con --exclude)
EXCLUDE_CHANNELS = [37, 40, 41, 94]

# Colori/marker in stile del plot di riferimento.
C_VBIAS = "#2b6cb4"   # blu   — V_bias
C_AMP   = "#c0392b"   # rosso — AP amplitude
C_RMS   = "#9b59b6"   # viola — OF RMS
C_SNR   = "#5aa02c"   # verde — OF SNR


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def read_load_curve(path: str) -> list:
    """Legge il CSV dei risultati BI e tiene le grandezze della curva di carico.
    Converte le ampiezze/rms da V a mV."""
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ch = _to_float(row.get("channel"))
            vb = _to_float(row.get("vbias"))
            amp = _to_float(row.get("signal_amp"))       # V
            rms = _to_float(row.get("sigma_analytic"))   # V
            snr = _to_float(row.get("SNR"))
            if ch is None or vb is None:
                continue
            rows.append({
                "channel": int(ch),
                "vbias": vb,
                "amp_mV": amp * 1e3 if amp is not None else None,
                "rms_mV": rms * 1e3 if rms is not None else None,
                "SNR": snr,
            })
    return rows


def _panel(ax, d, resistance):
    """Disegna la curva di carico di un canale (4 assi y) su un host axis dato."""
    d = sorted(d, key=lambda r: r["vbias"])
    I_nA = [r["vbias"] / resistance * 1e9 for r in d]   # I = V/R, in nA

    # host: V_bias (blu)
    ax.plot(I_nA, [r["vbias"] for r in d], "-o", color=C_VBIAS, ms=5, lw=1.3)
    ax.set_ylabel(r"V$_{bias}$ (V)", color=C_VBIAS, fontsize=11)
    ax.tick_params(axis="y", colors=C_VBIAS)
    ax.spines["left"].set_color(C_VBIAS)

    # right axis 1: AP amplitude (rosso)
    ax1 = ax.twinx()
    ax1.plot(I_nA, [r["amp_mV"] for r in d], "-s", color=C_AMP, ms=5, lw=1.3)
    ax1.set_ylabel("AP Amplitude (mV)", color=C_AMP, fontsize=11)
    ax1.tick_params(axis="y", colors=C_AMP)
    ax1.spines["right"].set_color(C_AMP)

    # right axis 2: OF RMS (viola), spina spostata in fuori
    ax2 = ax.twinx()
    ax2.spines["right"].set_position(("outward", 52))
    ax2.plot(I_nA, [r["rms_mV"] for r in d], "-v", color=C_RMS, ms=5, lw=1.3)
    ax2.set_ylabel("OF RMS (mV)", color=C_RMS, fontsize=11)
    ax2.tick_params(axis="y", colors=C_RMS)
    ax2.spines["right"].set_color(C_RMS)

    # right axis 3: OF SNR (verde, marker vuoto), spina piu' in fuori
    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("outward", 104))
    ax3.plot(I_nA, [r["SNR"] for r in d], "-s", color=C_SNR, ms=6, lw=1.3,
             mfc="white", mec=C_SNR, mew=1.3)
    ax3.set_ylabel("OF SNR", color=C_SNR, fontsize=11)
    ax3.tick_params(axis="y", colors=C_SNR)
    ax3.spines["right"].set_color(C_SNR)

    ax.grid(True, which="both", linestyle="--", alpha=0.3)


def plot_load_curves(rows, out_png, resistance):
    channels = sorted(set(r["channel"] for r in rows))
    n = len(channels)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.0 * n), squeeze=False, sharex=True)
    axf = axes.ravel()
    for ax, ch in zip(axf, channels):
        _panel(ax, [r for r in rows if r["channel"] == ch], resistance)
        ax.set_title(f"Ch {ch}", fontsize=12, fontweight="bold", loc="left")
    axf[-1].set_xlabel(r"I$_{bol}$ (nA)   [ = V$_{bias}$ / R,  R = "
                       f"{resistance/1e9:g} G$\\Omega$ ]", fontsize=12)
    fig.suptitle(f"Load curves — Measurement {MEAS_NAME}", fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.80, top=0.95, bottom=0.06, hspace=0.35)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"  → {os.path.basename(out_png)}  ({n} canali)")


def main():
    parser = argparse.ArgumentParser(description="Curve di carico per canale (m205), stile classico.")
    parser.add_argument("--bi-csv", default=DEFAULT_BI_CSV,
                        help="CSV dei risultati BI (contiene V_bias, ampiezza, RMS, SNR)")
    parser.add_argument("--outdir", default=None,
                        help="cartella di output (default: accanto al BI CSV)")
    parser.add_argument("--resistance", type=float, default=R_DEFAULT,
                        help="resistenza di carico in Ohm (default 2e9 = 2 GOhm)")
    parser.add_argument("--exclude", nargs="*", type=int, default=None,
                        help="canali da escludere, es. --exclude 91")
    args = parser.parse_args()

    if not os.path.exists(args.bi_csv):
        raise SystemExit(f"[ERROR] CSV BI non trovato: {args.bi_csv}")

    rows = read_load_curve(args.bi_csv)
    if not rows:
        raise SystemExit(f"[ERROR] Nessun dato leggibile in {args.bi_csv}")

    exclude = set(EXCLUDE_CHANNELS) | set(args.exclude or [])
    if exclude:
        before = len(rows)
        rows = [r for r in rows if r["channel"] not in exclude]
        print(f"Canali esclusi {sorted(exclude)}: rimosse {before - len(rows)} righe.")
    if not rows:
        raise SystemExit("[ERROR] Tutti i dati sono stati esclusi.")

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.bi_csv))
    os.makedirs(outdir, exist_ok=True)

    n_ch = len(set(r["channel"] for r in rows))
    print(f"Dati: {len(rows)} punti su {n_ch} canali (R = {args.resistance/1e9:g} GOhm).")
    print(f"Genero l'output in {outdir}:")
    plot_load_curves(rows, os.path.join(outdir, "load_curves_m205.png"), args.resistance)
    print("\nFatto.")


if __name__ == "__main__":
    main()
