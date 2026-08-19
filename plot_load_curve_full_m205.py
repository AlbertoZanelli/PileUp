"""
plot_load_curve_full_m205.py
============================
Curve di carico "complete" (stile figura di riferimento, 4 assi y colorati) per i
9 canali della m205, in una griglia 3x3. Un pannello per canale.

Grandezze e sorgenti (unite per (canale, Bias_V)):
  - V_bol (mV)        -> blu,  cerchi pieni     [Excel load curves 205, colonna V_Bol]
  - AP Amplitude (mV) -> rosso, quadrati pieni  [BI_results_m205.csv, signal_amp]
  - OF RMS (mV)       -> viola, triangoli giu'  [BI_results_m205.csv, sigma_analytic]
  - OF SNR            -> verde, quadrati vuoti   [BI_results_m205.csv, SNR]

Asse x: corrente del bolometro  I_bol = V_bias / R_load  (R_load = 2.069 GOhm), in nA.

Produce inoltre:
  - load_curve_power_m205.png : R_bol (asse y) vs potenza P = V_bol·I_calc (asse x),
    con I_calc = V_bias/R_load, in doppia logaritmica, griglia 3x3.
  - load_curve_data_m205.txt  : dump CSV di TUTTE le grandezze prese dall'Excel
    (V_bol, I_bol, R_bol, R_load) per (canale, bias), condiviso con altri programmi
    (es. plot_BI_results.py lo legge per R_bol) così non devono ri-parsare l'.xlsx.

Uso:
    python plot_load_curve_full_m205.py
"""

import os
import re
import csv
import zipfile
import argparse
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


# ── Lettura .xlsx con la sola libreria standard (niente openpyxl / pandas) ─────
def _col_index(cell_ref):
    letters = re.match(r"([A-Z]+)", cell_ref).group(1)
    n = 0
    for c in letters:
        n = n * 26 + (ord(c) - 64)
    return n - 1


def read_xlsx(path, sheet="xl/worksheets/sheet1.xml"):
    """Legge il primo foglio di un .xlsx -> lista di dict {header: valore}."""
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(NS + "si"):
            shared.append("".join(t.text or "" for t in si.iter(NS + "t")))
    ws = ET.fromstring(z.read(sheet))
    raw = []
    for row in ws.iter(NS + "row"):
        cells = {}
        for c in row.findall(NS + "c"):
            v = c.find(NS + "v")
            if v is not None:
                val = shared[int(v.text)] if c.get("t") == "s" else v.text
            else:
                isv = c.find(NS + "is")
                val = "".join(x.text or "" for x in isv.iter(NS + "t")) if isv is not None else None
            cells[_col_index(c.get("r"))] = val
        raw.append(cells)
    if not raw:
        return []
    ncol = max(raw[0]) + 1
    names = [(raw[0].get(i) or "").strip() for i in range(ncol)]
    return [{names[i]: r.get(i) for i in range(ncol)} for r in raw[1:]]


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _get(row, *aliases):
    low = {k.lower(): v for k, v in row.items() if k}
    for a in aliases:
        if a.lower() in low:
            return low[a.lower()]
    return None


BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BI_CSV = os.path.join(BASE_DIR, "m205_results_octopus", "BI_results_m205.csv")
# File Excel di input (nome reale sul disco; contiene V_bol/I_bol della run 205).
DEFAULT_XLSX   = os.path.join(BASE_DIR, "20260406_RUN14_load_curves_25mK.xlsx")
# Dump di tutte le grandezze prese dall'Excel, in un txt CSV condiviso nella root
# del progetto, così altri programmi (es. plot_BI_results.py per R_bol) lo leggono
# senza dover ri-parsare l'.xlsx con la stdlib.
DEFAULT_TXT    = os.path.join(BASE_DIR, "load_curve_data_m205.txt")
CHANNELS = [31, 34, 37, 40, 41, 71, 83, 91, 94]
R_LOAD   = 2.069e9        # Ohm

C_VBOL, C_AMP, C_RMS, C_SNR = "#2b6cb4", "#c0392b", "#9b59b6", "#5aa02c"
C_POWER = "#b8860b"       # curva Potenza - I_bol


def read_bi(path):
    """(canale, bias) -> (amp_mV, rms_mV, snr) da BI_results_m205.csv."""
    out = {}
    for row in csv.DictReader(open(path, newline="")):
        ch, vb = _to_float(row.get("channel")), _to_float(row.get("vbias"))
        amp, rms, snr = (_to_float(row.get("signal_amp")), _to_float(row.get("sigma_analytic")),
                         _to_float(row.get("SNR")))
        if ch is None or vb is None:
            continue
        out[(int(ch), round(vb, 3))] = (amp * 1e3 if amp else None,
                                        rms * 1e3 if rms else None, snr)
    return out


# Colonne prese dall'Excel (chiave interna -> alias nel foglio). bias_V esclusa: e'
# la chiave. Unita' come nell'Excel: V_bol in V, I_bol in A, R_bol/R_load in Ohm.
EXCEL_COLS = [("V_bol_V", ("V_Bol", "V_bol")), ("I_bol_A", ("I_Bol", "I_bol")),
              ("R_bol_Ohm", ("R_Bol", "R_bol")), ("R_load_Ohm", ("R_Load", "R_load"))]


def read_excel(path):
    """(canale, bias) -> {V_bol_V, I_bol_A, R_bol_Ohm, R_load_Ohm} dal file Excel
    delle load curves (run 205). Il canale e' l'intero prima del '-' in 'Name'."""
    out = {}
    for row in read_xlsx(path):
        name = _get(row, "Name")
        m = re.match(r"\s*(\d+)\s*-", name) if name else None
        bias = _to_float(_get(row, "Bias_V"))
        if not (m and bias is not None):
            continue
        out[(int(m.group(1)), round(bias, 3))] = {
            key: _to_float(_get(row, *aliases)) for key, aliases in EXCEL_COLS}
    return out


def write_excel_txt(excel, path):
    """Salva tutte le grandezze prese dall'Excel in un txt CSV (una riga per
    (canale, bias)), per condividerle con altri programmi senza ri-leggere l'.xlsx."""
    cols = [k for k, _ in EXCEL_COLS]
    with open(path, "w", newline="") as f:
        f.write("# Load-curve quantities extracted from the Excel (run 205, 25 mK), "
                "one row per (channel, bias). Units: V in V, I in A, R in Ohm.\n")
        w = csv.writer(f)
        w.writerow(["channel", "bias_V"] + cols)
        for (ch, bias) in sorted(excel):
            d = excel[(ch, bias)]
            w.writerow([ch, f"{bias:.3f}"] +
                       [f"{d[k]:.6g}" if d[k] is not None else "" for k in cols])
    print(f"  → {path}")


def build(bi, excel):
    """{canale: [righe ordinate per bias]} unendo BI ed Excel sui bias comuni. La
    corrente e' SEMPRE I_calc = V_bias / R_load; la potenza dissipata dal bolometro
    e' P = V_bol · I_calc (V_bol reale dall'Excel), da plottare vs R_bol."""
    data = {}
    for (ch, bias) in sorted(set(bi) & set(excel)):
        if ch not in CHANNELS:
            continue
        amp, rms, snr = bi[(ch, bias)]
        e = excel[(ch, bias)]
        vbol_V, rbol_Ohm = e["V_bol_V"], e["R_bol_Ohm"]
        i_calc_A = bias / R_LOAD                        # I_calc = V_bias / R_load
        data.setdefault(ch, []).append({
            "I": i_calc_A * 1e9,               # I_calc in nA (asse dei 4-assi)
            "vbol": vbol_V * 1e3 if vbol_V is not None else None,   # mV
            "amp": amp, "rms": rms, "snr": snr,
            "r_bol_MOhm": rbol_Ohm / 1e6 if rbol_Ohm is not None else None,
            "power_pW": vbol_V * i_calc_A * 1e12 if vbol_V is not None else None,   # P = V_bol·I_calc, pW
        })
    for ch in data:
        data[ch].sort(key=lambda r: r["I"])
    return data


def _panel(ax, d):
    """Un pannello a 4 assi y (stile figura di riferimento)."""
    x = [r["I"] for r in d]
    ax.plot(x, [r["vbol"] for r in d], "-o", color=C_VBOL, ms=5, lw=1.4)
    ax.set_ylabel(r"V$_{bol}$ (mV)", color=C_VBOL, fontsize=14)
    ax.tick_params(axis="y", colors=C_VBOL, labelsize=12)
    ax.tick_params(axis="x", labelsize=12)
    ax.spines["left"].set_color(C_VBOL)

    ax1 = ax.twinx()
    ax1.plot(x, [r["amp"] for r in d], "-s", color=C_AMP, ms=5, lw=1.4)
    ax1.set_ylabel("AP Amplitude (mV)", color=C_AMP, fontsize=14)
    ax1.tick_params(axis="y", colors=C_AMP, labelsize=12)
    ax1.spines["right"].set_color(C_AMP)

    ax2 = ax.twinx()
    ax2.spines["right"].set_position(("outward", 52))
    ax2.plot(x, [r["rms"] for r in d], "-v", color=C_RMS, ms=5, lw=1.4)
    ax2.set_ylabel("OF RMS (mV)", color=C_RMS, fontsize=14)
    ax2.tick_params(axis="y", colors=C_RMS, labelsize=12)
    ax2.spines["right"].set_color(C_RMS)

    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("outward", 108))
    ax3.plot(x, [r["snr"] for r in d], "-s", color=C_SNR, ms=6, lw=1.4,
             mfc="white", mec=C_SNR, mew=1.4)
    ax3.set_ylabel("OF SNR", color=C_SNR, fontsize=14)
    ax3.tick_params(axis="y", colors=C_SNR, labelsize=12)
    ax3.spines["right"].set_color(C_SNR)

    ax.grid(True, linestyle="--", alpha=0.3)


def plot_power(data, out_png):
    """Curva R_bol vs potenza dissipata: R_bol (asse y) vs P = V_bol·I_calc (asse x),
    in doppia logaritmica, griglia 3x3 (un pannello per canale). I_calc = V_bias /
    R_load; V_bol e R_bol sono i valori dell'Excel."""
    import math
    chs = sorted(data)
    ncols = 3
    nrows = math.ceil(len(chs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 4.1 * nrows), squeeze=False)
    axf = axes.ravel()
    for ax, ch in zip(axf, chs):
        d = [r for r in data[ch] if r["r_bol_MOhm"] not in (None, 0) and r["power_pW"] not in (None, 0)]
        d.sort(key=lambda r: r["power_pW"])
        if d:
            ax.loglog([r["power_pW"] for r in d], [r["r_bol_MOhm"] for r in d],
                      "-o", color=C_POWER, ms=5, lw=1.4)
        ax.set_title(f"Ch {ch}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Power (pW)", fontsize=11)
        ax.set_ylabel(r"R$_{bol}$ (M$\Omega$)", fontsize=11)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
    for ax in axf[len(chs):]:
        ax.axis("off")
    fig.suptitle(r"Load curves — R$_{bol}$ vs bolometer power  P = V$_{bol}\cdot$I$_{calc}$"
                 r"  (I$_{calc}$ = V$_{bias}$/R$_{load}$,  run 205, 25 mK)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    print(f"  → {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Curve di carico complete (griglia 3x3) per la m205.")
    parser.add_argument("--bi-csv", default=DEFAULT_BI_CSV)
    parser.add_argument("--xlsx", default=DEFAULT_XLSX)
    parser.add_argument("--outdir", default=os.path.join(BASE_DIR, "m205_load_curves"))
    parser.add_argument("--txt", default=DEFAULT_TXT, help="dump txt delle grandezze Excel")
    args = parser.parse_args()

    for label, path in (("BI", args.bi_csv), ("Excel", args.xlsx)):
        if not os.path.exists(path):
            raise SystemExit(f"[ERROR] file {label} non trovato: {path}")

    excel = read_excel(args.xlsx)
    # Dump txt di tutte le grandezze Excel (condiviso con altri programmi).
    write_excel_txt(excel, args.txt)

    data = build(read_bi(args.bi_csv), excel)
    if not data:
        raise SystemExit("[ERROR] nessuna coppia (canale, bias) in comune tra BI CSV ed Excel.")

    os.makedirs(args.outdir, exist_ok=True)
    chs = sorted(data)

    # Griglia 3x3 a posizionamento manuale: i pannelli host mantengono un aspetto
    # ~1.26:1 (come il grid V_bol-I_bol), mentre gap_x lascia spazio fisso ai 3 assi
    # y offset a destra di ciascun pannello (senza sovrapporsi alla colonna dopo).
    W, H = 21.0, 14.0                      # dimensioni figura (inch)
    left, top_pad, bot_pad = 0.85, 1.15, 1.25
    w, h = 3.9, 3.05                       # host: 3.9 x 3.05 -> aspetto 1.28
    gap_x, gap_y = 3.15, 1.2              # gap_x ospita gli assi offset (verde OF SNR incluso)
    fig = plt.figure(figsize=(W, H))
    for k, ch in enumerate(chs):
        r, c = divmod(k, 3)
        l_in = left + c * (w + gap_x)
        b_in = H - top_pad - r * (h + gap_y) - h
        ax = fig.add_axes([l_in / W, b_in / H, w / W, h / H])
        _panel(ax, data[ch])
        ax.set_title(f"Ch {ch}", fontsize=15, fontweight="bold", loc="left")
    fig.text(0.5, 0.035, r"I$_{bol}$ (nA)   [ = V$_{bias}$ / R$_{load}$,  R = 2.069 G$\Omega$ ]",
             ha="center", fontsize=16)
    fig.suptitle("Load curves — run 205 (25 mK) + OF quantities",
                 fontsize=19, fontweight="bold", y=0.98)
    out_png = os.path.join(args.outdir, "load_curve_full_m205.png")
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    print(f"Canali: {chs}  (R_Load = {R_LOAD/1e9:g} GOhm)")
    print(f"  → {out_png}")

    # Curva di potenza P = V_bol·I_bol vs I_bol (doppia logaritmica), griglia 3x3.
    plot_power(data, os.path.join(args.outdir, "load_curve_power_m205.png"))


if __name__ == "__main__":
    main()
