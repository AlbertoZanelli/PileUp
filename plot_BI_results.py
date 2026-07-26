"""
plot_BI_results.py
==================
Analisi e plot dei risultati di stima del BI (Background Index) della misura m205.

Quale set di risultati analizzare si sceglie con l'UNICA variabile SUFFIX in testa
al file (sezione CONFIGURAZIONE): da SUFFIX derivano la cartella dei risultati, il
CSV dei BI, la cartella dei filtri e il suffisso di tutti i file di output.
  SUFFIX = ""             -> filtro ottimo          (m205_results_octopus)
  SUFFIX = "_wiener"      -> Wiener lambda scalare   (m205_results_wiener)
  SUFFIX = "_wiener_freq" -> Wiener lambda(f)        (m205_results_wiener_freq)

Nessun argomento da linea di comando: si configura tutto dalle costanti in testa.

Sorgenti dati:
  - BI_results_m205<SUFFIX>.csv : channel, wp, vbias, signal_amp, sigma_analytic,
                                  SNR, BI, J_final; opzionali beta_Hz, rho_t
                                  (=SNR*beta) e lambda_wiener (Wiener scalare)
  - amplitudes_m205.csv (condiviso): channel, vbias_V, risetime_ms, decaytime_ms,
                                     amplitude_mV, hf_power
  - trained_filters/*.npy : filtri f1/f2 e lambda(f) addestrati (versioni Wiener)

Plot prodotti (nella cartella dei risultati, col suffisso):
  - BI_vs_parameters_m205.png            (3x3) BI vs V_bias/risetime/decaytime/SNR/
                                         amplitude/sigma/SNR-over-risetime/SNR*beta/HF-power
  - BI_vs_parameters_vbiascolor_m205.png (3x3) come sopra, hue=canale, shade=V_bias
  - params_vs_Vbias_m205.png             (2x2) amplitude/sigma/SNR/HF-power vs V_bias
  - BI_vs_Vbias_m205.png                 BI vs V_bias (figura dedicata)
  - lambda_vs_Vbias_m205.png             lambda del Wiener vs V_bias (solo Wiener scalare)
  I plot dei filtri finiscono nella sottocartella  filter_plots/  (uno per canale):
  - trained_filters_ch*_m205.png         f1, f2 vs frequenza (se ci sono i .npy)
  - trained_lambda_ch*_m205.png          lambda(f) vs frequenza (Wiener freq-dipendente)
  - total_filters_ch*_m205.png           filtri totali g_i = f_i · W (W ricostruito
                                         da ROOT + lambda); |g_i(f)|, freq positive

Modalità BAD_CHANNELS: se True analizza i 4 canali scartati (37, 40, 41, 94) invece
dei 5 buoni, aggiunge "_bad" al nome delle immagini per non sovrascrivere quelle
standard e usa una PALETTE DIVERSA (Dark2, tonalità scure) invece di quella dei
canali buoni (tab10), così le figure delle due modalità non si confondono.
Altrimenti analizza i 5 buoni (31, 34, 71, 83, 91).
"""

import os
import re
import csv
import glob
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE  —  quale set di risultati analizzare
# ═════════════════════════════════════════════════════════════════════════════
# Basta cambiare SUFFIX: da qui derivano la cartella dei risultati, il CSV dei BI,
# la cartella dei filtri e il suffisso di TUTTI i file di output.
#   ""             -> filtro ottimo         (cartella m205_results_octopus)
#   "_wiener"      -> Wiener lambda scalare  (cartella m205_results_wiener)
#   "_wiener_freq" -> Wiener lambda(f)       (cartella m205_results_wiener_freq)
SUFFIX = "_wiener"

# Modalità "bad channels": se True analizza i 4 canali normalmente scartati
# (37, 40, 41, 94) invece dei 5 buoni, e alle immagini viene aggiunto "_bad" nel
# nome per non sovrascrivere quelle standard.
BAD_CHANNELS = True

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "m205_results" + (SUFFIX or "_octopus"))
BI_CSV      = os.path.join(RESULTS_DIR, "BI_results_m205" + SUFFIX + ".csv")
FILTERS_DIR = os.path.join(RESULTS_DIR, "trained_filters")     # .npy di f1/f2/lambda(f)
TIMING_CSV  = os.path.join(RESULTS_DIR, "timing_SNR_m205.csv")  # fallback per rho_t
AMP_CSV     = os.path.join(BASE_DIR, "amplitudes_m205.csv")     # condiviso tra i set
OUTDIR      = RESULTS_DIR                                       # output accanto ai dati

MEAS_NAME     = "000205"
SAMPLING_RATE = 10_000.0   # m205: 10 kHz (asse frequenze dei filtri)
WINDOW_SIZE   = 10_000     # campioni per finestra (per ricostruire il kernel Wiener)
PROCESSED_DIR = os.path.join(BASE_DIR, "Processed")   # file ROOT con AP e NPS

# Canali da escludere, tag nel nome dei file e PALETTE, secondo la modalità
# BAD_CHANNELS. La palette e' diversa nelle due modalità (tab10 = brillante per i
# canali buoni, Dark2 = tonalità scure/smorzate per i cattivi) così le figure delle
# due modalità sono distinguibili a colpo d'occhio anche messe una accanto all'altra.
if BAD_CHANNELS:
    EXCLUDE_CHANNELS = [31, 34, 40, 71, 83, 91]   # tieni i "cattivi": 37, 40, 41, 94
    NAME_TAG = "_bad"
    CHANNEL_CMAP = "Dark2"
else:
    EXCLUDE_CHANNELS = [37, 40, 41, 94]        # tieni i "buoni": 31, 34, 71, 83, 91
    NAME_TAG = ""
    CHANNEL_CMAP = "tab10"


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
    La palette (CHANNEL_CMAP) dipende dalla modalità: tab10 per i canali buoni,
    Dark2 per i cattivi. Entrambe qualitative, con hue ben separate (necessario per
    la sfumatura per-canale by V_bias)."""
    cmap = plt.get_cmap(CHANNEL_CMAP)
    chs = sorted(set(channels))
    return {ch: cmap(i % cmap.N) for i, ch in enumerate(chs)}


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


def plot_trained_filters(rows, filters_dir, p):
    """Una immagine per canale coi filtri di banda ADDESTRATI f1, f2 (dai .npy di
    analyse_BI_m205_wiener.py). Ogni immagine e' una griglia: un pannello per WP,
    con f1 e f2 sovrapposti. I .npy contengono la meta' indipendente dello spettro
    (DC..Nyquist), quindi si plottano direttamente SOLO le frequenze positive.

    Se la cartella dei filtri non esiste (es. run col filtro ottimo) non fa nulla."""
    if not os.path.isdir(filters_dir):
        print(f"[INFO] cartella filtri non trovata ({filters_dir}): salto il plot di f1/f2.")
        return
    # (canale, wp) -> V_bias, per titolare i pannelli.
    vb_of = {(r["channel"], int(r["wp"])): r["vbias"]
             for r in rows if r.get("wp") is not None}
    wp_re = re.compile(r"_wp(\d+)\.npy$")
    n_imgs = 0
    for ch in sorted(set(r["channel"] for r in rows)):
        f1_files = glob.glob(os.path.join(filters_dir, f"f1_ch{ch}_wp*.npy"))
        wps = sorted(int(wp_re.search(os.path.basename(x)).group(1)) for x in f1_files)
        if not wps:
            continue
        ncols = min(5, len(wps))
        nrows = math.ceil(len(wps) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.5 * nrows),
                                 squeeze=False)
        axf = axes.ravel()
        for ax, wp in zip(axf, wps):
            f1 = np.load(os.path.join(filters_dir, f"f1_ch{ch}_wp{wp}.npy"))
            f2_path = os.path.join(filters_dir, f"f2_ch{ch}_wp{wp}.npy")
            f2 = np.load(f2_path) if os.path.exists(f2_path) else None
            freq = np.linspace(0.0, SAMPLING_RATE / 2.0, len(f1))  # positive freqs: DC..Nyquist
            ax.plot(freq, f1, lw=1.0, color="#1f77b4", label=r"$f_1$")
            if f2 is not None:
                ax.plot(freq, f2[:len(freq)], lw=1.0, color="#d62728", label=r"$f_2$")
            # usare scala logaritmica sull'asse y per evidenziare le differenze di ampiezza
            ax.set_yscale("log")
            ax.set_xscale("log")
            vb = vb_of.get((ch, wp))
            ax.set_title(f"WP {wp}" + (f"  ·  {vb:g} V" if vb is not None else ""), fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
        for ax in axf[len(wps):]:
            ax.axis("off")
        axf[0].legend(fontsize=9, loc="upper right")
        fig.suptitle(f"Trained band filters $f_1$, $f_2$ — Ch {ch}  ·  Measurement {MEAS_NAME}",
                     fontsize=14, fontweight="bold")
        fig.supxlabel("Frequency (Hz)", fontsize=11)
        fig.supylabel("Filter amplitude", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_png = p(f"trained_filters_ch{ch}_m205.png")
        fig.savefig(out_png, dpi=180)
        plt.close(fig)
        n_imgs += 1
        print(f"  → {os.path.basename(out_png)}  ({len(wps)} WP)")
    if n_imgs == 0:
        print(f"[INFO] nessun .npy di filtri in {filters_dir}: nessuna immagine f1/f2 prodotta.")


def plot_trained_lambda(rows, filters_dir, p):
    """Una immagine per canale con la curva lambda(f) ADDESTRATA (dai .npy di
    analyse_BI_m205_wiener_freq.py, filtro di Wiener a lambda dipendente dalla
    frequenza). Griglia: un pannello per WP, lambda(f) vs frequenza. I .npy sono la
    meta' indipendente dello spettro (DC..Nyquist), quindi si plottano SOLO le
    frequenze positive. Asse y log (lambda copre molte decadi); linea a lambda=1
    (Wiener standard). Se non ci sono file lambda_*.npy non fa nulla."""
    if not os.path.isdir(filters_dir):
        return
    vb_of = {(r["channel"], int(r["wp"])): r["vbias"]
             for r in rows if r.get("wp") is not None}
    wp_re = re.compile(r"_wp(\d+)\.npy$")
    n_imgs = 0
    for ch in sorted(set(r["channel"] for r in rows)):
        lam_files = glob.glob(os.path.join(filters_dir, f"lambda_ch{ch}_wp*.npy"))
        wps = sorted(int(wp_re.search(os.path.basename(x)).group(1)) for x in lam_files)
        if not wps:
            continue
        ncols = min(5, len(wps))
        nrows = math.ceil(len(wps) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.5 * nrows),
                                 squeeze=False)
        axf = axes.ravel()
        for ax, wp in zip(axf, wps):
            lam = np.load(os.path.join(filters_dir, f"lambda_ch{ch}_wp{wp}.npy"))
            freq = np.linspace(0.0, SAMPLING_RATE / 2.0, len(lam))  # positive freqs: DC..Nyquist
            ax.plot(freq, lam, lw=1.0, color="#2ca02c")
            ax.axhline(1.0, color="gray", ls=":", lw=0.8)   # lambda=1: Wiener standard
            ax.set_yscale("log")
            vb = vb_of.get((ch, wp))
            ax.set_title(f"WP {wp}" + (f"  ·  {vb:g} V" if vb is not None else ""), fontsize=9)
            ax.grid(True, which="both", alpha=0.3)
            ax.tick_params(labelsize=7)
        for ax in axf[len(wps):]:
            ax.axis("off")
        fig.suptitle(rf"Trained Wiener $\lambda(f)$ — Ch {ch}  ·  Measurement {MEAS_NAME}",
                     fontsize=14, fontweight="bold")
        fig.supxlabel("Frequency (Hz)", fontsize=11)
        fig.supylabel(r"$\lambda(f)$", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_png = p(f"trained_lambda_ch{ch}_m205.png")
        fig.savefig(out_png, dpi=180)
        plt.close(fig)
        n_imgs += 1
        print(f"  → {os.path.basename(out_png)}  ({len(wps)} WP)")


# ═════════════════════════════════════════════════════════════════════════════
# Filtri TOTALI  g_i(f) = f_i(f) · W(f)   (banda × kernel di Wiener)
# ═════════════════════════════════════════════════════════════════════════════
# Il filtro effettivamente applicato ai dati per ogni statistica non e' il solo
# filtro di banda f_i, ma il suo PRODOTTO col kernel di Wiener W:
#
#     g_i(f) = f_i(f) · W(f) ,     W(f) = S*(f) / ( |S(f)|^2 + lambda·NPS(f) )
#
# f_i (reale) e' salvato nei .npy; W (complesso) NON e' salvato, quindi lo si
# RICOSTRUISCE qui dai file ROOT (stessa AP/NPS e stessa normalizzazione del worker
# analyse_BI_m205_wiener.py) e dal lambda addestrato (scalare dal CSV, oppure
# lambda(f) dal .npy per la variante freq-dipendente). Poi g_i = f_i · W e nel plot
# si mostra |g_i(f)| sulle frequenze positive.
def _load_ap_nps(uproot_file, wp):
    """AP (meanpulse) e NPS two-sided da un file ROOT aperto, per un WP. La NPS usa
    la STESSA normalizzazione del worker (× 5.708 × window^2 × 1/sampling_time)."""
    try:
        mp = np.asarray(uproot_file[f"averagepulse_ap_wp{wp}_medianAP"].values(), dtype=float)
        nps = np.asarray(uproot_file[f"averagepowerspectrum_noise_wp{wp}_medianpower"].values(), dtype=float)
    except Exception:
        return None, None
    nps = np.concatenate([nps, nps[-2:0:-1]])                  # one-sided -> two-sided
    nps = nps * 5.708 * (WINDOW_SIZE ** 2) * (SAMPLING_RATE / WINDOW_SIZE)  # = ×(1/sampling_time)
    return mp, nps


def _wiener_kernel_half(meanpulse, nps, lam):
    """Kernel di Wiener W(f) sulla meta' indipendente (DC..Nyquist), ricostruito con
    la stessa formula di compute_W_torch (CUORE norm_type=0):

        AvgPS = |S|^2,  a = sum|S|,  b = sum sqrt(NPS)
        W = conj(S) / ( AvgPS/a + lam · NPS·a/b^2 ),   W[0] = 0

    S = FFT del pulse finestrato (Hanning), come compute_H. lam puo' essere uno
    scalare (Wiener a lambda unico) o un array sulla meta' indipendente (lambda(f))."""
    S = np.fft.fft(meanpulse * np.hanning(len(meanpulse)))
    AvgPS = np.abs(S) ** 2
    a = np.sum(np.sqrt(AvgPS))
    b = np.sum(np.sqrt(nps))
    AvgPS_n = AvgPS / a
    nps_n = nps * (a / b ** 2)
    lam = np.asarray(lam, dtype=float)
    if lam.ndim > 0:                                           # lambda(f): meta' -> spettro pieno
        lam = np.concatenate([lam, lam[-2:0:-1]])
    W = np.conj(S) / (AvgPS_n + lam * nps_n)
    W[0] = 0.0
    return W[: len(meanpulse) // 2 + 1]


def plot_total_filters(rows, filters_dir, p):
    """Una immagine per canale coi FILTRI TOTALI g_i = f_i · kernel (banda × kernel).
    Il kernel viene letto dal .npy `kernel_ch*_wp*` se presente (lo salvano i worker:
    W per il Wiener, H_unit per il filtro ottimo); altrimenti, per i run Wiener
    vecchi, viene RICOSTRUITO dai file ROOT (AP+NPS) + lambda. Griglia: un pannello
    per WP, |g_1| e |g_2| sovrapposti, solo frequenze positive (scala log)."""
    if not os.path.isdir(filters_dir):
        return
    lam_scalar = {(r["channel"], int(r["wp"])): r.get("lambda_wiener")
                  for r in rows if r.get("wp") is not None}
    vb_of = {(r["channel"], int(r["wp"])): r["vbias"] for r in rows if r.get("wp") is not None}
    wp_re = re.compile(r"_wp(\d+)\.npy$")
    n_imgs = 0
    for ch in sorted(set(r["channel"] for r in rows)):
        f1_files = glob.glob(os.path.join(filters_dir, f"f1_ch{ch}_wp*.npy"))
        wps = sorted(int(wp_re.search(os.path.basename(x)).group(1)) for x in f1_files)
        if not wps:
            continue
        # Il ROOT serve solo se manca qualche kernel .npy (fallback ricostruzione).
        need_root = any(not os.path.exists(os.path.join(filters_dir, f"kernel_ch{ch}_wp{wp}.npy"))
                        for wp in wps)
        rf = None
        if need_root:
            try:
                import uproot
                roots = glob.glob(os.path.join(PROCESSED_DIR, f"Processed_*_{MEAS_NAME}_{ch}.root"))
                rf = uproot.open(roots[0]) if roots else None
            except ImportError:
                rf = None
        ncols = min(5, len(wps))
        nrows = math.ceil(len(wps) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.5 * nrows), squeeze=False)
        axf = axes.ravel()
        try:
            for ax, wp in zip(axf, wps):
                f1 = np.load(os.path.join(filters_dir, f"f1_ch{ch}_wp{wp}.npy"))
                f2 = np.load(os.path.join(filters_dir, f"f2_ch{ch}_wp{wp}.npy"))
                kernel_path = os.path.join(filters_dir, f"kernel_ch{ch}_wp{wp}.npy")
                if os.path.exists(kernel_path):
                    W = np.load(kernel_path)                      # kernel salvato (meta' indip.)
                elif rf is not None:                              # fallback: ricostruisci W
                    lam_path = os.path.join(filters_dir, f"lambda_ch{ch}_wp{wp}.npy")
                    lam = np.load(lam_path) if os.path.exists(lam_path) else lam_scalar.get((ch, wp))
                    mp, nps = _load_ap_nps(rf, wp)
                    W = (_wiener_kernel_half(mp, nps, lam)
                         if (lam is not None and mp is not None) else None)
                else:
                    W = None
                if W is None:
                    ax.axis("off"); continue
                m = min(len(f1), len(f2), len(W))
                freq = np.linspace(0.0, SAMPLING_RATE / 2.0, m)   # solo frequenze positive
                ax.plot(freq, np.abs(f1[:m] * W[:m]), lw=1.0, color="#1f77b4", label=r"$g_1=f_1 W$")
                ax.plot(freq, np.abs(f2[:m] * W[:m]), lw=1.0, color="#d62728", label=r"$g_2=f_2 W$")
                ax.set_yscale("log"); ax.set_xscale("log")
                vb = vb_of.get((ch, wp))
                ax.set_title(f"WP {wp}" + (f"  ·  {vb:g} V" if vb is not None else ""), fontsize=9)
                ax.grid(True, which="both", alpha=0.3)
                ax.tick_params(labelsize=7)
        finally:
            if rf is not None:
                rf.close()
        for ax in axf[len(wps):]:
            ax.axis("off")
        axf[0].legend(fontsize=9, loc="upper right")
        fig.suptitle(rf"Total filters $g_i = f_i \cdot W$ — Ch {ch}  ·  Measurement {MEAS_NAME}",
                     fontsize=14, fontweight="bold")
        fig.supxlabel("Frequency (Hz)", fontsize=11)
        fig.supylabel(r"$|g_i(f)|$", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_png = p(f"total_filters_ch{ch}_m205.png")
        fig.savefig(out_png, dpi=180)
        plt.close(fig)
        n_imgs += 1
        print(f"  → {os.path.basename(out_png)}  ({len(wps)} WP)")
    if n_imgs == 0:
        print(f"[INFO] nessun filtro totale prodotto (mancano .npy/lambda/ROOT).")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    if not os.path.exists(BI_CSV):
        raise SystemExit(f"[ERROR] CSV BI non trovato: {BI_CSV}")

    rows = read_bi_results(BI_CSV)
    if not rows:
        raise SystemExit(f"[ERROR] Nessun dato leggibile in {BI_CSV}")

    amap = read_amplitudes(AMP_CSV)
    merge_timing(rows, amap)

    # rho_t = SNR*beta. I CSV arricchiti (OF e Wiener) lo contengono gia' inline:
    # il CSV timing serve solo da fallback per le righe che non ce l'hanno.
    if any(r.get("rho_t") is None for r in rows):
        tmap = read_timing(TIMING_CSV)
        for r in rows:
            if r.get("rho_t") is None:
                r["rho_t"] = tmap.get((r["channel"], round(r["vbias"], 3)))

    # ── Esclusione canali ──────────────────────────────────────────────────────
    if EXCLUDE_CHANNELS:
        before = len(rows)
        rows = [r for r in rows if r["channel"] not in EXCLUDE_CHANNELS]
        print(f"Canali esclusi {sorted(EXCLUDE_CHANNELS)}: rimosse {before - len(rows)} righe.")
        if not rows:
            raise SystemExit("[ERROR] Tutti i dati sono stati esclusi.")

    os.makedirs(OUTDIR, exist_ok=True)
    colors = channel_colors([r["channel"] for r in rows])

    def p(name):
        """Percorso di output: inserisce SUFFIX (set di risultati) e NAME_TAG
        (modalità bad-channels) prima dell'estensione, così tutti i file cambiano
        nome coerentemente e le due modalità non si sovrascrivono."""
        root, ext = os.path.splitext(name)
        return os.path.join(OUTDIR, f"{root}{SUFFIX}{NAME_TAG}{ext}")

    # Le immagini dei filtri (f1/f2, lambda(f), totali f·W) vanno in una cartella
    # dedicata, per canale/WP, dentro la cartella dei risultati.
    filter_dir = os.path.join(OUTDIR, "filter_plots")
    os.makedirs(filter_dir, exist_ok=True)

    def pf(name):
        """Come p(), ma nella sottocartella dedicata ai plot dei filtri."""
        root, ext = os.path.splitext(name)
        return os.path.join(filter_dir, f"{root}{SUFFIX}{NAME_TAG}{ext}")

    n_ch = len(set(r["channel"] for r in rows))
    print(f"Dati: {len(rows)} punti su {n_ch} canali.\nGenero i plot in {OUTDIR}:")

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

    # ── Plot dei filtri, nella sottocartella dedicata (un'immagine per canale) ──
    plot_trained_filters(rows, FILTERS_DIR, pf)   # filtri di banda f1, f2
    plot_trained_lambda(rows, FILTERS_DIR, pf)    # lambda(f) (Wiener freq-dipendente)
    plot_total_filters(rows, FILTERS_DIR, pf)     # filtri totali g_i = f_i · W

    print("\nFatto.")

if __name__ == "__main__":
    main()