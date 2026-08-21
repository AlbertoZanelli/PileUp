"""
compare_wiener_vs_optimum_m205.py
=================================
Confronta DUE set di risultati BI (m205) qualsiasi, punto per punto (canale,
V_bias). Generico: "A" e' il riferimento (baseline), "B" e' il risultato che si
vuole valutare.

Quali due set confrontare si sceglie con le DUE variabili SUFFIX_A e SUFFIX_B in
testa al file (sezione CONFIGURAZIONE), con gli stessi suffissi di
plot_BI_results.py: da ognuna derivano la cartella dei risultati e il CSV dei BI,
e dalla coppia derivano etichette, tag e cartella di output.
  ""             -> filtro ottimo          (m205_results_octopus)
  "_wiener"      -> Wiener lambda scalare   (m205_results_wiener)
  "_wiener_freq" -> Wiener lambda(f)        (m205_results_wiener_freq)
Casi d'uso tipici:
  - SUFFIX_A = ""        , SUFFIX_B = "_wiener"       -> ottimo vs Wiener scalare
  - SUFFIX_A = "_wiener" , SUFFIX_B = "_wiener_freq"  -> Wiener scalare vs lambda(f)

Nessun argomento da linea di comando: si configura tutto dalle costanti in testa.

Il BI e' da MINIMIZZARE: il "miglioramento" di B su A e' la DIMINUZIONE percentuale
del BI, punto per punto:

    improvement_% = 100 * (BI_A - BI_B) / BI_A

  > 0  -> B abbassa il BI rispetto ad A (meglio)
  < 0  -> B peggiora il BI

Output: una CARTELLA DEDICATA che esplicita il confronto,
    <root>/comparisons/<tag>/            (es. comparisons/OF_vs_WF/)
dove <tag> deriva dalla coppia di suffissi (OF, WF, WFfreq). Dentro, col <tag> nel
nome dei file:
  - BI_improvement_vs_Vbias_<tag>_m205.png    : miglioramento % vs V_bias, per canale
  - BI_improvement_per_channel_<tag>_m205.png : miglioramento medio per canale (bar)
  - BI_vs_Vbias_<tag>_m205.png                : griglia per canale, curve BI vs V_bias (A e B)
  - BI_vs_SNRbeta_<tag>_m205.png              : idem ma BI vs SNR*beta (rho_t condiviso)
  - BI_improvement_<tag>_m205.csv             : tabella punto-per-punto + riepilogo
  - total_filters/total_f1_compare_ch*_<tag>.png : confronto del filtro TOTALE
        g=f1·kernel dei due modelli (griglia per WP; sotto ogni pannello il rapporto
        B/A). Idem total_f2_compare_ch*. Richiede i .npy dei filtri di entrambi i
        modelli (in <dir del CSV>/trained_filters); il kernel e' letto dal .npy o
        ricostruito da ROOT + lambda.
A video: tabella dei miglioramenti medi per canale e miglioramento medio globale.

Modalità BAD_CHANNELS (come in plot_BI_results.py): se True confronta i 4 canali
scartati (37, 40, 41, 94) invece dei 5 buoni, aggiunge "_bad" al nome di tutti i
file e usa una palette diversa per i canali (Dark2 invece di tab10), così le figure
delle due modalità convivono nella stessa cartella senza confondersi.

I CSV dei risultati vengono cercati nella root del progetto (la cartella che
contiene m205_results_octopus), sia che lo script stia in quella root sia in una
sottocartella come m204_comparison/.
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

# Root del progetto = cartella che contiene m205_results_octopus. Cosi' lo script
# trova i dati sia se sta nella root sia se sta in una sottocartella (m204_comparison/).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_root():
    for cand in (_SCRIPT_DIR, os.path.dirname(_SCRIPT_DIR)):
        if os.path.isdir(os.path.join(cand, "m205_results_octopus")):
            return cand
    return _SCRIPT_DIR


ROOT = _find_root()
MEAS_NAME = "000205"

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE  —  quali due set confrontare
# ═════════════════════════════════════════════════════════════════════════════
# Stessi suffissi di plot_BI_results.py: A = riferimento (baseline), B = valutato.
#   ""             -> filtro ottimo         (cartella m205_results_octopus)
#   "_wiener"      -> Wiener lambda scalare  (cartella m205_results_wiener)
#   "_wiener_freq" -> Wiener lambda(f)       (cartella m205_results_wiener_freq)
SUFFIX_A = "_wiener"        # riferimento: Wiener su medianAP del ROOT
SUFFIX_B = "_wiener_fit"    # confronto:   Wiener su template FITTATO

# Modalità "bad channels": se True confronta i 4 canali normalmente scartati
# (37, 40, 41, 94) invece dei 5 buoni; ai file viene aggiunto "_bad" nel nome e i
# canali usano una palette diversa, per non confondere le due modalità.
BAD_CHANNELS = False

# Canali da escludere IN AGGIUNTA a quelli della modalità (es. [91] per lasciare
# fuori il canale anomalo).
EXTRA_EXCLUDE = [31, 34, 71, 83]   # lo scan dei fit ha girato solo sul canale 91

# Etichette (per titoli e legende, in inglese) e sigle (per il tag del confronto)
# dei set di risultati, indicizzate dal suffisso.
SET_LABELS = {"": "Optimum filter", "_wiener": "Wiener (scalar λ)",
              "_wiener_freq": "Wiener (λ(f))",
              "_wiener_fit": "Wiener on fitted template",     # analysis_BI_..._regolarized, TEMPLATE_SOURCE="fit"
              "_wiener_root_R": "Wiener + R(f), β=2"}         # idem, TEMPLATE_SOURCE="root" + USE_R
SET_CODES  = {"": "OF", "_wiener": "WF", "_wiener_freq": "WFfreq",
              "_wiener_fit": "WFfit", "_wiener_root_R": "WFrootR"}
# Un COLORE PER SUFFISSO, non per ruolo A/B: cosi' lo stesso set ha sempre lo stesso colore in
# tutti i confronti (in OF_vs_WF e in WF_vs_WFfit il Wiener scalare resta arancione), e figure
# di confronti diversi non si confondono fra loro.
SET_COLORS = {"": "#1f4e79",              # navy   - filtro ottimo
              "_wiener": "#e8871e",       # ambra  - Wiener lambda scalare
              "_wiener_freq": "#2ca02c",  # verde  - Wiener lambda(f)
              "_wiener_fit": "#8c1d9c",   # viola  - Wiener su template fittato
              "_wiener_root_R": "#c02b2b"}  # rosso - Wiener + R(f)


def _bi_csv(suffix: str) -> str:
    """CSV dei risultati BI del set identificato dal suffisso (stessa convenzione di
    plot_BI_results.py: il set base "" vive nella cartella ..._octopus)."""
    return os.path.join(ROOT, "m205_results" + (suffix or "_octopus"),
                        "BI_results_m205" + suffix + ".csv")


def _label(suffix: str) -> str:
    return SET_LABELS.get(suffix, suffix.lstrip("_") or "optimum")


def _code(suffix: str) -> str:
    return SET_CODES.get(suffix, suffix.lstrip("_") or "OF")


CSV_A,   CSV_B   = _bi_csv(SUFFIX_A), _bi_csv(SUFFIX_B)
LABEL_A, LABEL_B = _label(SUFFIX_A),  _label(SUFFIX_B)
TAG              = f"{_code(SUFFIX_A)}_vs_{_code(SUFFIX_B)}"   # es. OF_vs_WF
OUTDIR           = os.path.join(ROOT, "comparisons", TAG)

# Canali esclusi, tag nel nome dei file e palette, secondo la modalità BAD_CHANNELS
# (identico a plot_BI_results.py).
if BAD_CHANNELS:
    EXCLUDE_CHANNELS = [31, 34, 40, 71, 83, 91]   # tieni i "cattivi": 37, 40, 41, 94
    NAME_TAG = "_bad"
    CHANNEL_CMAP = "Dark2"
else:
    EXCLUDE_CHANNELS = [37, 40, 41, 94]        # tieni i "buoni": 31, 34, 71, 83, 91
    NAME_TAG = ""
    CHANNEL_CMAP = "tab10"

def _color(suffix: str) -> str:
    """Colore del set, dal suffisso (vedi SET_COLORS). Per un suffisso non elencato si pesca
    un colore dalla palette tab10, deterministico sul nome, cosi' resta stabile fra i lanci."""
    if suffix in SET_COLORS:
        return SET_COLORS[suffix]
    import matplotlib.pyplot as _plt
    palette = _plt.get_cmap("tab10").colors
    return palette[sum(map(ord, suffix)) % len(palette)]


# Colori dei due set: dipendono dal SUFFISSO, non dal ruolo A/B (vedi SET_COLORS).
A_COLOR = _color(SUFFIX_A)
B_COLOR = _color(SUFFIX_B)


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
    {'BI':..., 'rho_t':...}. rho_t (=SNR*beta) e' opzionale (puo' mancare in alcuni
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
    """Una tonalità distinta e stabile per canale (coerente con plot_BI_results.py):
    la palette dipende dalla modalità, tab10 per i canali buoni e Dark2 per i cattivi."""
    cmap = plt.get_cmap(CHANNEL_CMAP)
    chs = sorted(set(channels))
    return {ch: cmap(i % cmap.N) for i, ch in enumerate(chs)}


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
# Confronto dei FILTRI TOTALI  g_i = f_i · kernel  fra i due modelli
# ═════════════════════════════════════════════════════════════════════════════
# I filtri di banda f1/f2 e il kernel sono salvati come .npy in <dir del CSV>/
# trained_filters. Il kernel viene letto dal .npy se presente, altrimenti (run
# Wiener vecchi) ricostruito da ROOT + lambda riusando gli helper di
# plot_BI_results. La AP/NPS grezza dai ROOT e' la stessa per i due modelli: cambia
# solo il lambda, quindi si apre il ROOT una volta per canale.
def _filter_dir(csv_path):
    return os.path.join(os.path.dirname(os.path.abspath(csv_path)), "trained_filters")


def read_lambda_scalar(csv_path):
    """(canale, wp) -> lambda_wiener (scalare) dal CSV, se la colonna c'e'."""
    out = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            ch, wp = _to_float(row.get("channel")), _to_float(row.get("wp"))
            lam = _to_float(row.get("lambda_wiener"))
            if ch is not None and wp is not None and lam is not None:
                out[(int(ch), int(wp))] = lam
    return out


def _kernel_half(fdir, ch, wp, lam_scalar, rf, pbr):
    """Kernel (meta' indipendente) per (canale, wp): dal .npy se c'e', altrimenti
    ricostruito da ROOT (rf aperto) + lambda (lambda(f).npy o scalare)."""
    kpath = os.path.join(fdir, f"kernel_ch{ch}_wp{wp}.npy")
    if os.path.exists(kpath):
        return np.load(kpath)
    lam_path = os.path.join(fdir, f"lambda_ch{ch}_wp{wp}.npy")
    lam = np.load(lam_path) if os.path.exists(lam_path) else lam_scalar
    if lam is None or rf is None:
        return None
    mp, nps = pbr._load_ap_nps(rf, wp)
    return pbr._wiener_kernel_half(mp, nps, lam) if mp is not None else None


def _total_half(fdir, ch, wp, which, lam_scalar, rf, pbr):
    """|g| = |f_which · kernel| (meta' indipendente) per (canale, wp), o None."""
    fpath = os.path.join(fdir, f"{which}_ch{ch}_wp{wp}.npy")
    if not os.path.exists(fpath):
        return None
    f = np.load(fpath)
    W = _kernel_half(fdir, ch, wp, lam_scalar, rf, pbr)
    if W is None:
        return None
    m = min(len(f), len(W))
    return np.abs(f[:m] * W[:m])


def plot_total_filter_compare(which, fdir_a, fdir_b, lam_a, lam_b,
                              label_a, label_b, outdir, tag, exclude=frozenset()):
    """Una immagine per canale col confronto del FILTRO TOTALE g=f·kernel dei due
    modelli (which = 'f1' o 'f2'). Griglia: un pannello per WP; in alto |g_A| e
    |g_B| sovrapposti, sotto il loro rapporto |g_B|/|g_A|."""
    import plot_BI_results as pbr   # helper ROOT + ricostruzione kernel
    try:
        import uproot
    except ImportError:
        uproot = None
    wp_re = re.compile(r"_wp(\d+)\.npy$")
    n_imgs = 0
    # canali presenti in entrambi i modelli per questo filtro (esclusi quelli in exclude)
    def chans(fdir):
        return {int(re.search(r"_ch(\d+)_wp", os.path.basename(x)).group(1))
                for x in glob.glob(os.path.join(fdir, f"{which}_ch*_wp*.npy"))}
    for ch in sorted((chans(fdir_a) & chans(fdir_b)) - set(exclude)):
        wps_a = {int(wp_re.search(os.path.basename(x)).group(1))
                 for x in glob.glob(os.path.join(fdir_a, f"{which}_ch{ch}_wp*.npy"))}
        wps_b = {int(wp_re.search(os.path.basename(x)).group(1))
                 for x in glob.glob(os.path.join(fdir_b, f"{which}_ch{ch}_wp*.npy"))}
        wps = sorted(wps_a & wps_b)
        if not wps:
            continue
        # ROOT del canale (condiviso fra i due modelli): serve solo se manca il kernel
        rf = None
        if uproot is not None:
            roots = glob.glob(os.path.join(pbr.PROCESSED_DIR, f"Processed_*_{pbr.MEAS_NAME}_{ch}.root"))
            rf = uproot.open(roots[0]) if roots else None
        ncols = min(4, len(wps))
        nrows = math.ceil(len(wps) / ncols)
        fig = plt.figure(figsize=(3.7 * ncols, 3.0 * nrows))
        # Margini stretti impostati direttamente sul gridspec (niente tight_layout,
        # che coi gridspec annidati lascia molto spazio vuoto).
        outer = fig.add_gridspec(nrows, ncols, left=0.055, right=0.995,
                                 top=0.93, bottom=0.06, wspace=0.22, hspace=0.30)
        try:
            for i, wp in enumerate(wps):
                r, c = divmod(i, ncols)
                inner = outer[r, c].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
                axm = fig.add_subplot(inner[0])
                axr = fig.add_subplot(inner[1], sharex=axm)
                gA = _total_half(fdir_a, ch, wp, which, lam_a.get((ch, wp)), rf, pbr)
                gB = _total_half(fdir_b, ch, wp, which, lam_b.get((ch, wp)), rf, pbr)
                if gA is None or gB is None:
                    axm.axis("off"); axr.axis("off"); continue
                m = min(len(gA), len(gB))
                gA, gB = gA[:m], gB[:m]
                freq = np.linspace(0.0, pbr.SAMPLING_RATE / 2.0, m)   # freq positive
                axm.plot(freq, gA, lw=1.0, color=A_COLOR, label=label_a)
                axm.plot(freq, gB, lw=1.0, color=B_COLOR, label=label_b)
                axm.set_yscale("log"); axm.set_xscale("log")
                axm.set_title(f"WP {wp}", fontsize=9)
                axm.grid(True, which="both", alpha=0.3)
                axm.tick_params(labelsize=7, labelbottom=False)
                # rapporto B/A sotto
                ratio = np.divide(gB, gA, out=np.full(m, np.nan), where=gA > 0)
                axr.plot(freq, ratio, lw=0.9, color="#555555")
                axr.axhline(1.0, color="gray", ls=":", lw=0.8)
                axr.set_yscale("log"); axr.set_xscale("log")
                axr.grid(True, which="both", alpha=0.3)
                axr.tick_params(labelsize=6)
                axr.set_ylabel("B/A", fontsize=7)
        finally:
            if rf is not None:
                rf.close()
        sub = r"$f_1$" if which == "f1" else r"$f_2$"
        fig.suptitle(rf"Total filter {sub}$\cdot$kernel — {label_b} vs {label_a} — "
                     rf"Ch {ch}  ({MEAS_NAME})", fontsize=13, fontweight="bold")
        from matplotlib.lines import Line2D
        fig.legend(handles=[Line2D([0], [0], color=A_COLOR, lw=2, label=f"A: {label_a}"),
                            Line2D([0], [0], color=B_COLOR, lw=2, label=f"B: {label_b}")],
                   loc="upper right", fontsize=9, frameon=False)
        fig.supxlabel("Frequency (Hz)", fontsize=11, y=0.005)
        fig.supylabel(rf"$|g|=|${sub}$\cdot$kernel$|$", fontsize=11, x=0.005)
        out_png = os.path.join(outdir, f"total_{which}_compare_ch{ch}_{tag}.png")
        fig.savefig(out_png, dpi=170)
        plt.close(fig)
        n_imgs += 1
        print(f"  → {os.path.basename(out_png)}  ({len(wps)} WP)")
    if n_imgs == 0:
        print(f"[INFO] confronto filtri totali {which}: filtri mancanti per uno dei due modelli, salto.")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    if SUFFIX_A == SUFFIX_B:
        raise SystemExit("[ERROR] SUFFIX_A e SUFFIX_B sono uguali: niente da confrontare.")
    for lab, path in ((LABEL_A, CSV_A), (LABEL_B, CSV_B)):
        if not os.path.exists(path):
            raise SystemExit(f"[ERROR] CSV '{lab}' non trovato: {path}")

    map_a = read_bi_map(CSV_A)
    map_b = read_bi_map(CSV_B)
    if not map_a or not map_b:
        raise SystemExit("[ERROR] Uno dei due CSV non contiene BI validi.")

    exclude = set(EXCLUDE_CHANNELS) | set(EXTRA_EXCLUDE)
    rows = build_comparison(map_a, map_b, exclude)
    if not rows:
        raise SystemExit("[ERROR] Nessuna coppia (canale, V_bias) in comune tra i due CSV.")

    # ── Miglioramento medio per canale e globale ───────────────────────────────
    channels = sorted(set(r["channel"] for r in rows))
    per_ch = {ch: mean([r["improvement_pct"] for r in rows if r["channel"] == ch])
              for ch in channels}
    global_mean = mean([r["improvement_pct"] for r in rows])            # media su tutti i punti
    per_ch_mean = mean(list(per_ch.values()))                          # media delle medie per canale

    # Cartella dedicata che esplicita il confronto: comparisons/<tag>/.
    os.makedirs(OUTDIR, exist_ok=True)
    colors = channel_colors(channels)

    n_common = len(rows)
    n_a_only = len(set(map_a) - set(map_b))
    n_b_only = len(set(map_b) - set(map_a))
    print(f"Confronto:  A = {LABEL_A}   vs   B = {LABEL_B}")
    print(f"Punti in comune: {n_common} su {len(channels)} canali "
          f"(solo-A: {n_a_only}, solo-B: {n_b_only}).")
    if exclude:
        print(f"Canali esclusi: {sorted(exclude)}"
              + ("  (modalità bad channels)" if BAD_CHANNELS else ""))
    print(f"Genero l'output in {OUTDIR}:")

    # ── Tabella a video ────────────────────────────────────────────────────────
    print(f"\n  Miglioramento % del BI (B={LABEL_B} vs A={LABEL_A})  "
          f"[>0 = B abbassa il BI]")
    print(f"  {'Ch':>4} {'mean %':>9} {'min %':>9} {'max %':>9} {'n':>4}")
    for ch in channels:
        imps = [r["improvement_pct"] for r in rows if r["channel"] == ch]
        print(f"  {ch:>4} {per_ch[ch]:>+9.2f} {min(imps):>+9.2f} {max(imps):>+9.2f} {len(imps):>4}")
    print(f"  {'-'*40}")
    print(f"  Media per canale (media delle medie): {per_ch_mean:+.2f} %")
    print(f"  Media globale (su tutti i punti):     {global_mean:+.2f} %")

    # ── Plot + CSV ─────────────────────────────────────────────────────────────
    def p(name):
        """Percorso di output: aggiunge il TAG del confronto e l'eventuale "_bad"
        (modalità bad-channels) al nome, così le due modalità non si sovrascrivono."""
        root, ext = os.path.splitext(name)
        return os.path.join(OUTDIR, f"{root}_{TAG}{NAME_TAG}_m205{ext}")

    print()
    plot_improvement_vs_vbias(rows, colors, p("BI_improvement_vs_Vbias.png"),
                              LABEL_A, LABEL_B)
    plot_improvement_per_channel(per_ch, global_mean, colors,
                                 p("BI_improvement_per_channel.png"),
                                 LABEL_A, LABEL_B)
    # Confronto diretto delle due curve BI (A vs B), un pannello per canale.
    plot_bi_compare_grid(rows, p("BI_vs_Vbias.png"), "vbias",
                         r"$V_{bias}$ (V)", f"BI vs Bias Voltage — {LABEL_B} vs {LABEL_A}",
                         LABEL_A, LABEL_B, logx=False)
    plot_bi_compare_grid(rows, p("BI_vs_SNRbeta.png"), "rho_t",
                         r"SNR·$\beta$ (Hz)", f"BI vs SNR·$\\beta$ — {LABEL_B} vs {LABEL_A}",
                         LABEL_A, LABEL_B, logx=True)
    write_csv(rows, per_ch, global_mean, p("BI_improvement.csv"), LABEL_A, LABEL_B)

    # ── Confronto dei filtri totali g=f·kernel (una griglia per f1 e una per f2) ─
    fdir_a, fdir_b = _filter_dir(CSV_A), _filter_dir(CSV_B)
    if os.path.isdir(fdir_a) and os.path.isdir(fdir_b):
        tf_dir = os.path.join(OUTDIR, "total_filters")
        os.makedirs(tf_dir, exist_ok=True)
        lam_a, lam_b = read_lambda_scalar(CSV_A), read_lambda_scalar(CSV_B)
        print("Filtri totali (griglia per f1 e per f2, col rapporto B/A):")
        for which in ("f1", "f2"):
            plot_total_filter_compare(which, fdir_a, fdir_b, lam_a, lam_b,
                                      LABEL_A, LABEL_B, tf_dir, TAG + NAME_TAG,
                                      exclude=exclude)
    else:
        print("[INFO] cartelle trained_filters mancanti per uno dei due modelli: "
              "salto il confronto dei filtri totali.")

    print("\nFatto.")


if __name__ == "__main__":
    main()
