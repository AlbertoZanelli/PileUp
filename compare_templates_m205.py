#!/usr/bin/env python3
"""
compare_templates_m205.py
=========================
Confronta i BI e i FILTRI di campagne diverse, a parita' di tutto il resto.

In SETS si mettono i NOMI DELLE CARTELLE: etichette, colori, nomi dei file e titoli sono
DEDOTTI da li' (describe()), cosi' non si possono disallineare dal contenuto.

    m205_results_octopus_npsclean           -> OF-root      "optimum filter - train root"
    m205_results_wiener_fit_npsclean        -> W-fit        "Wiener - train fit"
    m205_results_wiener_root_npsclean_swna1 -> Wwna-root    "Wiener + s-penalty - train root"

Convenzione dei nomi: la NPS pulita e' il DEFAULT e non si scrive; compare solo se il set
usa quella di Octopus. Il template del TRAINING sta nell'etichetta di ogni set; il template
INIETTATO negli eventi del Monte Carlo (colonna `gen` del CSV) sta nel titolo e nel nome del
file, perche' di solito e' lo stesso per tutti i set confrontati.

Legge, per ogni set:
  - BI_results_*.csv          (BI analitico, sigma, SNR, J)          da analyse_BI_m205.py
  - BI_mc_error_m205.csv      (BI Monte Carlo + sigma_BI)            da simulate_BI_error_m205.py
  - trained_filters/f{1,2}_ch<ch>_wp<wp>.npy                          i filtri di banda
I set mancanti vengono saltati con un [INFO], cosi' si puo' girare a campagne incomplete.

Uso:
    KMP_DUPLICATE_LIB_OK=TRUE python3 compare_templates_m205.py
"""

import os, re, csv, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═════════════════════════════════════════════════════════════════════════════
# Config
# ═════════════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# I set da confrontare: SOLO i nomi delle cartelle. Tutto il resto e' dedotto.
SETS = ["m205_results_octopus_npsclean",
        "m205_results_wiener_root_npsclean",
        "m205_results_wiener_root_npsclean_swna1"]

# Coppie per i pannelli di differenza. None = ogni set contro il PRIMO della lista.
# Altrimenti una lista di (tag_riferimento, tag_confronto) con i tag corti di describe().
PAIRS = None

BI_SOURCE = "mc"         # "mc" = BI Monte Carlo con barre d'errore ; "analytic" = BI analitico
GRID = (5, 3)            # righe x colonne della griglia dei filtri (15 WP)
COLORS = {}              # override manuale {tag: colore}; default = ciclo C0, C1, ...
SAMPLING_RATE = 10_000
WINDOW        = 10_000

# Nomi lunghi per la legenda, a partire dal tag corto del filtro.
FILTER_LABEL = {"OF": "optimum filter", "W": "Wiener", "WR": "Wiener x R(f)"}
# suffisso della cartella -> (pezzo del tag corto, pezzo dell'etichetta). Il suffisso e'
# quello scritto da S_PENALTY in analysis_BI_m205_wiener_regolarized.py: "_swna<w>", "_sbar<s>".
PENALTY = {"swna": ("wna", " + s-penalty"), "sbar": ("sbar", " + s-barrier")}


def describe(folder):
    """(tag corto, etichetta di legenda) dal nome della cartella dei risultati.

    Stessa grammatica di simulate_BI_error_m205._parse_results_name, cosi' i due programmi
    non possono leggere il nome in due modi diversi. La NPS pulita e' il default e sparisce
    dal nome; quella di Octopus si vede, perche' e' l'eccezione."""
    for prefix, filt in (("m205_results_octopus", "OF"), ("m205_results_wiener", "W")):
        if folder.startswith(prefix):
            tag = folder[len(prefix):]
            break
    else:
        raise SystemExit(f"[ERROR] nome non riconosciuto: {folder}")
    m = re.search(r"_(sbar|swna)[0-9.eE+-]+$", tag)
    pen_tag, pen_label = PENALTY.get(m.group(1), ("", "")) if m else ("", "")
    tag = tag[:m.start()] if m else tag
    nps = "clean" if tag.endswith("_npsclean") else "octopus"
    tag = tag[:-len("_npsclean")] if nps == "clean" else tag
    if tag.endswith("_R"):
        filt, tag = filt + "R", tag[:-2]
    if tag in ("", "_root"):
        train = "root"
    elif tag == "_fit":
        train = "fit"
    elif tag.startswith("_sim_"):
        train = tag[len("_sim_"):]
    else:
        raise SystemExit(f"[ERROR] non so dedurre il template di training da '{folder}'")
    short = f"{filt}{pen_tag}-{train}" + ("-npsoct" if nps == "octopus" else "")
    label = (FILTER_LABEL.get(filt, filt) + pen_label
             + f" - train {train}" + ("  (Octopus NPS)" if nps == "octopus" else ""))
    return short, label


TAGS = [describe(f)[0] for f in SETS]
LABELS = dict(describe(f) for f in SETS)
FOLDERS = dict(zip(TAGS, SETS))
REFERENCE = TAGS[0]                      # riferimento di rapporti e differenze
PAIRS = PAIRS or [(REFERENCE, t) for t in TAGS[1:]]
# La cartella di output nomina i set confrontati con i tag corti: due confronti diversi
# non si sovrascrivono, e il nome resta leggibile.
OUT_DIR = os.path.join(BASE_DIR, "comparisons", "_vs_".join(TAGS))


def color(tag):
    return COLORS.get(tag, f"C{TAGS.index(tag)}")


def marker(tag):
    """Marker diverso per set: a bias alto i BI coincidono entro il tratto, e con lo stesso
    marker l'ultimo disegnato copre gli altri e sembrano spariti."""
    return "os^Dv"[TAGS.index(tag) % 5]


# ═════════════════════════════════════════════════════════════════════════════
# Lettura
# ═════════════════════════════════════════════════════════════════════════════
def load_set(folder):
    """{(ch, wp): dict} con BI analitico, BI MC e sigma, per un set. None se manca."""
    d = os.path.join(BASE_DIR, folder)
    files = glob.glob(os.path.join(d, "BI_results_*.csv"))
    if not files:
        return None
    out = {}
    for r in csv.DictReader(open(files[0])):
        k = (int(r["channel"]), int(r["wp"]))
        out[k] = dict(vbias=float(r["vbias"]), BI=float(r["BI"]),
                      sigma_analytic=float(r["sigma_analytic"]), SNR=float(r["SNR"]),
                      BI_mc=np.nan, sigma_BI=np.nan, gen="", nsim=np.nan, dir=d)
    mc = os.path.join(d, "BI_mc_error_m205.csv")
    if os.path.exists(mc):
        for r in csv.DictReader(open(mc)):
            k = (int(r["channel"]), int(r["wp"]))
            if k in out:
                out[k]["BI_mc"] = float(r["BI_mc"])
                out[k]["sigma_BI"] = float(r["sigma_BI"])
                # il template INIETTATO e la statistica del MC stanno solo qui: servono al
                # titolo e alla nota sul pannello delle differenze
                out[k]["gen"] = r.get("gen", "")
                out[k]["nsim"] = float(r.get("nsim") or np.nan)
    return out


def mc_info(data, keys, channel):
    """(template iniettato, NSIM) dei set disegnati, come stringhe pronte per il titolo.
    Se i set non concordano si elencano tutti: e' un'informazione, non un errore."""
    def collect(field, cast=str):
        v = {cast(rec[field]) for lab in keys for (c, _), rec in data[lab].items()
             if c == channel and rec[field] == rec[field] and rec[field] != ""}
        return "/".join(sorted(v)) or "?"
    return collect("gen"), collect("nsim", lambda x: f"{int(x)}")


def value(rec):
    """Il BI da usare nei grafici, con la sua incertezza (nan se non c'e')."""
    if BI_SOURCE == "mc" and np.isfinite(rec["BI_mc"]):
        return rec["BI_mc"], rec["sigma_BI"]
    return rec["BI"], np.nan


def significance(rec_ref, rec_sim):
    """z = (BI_sim - BI_ref) / sigma, con sigma dalle SOLE simulazioni Monte Carlo.

    Serve il Monte Carlo del set SIMULATO: e' quello che porta l'incertezza. Per il set di
    riferimento (root o fit) si usa il suo BI Monte Carlo, con la sua sigma, se c'e';
    altrimenti il BI ANALITICO, che non ha incertezza — e' una funzione deterministica di
    template, NPS e ampiezza, ripetendolo da' lo stesso numero — e quindi entra nel
    denominatore con zero.

    NB: sigma_BI e' l'errore STATISTICO del Monte Carlo e scala come 1/sqrt(NSIM), quindi
    alzando NSIM qualunque differenza diventa "significativa". E i due set condividono gli
    stessi eventi generati (stesso seed), quindi i BI sono correlati e questo z e'
    conservativo: la differenza e' misurata meglio di quanto dica."""
    if not np.isfinite(rec_sim["BI_mc"]):
        return None
    if np.isfinite(rec_ref["BI_mc"]):
        bi_ref, s_ref = rec_ref["BI_mc"], rec_ref["sigma_BI"]
    else:
        bi_ref, s_ref = rec_ref["BI"], 0.0
    den = float(np.hypot(rec_sim["sigma_BI"], s_ref))
    return None if den == 0 else (rec_sim["BI_mc"] - bi_ref) / den


def load_filters(rec, ch, wp):
    """(f1, f2) del set, meta' indipendente dello spettro. None se mancano."""
    p1 = os.path.join(rec["dir"], "trained_filters", f"f1_ch{ch}_wp{wp}.npy")
    p2 = os.path.join(rec["dir"], "trained_filters", f"f2_ch{ch}_wp{wp}.npy")
    if not (os.path.exists(p1) and os.path.exists(p2)):
        return None
    return np.load(p1).ravel(), np.load(p2).ravel()


# ═════════════════════════════════════════════════════════════════════════════
# Figure
# ═════════════════════════════════════════════════════════════════════════════
def plot_bi(data, channel, keys):
    """BI vs V_bias dei set, piu' differenza e compatibilita' dentro ogni coppia."""
    fig, ax = plt.subplots(3, 1, figsize=(9, 9.5), sharex=True,
                           gridspec_kw={"height_ratios": [2.4, 1, 1]})
    for lab in keys:
        v, b, e = [], [], []
        for (c, wp), rec in sorted(data[lab].items()):
            if c != channel:
                continue
            bi, s = value(rec)
            v.append(rec["vbias"]); b.append(bi); e.append(s)
        if not v:
            continue
        e = np.array(e, dtype=float)
        ax[0].errorbar(v, b, yerr=np.where(np.isfinite(e), e, 0), fmt=marker(lab), ms=5,
                       capsize=3, lw=1, mfc="none", mew=1.3,
                       color=color(lab), label=LABELS[lab])
    ax[0].set_ylabel("BI  [counts/keV/kg/yr]")
    # il titolo deve dire cosa e' stato DAVVERO disegnato: se il CSV del Monte Carlo manca,
    # value() ripiega sul BI analitico e dirlo "Monte Carlo" sarebbe falso.
    got = [np.isfinite(rec["BI_mc"]) for lab in keys for (c, _), rec in data[lab].items()
           if c == channel]
    src = ("simulated (Monte Carlo)" if BI_SOURCE == "mc" and got and all(got) else
           "analytic" if not any(got) else "Monte Carlo where available, else analytic")
    gen, nsim = mc_info(data, keys, channel)
    inj = (f"events injected from '{gen}' AP, {nsim} events/population"
           if "Monte Carlo" in src else "no events: BI = J x K from the model")
    ax[0].set_title(f"m205 Ch{channel} - background index vs bias\n"
                    f"BI: {src}   |   {inj}", fontsize=11)
    ax[0].legend(fontsize=9); ax[0].grid(True, ls="--", alpha=0.4)

    # compatibilita' dentro le coppie: z = (BI_a - BI_b) / sqrt(sa^2 + sb^2)
    for a, b in PAIRS:
        if a not in keys or b not in keys:
            continue
        v, z, vd, d = [], [], [], []
        for (c, wp), rec in sorted(data[a].items()):
            if c != channel or (c, wp) not in data[b]:
                continue
            zz = significance(rec, data[b][(c, wp)])
            if zz is not None:
                v.append(rec["vbias"]); z.append(zz)
            ba, _ = value(rec)
            bb, _ = value(data[b][(c, wp)])
            if ba:
                vd.append(rec["vbias"]); d.append(100.0 * (bb - ba) / ba)
        if v:
            ax[1].plot(v, z, "o-", ms=4, lw=1, color=color(b), label=f"{b} vs {a}")
        if vd:
            ax[2].plot(vd, d, "o-", ms=4, lw=1, color=color(b), label=f"{b} vs {a}")
    for k, c in ((1, "0.6"), (2, "0.8")):
        ax[1].axhspan(-k, k, color=c, alpha=0.25, zorder=0)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_ylabel(r"z")
    ax[1].text(0.012, 0.97, r"z = (BI $-$ BI$_{ref}$) / $\sqrt{\sigma^2 + \sigma_{ref}^2}$   "
               r"$\sigma$ = Monte Carlo statistics only (shaded: 1$\sigma$, 2$\sigma$)",
               transform=ax[1].transAxes, fontsize=7.5, color="0.25", va="top",
               bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_ylabel(r"$\Delta$BI  [%]"); ax[2].set_xlabel("V bias [V]")
    # Il pannello delle differenze deve dire COME e' calcolata la differenza e SU QUALE BI:
    # senza, "-4%" non e' interpretabile (BI analitico o simulato? rispetto a chi?).
    ax[2].text(0.012, 0.97,
               rf"$\Delta$BI = 100 $\cdot$ (BI $-$ BI$_{{ref}}$) / BI$_{{ref}}$   on the "
               rf"{'simulated' if 'Monte Carlo' in src else 'analytic'} BI   |   "
               rf"negative = better than the reference",
               transform=ax[2].transAxes, fontsize=7.5, color="0.25", va="top",
               bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))
    for a in ax[1:]:
        if a.get_legend_handles_labels()[0]:      # niente coppie disegnate -> niente legenda
            a.legend(fontsize=8)
        a.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    # Il nome dice la QUANTITA' (BI simulato o analitico), il canale e cosa e' stato iniettato.
    # Quali set sono confrontati lo dice gia' la cartella, non serve ripeterlo qui.
    kind = "mc" if "Monte Carlo" in src else "analytic"
    out = os.path.join(OUT_DIR, f"BI-{kind}_ch{channel}"
                       + (f"_inj-{gen}" if kind == "mc" else "") + ".png")
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


def smooth(y, size=51):
    """Mediana mobile: i filtri addestrati sono frastagliati e senza questa il confronto
    fra due set non si legge."""
    from scipy.ndimage import median_filter
    return median_filter(np.asarray(y, dtype=float), size=size, mode="nearest")


def total_filter(rec, ch, wp, which):
    """Filtro TOTALE applicato ai dati: g_i = f_i * kernel, meta' indipendente dello spettro.

    Sia i filtri di banda sia il kernel sono salvati come meta' indipendente (DC..Nyquist),
    quindi il prodotto si fa direttamente li'. Si restituisce |g_i|: la fase non serve al
    confronto e per il filtro ottimo e' comunque quella di S*."""
    d = os.path.join(rec["dir"], "trained_filters")
    paths = [os.path.join(d, f"{n}_ch{ch}_wp{wp}.npy") for n in (which, "kernel")]
    if not all(os.path.exists(x) for x in paths):
        return None
    f, k = (np.load(x).ravel() for x in paths)
    n = min(len(f), len(k))
    return np.abs(f[:n] * k[:n])


def plot_total_filters(data, channel, which, keys):
    """Una griglia per canale e per filtro: un pannello per WP, dentro tutti i set."""
    wps = sorted({wp for lab in keys for (c, wp) in data[lab] if c == channel})
    if not wps:
        return None
    rows, cols = GRID
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.1 * rows),
                             sharex=True, squeeze=False)
    fr = np.fft.rfftfreq(WINDOW, 1 / SAMPLING_RATE)
    drawn = False
    top, bot = 0.0, np.inf          # estremi su tutta la griglia, per una scala y comune
    for a, wp in zip(axes.ravel(), wps):
        for lab in keys:
            rec = data[lab].get((channel, wp))
            g = total_filter(rec, channel, wp, which) if rec else None
            if g is None:
                continue
            drawn = True
            n = min(len(fr), len(g))
            # i filtri addestrati sono frastagliati bin per bin: grezzo in trasparenza,
            # mediana mobile in pieno, altrimenti quattro curve sovrapposte non si leggono.
            sm = smooth(g[1:n])
            pos = sm[sm > 0]
            top = max(top, float(sm.max()))
            bot = min(bot, float(pos.min())) if pos.size else bot
            a.loglog(fr[1:n], g[1:n], lw=0.4, alpha=0.2, color=color(lab))
            a.loglog(fr[1:n], sm, lw=1.2, color=color(lab), label=LABELS[lab])
        a.set_title(f"WP{wp}", fontsize=10)
        a.grid(True, which="both", ls="--", alpha=0.3)
    for a in axes.ravel()[len(wps):]:
        a.axis("off")
    # Scala y comune a tutti i pannelli (senza, ognuno si autoscala e non sono confrontabili),
    # ma presa dagli estremi VERI delle curve lisciate: non taglia niente in basso.
    if np.isfinite(bot) and top > 0:
        for a in axes.ravel()[:len(wps)]:
            a.set_ylim(bot * 0.5, top * 3)
    if not drawn:
        plt.close(fig)
        return None
    for a in axes[-1]:
        a.set_xlabel("frequency [Hz]")
    for a in axes[:, 0]:
        a.set_ylabel(rf"$|{which} \cdot$ kernel$|$")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.972),
               ncol=max(1, len(l)), fontsize=11, frameon=False)
    fig.suptitle(f"m205 Ch{channel} - total filter applied to the data:  "
                 rf"$|{which} \times$ kernel$|$   (kernel = the trained filter's own)",
                 y=0.998, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.952])
    out = os.path.join(OUT_DIR, f"filter-{which}_ch{channel}.png")
    fig.savefig(out, dpi=110); plt.close(fig)
    return out


def summary_table(data, keys):
    """Mediane sui punti in comune: BI relativo al riferimento e compatibilita' per coppia."""
    got = [np.isfinite(rec["BI_mc"]) for lab in keys for rec in data[lab].values()]
    src = ("simulato (MC)" if BI_SOURCE == "mc" and got and all(got) else
           "analitico" if not (BI_SOURCE == "mc" and any(got)) else
           "simulato dove c'e', altrimenti analitico")
    print(f"\nBI usato nelle colonne 'BI mediano' e 'vs riferimento': {src}. "
          f"Riferimento: {REFERENCE}.")
    print(f"{'set':>16s} {'punti':>6s} {'BI mediano':>12s} {'vs ' + REFERENCE:>14s} "
          f"{'BI_mc/BI_an':>12s}")
    for lab in keys:
        bi, rel, rat = [], [], []
        for k, rec in data[lab].items():
            b, _ = value(rec)
            bi.append(b)
            if np.isfinite(rec["BI_mc"]):
                rat.append(rec["BI_mc"] / rec["BI"])
            if REFERENCE in keys and k in data[REFERENCE]:
                rel.append(b / value(data[REFERENCE][k])[0])
        print(f"{lab:>16s} {len(bi):>6d} {np.median(bi):12.4e} "
              f"{(np.median(rel) if rel else np.nan):14.3f} "
              f"{(np.median(rat) if rat else np.nan):12.3f}")
    for a, b in PAIRS:
        if a not in keys or b not in keys:
            continue
        z = [significance(rec, data[b][k]) for k, rec in data[a].items() if k in data[b]]
        z = [x for x in z if x is not None]
        if z:
            z = np.array(z)
            print(f"\ncoppia {b} vs {a}: {len(z)} punti, z mediano {np.median(z):+.2f}, "
                  f"|z|>2 in {(np.abs(z) > 2).mean():.0%} dei casi")
            print("  z calcolato con le sole sigma del Monte Carlo (il BI analitico non ha "
                  "incertezza).\n  Gli eventi sono gli STESSI nei due set (stesso seed), quindi i "
                  "BI sono correlati e lo z\n  e' conservativo. E sigma_BI e' errore di NSIM: "
                  "alzando NSIM ogni differenza diventa\n  'significativa'.")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data, keys = {}, []
    for tag, folder in FOLDERS.items():
        d = load_set(folder)
        if d is None:
            print(f"[INFO] salto '{tag}': nessun CSV in {folder}")
            continue
        data[tag], _ = d, keys.append(tag)
        print(f"[OK]   {tag:<14s} = {LABELS[tag]:<38s} ({len(d)} coppie da {folder})")
    if not keys:
        raise SystemExit("[ERROR] nessun set disponibile")

    for ch in sorted({c for lab in keys for (c, _) in data[lab]}):
        print(f"   -> {os.path.relpath(plot_bi(data, ch, keys), BASE_DIR)}")
        for which in ("f1", "f2"):
            out = plot_total_filters(data, ch, which, keys)
            if out:
                print(f"   -> {os.path.relpath(out, BASE_DIR)}")
    summary_table(data, keys)


if __name__ == "__main__":
    main()
