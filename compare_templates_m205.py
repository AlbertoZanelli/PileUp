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
                              puo' contenere piu' campagne: si sceglie con MC_GEN
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

# I set da confrontare: i nomi delle cartelle. Tutto il resto e' dedotto.
# Una cartella puo' comparire PIU' VOLTE con "@<gen>", per confrontare gli stessi filtri
# valutati su eventi iniettati da template diversi (e' la sistematica "pulse template
# (injection)" del paper). Senza "@" si usa MC_GEN.
#     "m205_results_wiener_root_npsclean"          -> MC_GEN
#     "m205_results_wiener_root_npsclean@fit"      -> righe con gen='fit' dello stesso CSV
SETS = ["m205_results_octopus_npsclean",
        "m205_results_wiener_root_npsclean_swna1",
        "m205_results_wiener_root_npsclean"]

# Coppie per i pannelli di differenza (z e Delta BI).
#   None                          -> ogni set contro il PRIMO della lista
#   ("OF-root", "W-root")         -> due set, sulla stessa sorgente
#   ("OF-root:analytic", "OF-root:mc") -> LO STESSO set, analitico contro simulato: e' il
#                                    rapporto MC/analitico WP per WP, cioe' quanto il modello
#                                    sbaglia. Si disegna tratteggiata, per distinguerla dai
#                                    confronti fra set.
# Le due forme si possono mescolare nella stessa lista, per esempio:
PAIRS = [("OF-root:mc", "W-root:mc"), ("OF-root:mc", "Wwna-root:mc")]
#            ("Wwna-root:analytic", "Wwna-root:mc")]
# Il default (None) mette solo i confronti fra set: le coppie mc-vs-analytic si aggiungono
# a mano, altrimenti i due pannelli diventano illeggibili con piu' di due set.
#PAIRS = [("OF-root:mc", "OF-root:analytic"), ("W-root:mc", "W-root:analytic")]

# Quale BI disegnare nel pannello di sopra:
#   "mc"       -> solo il BI Monte Carlo, con le sue barre d'errore (ripiega sull'analitico
#                 dove il CSV del MC manca, e il titolo lo dice);
#   "analytic" -> solo il BI analitico, J x K, che non ha incertezza;
#   "both"     -> tutti e due: MARKER = Monte Carlo, LINEA = analitico, stesso colore per set.
#                 E' il modo per vedere a colpo d'occhio dove i due divergono, cioe' dove il
#                 modello sta mentendo. I due pannelli di sotto (z e Delta BI) restano sul MC.
ONLY_CHANNELS = None     # lista di canali da disegnare, es. [34, 91]; None/[] = tutti quelli
                         # presenti nei CSV dei set

# QUALE campagna Monte Carlo leggere. Nel CSV di una cartella convivono piu' campagne, una
# riga per (canale, WP, gen): `gen` e' il template INIETTATO negli eventi (GEN_TEMPLATE di
# simulate_BI_error). Si tengono solo le righe che combaciano.
# Non confondere MC_GEN con il template del TRAINING, che sta nel nome della cartella: qui si
# sceglie cosa e' stato INIETTATO, la' cosa e' stato usato per addestrare.
MC_GEN = "root"          # "root" | "fit"
MC_CSV = "BI_mc_error_m205.csv"
BI_SOURCE = "both"       # "mc" | "analytic" | "both"
GRID = (5, 3)            # righe x colonne della griglia dei filtri (15 WP)
COLORS = {}              # override manuale {tag: colore}; default = ciclo C0, C1, ...
SAMPLING_RATE = 10_000
WINDOW        = 10_000

# Nomi lunghi per la legenda, a partire dal tag corto del filtro.
FILTER_LABEL = {"OF": "optimum filter", "W": "Wiener", "WR": "Wiener x R(f)"}
# suffisso della cartella -> (pezzo del tag corto, pezzo dell'etichetta). Il suffisso e'
# quello scritto da S_PENALTY in analysis_BI_m205_wiener_regolarized.py: "_swna<w>", "_sbar<s>".
PENALTY = {"swna": ("wna", " + s-penalty"), "sbar": ("sbar", " + s-barrier")}


def describe(spec):
    """(tag corto, etichetta di legenda) dalla voce di SETS: cartella, piu' un "@<gen>"
    opzionale che sceglie il template INIETTATO fra quelli presenti nel CSV del Monte Carlo.

    Stessa grammatica di simulate_BI_error_m205._parse_results_name, cosi' i due programmi
    non possono leggere il nome in due modi diversi. La NPS pulita e' il default e sparisce
    dal nome; quella di Octopus si vede, perche' e' l'eccezione."""
    folder, _, gen = spec.partition("@")
    gen = gen or MC_GEN
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
    # il template iniettato compare solo se NON e' quello di default: altrimenti sarebbe
    # scritto uguale su ogni set e non distinguerebbe niente
    inj = "" if gen == MC_GEN else gen
    short = (f"{filt}{pen_tag}-{train}" + ("-npsoct" if nps == "octopus" else "")
             + (f"-inj{inj}" if inj else ""))
    label = (FILTER_LABEL.get(filt, filt) + pen_label
             + f" - train {train}" + ("  (Octopus NPS)" if nps == "octopus" else "")
             + (f" - inj {inj}" if inj else ""))
    return short, label


TAGS = [describe(f)[0] for f in SETS]
LABELS = dict(describe(f) for f in SETS)
FOLDERS = {t: spec.partition("@")[0] for t, spec in zip(TAGS, SETS)}
GENS = {t: (spec.partition("@")[2] or MC_GEN) for t, spec in zip(TAGS, SETS)}
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
def read_csv_checked(path):
    """Righe del CSV, con un controllo sullo schema.

    Un file con UN header e righe di lunghezza diversa (tipico: campagna ripresa dopo aver
    aggiunto colonne al programma di training, senza riscrivere l'header) e' il modo peggiore
    di sbagliare: csv assegna per posizione, le colonne slittano e 'BI' legge il valore di
    un'altra colonna. Nessun errore, solo numeri sbagliati. Meglio fermarsi."""
    rows = list(csv.DictReader(open(path, newline="")))
    for n, r in enumerate(rows, start=2):
        if r.get(None) is not None or any(v is None for v in r.values()):
            raise SystemExit(
                f"[ERROR] {os.path.relpath(path, BASE_DIR)}: la riga {n} non ha lo stesso "
                f"numero di campi dell'header.\n"
                f"        Il file mescola due schemi: le colonne slittano e i valori finiscono "
                f"nella colonna sbagliata.\n"
                f"        Rigenera la campagna, oppure riscrivi il file con l'header completo.")
    return rows


def load_set(folder, gen):
    """{(ch, wp): dict} con BI analitico, BI MC e sigma, per un set. None se manca.

    Del CSV del Monte Carlo si prendono le SOLE righe con quel `gen`: nello stesso file
    possono convivere piu' campagne, distinte da quella colonna."""
    d = os.path.join(BASE_DIR, folder)
    files = glob.glob(os.path.join(d, "BI_results_*.csv"))
    if not files:
        return None
    out = {}
    for r in read_csv_checked(files[0]):
        k = (int(r["channel"]), int(r["wp"]))
        out[k] = dict(vbias=float(r["vbias"]), BI=float(r["BI"]),
                      sigma_analytic=float(r["sigma_analytic"]), SNR=float(r["SNR"]),
                      BI_mc=np.nan, sigma_BI=np.nan, gen="", nsim=np.nan, dir=d)
    mc = os.path.join(d, MC_CSV)
    n_mc = 0
    if os.path.exists(mc):
        for r in read_csv_checked(mc):
            if (r.get("gen") or "root") != gen:
                continue
            k = (int(r["channel"]), int(r["wp"]))
            if k in out:
                out[k]["BI_mc"] = float(r["BI_mc"])
                out[k]["sigma_BI"] = float(r["sigma_BI"])
                # il template INIETTATO e la statistica del MC stanno solo qui: servono al
                # titolo e alla nota sul pannello delle differenze
                out[k]["gen"] = r.get("gen", "")
                out[k]["nsim"] = float(r.get("nsim") or np.nan)
                n_mc += 1
    if os.path.exists(mc) and n_mc == 0:
        print(f"[INFO] {folder}: il CSV del Monte Carlo non ha righe con gen='{gen}'")
    return out


def mc_info(data, keys, channel):
    """(template iniettato, NSIM) dei set disegnati, come stringhe pronte per il titolo.
    Se i set non concordano si elencano tutti: e' un'informazione, non un errore."""
    def collect(field, cast=str):
        v = {cast(rec[field]) for lab in keys for (c, _), rec in data[lab].items()
             if c == channel and rec[field] == rec[field] and rec[field] != ""}
        return "+".join(sorted(v)) or "?"
    return collect("gen"), collect("nsim", lambda x: f"{int(x)}")


def bi_mc(rec):
    """(BI Monte Carlo, sigma) del punto, oppure None se il MC per quel punto non c'e'.

    Niente ripiego sull'analitico: sono due numeri diversi, e mescolarli in un grafico o in
    un confronto e' il modo piu' rapido per concludere qualcosa di falso. Se manca, il punto
    non si disegna e il programma lo dice."""
    return (rec["BI_mc"], rec["sigma_BI"]) if np.isfinite(rec["BI_mc"]) else None


# I pannelli di confronto (z e Delta BI) lavorano sul simulato, tranne quando si e' chiesto
# esplicitamente l'analitico. In "both" il simulato e' la sorgente: e' quello con l'incertezza.
COMPARE_ON = "analytic" if BI_SOURCE == "analytic" else "mc"


def bi_of(rec, src):
    """(BI, sigma) della sorgente richiesta.
    "mc" -> None se per quel punto il Monte Carlo non c'e';
    "analytic" -> sigma 0, perche' J x K e' deterministico: stessi input, stesso numero."""
    return bi_mc(rec) if src == "mc" else (rec["BI"], 0.0)


def series(spec):
    """Voce di PAIRS -> (tag del set, sorgente).

    "W-root"            -> sorgente di confronto di default (COMPARE_ON)
    "W-root:mc"         -> BI Monte Carlo di quel set
    "OF-root:analytic"  -> BI analitico di quel set
    Cosi' si puo' mettere in coppia lo STESSO set con le due sorgenti, es.
    ("OF-root:analytic", "OF-root:mc"), che e' il rapporto MC/analitico WP per WP."""
    tag, _, src = spec.partition(":")
    if src not in ("", "mc", "analytic"):
        raise SystemExit(f"[ERROR] sorgente '{src}' in PAIRS: usa 'mc' o 'analytic'")
    return tag, (src or COMPARE_ON)


def significance(rec_ref, rec, spec_ref, spec):
    """z = (BI - BI_ref) / sqrt(sigma^2 + sigma_ref^2), None se non e' calcolabile.

    Due casi, e la differenza conta:
      - STESSO set, analitico contro simulato: sigma_ref = 0 e' corretta, l'analitico e' la
        PREVISIONE deterministica e lo z e' il pull della simulazione contro il modello;
      - set DIVERSI: servono le sigma di tutti e due, quindi il Monte Carlo su entrambi.
        Con sigma = 0 su un lato si confronterebbe un BI simulato con uno analitico di un
        altro set, che e' il confronto che non si fa.

    NB: sigma_BI e' l'errore STATISTICO del Monte Carlo e scala come 1/sqrt(NSIM), quindi
    alzando NSIM qualunque differenza diventa "significativa". Fra set diversi gli eventi
    sono gli STESSI (stesso seed), i BI sono correlati e lo z e' conservativo."""
    ta, sa = series(spec_ref)
    tb, sb = series(spec)
    if ta != tb and not (sa == "mc" and sb == "mc"):
        return None
    a, b = bi_of(rec_ref, sa), bi_of(rec, sb)
    if a is None or b is None:
        return None
    den = float(np.hypot(a[1], b[1]))
    return None if den == 0 else (b[0] - a[0]) / den


def delta_pct(rec_ref, rec, spec_ref, spec):
    """100*(BI - BI_ref)/BI_ref. Fra set diversi le due sorgenti devono coincidere: analitico
    con analitico, simulato con simulato. Sullo stesso set possono differire, ed e' il punto."""
    ta, sa = series(spec_ref)
    tb, sb = series(spec)
    if ta != tb and sa != sb:
        return None
    a, b = bi_of(rec_ref, sa), bi_of(rec, sb)
    if a is None or b is None or not a[0]:
        return None
    return 100.0 * (b[0] - a[0]) / a[0]


def pair_label(spec_ref, spec):
    """Etichetta leggibile della coppia, che dice sempre quali sorgenti sono in gioco."""
    ta, sa = series(spec_ref)
    tb, sb = series(spec)
    if ta == tb:
        return f"{ta}: {sb} vs {sa}"
    if sa == sb:
        return f"{tb} vs {ta}"
    return f"{tb}:{sb} vs {ta}:{sa}"


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
    """BI vs V_bias dei set, piu' differenza e compatibilita' dentro ogni coppia.

    Regola: cio' che non c'e' non si disegna. Un set senza Monte Carlo non compare fra i
    marker (e lo si scrive a schermo), una coppia senza Monte Carlo non compare nei pannelli
    di sotto. Mai un punto analitico travestito da simulato."""
    fig, ax = plt.subplots(3, 1, figsize=(9, 9.5), sharex=True,
                           gridspec_kw={"height_ratios": [2.4, 1, 1]})
    no_mc = []
    for lab in keys:
        vm, bm, em, va, ba = [], [], [], [], []
        for (c, wp), rec in sorted(data[lab].items()):
            if c != channel:
                continue
            va.append(rec["vbias"]); ba.append(rec["BI"])
            m = bi_mc(rec)
            if m is not None:
                vm.append(rec["vbias"]); bm.append(m[0]); em.append(m[1])
        if not va:
            continue
        if BI_SOURCE != "analytic":
            if vm:
                ax[0].errorbar(vm, bm, yerr=em, fmt=marker(lab), ms=5, capsize=3, lw=1,
                               mfc="none", mew=1.3, color=color(lab), label=LABELS[lab])
            else:
                no_mc.append(lab)
        if BI_SOURCE != "mc":
            # in "both" la linea non prende etichetta, ce l'ha gia' il marker dello stesso
            # colore; ma se quel set il Monte Carlo non ce l'ha, l'etichetta va sulla linea,
            # altrimenti il set sparirebbe dalla legenda senza che si capisca perche'.
            ax[0].plot(va, ba, "-", lw=1.1, alpha=0.75, color=color(lab),
                       label=LABELS[lab] + ("  [analytic only]" if lab in no_mc else "")
                       if (BI_SOURCE == "analytic" or lab in no_mc) else None)
    ax[0].set_ylabel("BI  [counts/keV/kg/yr]")

    n_mc = sum(1 for lab in keys
               if any(bi_mc(rec) is not None for (c, _), rec in data[lab].items() if c == channel))
    if BI_SOURCE == "analytic" or n_mc == 0:
        src = "analytic"
    elif BI_SOURCE == "both":
        src = "simulated (markers) and analytic (lines)"
    else:
        src = "simulated (Monte Carlo)"
    used = "simulated" if COMPARE_ON == "mc" and n_mc else "analytic"
    gen, nsim = mc_info(data, keys, channel)
    inj = (f"events injected from '{gen}' AP, {nsim} events/pop."
           if src != "analytic" else "no events: BI = J x K from the model")
    ax[0].set_title(f"m205 Ch{channel} - background index vs bias\n"
                    f"BI: {src}   |   {inj}", fontsize=11)
    h, l = ax[0].get_legend_handles_labels()
    if BI_SOURCE == "both" and n_mc:
        from matplotlib.lines import Line2D
        h += [Line2D([], [], color="0.35", marker="o", ls="none", mfc="none"),
              Line2D([], [], color="0.35", ls="-")]
        l += ["markers: Monte Carlo", "lines: analytic"]
    ax[0].legend(h, l, fontsize=9); ax[0].grid(True, ls="--", alpha=0.4)

    drawn_pairs = []
    for spec_a, spec_b in PAIRS:
        ta, tb = series(spec_a)[0], series(spec_b)[0]
        if ta not in keys or tb not in keys:
            continue
        # coppia con le sorgenti scritte a mano (es. "OF-root:analytic" vs "OF-root:mc"):
        # e' UNA sola serie, e va disegnata come tale. Senza sorgenti esplicite invece la
        # coppia si disegna due volte, una per sorgente, come il pannello di sopra.
        explicit = ":" in spec_a or ":" in spec_b
        lab = pair_label(spec_a, spec_b)
        v, z, dm, da = [], [], [], []
        for (c, wp), rec in sorted(data[ta].items()):
            if c != channel or (c, wp) not in data[tb]:
                continue
            other = data[tb][(c, wp)]
            zz = significance(rec, other, spec_a, spec_b)
            if zz is not None:
                v.append(rec["vbias"]); z.append(zz)
            if explicit:
                dd = delta_pct(rec, other, spec_a, spec_b)
                if dd is not None:
                    dm.append((rec["vbias"], dd))
            else:
                for lst, sname in ((dm, "mc"), (da, "analytic")):
                    dd = delta_pct(rec, other, f"{ta}:{sname}", f"{tb}:{sname}")
                    if dd is not None:
                        lst.append((rec["vbias"], dd))
        col = color(tb)
        if v or dm or da:
            drawn_pairs.append((spec_a, spec_b))
        if v:
            ax[1].plot(v, z, "o--" if explicit else "o-", ms=4, lw=1, color=col, label=lab)
        # Delta BI: stessa convenzione del pannello di sopra, marker = simulato, linea =
        # analitico. Ognuno sulla propria sorgente, mai una miscela delle due.
        if explicit:
            if dm:
                ax[2].plot(*zip(*dm), ls="--", marker=marker(tb), ms=5, mfc="none", mew=1.3,
                           lw=1, color=col, label=lab)
        else:
            if BI_SOURCE != "analytic" and dm:
                ax[2].plot(*zip(*dm), ls="none", marker=marker(tb), ms=5, mfc="none", mew=1.3,
                           color=col, label=lab)
            if BI_SOURCE != "mc" and da:
                ax[2].plot(*zip(*da), "-", lw=1.1, alpha=0.75, color=col,
                           label=lab if (BI_SOURCE == "analytic" or not dm) else None)
    for k, c in ((1, "0.6"), (2, "0.8")):
        ax[1].axhspan(-k, k, color=c, alpha=0.25, zorder=0)
    # Nomi sugli assi, gli stessi che stanno in legenda, cosi' l'asse dice come e' fatto il
    # conto senza doverla leggere. Il riferimento si scrive quando e' unico fra le coppie
    # disegnate; il termine al numeratore solo se la coppia disegnata e' UNA, altrimenti
    # cambia da curva a curva ed e' la legenda a dire quale.
    def name(spec):
        t, sr = series(spec)
        return f"{t},{sr}" if ":" in spec else t

    refs = {name(a) for a, _ in drawn_pairs}
    ref = refs.pop() if len(refs) == 1 else "ref"
    num = f"BI_{name(drawn_pairs[0][1])}" if len(drawn_pairs) == 1 else "BI"
    # Etichette in testo semplice, non mathtext: dentro $...$ il trattino di "OF-root"
    # diventa un segno meno e il nome del set si legge male.
    ax[1].axhline(0, color="k", lw=0.8)
    # Un pannello basso regge ~32 caratteri a fontsize 8.5: con i nomi dei set per esteso
    # l'etichetta e' piu' alta del pannello e sborda in quello sopra. Si spezza e si stringe.
    def ylab(a, lines):
        lines = [x for x in lines if x]
        n = max(len(x) for x in lines)
        a.set_ylabel("\n".join(lines), fontsize=8.5 if n <= 30 else 7.5 if n <= 36 else 6.5)

    zt = f"z = ({num} \u2212 BI_{ref}) / \u03c3"
    ylab(ax[1], [zt] if len(zt) <= 36 else [f"z = ({num} \u2212 BI_{ref})", "/ \u03c3"])
    ax[1].text(0.012, 0.97, r"$\sigma = \sqrt{\sigma^2 + \sigma_{ref}^2}$, Monte Carlo "
               r"statistics only (shaded: 1$\sigma$, 2$\sigma$)",
               transform=ax[1].transAxes, fontsize=7.5, color="0.25", va="top",
               bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))
    ax[2].axhline(0, color="k", lw=0.8)
    # su due righe: su una sola la formula e' piu' alta del pannello e sborda
    dt = f"({num} \u2212 BI_{ref}) / BI_{ref}"
    ylab(ax[2], ["\u0394BI  [%]  =  100 \u00b7"] + ([dt] if len(dt) <= 36 else
                [f"({num} \u2212 BI_{ref})", f"/ BI_{ref}"]))
    ax[2].set_xlabel("V bias [V]")
    # Il pannello delle differenze deve dire COME e' calcolata la differenza e SU QUALE BI:
    # senza, "-4%" non e' interpretabile (BI analitico o simulato? rispetto a chi?).
    # la nota deve descrivere cio' che e' stato DISEGNATO: in "both" senza Monte Carlo i
    # marker non ci sono, e annunciarli sarebbe una bugia
    how = ("markers: simulated, line: analytic" if BI_SOURCE == "both" and n_mc
           else f"on the {used} BI")
    ax[2].text(0.012, 0.97, rf"{how}   |   negative = better than {ref}",
               transform=ax[2].transAxes, fontsize=7.5, color="0.25", va="top",
               bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))
    for a, why in ((ax[1], "no Monte Carlo for these pairs: z needs both sets simulated"),
                   (ax[2], "no pair with both sets available")):
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=8)
        else:
            a.text(0.5, 0.45, why, transform=a.transAxes, ha="center", fontsize=9, color="0.45")
        a.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    # Il nome dice la QUANTITA' (BI simulato o analitico), il canale e cosa e' stato iniettato.
    # Quali set sono confrontati lo dice gia' la cartella, non serve ripeterlo qui.
    kind = {"both": "mc+analytic", "mc": "mc", "analytic": "analytic"}[BI_SOURCE]
    kind = "analytic" if src == "analytic" else kind          # nessun MC disponibile
    out = os.path.join(OUT_DIR, f"BI-{kind}_ch{channel}"
                       + (f"_inj-{gen}" if kind != "analytic" else "") + ".png")
    fig.savefig(out, dpi=130); plt.close(fig)
    if no_mc:
        print(f"   [INFO] ch{channel}: nessun Monte Carlo per {', '.join(no_mc)}"
              + (" (disegnati solo come linea analitica)" if BI_SOURCE == "both"
                 else " (non disegnati)"))
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
        seen = set()
        for lab in keys:
            rec = data[lab].get((channel, wp))
            # due set che differiscono solo per il template iniettato condividono i filtri:
            # disegnarli due volte darebbe due curve identiche sovrapposte
            if rec and rec["dir"] in seen:
                continue
            g = total_filter(rec, channel, wp, which) if rec else None
            if g is None:
                continue
            seen.add(rec["dir"])
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
    """Mediane sui punti in comune. Le colonne analitiche e simulate sono SEPARATE: se il
    Monte Carlo non c'e' la sua colonna resta vuota, non prende il valore analitico."""
    print(f"\n{'set':>16s} {'N':>4s} {'BI analitico':>13s} {'N mc':>5s} {'BI simulato':>13s} "
          f"{'mc/an':>7s} {'vs ref (an)':>12s} {'vs ref (mc)':>12s}")
    for lab in keys:
        an = [rec["BI"] for rec in data[lab].values()]
        mc = [bi_mc(rec)[0] for rec in data[lab].values() if bi_mc(rec) is not None]
        rat = [bi_mc(rec)[0] / rec["BI"] for rec in data[lab].values()
               if bi_mc(rec) is not None and rec["BI"]]
        rel = {"analytic": [], "mc": []}
        for k, rec in data[lab].items():
            if REFERENCE not in keys or k not in data[REFERENCE]:
                continue
            for srcname, lst in rel.items():
                d = delta_pct(data[REFERENCE][k], rec, f"{REFERENCE}:{srcname}",
                              f"{lab}:{srcname}")
                if d is not None:
                    lst.append(1 + d / 100.0)
        med = lambda x: np.median(x) if len(x) else np.nan
        print(f"{lab:>16s} {len(an):>4d} {med(an):13.4e} {len(mc):>5d} {med(mc):13.4e} "
              f"{med(rat):7.3f} {med(rel['analytic']):12.3f} {med(rel['mc']):12.3f}")
    print(f"  'vs ref' = rapporto mediano rispetto a {REFERENCE}, calcolato separatamente sul "
          f"BI analitico e su quello simulato,\n  sui soli punti in cui la sorgente esiste per "
          f"entrambi i set. Vuoto = quella sorgente non c'e'.")
    for a, b in PAIRS:
        ta, tb = series(a)[0], series(b)[0]
        if ta not in keys or tb not in keys:
            continue
        z = [significance(rec, data[tb][k], a, b) for k, rec in data[ta].items()
             if k in data[tb]]
        z = [x for x in z if x is not None]
        if not z:
            print(f"\ncoppia {pair_label(a, b)}: nessun punto con il Monte Carlo dove serve, "
                  f"z non calcolabile.")
            continue
        z = np.array(z)
        print(f"\ncoppia {pair_label(a, b)}: {len(z)} punti, z mediano {np.median(z):+.2f}, "
              f"|z|>2 in {(np.abs(z) > 2).mean():.0%} dei casi")
        print("  z calcolato con le sole sigma del Monte Carlo, e solo dove il MC c'e' per "
              "tutti e due i set.\n  Gli eventi sono gli STESSI nei due set (stesso seed), "
              "quindi i BI sono correlati e lo z\n  e' conservativo. E sigma_BI e' errore di "
              "NSIM: alzando NSIM ogni differenza diventa\n  'significativa'.")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data, keys = {}, []
    for tag, folder in FOLDERS.items():
        d = load_set(folder, GENS[tag])
        if d is None:
            print(f"[INFO] salto '{tag}': nessun CSV in {folder}")
            continue
        data[tag], _ = d, keys.append(tag)
        print(f"[OK]   {tag:<14s} = {LABELS[tag]:<38s} ({len(d)} coppie da {folder})")
    if not keys:
        raise SystemExit("[ERROR] nessun set disponibile")

    chans = sorted({c for lab in keys for (c, _) in data[lab]})
    if ONLY_CHANNELS:
        missing = [c for c in ONLY_CHANNELS if c not in chans]
        if missing:
            print(f"[INFO] canali richiesti e non presenti in nessun set: {missing}")
        chans = [c for c in chans if c in ONLY_CHANNELS]
        if not chans:
            raise SystemExit(f"[ERROR] ONLY_CHANNELS={ONLY_CHANNELS} non seleziona niente")
    for ch in chans:
        print(f"   -> {os.path.relpath(plot_bi(data, ch, keys), BASE_DIR)}")
        for which in ("f1", "f2"):
            out = plot_total_filters(data, ch, which, keys)
            if out:
                print(f"   -> {os.path.relpath(out, BASE_DIR)}")
    summary_table(data, keys)


if __name__ == "__main__":
    main()
