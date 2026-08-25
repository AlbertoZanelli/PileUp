#!/usr/bin/env python3
"""
compare_templates_m205.py
=========================
Confronta i BI e i FILTRI addestrati su template diversi, a parita' di tutto il resto.

I quattro set:
    root        -> template = medianAP di Octopus (l'AP vero)
    fit         -> template = bestfit dello scan (liscio, senza rumore finito-N)
    sim rootinj -> template = AP SIMULATO generato dal medianAP, rumore vero iniettato
    sim fitinj  -> template = AP SIMULATO generato dal fit, rumore vero iniettato

Le due COPPIE sono quelle che isolano l'effetto del rumore di template, perche' dentro
ogni coppia la FORMA del template e' la stessa e cambia solo la realizzazione del rumore:
    root <-> sim rootinj      e      fit <-> sim fitinj
Se il BI di un set simulato e' compatibile con quello del suo originale, il rumore di
template non sta influenzando il risultato; se non lo e', la differenza e' il costo di
addestrare su una realizzazione diversa (cioe' il guadagno fittizio dell'auto-consistenza).

Legge, per ogni set:
  - BI_results_*.csv          (BI analitico, sigma, SNR, J)          da analyse_BI_m205.py
  - BI_mc_error_m205.csv      (BI Monte Carlo + sigma_BI)            da simulate_BI_error_m205.py
  - trained_filters/f{1,2}_ch<ch>_wp<wp>.npy                          i filtri di banda
I set mancanti vengono saltati con un [INFO], cosi' si puo' girare a campagne incomplete.

Uso:
    KMP_DUPLICATE_LIB_OK=TRUE python3 compare_templates_m205.py
"""

import os, csv, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═════════════════════════════════════════════════════════════════════════════
# Config
# ═════════════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NPS_TAG  = ""            # "" (NPS di Octopus) oppure "_npsclean": si appende a tutte le cartelle

#SETS = [("root",        "m205_results_wiener_root_npsclean"),
#        ("sim rootinj", "m205_results_wiener_sim_rootinj_npsclean"),
#        ("root_swna1",        "m205_results_wiener_root_npsclean_swna1"),
#        ("sim rootinj_swna1", "m205_results_wiener_sim_rootinj_npsclean_swna1"),]
#PAIRS = [("root", "sim rootinj"), ("root_swna1", "sim rootinj_swna1")]

SETS = [("sim rootinj", "m205_results_octopus_sim_rootinj_npsclean"),
        ("sim rootinj_swna1", "m205_results_wiener_sim_rootinj_npsclean_swna1"),]

PAIRS = [("sim rootinj", "sim rootinj_swna1")]

REFERENCE = "root"       # set rispetto a cui si misurano rapporti e distanze fra filtri

BI_SOURCE = "mc"         # "mc" = BI Monte Carlo con barre d'errore ; "analytic" = BI analitico
GRID = (5, 3)            # righe x colonne della griglia dei filtri (15 WP)
# La cartella di output nomina le CARTELLE confrontate (non solo le etichette), cosi'
# convivono confronti fra set diversi: filtro ottimo, Wiener, NPS pulita, combinazioni.
OUT_DIR  = os.path.join(BASE_DIR, "comparisons",
                        "-".join(f.replace("m205_results_", "") + NPS_TAG for _, f in SETS))
COLORS   = {"root": "tab:blue", "root_swna1": "tab:green",
            "sim rootinj": "tab:orange", "sim rootinj_swna1": "tab:red"}
SAMPLING_RATE = 10_000
WINDOW        = 10_000


# ═════════════════════════════════════════════════════════════════════════════
# Lettura
# ═════════════════════════════════════════════════════════════════════════════
def load_set(folder):
    """{(ch, wp): dict} con BI analitico, BI MC e sigma, per un set. None se manca."""
    d = os.path.join(BASE_DIR, folder + NPS_TAG)
    files = glob.glob(os.path.join(d, "BI_results_*.csv"))
    if not files:
        return None
    out = {}
    for r in csv.DictReader(open(files[0])):
        k = (int(r["channel"]), int(r["wp"]))
        out[k] = dict(vbias=float(r["vbias"]), BI=float(r["BI"]),
                      sigma_analytic=float(r["sigma_analytic"]), SNR=float(r["SNR"]),
                      BI_mc=np.nan, sigma_BI=np.nan, dir=d)
    mc = os.path.join(d, "BI_mc_error_m205.csv")
    if os.path.exists(mc):
        for r in csv.DictReader(open(mc)):
            k = (int(r["channel"]), int(r["wp"]))
            if k in out:
                out[k]["BI_mc"] = float(r["BI_mc"])
                out[k]["sigma_BI"] = float(r["sigma_BI"])
    return out


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
def plot_bi(data, channel, keys, tag):
    """BI vs V_bias per i quattro set, piu' la compatibilita' dentro ogni coppia."""
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
        ax[0].errorbar(v, b, yerr=np.where(np.isfinite(e), e, 0), fmt="o", ms=5, capsize=3,
                       lw=1, color=COLORS[lab], label=lab)
    ax[0].set_ylabel("BI  [counts/keV/kg/yr]")
    # il titolo deve dire cosa e' stato DAVVERO disegnato: se il CSV del Monte Carlo manca,
    # value() ripiega sul BI analitico e dirlo "Monte Carlo" sarebbe falso.
    got = [np.isfinite(rec["BI_mc"]) for lab in keys for (c, _), rec in data[lab].items()
           if c == channel]
    src = ("Monte Carlo" if BI_SOURCE == "mc" and all(got) and got else
           "analytic" if not any(got) else "Monte Carlo where available, else analytic")
    ax[0].set_title(f"m205 Ch{channel} - BI vs bias, four training templates"
                    f"  ({src}{', clean NPS' if NPS_TAG else ''})")
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
            ax[1].plot(v, z, "o-", ms=4, lw=1, color=COLORS[b], label=f"{b} vs {a}")
        if vd:
            ax[2].plot(vd, d, "o-", ms=4, lw=1, color=COLORS[b], label=f"{b} vs {a}")
    for k, c in ((1, "0.6"), (2, "0.8")):
        ax[1].axhspan(-k, k, color=c, alpha=0.25, zorder=0)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_ylabel(r"z = $\Delta$BI / $\sigma_{MC}$")
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_ylabel(r"$\Delta$BI  [%]"); ax[2].set_xlabel("V bias [V]")
    for a in ax[1:]:
        if a.get_legend_handles_labels()[0]:      # niente coppie disegnate -> niente legenda
            a.legend(fontsize=8)
        a.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"BI_vs_vbias_ch{channel}_{tag}.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


def sets_tag(keys):
    """Identifica il confronto, cosi' due confronti diversi non si sovrascrivono i file."""
    return "_".join(l.replace(" ", "") for l in keys) + NPS_TAG


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


def plot_total_filters(data, channel, which, keys, tag):
    """Una griglia per canale e per filtro: un pannello per WP, dentro i quattro set."""
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
            a.loglog(fr[1:n], g[1:n], lw=0.4, alpha=0.2, color=COLORS[lab])
            a.loglog(fr[1:n], sm, lw=1.2, color=COLORS[lab], label=lab)
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
    fig.suptitle(f"m205 Ch{channel} - total filter {which} = {which} x kernel"
                 + (", clean NPS" if NPS_TAG else ""), y=0.998, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.952])
    out = os.path.join(OUT_DIR, f"totalfilter_{which}_ch{channel}_{tag}.png")
    fig.savefig(out, dpi=110); plt.close(fig)
    return out


def summary_table(data, keys):
    """Mediane sui punti in comune: BI relativo al riferimento e compatibilita' per coppia."""
    print(f"\n{'set':>12s} {'punti':>6s} {'BI mediano':>12s} {'vs ' + REFERENCE:>10s} "
          f"{'MC/analitico':>13s}")
    for lab in keys:
        bi, rel, rat = [], [], []
        for k, rec in data[lab].items():
            b, _ = value(rec)
            bi.append(b)
            if np.isfinite(rec["BI_mc"]):
                rat.append(rec["BI_mc"] / rec["BI"])
            if REFERENCE in keys and k in data[REFERENCE]:
                rel.append(b / value(data[REFERENCE][k])[0])
        print(f"{lab:>12s} {len(bi):>6d} {np.median(bi):12.4e} "
              f"{(np.median(rel) if rel else np.nan):10.3f} "
              f"{(np.median(rat) if rat else np.nan):13.3f}")
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
    for lab, folder in SETS:
        d = load_set(folder)
        if d is None:
            print(f"[INFO] salto '{lab}': nessun CSV in {folder + NPS_TAG}")
            continue
        data[lab], _ = d, keys.append(lab)
        print(f"[OK]   '{lab}': {len(d)} coppie da {folder + NPS_TAG}")
    if not keys:
        raise SystemExit("[ERROR] nessun set disponibile")

    tag = sets_tag(keys)
    for ch in sorted({c for lab in keys for (c, _) in data[lab]}):
        print(f"   -> {os.path.relpath(plot_bi(data, ch, keys, tag), BASE_DIR)}")
        for which in ("f1", "f2"):
            out = plot_total_filters(data, ch, which, keys, tag)
            if out:
                print(f"   -> {os.path.relpath(out, BASE_DIR)}")
    summary_table(data, keys)


if __name__ == "__main__":
    main()
