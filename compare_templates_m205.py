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

SETS = [("root",        "m205_results_octopus"),
        ("fit",         "m205_results_octopus_fit"),
        ("sim rootinj", "m205_results_octopus_sim_rootinj"),
        ("sim fitinj",  "m205_results_octopus_sim_fitinj")]
PAIRS = [("root", "sim rootinj"), ("fit", "sim fitinj")]
REFERENCE = "root"       # set rispetto a cui si misurano rapporti e distanze fra filtri

BI_SOURCE = "mc"         # "mc" = BI Monte Carlo con barre d'errore ; "analytic" = BI analitico
FILTER_WPS = [15]        # WP per cui disegnare i filtri in dettaglio
OUT_DIR  = os.path.join(BASE_DIR, "comparisons", "templates" + NPS_TAG)
COLORS   = {"root": "tab:blue", "fit": "tab:green",
            "sim rootinj": "tab:orange", "sim fitinj": "tab:red"}
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
def plot_bi(data, channel, keys):
    """BI vs V_bias per i quattro set, piu' la compatibilita' dentro ogni coppia."""
    fig, ax = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True,
                           gridspec_kw={"height_ratios": [2.2, 1]})
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
        v, z = [], []
        for (c, wp), rec in sorted(data[a].items()):
            if c != channel or (c, wp) not in data[b]:
                continue
            zz = significance(rec, data[b][(c, wp)])
            if zz is None:
                continue
            v.append(rec["vbias"]); z.append(zz)
        if v:
            ax[1].plot(v, z, "o-", ms=4, lw=1, color=COLORS[b], label=f"{b} vs {a}")
    for k, c in ((1, "0.6"), (2, "0.8")):
        ax[1].axhspan(-k, k, color=c, alpha=0.25, zorder=0)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_ylabel(r"z = $\Delta$BI / $\sigma_{MC}$"); ax[1].set_xlabel("V bias [V]")
    ax[1].legend(fontsize=8); ax[1].grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"BI_vs_vbias_ch{channel}.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


def smooth(y, size=51):
    """Mediana mobile: i filtri addestrati sono frastagliati e senza questa il confronto
    fra due set non si legge."""
    from scipy.ndimage import median_filter
    return median_filter(np.asarray(y, dtype=float), size=size, mode="nearest")


def plot_filters(data, channel, wp, keys):
    """f1 e f2 dei quattro set sovrapposti, piu' il rapporto rispetto al set di riferimento."""
    fr = np.fft.rfftfreq(WINDOW, 1 / SAMPLING_RATE)
    ref = load_filters(data[REFERENCE][(channel, wp)], channel, wp) \
        if REFERENCE in keys and (channel, wp) in data[REFERENCE] else None
    fig, ax = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True)
    drawn = False
    for lab in keys:
        rec = data[lab].get((channel, wp))
        f = load_filters(rec, channel, wp) if rec else None
        if f is None:
            continue
        drawn = True
        for i in (0, 1):
            n = min(len(fr), len(f[i]))
            y = np.abs(f[i][1:n])
            # i filtri addestrati sono frastagliati bin per bin: la curva grezza resta in
            # trasparenza, quella lisciata (mediana mobile) e' l'unica leggibile nel rapporto.
            ax[0][i].semilogx(fr[1:n], y, lw=0.4, alpha=0.25, color=COLORS[lab])
            ax[0][i].semilogx(fr[1:n], smooth(y), lw=1.3, color=COLORS[lab], label=lab)
            if ref is not None:
                r = y / np.abs(ref[i][1:n])
                ax[1][i].semilogx(fr[1:n], r, lw=0.4, alpha=0.2, color=COLORS[lab])
                ax[1][i].semilogx(fr[1:n], smooth(r), lw=1.3, color=COLORS[lab])
    if not drawn:
        plt.close(fig)
        return None
    for i, name in enumerate(("f1", "f2")):
        ax[0][i].set_title(f"{name}  -  Ch{channel} WP{wp}")
        ax[0][i].set_yscale("log"); ax[0][i].grid(True, which="both", ls="--", alpha=0.3)
        ax[1][i].axhline(1, color="k", lw=0.8)
        ax[1][i].set_ylim(0.3, 2.2); ax[1][i].set_xlabel("frequency [Hz]")
        ax[1][i].set_ylabel(f"ratio to '{REFERENCE}'"); ax[1][i].grid(True, ls="--", alpha=0.3)
    ax[0][0].legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"filters_ch{channel}_wp{wp}.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


def f90(f):
    """Frequenza entro cui sta il 90% del peso del filtro: una misura di banda in un numero."""
    fr = np.fft.rfftfreq(WINDOW, 1 / SAMPLING_RATE)[:len(f)]
    c = np.cumsum(np.abs(f))
    return float(np.interp(0.9 * c[-1], c, fr))


def plot_filter_summary(data, keys):
    """Due numeri per filtro e per WP: la banda (f90) e la distanza dal set di riferimento."""
    chans = sorted({c for lab in keys for (c, _) in data[lab]})
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for lab in keys:
        v, band, dist = [], [], []
        for (c, wp), rec in sorted(data[lab].items()):
            f = load_filters(rec, c, wp)
            g = load_filters(data[REFERENCE].get((c, wp)), c, wp) if REFERENCE in keys else None
            if f is None:
                continue
            v.append(rec["vbias"]); band.append(f90(f[0]))
            dist.append(np.nan if g is None else
                        float(np.linalg.norm(f[0] - g[0]) / np.linalg.norm(g[0])))
        if not v:
            continue
        ax[0].plot(v, band, "o", ms=5, color=COLORS[lab], label=lab)
        ax[1].plot(v, dist, "o", ms=5, color=COLORS[lab], label=lab)
    ax[0].set_ylabel("f90 of f1  [Hz]"); ax[0].set_title("filter bandwidth")
    ax[1].set_ylabel(f"|f1 - f1({REFERENCE})| / |f1({REFERENCE})|")
    ax[1].set_title("distance from the reference filters")
    for a in ax:
        a.set_xlabel("V bias [V]"); a.grid(True, ls="--", alpha=0.4); a.legend(fontsize=9)
    fig.suptitle(f"m205 - band filters, {len(chans)} channels" + (", clean NPS" if NPS_TAG else ""))
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "filter_summary.png")
    fig.savefig(out, dpi=130); plt.close(fig)
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

    for ch in sorted({c for lab in keys for (c, _) in data[lab]}):
        print(f"   -> {os.path.relpath(plot_bi(data, ch, keys), BASE_DIR)}")
        for wp in FILTER_WPS:
            out = plot_filters(data, ch, wp, keys)
            if out:
                print(f"   -> {os.path.relpath(out, BASE_DIR)}")
    print(f"   -> {os.path.relpath(plot_filter_summary(data, keys), BASE_DIR)}")
    summary_table(data, keys)


if __name__ == "__main__":
    main()
