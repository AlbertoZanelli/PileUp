#!/usr/bin/env python3
"""
plot_BI_distributions_m205.py
=============================
Distribuzioni SUI SEED del BI Monte Carlo e del Delta BI, per ogni (canale, WP).

Legge gli stessi set di compare_templates_m205.py -- SETS, PAIRS, MC_CSV, ONLY_CHANNELS: la
config sta LA' -- e ne riusa lettura, allineamento dei seed e Delta BI appaiato, cosi' i numeri
sono quelli dei grafici di confronto. Una figura per canale e per quantita', un pannello per WP:
  dist_BI_ch<ch>.png   istogramma di BI_s per set, tratteggio = mediana. Nel pannello
                       sigma_p(BI_s) / sigma della formula (compute_BI_uncertainty, per UNA
                       run), con sigma_p = (P84-P16)/2 dei seed: ~1 se l'errore per run e' giusto
  dist_dBI_ch<ch>.png  istogramma di Delta BI_s [%] per coppia, seed per seed (stessi eventi);
                       linea = mediana, banda = errore della mediana, cioe' la barra dei
                       grafici di confronto

Uso:
    KMP_DUPLICATE_LIB_OK=TRUE python3 plot_BI_distributions_m205.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import compare_templates_m205 as ct


def grid(n):
    rows, cols = ct.GRID
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.0 * rows), squeeze=False)
    for a in axes.ravel()[n:]:
        a.axis("off")
    return fig, axes.ravel()


def save(fig, name, title):
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(ct.OUT_DIR, name)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_bi(data, keys, ch, wps):
    """Istogrammi di BI_s. Restituisce i rapporti std/sigma_formula, per il riepilogo."""
    fig, axes = grid(len(wps))
    ratios = {lab: [] for lab in keys}
    for a, wp in zip(axes, wps):
        seeds = {lab: np.array(list(data[lab][(ch, wp)]["seeds"].values())).T
                 for lab in keys if len(data[lab].get((ch, wp), {}).get("seeds", {})) > 1}
        if not seeds:
            a.axis("off")
            continue
        edges = np.histogram_bin_edges(np.concatenate([v[0] for v in seeds.values()]), bins=15)
        for i, (lab, (bi, sg, _)) in enumerate(seeds.items()):
            r = ct.robust_sigma(bi) / sg.mean()
            ratios[lab].append(r)
            a.hist(bi, bins=edges, histtype="step", lw=1.5, color=ct.color(lab), label=lab)
            a.axvline(np.median(bi), color=ct.color(lab), ls="--", lw=1)
            a.text(0.02, 0.97 - 0.09 * i, f"σ_p/σ = {r:.2f}", transform=a.transAxes, va="top",
                   fontsize=8, color=ct.color(lab))
        a.set_title(f"WP{wp}  ({data[keys[0]][(ch, wp)]['vbias']:g} V)", fontsize=10)
        a.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        a.set_xlabel("BI  [counts/keV/kg/yr]", fontsize=8)
    axes[0].legend(fontsize=8)
    n = len(next(iter(seeds.values()))[0]) if seeds else 0
    return save(fig, f"dist_BI_ch{ch}.png",
                f"m205 Ch{ch} — Monte Carlo BI over {n} seeds (dashed: median; σ_p = (P84−P16)/2 "
                f"of the seeds, σ = per-run formula error)"), ratios


def plot_dbi(data, keys, ch, wps):
    """Istogrammi di Delta BI_s [%] per ogni coppia di PAIRS con tutti e due i set."""
    pairs = [(a, b) for a, b in ct.PAIRS
             if ct.series(a)[0] in keys and ct.series(b)[0] in keys]
    fig, axes = grid(len(wps))
    drawn = False
    for a, wp in zip(axes, wps):
        txt = []
        for sa, sb in pairs:
            ta, tb = ct.series(sa)[0], ct.series(sb)[0]
            if (ch, wp) not in data[ta] or (ch, wp) not in data[tb]:
                continue
            d = ct.paired(data[ta][(ch, wp)], data[tb][(ch, wp)], sa, sb)
            if d is None:
                continue
            drawn = True
            col = ct.DELTA_COLOR if len(pairs) == 1 else ct.color(tb)
            m, e = ct.med_err(d)
            a.hist(d, bins=15, histtype="stepfilled", alpha=0.35, color=col,
                   label=ct.pair_label(sa, sb))
            a.axvline(m, color=col, lw=1.6)
            a.axvspan(m - e, m + e, color=col, alpha=0.25, lw=0)
            txt.append(f"{m:+.2f} ± {e:.2f} %")
        if not txt:
            a.axis("off")
            continue
        a.text(0.02, 0.97, "\n".join(txt), transform=a.transAxes, va="top", fontsize=8,
               color="0.25")
        a.set_title(f"WP{wp}  ({data[keys[0]][(ch, wp)]['vbias']:g} V)", fontsize=10)
        a.set_xlabel("ΔBI  [%]", fontsize=8)
    if not drawn:
        plt.close(fig)
        return None
    axes[0].legend(fontsize=8)
    return save(fig, f"dist_dBI_ch{ch}.png",
                f"m205 Ch{ch} — ΔBI seed by seed, same events (line: median, "
                f"band: error of the median)")


def main():
    data, keys = {}, []
    for tag, folder in ct.FOLDERS.items():
        d = ct.load_set(folder, ct.GENS[tag])
        if d is not None:
            data[tag] = d
            keys.append(tag)
    if not keys:
        raise SystemExit("[ERROR] nessun set disponibile")
    ct.align_seeds(data, keys)
    os.makedirs(ct.OUT_DIR, exist_ok=True)
    chans = sorted({c for lab in keys for (c, _) in data[lab]})
    ratios = {lab: [] for lab in keys}
    for ch in chans:
        if ct.ONLY_CHANNELS and ch not in ct.ONLY_CHANNELS:
            continue
        wps = sorted({wp for lab in keys for (c, wp) in data[lab] if c == ch})
        out, r = plot_bi(data, keys, ch, wps)
        print(f"   -> {os.path.relpath(out, ct.BASE_DIR)}")
        for lab in keys:
            ratios[lab] += r[lab]
        out = plot_dbi(data, keys, ch, wps)
        if out:
            print(f"   -> {os.path.relpath(out, ct.BASE_DIR)}")
    # il numero da citare: la formula dell'errore per run regge se questo e' ~1
    for lab in keys:
        if ratios[lab]:
            print(f"   {lab}: sigma_p(BI_s)/sigma_formula, mediana su {len(ratios[lab])} punti = "
                  f"{np.median(ratios[lab]):.2f}")


if __name__ == "__main__":
    main()
