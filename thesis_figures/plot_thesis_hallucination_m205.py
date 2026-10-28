r"""
plot_thesis_hallucination_m205.py
=================================
Thesis figures for the "hallucination" study (Chapter 4, Sec. 4.2 of the skeleton): the original
algorithm (OF, 500 steps) trained on the Octopus average pulse ("root") vs on the fit, MC events
always generated from the fit, 50 seeds x 50 000 events per population, same events for both.

  BI_root_vs_fit_ch<ch>   top: BI vs V_bias, Monte Carlo (points, error = WIDTH (P84-P16)/2 of the
                          50-seed distribution) and analytic (line), for both templates, CUPID target;
                          bottom: Delta BI = BI(Octopus AP)/BI(fit) - 1, analytic and MC (paired seed
                          by seed, error = width of the per-seed Delta). Symlog: linear within +-10%.
  MC_over_analytic_vs_SNR all 75 points: BI_MC / BI_analytic vs SNR_OF, one marker per template.

Data loading, seed alignment and the width come from compare_templates_m205.py; style, channels and
the V_bias axis from plot_thesis_loadcurve_m205.py. PDF in OUTDIR.

Run:  KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/bin/python3.13 thesis_figures/plot_thesis_hallucination_m205.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter, FuncFormatter

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # project root (this script is in thesis_figures/)
import plot_thesis_loadcurve_m205 as st     # thesis style (rcParams), channels, V_bias axis
import compare_templates_m205 as ct         # campaign CSVs, common seeds, width

OUTDIR = os.path.join(st.BASE, "thesis_figures", "hallucination")
SETS = {"root": "m205_results_octopus_npsclean",        # trained on the Octopus AP
        "fit":  "m205_results_octopus_fit_npsclean"}    # trained on the fit
LABEL = {"root": "Octopus AP", "fit": "fit"}
COLOR = {"root": "#2a78d6", "fit": "#eb6834"}
MARK = {"root": "o", "fit": "s"}
C_MC, C_AN = "#0d366b", "0.45"                          # Delta BI: MC, analytic
BI_TARGET = 5e-5                                        # CUPID pile-up target
BI_UNIT = 1e-5
DBI_LIN = 10                                            # %: symlog linear range
DBI_TICKS = [-10, -5, 0, 5, 10, 30, 100]                # small ranges get [-5, -2, 0, 2, 5]

ct.ERROR_BAR = "width"
SHOW_ANALYTIC = False     # thesis (11/10): Monte Carlo only (analytic vs MC includes the model error, see verifiche_allucinazione)
SHOW_ANALYTIC_DBI = False # bottom panels: only the MC Delta BI between the two trainings (11/10)


def load():
    data = {k: ct.load_set(f, "fit") for k, f in SETS.items()}
    ct.align_seeds(data, list(data))
    return data


def delta(r, f):
    """(analytic, MC median, MC width) of 100 (BI_root/BI_fit - 1); MC paired seed by seed."""
    s = sorted(r["seeds"].keys() & f["seeds"].keys())
    d = np.array([100 * (r["seeds"][k][0] / f["seeds"][k][0] - 1) for k in s])
    return (100 * (r["BI"] / f["BI"] - 1), *ct.med_err(d))


def plot_channel(data, ch):
    wps = list(st.WPS)
    v = [st.vbias(wp) for wp in wps]
    fig, (ax, axd) = plt.subplots(2, 1, figsize=(3.4, 3.9), sharex=True,
                                  gridspec_kw={"height_ratios": [1.6, 1], "hspace": 0.08})
    for k in SETS:
        rec = [data[k][(ch, wp)] for wp in wps]
        if SHOW_ANALYTIC:
            ax.plot(v, [r["BI"] / BI_UNIT for r in rec], "-", color=COLOR[k], lw=1,
                    label=f"{LABEL[k]}, analytic")
        ax.errorbar(v, [r["BI_mc"] / BI_UNIT for r in rec], [r["sigma_BI"] / BI_UNIT for r in rec],
                    fmt=MARK[k], color=COLOR[k], ms=3.5, mec="white", mew=0.5, elinewidth=0.8,
                    capsize=1.5, label=f"{LABEL[k]}, MC")
    ax.axhline(BI_TARGET / BI_UNIT, color="0.3", ls="--", lw=0.8)
    ax.text(0.02, BI_TARGET / BI_UNIT, "CUPID target", transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=7, color="0.3")
    ax.set_ylabel(r"BI [$10^{-5}$ c/(keV kg yr)]")

    an, mc, w = np.array([delta(data["root"][(ch, wp)], data["fit"][(ch, wp)]) for wp in wps]).T
    axd.axhline(0, color="0.6", lw=0.6)
    if SHOW_ANALYTIC_DBI:
        axd.plot(v, an, "--o", color=C_AN, lw=0.9, ms=3.2, mfc="white", label=r"$\Delta$BI analytic")
    else:
        an = mc                                   # tick range from the MC only
    axd.errorbar(v, mc, w, fmt="-o", color=C_MC, lw=0.9, ms=3.2, mec="white", mew=0.5,
                 elinewidth=0.8, capsize=1.5, label=r"$\Delta$BI MC")
    axd.set_yscale("symlog", linthresh=DBI_LIN, linscale=1)
    lo, hi = min(an.min(), (mc - w).min()), max(an.max(), (mc + w).max())
    ticks = DBI_TICKS if max(-lo, hi) > 6 else [-5, -2, 0, 2, 5]
    axd.yaxis.set_major_locator(FixedLocator([t for t in ticks if lo - 1 <= t <= hi + 1]))
    axd.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:+g}" if x else "0"))
    axd.yaxis.set_minor_locator(FixedLocator([]))
    axd.set_ylabel(r"$\Delta$BI [\%]")
    h = [x for a in (ax, axd) for x in zip(*a.get_legend_handles_labels())]
    ax.legend(*zip(*h), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7,
              handlelength=1.5, handletextpad=0.4, columnspacing=0.8, borderaxespad=0.2)
    ax.set_title(f"Ch {ch}", fontsize=10, loc="left", pad=28)
    st.log_vbias_axis(axd)
    return fig


def plot_ratio(data):
    fig, ax = plt.subplots(figsize=st.SMALL)
    ax.axhline(1, color="0.6", lw=0.6)
    for k in SETS:
        rec = list(data[k].values())
        ax.errorbar([r["SNR"] for r in rec], [r["BI_mc"] / r["BI"] for r in rec],
                    [r["sigma_BI"] / r["BI"] for r in rec], fmt=MARK[k], color=COLOR[k],
                    ms=3.5, mec="white", mew=0.5, elinewidth=0.8, capsize=0,
                    label=f"trained on {LABEL[k] if k == 'root' else 'the fit'}")
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 50, 100, 150])
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel(r"SNR$_{\rm OF}$")
    ax.set_ylabel(r"BI$_{\rm MC}$ / BI$_{\rm analytic}$")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, borderaxespad=0.3)
    return fig


def save(fig, name):
    fig.savefig(os.path.join(OUTDIR, f"{name}.pdf"))
    plt.close(fig)
    print(f"  -> {os.path.relpath(os.path.join(OUTDIR, name), st.BASE)}.pdf")


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    data = load()
    # the paired Delta must reproduce the numbers quoted in HANDOFF.md (75/75 worse on the MC)
    assert all(delta(data["root"][k], data["fit"][k])[1] > 0 for k in data["root"])
    for ch in st.CHANNELS:
        save(plot_channel(data, ch), f"BI_root_vs_fit_ch{ch}")
    save(plot_ratio(data), "MC_over_analytic_vs_SNR")
