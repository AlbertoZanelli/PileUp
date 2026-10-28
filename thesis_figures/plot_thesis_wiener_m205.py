r"""
plot_thesis_wiener_m205.py
==========================
Thesis figures for Chapter 4, Sec. "Trainable Wiener Filter" (request: HANDOFF_richiesta_figure_wiener.md).
No training: everything is read from the trained folders, and the few event simulations (rejection of
fig. 3, cancellation check) are cached in OUTDIR as CSV. Prints every number quoted in the text and
asserts the ones of the request.

  wiener_kernel    (a) |S|^2 and lambda N (CUORE normalisation) for lambda = 1, 1e-4; (b) noiseless pulse
                   after OF, W_1, W_1e-4, with s = filtered-noise std / amplitude          ch34 WP3
  wiener_nopen     Wiener WITHOUT penalty, 75 points: (a) Delta BI vs OF, analytic + MC, ch34 and ch91;
                   (b) lambda vs BI_MC of the OF; (c) s1 vs lambda; (d) 1 - rho vs s1, with sigma_Y,0 lines
  wiener_events    one simulated event (single / pile-up r = 0.5, dt = 0.8 ms, same noise) after the two
                   filters of OF, Wiener without and with penalty                          ch34 WP3
  wiener_training  Wiener WITH penalty, training history at ch31 WP7, ch34 WP15, ch91 WP15
  wiener_vs_of     Delta BI MC, Wiener with penalty vs OF, 75 points

s1, s2, rho, sigma_Y,0 come from an.compute_vars_wiener (production code) with the trained f1, f2, the
kernel recomputed from the CSV lambda and checked against the saved one (mc.load_filters), the training
template (APsimfit10000led) and the clean NPS. Errors: width (P84 - P16)/2 of the 50 paired seeds.

Run:  KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/bin/python3.13 thesis_figures/plot_thesis_wiener_m205.py [--resim]
      --resim  redo the cached event simulations (~15 min)
"""

import os
import sys
import csv

# torch's OpenMP aborts on this Mac (pthread_mutex_init) once the other libraries are loaded
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, NullFormatter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # project root (this script is in thesis_figures/)
import plot_thesis_loadcurve_m205 as st     # thesis style (rcParams), channel colours, V_bias axis
import compare_templates_m205 as ct         # campaign CSVs, common seeds, width
import simulate_BI_error_m205 as mc         # templates, NPS, trained filters (kernel checked)
import src.analysis as an
import src.dataset as ds

OUTDIR = os.path.join(st.BASE, "thesis_figures", "wiener")
FOLD = {"OF": "m205_results_octopus_APsimfit10000led_npsclean",
        "W": "m205_results_wiener_APsimfit10000led_npsclean",              # no penalty
        "Wwna": "m205_results_wiener_APsimfit10000led_npsclean_swna1"}     # penalty w = 1
HIST_DIR = os.path.join(st.BASE, "m205_results_wiener_APsimfit10000led_npsclean_swna1_hist")
KERNEL_PT = (34, 3)                     # figs 1 and 3
LAMS = [1.0, 1e-4]
EVENT = dict(r=0.5, dt=0.8e-3, seed=3)  # fig. 3: example event
N_REJ = 20_000                          # fig. 3: events per population for the rejection
N_CHECK = 5_000                         # singles per point for the sigma_Y,0 check
TRAIN_PTS = [(31, 7), (34, 15), (91, 15)]
LAM_LOW = 0.1
FULL, GRID22, GRID32 = (5.9, 2.6), (5.9, 4.8), (5.9, 5.2)
C_OF, C_W = "0.25", {1.0: "#2a78d6", 1e-4: "#eb6834"}
C_RAW, C_F1, C_F2 = "0.6", "#2a78d6", "#eb6834"
ct.ERROR_BAR = "width"


def label(ax, s):
    ax.set_title(f"({s})", loc="left", fontsize=10, pad=3)


def chlab(ch):
    return f"ch. {ch}"


def rows(folder):
    return {(int(r["channel"]), int(r["wp"])): r
            for r in csv.DictReader(open(mc.folder_info(folder)["bi_csv"]))}


def save(fig, name):
    fig.savefig(os.path.join(OUTDIR, f"{name}.pdf"))
    plt.close(fig)
    print(f"  -> {os.path.relpath(os.path.join(OUTDIR, name), st.BASE)}.pdf")


FAILED = []


def check(name, value, expected, tol):
    """Number of the request vs the data, within tol. Mismatches are collected and asserted at the
    end, so that one run reports all of them."""
    ok = abs(value - expected) <= tol
    print(f"  {'ok ' if ok else 'XX '} {name:52s} {value:12.4g}   (request {expected:g})")
    if not ok:
        FAILED.append(f"{name}: data {value:.4g}, request {expected:g}")


# ── inputs of a point ───────────────────────────────────────────────────────────────────────
def point(folder, ch, wp, row):
    """(K, f1, f2, S, w, nps, A) with the production loaders; K checked against the saved kernel."""
    f = mc.folder_info(folder)
    nps = mc.load_noise(ch, wp)
    S, w, _ = an.compute_H(mc.template_pulse(ch, wp, f["train"], f["sim"]), nps, np.hanning,
                           sampling_rate=mc.SAMPLING_RATE)
    K, f1, f2 = mc.load_filters(f, ch, wp, nps, row)
    return K, f1, f2, S, w, nps, float(row["signal_amp"])


def filter_stats(folder):
    """{(ch, wp): dict(lam, s1, s2, rho, sY0, norm1, norm2)} from compute_vars_wiener."""
    out = {}
    for k, r in rows(folder).items():
        K, f1, f2, S, w, nps, A = point(folder, *k, r)
        Kt, St = torch.tensor(K), torch.tensor(S)
        v1, v2, c = (float(x) for x in an.compute_vars_wiener(Kt, St, nps, torch.tensor(f1),
                                                              torch.tensor(f2)))
        s1, s2, rho = v1 ** 0.5 / A, v2 ** 0.5 / A, c / (v1 * v2) ** 0.5
        out[k] = dict(lam=float(r.get("lambda_wiener") or np.inf), s1=s1, s2=s2, rho=rho,
                      sY0=np.sqrt(max(s1 ** 2 + s2 ** 2 - 2 * rho * s1 * s2, 0)),
                      norm1=np.mean(np.abs(f1 * K * S)), norm2=np.mean(np.abs(f2 * K * S)),
                      s1_csv=float(r["s1"]) if r.get("s1") else np.nan)
    return out


def load_mc():
    data = {k: ct.load_set(f, "fit") for k, f in FOLD.items()}
    ct.align_seeds(data, list(data))
    return data


def dbi(a, b):
    """(analytic, MC median, MC width) of 100 (BI_a / BI_b - 1), MC paired seed by seed."""
    s = sorted(a["seeds"].keys() & b["seeds"].keys())
    d = np.array([100 * (a["seeds"][k][0] / b["seeds"][k][0] - 1) for k in s])
    return (100 * (a["BI"] / b["BI"] - 1), *ct.med_err(d))


def filtered(x, g):
    """Event x (time) through the total filter g, peak brought to the centre (as the MC)."""
    n = len(x)
    x = x - np.mean(x[:n // 2 - 100])
    return np.roll(np.fft.ifft(g * np.fft.fft(x)).real, n // 2)


def psd(events, K, f1, f2):
    d = ds.NumpyDataset(np.asarray(events, np.float32))
    d.win_length = events.shape[1]
    return [np.asarray(x).ravel() for x in an.get_PSD_interpole(d, K, f1, f2)]   # Y, A1, A2


def make_events(S, w, nps, A, n, r, dt, rng):
    noise = (rng.normal(size=(n, len(S))) + 1j * rng.normal(size=(n, len(S)))) * np.sqrt(nps)
    return np.fft.ifft(A * S * ((1 - r) + r * np.exp(-1j * w * dt)) + noise, axis=1).real


# ── fig. 1 ──────────────────────────────────────────────────────────────────────────────────
def fig_kernel(snr):
    ch, wp = KERNEL_PT
    r = rows(FOLD["OF"])[(ch, wp)]
    K, _, _, S, w, nps, A = point(FOLD["OF"], ch, wp, r)
    n, fs = len(S), mc.SAMPLING_RATE
    fr = np.fft.fftfreq(n, 1 / fs)
    pos = fr > 0
    P = np.abs(S) ** 2
    a, b = np.sum(np.sqrt(P)), np.sum(np.sqrt(nps))
    Pn, Nn = P / a, nps * a / b ** 2                       # compute_W_torch normalisation
    unit = lambda g: g / np.mean((g * S).real)              # filtered pulse peak = 1
    kern = [("optimum filter", C_OF, unit(K), "-")]
    for lam in LAMS:
        Wl = an.compute_W_torch(torch.tensor(S), torch.tensor(nps), torch.tensor(lam)).numpy()
        kern.append((rf"Wiener, $\lambda = {'1' if lam == 1 else '10^{-4}'}$", C_W[lam], unit(Wl),
                     "-" if lam == 1 else "--"))
    sig = [np.sqrt(np.sum(np.abs(g) ** 2 * nps)) / n / A for _, _, g, _ in kern]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=FULL, gridspec_kw={"wspace": 0.32})
    a1.loglog(fr[pos], Pn[pos], color=C_RAW, lw=1.2, label=r"$|S|^2$")
    cross = {}
    for lam in LAMS:
        a1.loglog(fr[pos], lam * Nn[pos], color=C_W[lam], lw=0.8,
                  label=rf"$\lambda N$, $\lambda = {'1' if lam == 1 else '10^{-4}'}$")
        cross[lam] = fr[pos][np.argmax(Pn[pos] < lam * Nn[pos])]
        a1.axvline(cross[lam], color=C_W[lam], ls=":", lw=0.8)
        a1.text(cross[lam] * 1.12, 0.97, f"{cross[lam]:.0f} Hz", color=C_W[lam], fontsize=7,
                transform=a1.get_xaxis_transform(), va="top")
    a1.set_xlim(1, fs / 2)
    a1.set_ylim(bottom=max(min(Pn[pos].min(), 1e-4 * Nn[pos].min()), 1e-30))
    a1.set_xlabel("Frequency [Hz]")
    a1.set_ylabel("Power [a.u.]")
    a1.legend(loc="lower left", fontsize=7)
    label(a1, "a")

    t = (np.arange(n) - n // 2) / fs * 1e3
    m = (t >= -5) & (t <= 15)
    clean = np.fft.ifft(S).real
    for (lab, c, g, ls), s in zip(kern, sig):
        y = filtered(clean, g)
        a2.plot(t[m], y[m] / y.max(), color=c, ls=ls, lw=1.1, label=f"{lab} ($s = {s:.2g}$)")
    a2.axhline(0, color="0.8", lw=0.5, zorder=0)
    a2.set_xlim(-5, 15)
    a2.set_xlabel("Time [ms]")
    a2.set_ylabel("Normalized amplitude")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=7)
    label(a2, "b")
    save(fig, "wiener_kernel")

    print(f"\n[fig 1] ch{ch} WP{wp}: SNR_OF = {snr:.1f}, crossings " +
          ", ".join(f"lambda={l:g}: {cross[l]:.0f} Hz" for l in LAMS) +
          "; noise/amplitude " + ", ".join(f"{s:.3g}" for s in sig))
    check("fig1: OF noise/amplitude = 1/SNR_OF", sig[0], 1 / snr, 0.002)
    check("fig1: crossing, lambda = 1 [Hz]", cross[1.0], 160, 5)                     # request: 140
    check("fig1: crossing, lambda = 1e-4 [Hz]", cross[1e-4], 860, 5)
    check("fig1: noise/amplitude, lambda = 1", sig[1], 0.07, 0.005)


# ── fig. 2 ──────────────────────────────────────────────────────────────────────────────────
def fig_nopen(data, sW):
    keys = sorted(sW)
    fig, ax = plt.subplots(2, 2, figsize=GRID22, gridspec_kw={"hspace": 0.45, "wspace": 0.38})
    a, b, c, d = ax.ravel()

    for ch in (34, 91):
        v = [st.vbias(wp) for wp in st.WPS]
        an_, mc_, wd = np.array([dbi(data["W"][(ch, wp)], data["OF"][(ch, wp)]) for wp in st.WPS]).T
        a.plot(v, an_, "--" + st.CH_MARK[ch], color=st.CH_COLOR[ch], mfc="white", lw=0.8, ms=3.5)
        a.errorbar(v, mc_, wd, fmt="-" + st.CH_MARK[ch], color=st.CH_COLOR[ch], mec="white",
                   mew=0.4, lw=0.8, ms=3.5, elinewidth=0.8, capsize=1.5, label=chlab(ch))
    a.axhline(0, color="0.6", lw=0.6)
    st.log_vbias_axis(a)
    a.set_ylabel(r"$\Delta$BI [\%]")
    h = [plt.Line2D([], [], color="0.4", ls="--", marker="o", mfc="white", ms=3.5, lw=0.8),
         plt.Line2D([], [], color="0.4", ls="-", marker="o", ms=3.5, lw=0.8)]
    hc, lc = a.get_legend_handles_labels()
    a.legend(hc + h, lc + ["analytic", "Monte Carlo"], fontsize=7, ncol=2, loc="upper left",
             handlelength=1.6, columnspacing=0.8)
    label(a, "a")

    for ch in st.CHANNELS:
        ks = [k for k in keys if k[0] == ch]
        kw = dict(color=st.CH_COLOR[ch], marker=st.CH_MARK[ch], ls="none", ms=3.5, mec="white",
                  mew=0.4, label=chlab(ch))
        b.plot([data["OF"][k]["BI_mc"] / 1e-5 for k in ks], [sW[k]["lam"] for k in ks], **kw)
        c.plot([sW[k]["lam"] for k in ks], [sW[k]["s1"] for k in ks], **kw)
        d.plot([sW[k]["s1"] for k in ks], [1 - sW[k]["rho"] for k in ks], **kw)
    for y in (LAM_LOW, 1):
        b.axhline(y, color="0.5", ls=":", lw=0.8)
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlabel("BI$_{\\rm MC}$, original algorithm\n[$10^{-5}$ c/(keV kg yr)]")
    b.set_ylabel(r"$\lambda$")
    b.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    b.xaxis.set_minor_formatter(NullFormatter())
    b.set_xticks([5, 10, 20])
    b.legend(fontsize=6.5, loc="lower left", handletextpad=0.2, borderaxespad=0.3)
    label(b, "b")
    c.set_xscale("log"); c.set_yscale("log")
    c.set_xlabel(r"$\lambda$"); c.set_ylabel(r"$s_1$")
    label(c, "c")
    s = np.logspace(np.log10(min(sW[k]["s1"] for k in keys) / 1.5), np.log10(max(sW[k]["s1"] for k in keys) * 1.5), 50)
    lo = min(1 - sW[k]["rho"] for k in keys) / 3
    for sig in (0.01, 0.02, 0.05):                          # labelled at s1 = 4: no points there
        d.plot(s, sig ** 2 / (2 * s ** 2), color="0.55", ls="--", lw=0.7, zorder=0)
        d.text(4.4, sig ** 2 / 32, f"{sig:g}", fontsize=6.5, color="0.4", va="center",
               bbox=dict(fc="white", ec="none", pad=0.5))
    d.text(0.97, 0.95, r"dashed: constant $\sigma_{Y,0}$", transform=d.transAxes, ha="right",
           va="top", fontsize=6.5, color="0.4")
    d.set_xscale("log"); d.set_yscale("log")
    d.set_ylim(lo, 1)
    d.set_xlabel(r"$s_1$"); d.set_ylabel(r"$1-\rho$")
    label(d, "d")
    save(fig, "wiener_nopen")


# ── fig. 3 ──────────────────────────────────────────────────────────────────────────────────
def fig_events(resim):
    ch, wp = KERNEL_PT
    algos = [(k, rows(FOLD[k])[(ch, wp)]) for k in ("OF", "W", "Wwna")]
    pts = {k: point(FOLD[k], ch, wp, r) for k, r in algos}
    _, _, _, _, w, nps, A = pts["OF"]
    S = np.fft.fft(mc.template_pulse(ch, wp, "fit") * np.hanning(len(nps)))   # events from the fit
    rng = np.random.default_rng(EVENT["seed"])
    noise = (rng.normal(size=len(S)) + 1j * rng.normal(size=len(S))) * np.sqrt(nps)
    ev = [np.fft.ifft(A * S * ((1 - r) + r * np.exp(-1j * w * dt)) + noise).real
          for r, dt in ((0, 0), (EVENT["r"], EVENT["dt"]))]

    cache = os.path.join(OUTDIR, "wiener_events_rejection.csv")
    if resim or not os.path.exists(cache):
        g = np.random.default_rng(2026)
        rej = {}
        for k, _ in algos:
            rej[k] = []
        ys = {k: [] for k, _ in algos}
        yp = {k: [] for k, _ in algos}
        for i in range(0, N_REJ, 500):
            e1 = make_events(S, w, nps, A, 500, 0, 0, g)
            e2 = make_events(S, w, nps, A, 500, EVENT["r"], EVENT["dt"], g)
            for k, _ in algos:
                K, f1, f2 = pts[k][:3]
                ys[k].append(psd(e1, K, f1, f2)[0]); yp[k].append(psd(e2, K, f1, f2)[0])
        with open(cache, "w") as fh:
            fh.write("algo,rejected,n\n")
            for k, _ in algos:
                cut = np.percentile(np.concatenate(ys[k]), 10)
                fh.write(f"{k},{np.mean(np.concatenate(yp[k]) < cut):.4f},{N_REJ}\n")
    rej = {r["algo"]: float(r["rejected"]) for r in csv.DictReader(open(cache))}

    t = (np.arange(len(S)) - len(S) // 2) / mc.SAMPLING_RATE * 1e3
    m = (t >= -10) & (t <= 40)
    fig, ax = plt.subplots(3, 2, figsize=GRID32, sharex=True, sharey="row",
                           gridspec_kw={"hspace": 0.25, "wspace": 0.08})
    print(f"\n[fig 3] ch{ch} WP{wp}, r = {EVENT['r']}, dt = {EVENT['dt']*1e3} ms, A = {A:.3g} V")
    for i, (k, r) in enumerate(algos):
        K, f1, f2 = pts[k][:3]
        Y, X1, X2 = psd(np.array(ev), K, f1, f2)
        for j, x in enumerate(ev):
            a = ax[i, j]
            # the raw event already has the pulse at the centre: baseline only, no roll
            raw = x - np.mean(x[:len(x) // 2 - 100])
            a.plot(t[m], raw[m] * 1e3, color=C_RAW, lw=0.6, label="event")
            a.plot(t[m], filtered(x, f1 * K)[m] * 1e3, color=C_F1, lw=0.9, label=r"filter 1")
            a.plot(t[m], filtered(x, f2 * K)[m] * 1e3, color=C_F2, lw=0.9, label=r"filter 2")
            a.axhline(0, color="0.85", lw=0.5, zorder=0)
            a.text(0.98, 0.95, rf"$X_1/A = {X1[j]/A:.2f}$" "\n" rf"$X_2/A = {X2[j]/A:.2f}$" "\n"
                   rf"$Y = {Y[j]:.3f}$", transform=a.transAxes, ha="right", va="top", fontsize=7,
                   bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
            label(a, "abcdef"[2 * i + j])
            print(f"  {k:5s} {'single' if j == 0 else 'pile-up':8s} X1/A = {X1[j]/A:.2f}  "
                  f"X2/A = {X2[j]/A:.2f}  Y = {Y[j]:.3f}")
        ax[i, 0].set_ylabel("Amplitude [mV]")
        lam = r.get("lambda_wiener")
        print(f"  {k:5s} lambda = {float(lam):.3g}" if lam else f"  {k:5s}",
              f"  rejected pile-ups (r = {EVENT['r']}, dt = {EVENT['dt']*1e3} ms): {100*rej[k]:.1f}%")
    for a in ax[-1]:
        a.set_xlabel("Time [ms]")
    ax[0, 0].legend(loc="lower center", bbox_to_anchor=(1.04, 1.12), ncol=3, fontsize=8)
    save(fig, "wiener_events")
    check("fig3: rejected, OF [%]", 100 * rej["OF"], 85, 1)                    # request: 82
    check("fig3: rejected, Wiener no penalty [%]", 100 * rej["W"], 7, 1)       # request: 6
    check("fig3: rejected, Wiener + penalty [%]", 100 * rej["Wwna"], 84, 1)    # request: 80


# ── fig. 4 ──────────────────────────────────────────────────────────────────────────────────
def fig_training():
    hdir = os.path.join(st.BASE, FOLD["Wwna"], "training_history")
    fig, (a, b) = plt.subplots(1, 2, figsize=FULL, gridspec_kw={"wspace": 0.3})
    for ch, wp in TRAIN_PTS:
        h = np.load(os.path.join(hdir, f"hist_ch{ch}_wp{wp}.npz"))
        step = np.arange(1, len(h["J"]) + 1)
        kw = dict(color=st.CH_COLOR[ch], lw=1, label=f"{chlab(ch)}, {st.vbias(wp):g} V")
        a.plot(step, h["J"] / h["J"][-1], **kw)
        b.plot(step, h["lam"], **kw)
    a.set_ylabel(r"$\langle M\rangle / \langle M\rangle_{\rm final}$")
    b.set_yscale("log")
    b.set_ylabel(r"$\lambda$")
    for x, s in ((a, "a"), (b, "b")):
        x.set_xlabel("Training step")
        x.set_xlim(0, 500)
        label(x, s)
    a.legend(fontsize=7, loc="upper right")
    save(fig, "wiener_training")

    lam, grow, dJ, flat = [], 0, [], []
    for f in sorted(os.listdir(hdir)):
        h = np.load(os.path.join(hdir, f))
        lam.append(h["lam"][-1])
        grow += h["lam"][-1] > h["lam"][-101]
        dJ.append(abs(h["J"][-1] / h["J"][-101] - 1))
        flat.append(np.argmax(np.abs(h["J"] / h["J"][-1] - 1) < 0.01) + 1)   # 1% of the final cost
    lam = np.array(lam)
    print(f"\n[fig 4] {len(lam)} histories: lambda final median {np.median(lam):.2f}, "
          f"min {lam.min():.2f}, max {lam.max():.0f}, > 1 in {np.sum(lam > 1)}; growing over the "
          f"last 100 steps in {grow}; cost change over the last 100 steps median {100*np.median(dJ):.3f}%; "
          f"step where the cost is within 1% of its final value: median {np.median(flat):.0f}, "
          f"P10-P90 {np.percentile(flat, 10):.0f}-{np.percentile(flat, 90):.0f}")
    for ch, wp in TRAIN_PTS:
        h = np.load(os.path.join(hdir, f"hist_ch{ch}_wp{wp}.npz"))
        print(f"  ch{ch} WP{wp}: cost within 1% of final at step "
              f"{np.argmax(np.abs(h['J'] / h['J'][-1] - 1) < 0.01) + 1}, lambda final {h['lam'][-1]:.2f}")
    check("fig4: lambda final median", np.median(lam), 7.6, 0.1)
    check("fig4: lambda > 1 [points]", np.sum(lam > 1), 71, 0)
    check("fig4: lambda max", lam.max(), 776, 1)
    check("fig4: lambda min", lam.min(), 0.78, 0.01)
    check("fig4: lambda growing in last 100 steps", grow, 63, 0)
    check("fig4: cost change last 100 steps [%]", 100 * np.median(dJ), 0.09, 0.01)

    # share of the penalty in the loss: only the _hist training stored the loss
    hh = os.path.join(HIST_DIR, "training_history")
    share = []
    for f in sorted(os.listdir(hh)):
        h = np.load(os.path.join(hh, f))
        share.append((h["loss"][-1] - h["J"][-1]) / h["loss"][-1])
    share = 100 * np.array(share)
    print(f"  penalty / loss at the last step ({len(share)} points of {os.path.basename(HIST_DIR)}): "
          f"median {np.median(share):.2f}%, min {share.min():.2f}%, max {share.max():.2f}%, "
          f"P10-P90 {np.percentile(share, 10):.2f}-{np.percentile(share, 90):.2f}%")


# ── fig. 5 ──────────────────────────────────────────────────────────────────────────────────
def fig_vs_of(data):
    fig, a = plt.subplots(figsize=FULL)
    res = {}
    for n, ch in enumerate(st.CHANNELS):
        v = np.array([st.vbias(wp) for wp in st.WPS]) * 1.035 ** (n - 2)     # dodge
        _, m, wd = np.array([dbi(data["Wwna"][(ch, wp)], data["OF"][(ch, wp)]) for wp in st.WPS]).T
        res[ch] = (m, wd)
        a.errorbar(v, m, wd, fmt=st.CH_MARK[ch], color=st.CH_COLOR[ch], ms=3.5, mec="white",
                   mew=0.4, elinewidth=0.8, capsize=1.2, label=chlab(ch))
    a.axhline(0, color="0.5", lw=0.6)
    st.log_vbias_axis(a)
    a.set_ylabel(r"$\Delta$BI [\%]")
    a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, fontsize=8)
    save(fig, "wiener_vs_of")

    m = np.concatenate([res[ch][0] for ch in st.CHANNELS])
    wd = np.concatenate([res[ch][1] for ch in st.CHANNELS])
    print(f"\n[fig 5] Wwna vs OF: median {np.median(m):+.2f}%, better in {np.sum(m < 0)}, better beyond "
          f"2 widths {np.sum(m < -2 * wd)}, worse beyond 2 widths {np.sum(m > 2 * wd)}; per channel "
          + " / ".join(f"{np.median(res[ch][0]):+.2f}" for ch in st.CHANNELS))
    check("fig5: median Delta BI [%]", np.median(m), 0.30, 0.01)
    check("fig5: Wiener better [points]", np.sum(m < 0), 11, 0)
    check("fig5: better beyond two widths", np.sum(m < -2 * wd), 0, 0)
    check("fig5: worse beyond two widths", np.sum(m > 2 * wd), 28, 0)
    for ch, e in zip(st.CHANNELS, (0.24, 0.31, 0.29, 0.42, 0.06)):
        check(f"fig5: median ch{ch} [%]", np.median(res[ch][0]), e, 0.01)


# ── numbers without a figure ────────────────────────────────────────────────────────────────
def numbers_nopen(data, sW, sWwna):
    keys = sorted(sW)
    low = [k for k in keys if sW[k]["lam"] < LAM_LOW]
    rest = [k for k in keys if k not in low]
    lam = np.array([sW[k]["lam"] for k in keys])
    snr = {k: float(rows(FOLD["OF"])[k]["SNR"]) for k in keys}
    d = {k: dbi(data["W"][k], data["OF"][k]) for k in keys}
    an_, m = np.array([d[k][0] for k in keys]), np.array([d[k][1] for k in keys])
    ratio = {k: data["W"][k]["BI_mc"] / data["W"][k]["BI"] for k in keys}
    kmax = max(ratio, key=ratio.get)
    med = lambda ks, f: np.median([f(k) for k in ks])
    print(f"\n[fig 2] Wiener without penalty, {len(keys)} points")
    print(f"  lambda < 0.1 in {len(low)}: " + ", ".join(f"ch{c} WP{w}" for c, w in low))
    print(f"  lambda min {lam.min():.2g} at ch{keys[lam.argmin()][0]} WP{keys[lam.argmin()][1]}; "
          f"lambda < 1 in {np.sum(lam < 1)}; median {np.median(lam):.2f}")
    print(f"  BI_MC of OF: median {med(low, lambda k: data['OF'][k]['BI_mc']):.2e} (lambda < 0.1) vs "
          f"{med(rest, lambda k: data['OF'][k]['BI_mc']):.2e}; SNR_OF median {med(low, snr.get):.1f} vs "
          f"{med(rest, snr.get):.1f}")
    l31 = [sW[k]["lam"] for k in keys if k[0] == 31]
    print(f"  ch31 lambda {min(l31):.2f} - {max(l31):.2f}")
    s1l = sorted(sW[k]["s1"] for k in low)
    print(f"  s1 where lambda < 0.1: {s1l[0]:.3f} ... {s1l[-2]:.3f} and {s1l[-1]:.1f} "
          f"(ch{max(low, key=lambda k: sW[k]['s1'])}); others median {med(rest, lambda k: sW[k]['s1']):.3f}")
    print(f"  1 - rho median {med(low, lambda k: 1 - sW[k]['rho']):.1e} vs {med(rest, lambda k: 1 - sW[k]['rho']):.3f}; "
          f"sigma_Y0 median {med(low, lambda k: sW[k]['sY0']):.4f} vs {med(rest, lambda k: sW[k]['sY0']):.4f}")
    print("  ch91 WP13-21: lambda " + ", ".join(f"{sW[(91, w)]['lam']:.2g}" for w in range(13, 22, 2))
          + "; s1 " + ", ".join(f"{sW[(91, w)]['s1']:.3f}" for w in range(13, 22, 2)))
    print(f"  Delta BI analytic: better in {np.sum(an_ < 0)}, min {an_.min():+.1f}% "
          f"(ch{keys[an_.argmin()][0]} WP{keys[an_.argmin()][1]}); MC: worse in {np.sum(m > 0)}, "
          f"median {np.median(m):+.2f}%, ch91 median {np.median([d[k][1] for k in keys if k[0] == 91]):+.1f}%, "
          f"max {m.max():+.1f}%")
    print(f"  ch34 WP3: analytic {d[(34, 3)][0]:+.1f}%, MC {d[(34, 3)][1]:+.1f}%; BI_MC/BI_an max "
          f"{ratio[kmax]:.2f} at ch{kmax[0]} WP{kmax[1]}")
    nW = np.array([[sW[k]["norm1"], sW[k]["norm2"]] for k in keys])
    nP = np.array([[sWwna[k]["norm1"], sWwna[k]["norm2"]] for k in keys])
    print(f"  mean|f W S| (should be 1): no penalty {nW.min():.4f}-{nW.max():.4f}, "
          f"penalty {nP.min():.4f}-{nP.max():.4f}")
    ds1 = np.array([sW[k]["s1"] / sW[k]["s1_csv"] - 1 for k in keys])
    print(f"  s1 recomputed vs CSV of the training: max |diff| {100*np.abs(ds1).max():.2f}%")
    check("nopen: lambda < 0.1 [points]", len(low), 15, 0)
    check("nopen: lambda min", lam.min(), 1.0e-7, 0.1e-7)
    check("nopen: lambda < 1 [points]", np.sum(lam < 1), 33, 0)
    check("nopen: lambda median", np.median(lam), 1.3, 0.05)
    check("nopen: BI_MC OF median, lambda < 0.1 [1e-4]", med(low, lambda k: data['OF'][k]['BI_mc']) / 1e-4, 1.5, 0.05)
    check("nopen: BI_MC OF median, others [1e-5]", med(rest, lambda k: data['OF'][k]['BI_mc']) / 1e-5, 7.1, 0.05)  # request: 7.2
    check("nopen: SNR_OF median, lambda < 0.1", med(low, snr.get), 36.5, 0.5)
    check("nopen: SNR_OF median, others", med(rest, snr.get), 37.5, 0.5)
    check("nopen: ch31 lambda min", min(l31), 1.0, 0.05)
    check("nopen: ch31 lambda max", max(l31), 6.3, 0.05)
    check("nopen: s1 min where lambda < 0.1", s1l[0], 0.10, 0.01)
    check("nopen: s1 second largest where lambda < 0.1", s1l[-2], 2.3, 0.1)
    check("nopen: s1 largest where lambda < 0.1", s1l[-1], 63, 2)
    check("nopen: s1 median, others", med(rest, lambda k: sW[k]["s1"]), 0.06, 0.005)
    check("nopen: 1 - rho median, lambda < 0.1 [1e-4]", med(low, lambda k: 1 - sW[k]["rho"]) / 1e-4, 6, 1)
    check("nopen: 1 - rho median, others", med(rest, lambda k: 1 - sW[k]["rho"]), 0.05, 0.005)
    check("nopen: sigma_Y0 median, lambda < 0.1", med(low, lambda k: sW[k]["sY0"]), 0.022, 0.001)
    check("nopen: sigma_Y0 median, others", med(rest, lambda k: sW[k]["sY0"]), 0.021, 0.001)
    check("nopen: Delta BI an, better [points]", np.sum(an_ < 0), 25, 0)
    check("nopen: Delta BI an, min [%]", an_.min(), -25, 1)
    check("nopen: Delta BI MC, worse [points]", np.sum(m > 0), 75, 0)
    check("nopen: Delta BI MC median [%]", np.median(m), 1.05, 0.01)
    check("nopen: Delta BI MC ch91 median [%]", np.median([d[k][1] for k in keys if k[0] == 91]), 13, 0.5)
    check("nopen: Delta BI MC max [%]", m.max(), 41, 1)
    check("nopen: ch34 WP3 analytic [%]", d[(34, 3)][0], -9.0, 0.1)
    check("nopen: ch34 WP3 MC [%]", d[(34, 3)][1], 36, 1)
    check("nopen: BI_MC/BI_an max", ratio[kmax], 1.64, 0.01)


def numbers_penalty(data, sWwna):
    keys = sorted(sWwna)
    rW = [data["Wwna"][k]["BI_mc"] / data["Wwna"][k]["BI"] for k in keys]
    rO = [data["OF"][k]["BI_mc"] / data["OF"][k]["BI"] for k in keys]
    s1 = [sWwna[k]["s1"] for k in keys]
    print(f"\n[penalty] BI_MC/BI_an Wwna {min(rW):.3f}-{max(rW):.3f}, OF {min(rO):.3f}-{max(rO):.3f}; "
          f"s1 max {max(s1):.3f}; ch34 WP3 lambda {sWwna[(34, 3)]['lam']:.3g}")
    check("penalty: MC/an min", min(rW), 1.003, 0.001)
    check("penalty: MC/an max", max(rW), 1.12, 0.005)
    check("OF: MC/an min", min(rO), 1.02, 0.005)
    check("OF: MC/an max", max(rO), 1.13, 0.005)
    assert max(s1) < 0.1

    # two identical trainings: _swna1 (50 seeds, take seed 1234) vs _swna1_hist (single seed)
    def single(path):
        return {(int(r["channel"]), int(r["wp"])): r for r in csv.DictReader(open(path))
                if r.get("gen") == "fit" and int(r.get("seed") or 1234) == 1234}
    a = single(os.path.join(st.BASE, FOLD["Wwna"], "BI_mc_error_m205_seeds50.csv"))
    b = single(os.path.join(HIST_DIR, "BI_mc_error_m205.csv"))
    ks = sorted(a.keys() & b.keys())
    dm = np.array([100 * (float(b[k]["BI_mc"]) / float(a[k]["BI_mc"]) - 1) for k in ks])
    da = np.array([100 * (float(b[k]["BI_analytic"]) / float(a[k]["BI_analytic"]) - 1) for k in ks])
    nsim = {(a[k]["nsim"], b[k]["nsim"]) for k in ks}
    km, ka = ks[np.abs(dm).argmax()], ks[np.abs(da).argmax()]
    print(f"  two identical Wwna trainings ({len(ks)} points, seed 1234, nsim {nsim}): "
          f"MC median {np.median(dm):+.2f}%, max |.| {np.abs(dm).max():.2f}% (ch{km[0]} WP{km[1]}); "
          f"analytic median {np.median(da):+.2f}%, max |.| {np.abs(da).max():.2f}% (ch{ka[0]} WP{ka[1]})")


def shift_check():
    """Where the model breaks at the example point: the shift of Y produced by the pile-up of fig. 3,
    noiseless (what the model uses for mu_Y) vs median over simulated events, in units of the
    measured width of Y on singles."""
    ch, wp = KERNEL_PT
    r, dt = EVENT["r"], EVENT["dt"]
    print(f"\n[shift] ch{ch} WP{wp}, pile-up r = {r}, dt = {dt*1e3} ms, 4000 events per population")
    for k in ("OF", "W", "Wwna"):
        K, f1, f2, _, w, nps, A = point(FOLD[k], ch, wp, rows(FOLD[k])[(ch, wp)])
        S = np.fft.fft(mc.template_pulse(ch, wp, "fit") * np.hanning(len(nps)))
        clean = np.array([np.fft.ifft(A * S * ((1 - x) + x * np.exp(-1j * w * d))).real
                          for x, d in ((0, 0), (r, dt))])
        y0 = psd(clean, K, f1, f2)[0]
        g = np.random.default_rng(7)
        ys = psd(make_events(S, w, nps, A, 4000, 0, 0, g), K, f1, f2)[0]
        yp = psd(make_events(S, w, nps, A, 4000, r, dt, g), K, f1, f2)[0]
        wd = (np.percentile(ys, 84.13) - np.percentile(ys, 15.87)) / 2
        sh = np.median(ys) - np.median(yp)
        print(f"  {k:5s} shift noiseless {y0[0] - y0[1]:.4f} ({(y0[0] - y0[1]) / wd:.1f} widths), "
              f"simulated {sh:.4f} ({sh / wd:.1f} widths); width of Y on singles {wd:.4f}, "
              f"median Y of singles {np.median(ys):.4f}")


def cancellation_check(sW, sWwna, resim):
    """sigma_Y,0 analytic vs the width (P84 - P16)/2 of Y measured on simulated singles."""
    cache = os.path.join(OUTDIR, "wiener_sigmaY_check.csv")
    if resim or not os.path.exists(cache):
        out = []
        for name, sx in (("W", sW), ("Wwna", sWwna)):
            rr = rows(FOLD[name])
            for k in sorted(sx):
                K, f1, f2, _, w, nps, A = point(FOLD[name], *k, rr[k])
                S = np.fft.fft(mc.template_pulse(*k, "fit") * np.hanning(len(nps)))
                g = np.random.default_rng(1)
                y = np.concatenate([psd(make_events(S, w, nps, A, 500, 0, 0, g), K, f1, f2)[0]
                                    for _ in range(N_CHECK // 500)])
                p16, p50, p84 = np.percentile(y, [15.87, 50, 84.13])
                out.append(dict(algo=name, channel=k[0], wp=k[1], lam=sx[k]["lam"],
                                sY0_an=sx[k]["sY0"], sY0_mc=(p84 - p16) / 2 / p50))
                print(f"  {name} ch{k[0]} WP{k[1]}: {out[-1]['sY0_an']:.4f} vs {out[-1]['sY0_mc']:.4f}",
                      flush=True)
        with open(cache, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(out[0]))
            wr.writeheader(); wr.writerows(out)
    rr = list(csv.DictReader(open(cache)))
    print(f"\n[check] sigma_Y,0 analytic / measured on {N_CHECK} simulated singles (width / median):")
    for name in ("W", "Wwna"):
        for lab, sel in (("lambda < 0.1", lambda l: l < LAM_LOW), ("lambda >= 0.1", lambda l: l >= LAM_LOW)):
            q = np.array([float(r["sY0_an"]) / float(r["sY0_mc"]) for r in rr
                          if r["algo"] == name and sel(float(r["lam"]))])
            if len(q):
                print(f"  {name:5s} {lab:14s} {len(q):2d} points: ratio median {np.median(q):.3f}, "
                      f"range {q.min():.3g}-{q.max():.3g}")


if __name__ == "__main__":
    resim = "--resim" in sys.argv
    os.makedirs(OUTDIR, exist_ok=True)
    data = load_mc()
    sW, sWwna = filter_stats(FOLD["W"]), filter_stats(FOLD["Wwna"])
    fig_kernel(float(rows(FOLD["OF"])[KERNEL_PT]["SNR"]))
    fig_nopen(data, sW)
    numbers_nopen(data, sW, sWwna)
    fig_events(resim)
    fig_training()
    fig_vs_of(data)
    numbers_penalty(data, sWwna)
    shift_check()
    cancellation_check(sW, sWwna, resim)
    print("\n" + ("all numbers of the request confirmed" if not FAILED else
                  "NUMBERS THAT DO NOT MATCH THE REQUEST:\n  " + "\n  ".join(FAILED)))
    assert not FAILED
