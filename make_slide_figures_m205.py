#!/usr/bin/env python3
"""
make_slide_figures_m205.py
==========================
Figure per la parte PILE-UP della discussione di tesi, nell'ordine della scaletta del relatore.
Tutte con lo stesso stile da slide (font grandi, cornice pulita) e salvate in slides_figures/.
Testo delle figure in inglese.

  fig01_spectrum          7.1  spettro: 2vbb, pile-up (autoconvoluzione), picco 0vbb, ROI
  fig02_pileup_pulses     7.2  schema: impulso singolo e pile-up con le due componenti
  fig03_ingredients       8.1  gli ingredienti: average pulse + NPS, un solo canale
  fig04_model             8.3  SOLO la formula del modello, con S e NPS in miniatura
  fig05_discriminator     8.5  distribuzioni di Y, taglio al 90%, i due eventi di fig09
  fig06_BI_vs_V           9.3  BI in funzione del working point, 5 rivelatori (filtro ottimo)
  fig07_overtraining     10    Wiener non regolarizzato: atteso (linea) contro validazione (punti)
  fig08_regularization   11.2  stesso confronto dopo la regolarizzazione
  fig09_mc_event               due eventi del Monte Carlo: singolo e pile-up, stesso rumore
  fig10_filtered_event         gli stessi due eventi dopo i due filtri: A1, A2, Y
  fig11_ingredients_mini       la striscia degli ingredienti, tutta azzurra, trasparente
  fig12_pulses_alone           i due impulsi rumorosi da soli, PNG trasparenti
  fig13_filtered_alone         i quattro filtrati da soli, con l'ampiezza, PNG trasparenti

Uso:  KMP_DUPLICATE_LIB_OK=TRUE python3 make_slide_figures_m205.py [nome_figura ...]
"""

import os, csv, glob, sys
import numpy as np
import uproot

trapz = getattr(np, "trapezoid", None) or np.trapz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
OUT = os.path.join(BASE, "slides_figures")

CH_REF, WP_REF = 34, 15          # canale/WP usato da tutte le figure di esempio
CHANS = [31, 34, 71, 83, 91]
VBIAS = np.array([0.6, 1.0, 1.4, 1.8, 2, 3, 4, 5, 6, 8, 10, 20, 26, 30, 40])
WPS = list(range(1, 30, 2))
FS, WIN = 10_000, 10_000

OF_SET = "m205_results_octopus_npsclean"                    # riferimento, gen=root
# Le figure 4, 5, 9, 10 sono illustrative: servono i filtri addestrati, e in OF_SET manca ch34.
# Si usa la campagna OF che li ha per tutti i canali (template APsimfit10000led).
FILT_SET = "m205_results_octopus_APsimfit10000led_npsclean"
WF_RAW = "m205_results_wiener_root_npsclean"                # Wiener senza penalita'
WF_REG = "m205_results_wiener_root_npsclean_swna1"          # Wiener con penalita'
GEN = "root"

# Target CUPID sul pile-up: meta' del budget totale di 1e-4 counts/(keV kg yr).
# None per non disegnarlo.
BI_TARGET = 5e-5

C_SINGLE = "#04254F"   # blu scuro: l'impulso singolo in tutte le figure
C_LIGHT = "tab:blue"   # azzurro: gli ingredienti (AP, NPS) della slide del modello

RC = {"font.size": 17, "axes.titlesize": 21, "axes.labelsize": 18,
      "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15,
      "axes.linewidth": 1.3, "lines.linewidth": 2.6, "lines.markersize": 9,
      "axes.spines.top": False, "axes.spines.right": False,
      "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.5, "figure.dpi": 150}


def save(fig, name, transparent=False):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name + ".png")
    fig.savefig(p, dpi=150, bbox_inches="tight", transparent=transparent)
    plt.close(fig)
    print(f"   -> {os.path.relpath(p, BASE)}")
    return p


def root_file(ch):
    return glob.glob(os.path.join(BASE, "Processed", f"Processed_*_000205_{ch}.root"))[0]


def ap(ch, wp):
    with uproot.open(root_file(ch)) as f:
        v = np.asarray(f[f"averagepulse_ap_wp{wp}_medianAP"].values(), float)
    return v / v.max()


def nps(ch, wp):
    return np.load(os.path.join(BASE, "m205_NPS_clean", f"ch{ch}", f"nps_ch{ch}_wp{wp}.npy"))


def fit_tpl(ch, wp):
    v = np.load(os.path.join(BASE, "residual_scan_bessel", "fits_octopus",
                             f"bestfit_ch{ch}_wp{wp}.npy"))
    return v / v.max()


def load_csv(d, gen=GEN):
    """{(ch,wp): {...}} con BI analitico e BI Monte Carlo della campagna."""
    an = {(int(r["channel"]), int(r["wp"])): r
          for r in csv.DictReader(open(glob.glob(os.path.join(BASE, d, "BI_results_*.csv"))[0]))}
    out = {}
    for r in csv.DictReader(open(os.path.join(BASE, d, "BI_mc_error_m205.csv"))):
        if (r.get("gen") or "root") != gen:
            continue
        k = (int(r["channel"]), int(r["wp"]))
        out[k] = dict(vbias=float(r["vbias"]), BI_mc=float(r["BI_mc"]),
                      sigma=float(r["sigma_BI"]), BI_an=float(an[k]["BI"]))
    return out


# ═════════════════════════════════════════════════════════════════════════════
def fig01_spectrum():
    """Spettro del 2vbb, del pile-up di due 2vbb e picco 0vbb, con la ROI. Scala lineare."""
    from utility.double_beta_spectrum import g, L
    E = np.linspace(1e-3, L, 4000)
    s = np.asarray(g(E), float); s = np.clip(s, 0, None); s /= trapz(s, E)
    # pile-up = autoconvoluzione dello spettro di somma (due decadimenti non risolti)
    pu = np.convolve(s, s) * (E[1] - E[0])
    Epu = np.linspace(2 * E[0], 2 * L, len(pu))
    s, pu = s / s.max(), pu / pu.max()          # scala lineare: ognuno al proprio massimo
    roi = (L * 0.985, L * 1.015)
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(11.5, 6.8))
        ax.fill_between(E, 0, s, color=C_SINGLE, alpha=0.18, lw=0)
        ax.fill_between(Epu, 0, pu, color="tab:orange", alpha=0.22, lw=0)
        ax.plot(E, s, color=C_SINGLE, lw=3.0, label="2νββ  (one decay)")
        ax.plot(Epu, pu, color="tab:orange", lw=3.0, label="pile-up of two 2νββ")
        ax.axvline(L, color="tab:red", lw=3,
                   label=f"0νββ  ($Q_{{\\beta\\beta}}$ = {L:.3f} MeV)")
        ax.axvspan(*roi, color="tab:red", alpha=0.15, lw=0, label="ROI")
        ax.set_xlim(0, 2 * L * 0.80)
        ax.set_ylim(0, 1.15)
        ax.set_xlabel("deposited energy  [MeV]")
        ax.set_ylabel("intensity  [a.u.]")
        ax.set_title("Two pile-up 2νββ fall in the 0νββ ROI")
        ax.legend(frameon=False, loc="upper right")
        y_roi = float(np.interp(L, Epu, pu))
        ax.annotate("the pile-up leaks\ninto the ROI", xy=(L, y_roi),
                    xytext=(L * 1.13, y_roi + 0.30), color="tab:orange", fontsize=17,
                    fontweight="bold", ha="left",
                    arrowprops=dict(arrowstyle="->", color="tab:orange", lw=2.4))
        ax.text(0.015, -0.155, "each curve normalized to its own maximum",
                transform=ax.transAxes, fontsize=14, color="0.45")
    return save(fig, "fig01_spectrum")


def fig02_pileup_pulses():
    """Un impulso singolo e un pile-up con lo stesso rilascio totale di energia."""
    t = (np.arange(WIN) - WIN // 2) / FS * 1e3          # ms, zero sul trigger
    tpl = fit_tpl(CH_REF, WP_REF)
    tpl = np.roll(tpl, -int(np.argmax(tpl)) + WIN // 2)
    dt_ms, r = 0.45, 0.45
    shift = int(dt_ms * 1e-3 * FS)
    a, b = (1 - r) * tpl, r * np.roll(tpl, shift)
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(11, 6.6))
        m = (t > -1) & (t < 12)
        ax.plot(t[m], tpl[m], color=C_SINGLE, label="single pulse")
        ax.plot(t[m], (a + b)[m], color="tab:orange", label=f"pile-up, Δt = {dt_ms:.2f} ms")
        ax.plot(t[m], a[m], color="0.6", lw=1.6, ls="--")
        ax.plot(t[m], b[m], color="0.6", lw=1.6, ls="--", label="the two components")
        ax.set_xlabel("time  [ms]")
        ax.set_ylabel("normalized amplitude")
        ax.set_title("Same total energy, different shape")
        ax.legend(frameon=False)
    return save(fig, "fig02_pileup_pulses")


def fig03_ingredients():
    """I due ingredienti dell'algoritmo, per un solo canale: average pulse e NPS."""
    with plt.rc_context(RC):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.2))
        t = (np.arange(WIN) - WIN // 2) / FS * 1e3
        cmap = plt.get_cmap("viridis")
        for i, wp in enumerate(WPS):
            v = ap(CH_REF, wp)
            m = (t > -2) & (t < 30)
            a1.plot(t[m], v[m], color=cmap(i / (len(WPS) - 1)), lw=1.8)
        a1.set_xlabel("time  [ms]"); a1.set_ylabel("normalized amplitude")
        a1.set_title(f"Average pulse  $s(t)\\;\\rightarrow\\;S(f)$ — Ch {CH_REF}")
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(VBIAS.min(), VBIAS.max()))
        cb = fig.colorbar(sm, ax=a1); cb.set_label("bias  [V]")
        fr = np.fft.rfftfreq(WIN, 1 / FS)
        for i, wp in enumerate(WPS):
            p = nps(CH_REF, wp)[:len(fr)]
            a2.loglog(fr[1:], p[1:], color=cmap(i / (len(WPS) - 1)), lw=1.4)
        a2.set_xlabel("frequency  [Hz]"); a2.set_ylabel("$\\mathrm{NPS}(f)$  [a.u.]")
        a2.set_title(f"Noise power spectrum  $\\mathrm{{NPS}}(f)$ — Ch {CH_REF}")
        fig.tight_layout()
    return save(fig, "fig03_ingredients")


def _filters():
    """Ingredienti (S, NPS, ampiezza) e filtri addestrati del punto di riferimento."""
    import src.analysis as an
    d = os.path.join(BASE, FILT_SET, "trained_filters")
    def full(h):
        h = np.asarray(h).ravel(); m = h[-2:0:-1]
        return np.concatenate([h, np.conj(m) if np.iscomplexobj(h) else m])
    row = {(int(r["channel"]), int(r["wp"])): r for r in csv.DictReader(
        open(glob.glob(os.path.join(BASE, FILT_SET, "BI_results_*.csv"))[0]))}[(CH_REF, WP_REF)]
    p_nps = nps(CH_REF, WP_REF)
    S, w, _ = an.compute_H(fit_tpl(CH_REF, WP_REF), p_nps, np.hanning, sampling_rate=FS)
    H, f1, f2 = (full(np.load(os.path.join(d, f"{k}_ch{CH_REF}_wp{WP_REF}.npy")))
                 for k in ("kernel", "f1", "f2"))
    return dict(S=S, w=w, nps=p_nps, H=H, f1=f1, f2=f2, A=float(row["signal_amp"]), row=row)


def fig04_model():
    """La formula del modello, con S e NPS richiamati dalla slide degli ingredienti."""
    t = (np.arange(WIN) - WIN // 2) / FS * 1e3
    v = ap(CH_REF, WP_REF)
    mask = (t > -2) & (t < 30)
    fr = np.fft.rfftfreq(WIN, 1 / FS)
    p = nps(CH_REF, WP_REF)[:len(fr)]
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(13, 8.0))
        # i due ingredienti in miniatura, come nella slide precedente
        for rect, (x, y, col, ttl, logy) in zip(
                ([0.10, 0.72, 0.30, 0.22], [0.52, 0.72, 0.30, 0.22]),
                [(t[mask], v[mask], C_LIGHT, r"$s(t)\;\rightarrow\;S(f)$", False),
                 (fr[1:], p[1:], C_LIGHT, r"$\mathrm{NPS}(f)$", True)]):
            a = fig.add_axes(rect)
            (a.loglog if logy else a.plot)(x, y, color=col, lw=2.0)
            a.set_title(ttl, fontsize=20, color=col, pad=6)
            a.set_xticklabels([]); a.set_yticklabels([])
            a.tick_params(length=0); a.grid(ls=":", alpha=0.5)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
        fig.text(0.46, 0.655, "both measured on the detector — previous slide",
                 ha="center", va="center", fontsize=17, color="0.35")
        fig.text(0.5, 0.46, r"$H_\lambda(f)=\dfrac{S^*(f)}{|S(f)|^2+\lambda\,\mathrm{NPS}(f)}$",
                 ha="center", va="center", fontsize=36)
        fig.text(0.5, 0.24, r"$A_i=\max_t\;\mathcal{F}^{-1}\!\left[\,m_i(f)\,"
                            r"H_\lambda(f)\,X(f)\,\right]$",
                 ha="center", va="center", fontsize=29)
        fig.text(0.5, 0.155, "the two weights $m_1,m_2$ are trained", ha="center", va="center",
                 fontsize=17, color="0.35")
        fig.text(0.5, 0.075, r"$Y=A_1/A_2$", ha="center", va="center", fontsize=31,
                 bbox=dict(fc="#eef3fb", ec="tab:blue", pad=10))
        fig.text(0.97, 0.075, "a ratio:\nindependent of the energy", ha="right", va="center",
                 fontsize=16, color="0.35")
    return save(fig, "fig04_model")


def _filtered(g, trace):
    """Le due tracce filtrate con m_i*H, nella stessa convenzione di get_PSD_interpole."""
    n = len(g["S"])
    tgt = n // 2
    x = np.asarray(trace, float)
    x = x - np.mean(x[:tgt - 100])
    fx = np.fft.fft(x)
    return [np.roll(np.fft.ifft(f * g["H"] * fx).real, tgt) for f in (g["f1"], g["f2"])]


def fig10_filtered_event():
    """Gli stessi due eventi dopo i due filtri addestrati: le ampiezze A1, A2 e Y."""
    import src.analysis as an, src.dataset as ds
    g = _filters()
    ev1, ev2 = _mc_pair(g)
    dd = ds.NumpyDataset(np.asarray([ev1["y"], ev2["y"]], np.float32))
    dd.win_length = len(g["S"])
    Yv, A1, A2 = an.get_PSD_interpole(dd, g["H"], g["f1"], g["f2"])
    F = [_filtered(g, e["y"]) for e in (ev1, ev2)]
    n = len(g["S"])
    t = (np.arange(n) - n // 2) / FS * 1e3
    m = (t > -6) & (t < 6)
    with plt.rc_context(RC):
        fig, axs = plt.subplots(1, 2, figsize=(16, 6.8))
        for i, (ax, ttl, sub) in enumerate(zip(axs, [
                r"after the first filter   $m_1 H_\lambda$",
                r"after the second filter   $m_2 H_\lambda$"],
                ["enhances the high frequencies", "weighs the overall shape"])):
            for j, (col, lab, A) in enumerate(((C_SINGLE, "single pulse", A1 if i == 0 else A2),
                                               ("tab:orange", "pile-up", A1 if i == 0 else A2))):
                y = F[j][i] * 1e3
                ax.plot(t[m], y[m], color=col, lw=2.2, label=lab)
                amp = A[j] * 1e3
                k = int(np.argmax(F[j][i][n // 2 - 20:n // 2 + 20])) + n // 2 - 20
                ax.plot(t[k], amp, "o", color=col, ms=11, mec="white", mew=1.6)
                ax.annotate(f"$A_{i+1}$ = {amp:.3f} mV", xy=(t[k], amp),
                            xytext=(2.0, amp * (1.02 if j == 0 else 0.80)),
                            color=col, fontsize=16, fontweight="bold", va="center",
                            arrowprops=dict(arrowstyle="-", color=col, lw=1.4, alpha=0.6))
            ax.set_xlabel("time  [ms]")
            ax.set_title(f"{ttl}\n{sub}", fontsize=19, pad=10)
            ax.set_facecolor("#fbfbfd")
        axs[0].set_ylabel("filtered amplitude  [mV]")
        axs[0].legend(frameon=False, loc="upper left")
        fig.suptitle(f"$Y=A_1/A_2$ :   single pulse  {Yv[0]:.4f}       "
                     f"pile-up  {Yv[1]:.4f}", fontsize=22)
        fig.tight_layout()
    return save(fig, "fig10_filtered_event")


def fig05_discriminator():
    """Distribuzioni del discriminante Y per singoli e pile-up, col taglio al 90%."""
    import src.analysis as an, src.dataset as ds, src.simulation as sim
    g = _filters()
    S, w, p_nps, A = g["S"], g["w"], g["nps"], g["A"]
    H, f1, f2 = g["H"], g["f1"], g["f2"]

    def Y(traces):
        """Il discriminante Y = A1/A2 su tracce nel tempo."""
        dd = ds.NumpyDataset(np.asarray(traces, np.float32))
        dd.win_length = len(S)
        return an.get_PSD_interpole(dd, H, f1, f2)[0]

    def psd(dt_max, n=20000, chunk=2000):
        acc = []
        for k in range(0, n, chunk):
            fp, *_ = sim.simulate_frequency_pulses(S, p_nps, 0.0, w, nsim=chunk,
                                                   seed=1234 + k // chunk,
                                                   signal_scale=A, dt_max=dt_max)
            acc.append(Y(np.fft.ifft(fp, axis=1).real))
        return np.concatenate(acc)

    single, pile = psd(0.0), psd(8e-4)
    ev1, ev2 = _mc_pair(g)                      # i due eventi della figura precedente
    y_ev = Y([ev1["y"], ev2["y"]])
    cut = np.percentile(single, 10)
    rej = float(np.mean(pile < cut))
    lo, hi = np.percentile(np.concatenate([single, pile]), [0.2, 99.8])
    bins = np.linspace(lo, hi, 110)
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(11.5, 6.8))
        ax.hist(single, bins=bins, histtype="step", lw=2.6, color=C_SINGLE,
                label="single pulses")
        ax.hist(pile, bins=bins, histtype="step", lw=2.6, color="tab:orange",
                label="pile-up")
        ax.axvline(cut, color="0.2", ls="--", lw=2.4,
                   label="cut at 90% acceptance")
        top = ax.get_ylim()[1]
        dx = (hi - lo) * 0.09          # il singolo cade sul picco: etichetta spostata a destra
        for yv, col, txt, xt, yt, verdict in (
                (y_ev[0], C_SINGLE, "the single pulse\nfrom before", dx, 1.02,
                 "above threshold: accepted"),
                (y_ev[1], "tab:orange", "the pile-up\nfrom before", 0.0, 0.78,
                 "below threshold: rejected")):
            ax.annotate(f"{txt}\nY = {yv:.4f}\n{verdict}", xy=(yv, 0), xytext=(yv + xt, top * yt),
                        ha="center", fontsize=15, color=col, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=col, lw=2.4))
        ax.set_ylim(0, top * 1.30)
        ax.set_xlabel("discriminator  $Y=A_1/A_2$"); ax.set_ylabel("events")
        ax.set_title(f"Ch {CH_REF}, {VBIAS[WP_REF // 2]:g} V — rejected pile-up: {rej:.0%}")
        ax.legend(frameon=False, loc="upper left")
    return save(fig, "fig05_discriminator")


MC_DT, MC_R = 6e-4, 0.40      # i due eventi di esempio: 0.6 ms, 40% dell'energia sul secondo


def _mc_pair(g=None, dt=None, r=None):
    """I due eventi Monte Carlo delle figure: un impulso singolo e un pile-up, stesso rumore."""
    import src.simulation as sim
    g = g or _filters()

    def event(r, dt):
        # il jitter di trigger del simulatore usa np.random: lo si semina per poterlo
        # riprodurre e separare le due componenti esattamente come le ha generate.
        np.random.seed(3)
        fp, sf, nf = sim.simulate_frequency_pulses_fixed_dt_r(
            g["S"], g["nps"], g["w"], dt, r, nsim=1, seed=7, signal_scale=g["A"])
        np.random.seed(3)
        jit = np.exp(-1j * g["w"] * np.random.random(1)[0] * 1e-4)
        ifft = lambda x: np.fft.ifft(np.atleast_2d(x), axis=1).real[0]
        e = dict(y=ifft(fp), clean=ifft(fp - nf), p1=ifft(sf * (1 - r) * jit),
                 p2=ifft(sf * r * np.exp(-1j * g["w"] * dt) * jit))
        assert np.allclose(e["p1"] + e["p2"], e["clean"], atol=1e-12), \
            "le due componenti non ricostruiscono l'evento"
        return e

    return event(0.0, 0.0), event(MC_R if r is None else r, MC_DT if dt is None else dt)


def fig09_mc_event():
    """Due eventi del Monte Carlo: un impulso singolo e un pile-up, stessa energia e rumore."""
    g = _filters()
    ev1, ev2 = _mc_pair(g)
    mV = lambda x: x * 1e3
    y1, c1 = mV(ev1["y"]), mV(ev1["clean"])
    y2, c2, p1, p2 = mV(ev2["y"]), mV(ev2["clean"]), mV(ev2["p1"]), mV(ev2["p2"])
    t = (np.arange(WIN) - int(np.argmax(c1))) / FS * 1e3          # ms, zero sul picco singolo
    m = (t > -3) & (t < 25)
    zoom = (t > -1.5) & (t < 4)

    with plt.rc_context(RC):
        fig, axs = plt.subplots(1, 2, figsize=(16, 6.8), sharey=True)
        for ax, (y, c, col, ttl) in zip(axs, [
                (y1, c1, C_SINGLE, "single pulse"),
                (y2, c2, "tab:orange",
                 f"pile-up  —  Δt = {MC_DT*1e3:.1f} ms,  {MC_R:.0%} / {1-MC_R:.0%}")]):
            ax.plot(t[m], y[m], color=col, lw=1.5, alpha=0.75, label="simulated event")
            ax.plot(t[m], c[m], color="0.15", lw=2.4, label="noiseless signal")
            ax.set_xlabel("time  [ms]")
            ax.set_title(ttl, color=col, fontweight="bold", pad=12)
            ax.set_facecolor("#fbfbfd")
            ins = ax.inset_axes([0.42, 0.40, 0.56, 0.52])
            ins.plot(t[zoom], y[zoom], color=col, lw=1.4, alpha=0.75)
            if ax is axs[1]:
                ax.plot([], [], color="0.55", lw=1.8, ls="--",
                        label="single-pulse shape")     # solo per la legenda
                ins.plot(t[zoom], c1[zoom], color="0.55", lw=1.8, ls="--")
            ins.plot(t[zoom], c[zoom], color="0.15", lw=2.2)
            ins.set_title("zoom on the rising edge", fontsize=13, pad=4)
            ins.tick_params(labelsize=11); ins.grid(ls=":", alpha=0.5)
            for s in ("top", "right"):
                ins.spines[s].set_visible(False)
        for a, lw, k in ((axs[1], 2.0, m), (axs[1].child_axes[0], 1.6, zoom)):
            a.plot(t[k], p1[k], color="tab:green", lw=lw, ls="-.",
                   label=f"1st event  ({1-MC_R:.0%})")
            a.plot(t[k], p2[k], color="tab:purple", lw=lw, ls="-.",
                   label=f"2nd event  ({MC_R:.0%}),  +{MC_DT*1e3:.1f} ms")
        # legende sotto ai pannelli: nessuna sovrapposizione con le tracce
        LEG = dict(loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, ncol=2)
        axs[1].legend(fontsize=14, **LEG)
        axs[0].set_ylabel("amplitude  [mV]")
        axs[0].legend(fontsize=14, **LEG)
        fig.suptitle(f"Monte Carlo events — Ch {CH_REF}, {VBIAS[WP_REF // 2]:g} V  "
                     f"(SNR = {float(g['row']['SNR']):.0f})", fontsize=21)
        fig.tight_layout()
    return save(fig, "fig09_mc_event")


def fig06_BI_vs_V():
    """BI contro working point per i cinque rivelatori, con il filtro ottimo."""
    data = load_csv(OF_SET)
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(11, 6.8))
        for i, ch in enumerate(CHANS):
            v = sorted((r["vbias"], r) for (c, wp), r in data.items() if c == ch)
            x = [a for a, _ in v]
            y = [b["BI_mc"] for _, b in v]
            e = [b["sigma"] for _, b in v]
            ax.errorbar(x, y, yerr=e, marker="os^Dv"[i], ms=9, lw=2.4, color=f"C{i}",
                        mfc=f"C{i}", mec="white", mew=1.2, capsize=5, elinewidth=1.7,
                        label=f"Ch {ch}")
        ax.set_xscale("log")
        ax.set_xticks([0.6, 1, 2, 5, 10, 20, 40])
        ax.set_xticklabels(["0.6", "1", "2", "5", "10", "20", "40"])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize(RC["ytick.labelsize"])
        ax.set_xlabel("bias voltage  [V]")
        ax.set_ylabel("BI  [counts / keV / kg / yr]")
        if BI_TARGET:
            ax.axhline(BI_TARGET, color="tab:red", ls="--", lw=2.2, zorder=1,
                       label="CUPID pile-up target")
        ax.set_title("Background index vs working point — optimum filter")
        ax.legend(frameon=False, ncol=2)
    return save(fig, "fig06_BI_vs_V")


def _train_vs_valid(ax, data, ch, title):
    v = sorted((r["vbias"], r) for (c, wp), r in data.items() if c == ch)
    x = [a for a, _ in v]
    ax.plot(x, [b["BI_an"] for _, b in v], color="tab:blue", lw=2.8,
            label="expected from training  (analytic)")
    ax.errorbar(x, [b["BI_mc"] for _, b in v], yerr=[b["sigma"] for _, b in v],
                ls="none", marker="o", ms=9, color="tab:red", mfc="tab:red",
                mec="white", mew=1.2, capsize=5, elinewidth=1.7,
                label="validation  (simulation)")
    ax.set_xscale("log")
    ax.set_xticks([0.6, 1, 2, 5, 10, 20, 40])
    ax.set_xticklabels(["0.6", "1", "2", "5", "10", "20", "40"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax.yaxis.get_offset_text().set_fontsize(RC["ytick.labelsize"])
    ax.set_xlabel("bias voltage  [V]")
    ax.set_title(title)


def fig07_overtraining():
    """Wiener senza vincolo: l'atteso dal training e la validazione divergono. Ch di riferimento."""
    data = load_csv(WF_RAW)
    ch = CH_REF
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(11, 6.8))
        _train_vs_valid(ax, data, ch, f"Wiener without constraint — Ch {ch}")
        ax.set_ylabel("BI  [counts / keV / kg / yr]")
        ax.legend(frameon=False, loc="best")
        ax.text(0.03, 0.06, "overtraining", transform=ax.transAxes, fontsize=18,
                color="tab:red", va="bottom", fontweight="bold")
    return save(fig, "fig07_overtraining")


def fig08_regularization():
    """Lo stesso confronto prima e dopo la regolarizzazione, sul canale di riferimento."""
    raw, reg = load_csv(WF_RAW), load_csv(WF_REG)
    ch = CH_REF
    with plt.rc_context(RC):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6.6), sharey=True)
        _train_vs_valid(a1, raw, ch, "without constraint")
        _train_vs_valid(a2, reg, ch, "with regularization")
        a1.set_ylabel("BI  [counts / keV / kg / yr]")
        a1.legend(frameon=False, loc="best")
        fig.suptitle(f"Ch {ch} — regularization brings the training back to reality",
                     fontsize=21)
        fig.tight_layout()
    return save(fig, "fig08_regularization")


def fig11_ingredients_mini():
    """I due ingredienti in miniatura, tutti in azzurro: la striscia da mettere in alto."""
    t = (np.arange(WIN) - WIN // 2) / FS * 1e3
    v = ap(CH_REF, WP_REF)
    mask = (t > -2) & (t < 30)
    fr = np.fft.rfftfreq(WIN, 1 / FS)
    p = nps(CH_REF, WP_REF)[:len(fr)]
    with plt.rc_context(RC):
        fig, axs = plt.subplots(1, 2, figsize=(14, 3.4))
        for a, (x, y, ttl, logy) in zip(axs, [
                (t[mask], v[mask], r"$s(t)\;\rightarrow\;S(f)$", False),
                (fr[1:], p[1:], r"$\mathrm{NPS}(f)$", True)]):
            (a.loglog if logy else a.plot)(x, y, color=C_LIGHT, lw=2.4)
            a.set_title(ttl, fontsize=24, color=C_LIGHT, pad=8)
            a.set_xticklabels([]); a.set_yticklabels([])
            a.tick_params(length=0); a.grid(ls=":", alpha=0.5)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
        fig.tight_layout()
    return save(fig, "fig11_ingredients_mini", transparent=True)


def fig12_pulses_alone():
    """Solo le due tracce rumorose, sfondo trasparente: blu scuro il singolo, arancio il pile-up."""
    ev1, ev2 = _mc_pair()                       # gli stessi eventi di fig09: dt = 0.6 ms
    t = (np.arange(WIN) - int(np.argmax(ev1["clean"]))) / FS * 1e3
    m = (t > -3) & (t < 30)
    out = []
    for ev, col, name in ((ev1, C_SINGLE, "fig12_pulse_single"),
                          (ev2, "tab:orange", "fig12_pulse_pileup")):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(t[m], ev["y"][m] * 1e3, color=col, lw=1.8)
        ax.axis("off")
        ax.margins(x=0.01, y=0.06)
        out.append(save(fig, name, transparent=True))
    return out


def fig13_filtered_alone():
    """I quattro impulsi filtrati, uno per PNG, con la sua ampiezza. Sfondo trasparente."""
    import src.analysis as an, src.dataset as ds
    g = _filters()
    evs = _mc_pair(g)
    dd = ds.NumpyDataset(np.asarray([e["y"] for e in evs], np.float32))
    dd.win_length = len(g["S"])
    _, A1, A2 = an.get_PSD_interpole(dd, g["H"], g["f1"], g["f2"])
    n = len(g["S"])
    t = (np.arange(n) - n // 2) / FS * 1e3
    m = (t > -6) & (t < 6)
    out = []
    for j, (ev, col, who) in enumerate(((evs[0], C_SINGLE, "single"),
                                        (evs[1], "tab:orange", "pileup"))):
        F = _filtered(g, ev["y"])
        for i, A in enumerate((A1, A2)):
            amp = A[j] * 1e3
            y = F[i] * 1e3
            k = int(np.argmax(F[i][n // 2 - 20:n // 2 + 20])) + n // 2 - 20
            fig, ax = plt.subplots(figsize=(7.5, 4.6))
            ax.plot(t[m], y[m], color=col, lw=3.4, solid_capstyle="round")
            ax.plot(t[k], amp, "o", color=col, ms=13, mec="white", mew=2.0)
            ax.annotate(f"$A_{i+1}$ = {amp:.3f} mV", xy=(t[k], amp), xytext=(t[k] + 1.2, amp),
                        color=col, fontsize=22, fontweight="bold", va="center",
                        arrowprops=dict(arrowstyle="-", color=col, lw=1.6, alpha=0.6))
            ax.axis("off")
            ax.margins(x=0.16, y=0.10)
            out.append(save(fig, f"fig13_filt_{who}_m{i+1}", transparent=True))
    return out


ALL = [fig01_spectrum, fig02_pileup_pulses, fig03_ingredients, fig04_model,
       fig05_discriminator, fig06_BI_vs_V, fig07_overtraining, fig08_regularization,
       fig09_mc_event, fig10_filtered_event, fig11_ingredients_mini,
       fig12_pulses_alone, fig13_filtered_alone]

if __name__ == "__main__":
    want = sys.argv[1:]
    for f in ALL:
        if want and not any(w in f.__name__ for w in want):
            continue
        print(f.__name__)
        try:
            f()
        except Exception as e:
            print(f"   [ERROR] {type(e).__name__}: {e}")
