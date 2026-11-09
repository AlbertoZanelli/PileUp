"""Parametri del BDT di AMoRE sul calore (Y. Oh, poster Neutrino 2024; slide 5 di S. Kim, CUPID/AMoRE nov 2026)
calcolati sugli eventi del Monte Carlo CROSS (simulate_psd_v2, generazione v2, template "fit"): distribuzioni
di singoli e pile-up, come nella slide 5.

La slide da' solo i NOMI. Interpretazione usata qui (coerente con i valori della slide, dove
RT_raw10-f50 ~ 0.035 ms e RT_raw50-f50 = RT_raw05-50 - 0.06 ms):
  raw    impulso non filtrato, baseline = media del pretrigger; tXX = attraversamento in salita di XX% del massimo
  f0     filtro ottimo H (lo stesso dell'OF): Amp_f0 = massimo dell'uscita
  f1,f2  passa-banda Butterworth (ordine 2, fase zero) 5-800 Hz e 5-200 Hz
         Amp_fk/Amp_f0 diviso per lo stesso rapporto sul template senza rumore: i singoli stanno a 1
  f50    attraversamento al 50% in salita dell'uscita di f0 (tempo di riferimento)
  f1max  tempo del massimo dell'uscita di f1
Light1/Heat0 non esiste qui (un solo canale): al suo posto il rapporto dei massimi dell'OF della campagna.
Le linee tratteggiate tengono il 90% centrale dei singoli (5%-95%): R = pile-up fuori, parametro per parametro.
Nel titolo: BDT (HistGradientBoosting) sui 10 parametri, meta' eventi per addestrare e meta' per il test,
taglio al 90% dei singoli, contro l'OF sugli stessi eventi di test.

    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/bin/python3.13 test/amore/amore_features_m205.py 71 29 34 1 91 29
"""
import os, sys, csv
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import simulate_BI_error_m205 as s
from scipy.signal import butter, sosfiltfilt

NSIM, SEED, FS = 20_000, 1234, s.SAMPLING_RATE
OF_DIR = "m205_results_octopus_APsimfit10000ledv2tf_npsclean_dtrise"
OUT = os.path.join(ROOT, "comparisons", "amore_features")
BANDS = {"f1": 800, "f2": 200}
NAMES = ["Amp_f1/Amp_f0", "Amp_f2/Amp_f0", "RT_raw05-95 (ms)", "RT_raw05-10 (ms)", "RT_raw05-50 (ms)",
         "RT_raw10-f50 (ms)", "RT_raw50-f50 (ms)", "RT_raw90-f50 (ms)", "RT_raw95-f50 (ms)", "RT_f1max-f50 (ms)"]


def crossing(F, p, l):
    """Attraversamento in salita del livello l*max, interpolato: ultimo campione sotto soglia prima del picco p."""
    n, L = F.shape
    r = np.arange(n)
    h = F[r, p]
    below = (F < l * h[:, None]) & (np.arange(L)[None] < p[:, None])
    j = np.clip(L - 1 - np.argmax(below[:, ::-1], 1), 0, L - 2)
    x0, x1 = F[r, j], F[r, j + 1]
    return j + (l * h - x0) / np.where(x1 != x0, x1 - x0, 1)


def peak(F, mask=None):
    """(indice del massimo, tempo sub-campione con la parabola sui 3 campioni); mask = dove cercarlo."""
    p = np.clip((F if mask is None else np.where(mask, F, -np.inf)).argmax(1), 1, F.shape[1] - 2)
    r = np.arange(len(F))
    a, b, c = F[r, p - 1], F[r, p], F[r, p + 1]
    den = a - 2 * b + c
    return p, p + np.where(den < 0, 0.5 * (a - c) / np.where(den < 0, den, 1), 0)


class Features:
    def __init__(self, H, ipk, n_after, n_search):
        self.H, self.ipk, self.n_search = H, ipk, n_search
        self.sos = {k: butter(2, [5, f], btype="band", fs=FS, output="sos") for k, f in BANDS.items()}
        self.A, self.B = ipk - 300, ipk + n_after      # finestra comune, indici assoluti dell'impulso
        self.norm = np.ones(2)

    def __call__(self, P):
        P = P.astype(np.float64)
        P -= P[:, :4900].mean(1, keepdims=True)
        sl = slice(self.A, self.B)
        # uscita di H: ritardo zero all'indice 0, la sposto sul picco del template -> stessa scala dei tempi di raw
        f0 = np.roll(np.fft.ifft(np.fft.fft(P, axis=1) * self.H, axis=1).real, self.ipk, axis=1)[:, sl]
        raw = P[:, sl]
        f1 = sosfiltfilt(self.sos["f1"], P, axis=1)[:, sl]
        f2 = sosfiltfilt(self.sos["f2"], P, axis=1)[:, sl]
        p0, _ = peak(f0)
        f50 = crossing(f0, p0, 0.5)
        a0 = f0[np.arange(len(P)), p0]
        # i massimi di raw, f1, f2 si cercano vicino all'evento (picco di f0): con poco SNR, su tutta la
        # finestra vincerebbe spesso un picco di rumore lontano
        k = np.arange(raw.shape[1])[None]
        near = (k >= p0[:, None] - 10) & (k <= p0[:, None] + self.n_search)
        pr, _ = peak(raw, near)
        t = {l: crossing(raw, pr, l / 100) for l in (5, 10, 50, 90, 95)}
        p1, t1max = peak(f1, near)
        p2, _ = peak(f2, near)
        r = np.arange(len(P))
        ms = 1e3 / FS
        X = np.column_stack([f1[r, p1] / a0 / self.norm[0], f2[r, p2] / a0 / self.norm[1],
                             (t[95] - t[5]) * ms, (t[10] - t[5]) * ms, (t[50] - t[5]) * ms,
                             (t[10] - f50) * ms, (t[50] - f50) * ms, (t[90] - f50) * ms, (t[95] - f50) * ms,
                             (t1max - f50) * ms])
        return X


def simulate(ch, wp):
    of = s.folder_info(OF_DIR)
    row = next(r for r in csv.DictReader(open(of["bi_csv"])) if int(r["channel"]) == ch and int(r["wp"]) == wp)
    dt_max, K, jitter = s.point_dt(ch, float(row["vbias"]))
    meanpulse, nps = s.load_row_inputs(ch, wp)
    S, w, _ = s.an.compute_H(meanpulse, nps, np.hanning, sampling_rate=FS)
    H, f1, f2 = s.load_filters(of, ch, wp, nps, row)
    _, _, H0 = s.an.compute_H(meanpulse, nps, np.hanning, sampling_rate=FS)      # f0 = filtro ottimo
    feat = Features(H0, int(np.argmax(meanpulse)), 400, int(2 * dt_max * FS) + 20)   # fino a 2 dt_max dopo f0
    tmpl = feat(np.asarray(meanpulse)[None] * float(row["signal_amp"]) / np.max(meanpulse))
    feat.norm = tmpl[0, :2]                                                         # singoli -> 1
    kw = dict(n_photons=s.fn.n_photons(ch, cupid=s.CUPID), e_roi=s.fn.E_ROI_MEV, shift_max=1.0 / FS,
              jitter_max=jitter, nsim=NSIM)
    out = {}
    for k, dtm in (("s", 0.0), ("p", dt_max)):
        store, info = [], []
        cap = lambda x: (store.append(feat(x)), np.zeros(len(x)))[1]
        ofr, _ = s.simulate_psd_v2(S, nps, w, [(H, f1, f2)], float(row["signal_amp"]), dtm, SEED, (cap,),
                                   info_out=info, **kw)
        out[k] = dict(X=np.concatenate(store), of=ofr, dt=np.concatenate([d["dt"] for d in info]),
                      r=np.concatenate([d["r"] for d in info]))
    return out, dict(vbias=float(row["vbias"]), dt_max=dt_max, SNR=float(row["SNR"]))


def bdt_R(D):
    from sklearn.ensemble import HistGradientBoostingClassifier
    tr, te = slice(0, None, 2), slice(1, None, 2)
    Xs, Xp = (np.nan_to_num(D[k]["X"], posinf=0, neginf=0) for k in "sp")
    g = HistGradientBoostingClassifier(max_iter=400, random_state=0).fit(
        np.vstack([Xs[tr], Xp[tr]]), np.r_[np.zeros(len(Xs[tr])), np.ones(len(Xp[tr]))])
    ps, pp = g.predict_proba(Xs[te])[:, 1], g.predict_proba(Xp[te])[:, 1]
    R_bdt = np.mean(pp > np.percentile(ps, 90))
    R_of = np.mean(D["p"]["of"][te] < np.percentile(D["s"]["of"][te], 10))
    return 100 * R_bdt, 100 * R_of


def plot(ch, wp, D, meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE = "#1f5fbf", "#e8963c"
    fig, axes = plt.subplots(3, 4, figsize=(17, 10.5))
    cols = [(NAMES[i], D["s"]["X"][:, i], D["p"]["X"][:, i]) for i in range(10)]
    cols.append(("OF ratio max(f1 H x)/max(f2 H x)  [campaign]", D["s"]["of"], D["p"]["of"]))
    for ax, (name, xs, xp) in zip(axes.ravel(), cols):
        xs, xp = xs[np.isfinite(xs)], xp[np.isfinite(xp)]
        lo, hi = np.percentile(xs, [5, 95])
        span = hi - lo
        bins = np.linspace(np.percentile(np.r_[xs, xp], 0.5) if span == 0 else lo - 2.5 * span,
                           np.percentile(np.r_[xs, xp], 99.5) if span == 0 else hi + 2.5 * span, 80)
        w = lambda x: np.full(len(x), 1 / len(x))
        ax.hist(xs, bins, weights=w(xs), color=BLUE, alpha=0.9, label="single")
        ax.hist(xp, bins, weights=w(xp), color=ORANGE, alpha=0.7, label="pile-up")
        for v in (lo, hi):
            ax.axvline(v, color="#d62728", ls=":", lw=1.2)
        R = 100 * np.mean((xp < lo) | (xp > hi))
        ax.set(yscale="log", xlabel=name, ylim=(1e-4, 1))
        ax.text(0.98, 0.95, f"R = {R:.0f}%", transform=ax.transAxes, ha="right", va="top", fontsize=10,
                bbox=dict(fc="white", ec="0.8"))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(loc="upper left", fontsize=9, frameon=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("probability / bin")
    ax = axes[2, 3]
    ax.scatter(D["p"]["r"], D["p"]["dt"] * 1e3, s=1, color=ORANGE, alpha=0.3, rasterized=True)
    ax.set(xlabel="E1 / (E1 + E2)", ylabel="dt [ms]", title="pile-up samples", xlim=(0, 1))
    ax.spines[["top", "right"]].set_visible(False)
    R_bdt, R_of = bdt_R(D)
    fig.suptitle(f"Ch {ch} WP {wp}: V_bias = {meta['vbias']:g} V, rise time = dt_max = {meta['dt_max'] * 1e3:.2f} ms, "
                 f"SNR = {meta['SNR']:.0f}.   AMoRE heat parameters on CROSS MC events ({NSIM} + {NSIM})\n"
                 f"red dotted: central 90% of singles, R = pile-ups outside.   "
                 f"BDT on the 10 parameters: R = {R_bdt:.1f}%   vs   OF (campaign): R = {R_of:.1f}%   "
                 f"(test half, 90% singles acceptance)", fontsize=11)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"amore_features_ch{ch}_wp{wp}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"ch{ch} wp{wp}: BDT R {R_bdt:.1f}%, OF R {R_of:.1f}%  -> {os.path.relpath(out, ROOT)}", flush=True)


if __name__ == "__main__":
    a = list(map(int, sys.argv[1:]))
    for ch, wp in zip(a[::2], a[1::2]):
        D, meta = simulate(ch, wp)
        plot(ch, wp, D, meta)
