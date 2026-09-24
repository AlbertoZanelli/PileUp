"""
show_likelihood_m205.py
=======================
Cosa fa lo stimatore M (src/pileup_likelihood.py) su UN evento, passo per passo, e dove quell'evento
cade rispetto alle distribuzioni di M di singoli e pile-up.

Si sceglie un pile-up (r, dt) qui sotto. Il programma:
  1. genera l'evento come il Monte Carlo della campagna (template "fit" x ampiezza ROI + rumore
     dalla NPS), con r e dt FISSATI;
  2. calcola y(t), la correlazione del filtro ottimo, e il fit a UN impulso: L1 = max y^2 / R(0);
  3. per ogni (dt', r') della griglia fa il fit a DUE impulsi: L(dt', r'), e dL = L - L1;
  4. media i pile-up pesati col prior: M = log sum pi(dt', r') exp(dL / 2);
  5. genera singoli e pile-up come il MC (e pile-up con gli STESSI r, dt dell'evento), calcola M,
     mette il taglio al 90% dei singoli e mostra dove cade l'evento.
Il template di M e' quello di training (AP simulato), gli eventi nascono dal fit: come nel MC.

Uscita: likelihood_explained_ch<ch>_wp<wp>_r<r>_dt<dt>.png nella cartella del progetto, e i
numeri a terminale. ~1 min con N_DIST = 5000.

    python show_likelihood_m205.py
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.special import logsumexp

import simulate_BI_error_m205 as mc          # stessi template, NPS, T_MAX, ACCEPTANCE del MC
import src.analysis as an
import src.simulation as sim
from src.pileup_likelihood import PileupLikelihood

# ── CONFIGURAZIONE ────────────────────────────────────────────────────────────────────────────
CHANNEL, WP = 31, 7
R, DT = 0.3, 0.4e-3          # l'evento d'esempio: frazione d'energia del SECONDO impulso, ritardo [s]
EVENT_SEED = 7               # rumore dell'evento d'esempio
N_DIST = 5000                # eventi per popolazione negli istogrammi
DIST_SEED = 1234
TRAIN_SIM = "APsimfit10000led"   # template di M (quello di training delle campagne)

C_SINGLE, C_PILE, C_FIXED = "#2a78d6", "#eb6834", "#1baf7a"
FS = mc.SAMPLING_RATE


def signal_amplitude(channel, wp):
    """Ampiezza ROI e SNR del punto, dalla riga del CSV di training di RESULTS_NAME."""
    for row in csv.DictReader(open(mc.FOLDERS[0]["bi_csv"])):
        if int(row["channel"]) == channel and int(row["wp"]) == wp:
            return float(row["signal_amp"]), float(row["SNR"])
    raise SystemExit(f"[ERROR] ch {channel} wp {wp} non trovato in {mc.FOLDERS[0]['bi_csv']}")


def dissect(lik, x):
    """Lo stesso conto di PileupLikelihood.statistic, per UN evento, tenendo i pezzi."""
    y = lik.correlation(x[None, :])[0]
    y1 = y[:lik.n_t1]
    i1 = np.argmax(np.clip(y1, 0, None))
    L1 = max(y1[i1], 0) ** 2 / lik.R0
    L = np.empty((len(lik.d_steps), len(lik.r)))
    it = np.empty_like(L, dtype=int)
    for j, d in enumerate(lik.d_steps):
        c = (1 - lik.r)[:, None] * y1[None, :] + lik.r[:, None] * y[d:d + lik.n_t1][None, :]
        cc = np.clip(c, 0, None) ** 2
        it[j] = np.argmax(cc, axis=1)
        L[j] = cc[np.arange(len(lik.r)), it[j]] / lik.normP[j]
    dL = L - L1
    logw = lik.log_prior[None, :] + 0.5 * dL
    M = logsumexp(logw)
    assert np.isclose(M, lik.statistic(x[None, :])[0]), "dissect() diverso da statistic()"
    return dict(y=y, i1=i1, L1=L1, A1=y1[i1] / lik.R0, L=L, dL=dL, it=it,
                post=np.exp(logw - M), M=M)


def pulse(S, w, amp, t0, r=0.0, dt=0.0):
    """amp * [(1-r) s(t - t0) + r s(t - t0 - dt)] nel tempo, dal template S (anche fra i campioni)."""
    return np.fft.ifft(amp * S * np.exp(-1j * w * t0) * ((1 - r) + r * np.exp(-1j * w * dt))).real


def m_values(lik, S, nps, w, amp, dt_max, seed, fixed=None):
    """M di N_DIST eventi generati come nel MC (a blocchi di mc.CHUNK). fixed=(r, dt): pile-up con
    r e dt fissati invece che estratti."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(0, N_DIST, mc.CHUNK):
        n = min(mc.CHUNK, N_DIST - k)
        if fixed is None:
            fp, *_ = sim.simulate_frequency_pulses(S, nps, 0.0, w, nsim=n, seed=rng, signal_scale=amp,
                                                   dt_max=dt_max, fold_ratio=False)
        else:
            r, dt = fixed
            noise = (rng.normal(size=(n, len(S))) + 1j * rng.normal(size=(n, len(S)))) * np.sqrt(nps)
            fp = amp * S * ((1 - r) + r * np.exp(-1j * w * dt)) + noise
        out.append(lik.statistic(np.fft.ifft(fp, axis=1).real))
    return np.concatenate(out)


def main():
    amp, snr = signal_amplitude(CHANNEL, WP)
    nps = mc.load_noise(CHANNEL, WP)
    S_gen, w, _ = an.compute_H(mc.template_pulse(CHANNEL, WP, "fit"), nps, np.hanning, sampling_rate=FS)
    S_lik, _, _ = an.compute_H(mc.template_pulse(CHANNEL, WP, "sim", TRAIN_SIM), nps, np.hanning,
                               sampling_rate=FS)
    lik = PileupLikelihood(S_lik, nps, w, dt_max=mc.T_MAX)
    N = len(S_gen)

    # ── 1. l'evento: stessa convenzione di rumore del MC (E|N_k|^2 = 2 nps, poi .real) ──────────
    rng = np.random.default_rng(EVENT_SEED)
    noise = (rng.normal(size=N) + 1j * rng.normal(size=N)) * np.sqrt(nps)
    x = np.fft.ifft(amp * S_gen * ((1 - R) + R * np.exp(-1j * w * DT)) + noise).real

    # ── 2-4. i pezzi di M ──────────────────────────────────────────────────────────────────────
    d = dissect(lik, x)
    jb, rb = np.unravel_index(np.argmax(d["L"]), d["L"].shape)
    dt_best, r_best = lik.d_steps[jb] * (lik.t[1] - lik.t[0]), lik.r[rb]
    t_best = lik.t[d["it"][jb, rb]]
    c_best = (1 - r_best) * d["y"][d["it"][jb, rb]] + r_best * d["y"][d["it"][jb, rb] + lik.d_steps[jb]]
    A_best = c_best / lik.normP[jb, rb]
    t1 = lik.t[d["i1"]]
    dts = lik.d_steps * (lik.t[1] - lik.t[0])
    dt_mean = np.sum(d["post"] * dts[:, None])
    r_mean = np.sum(d["post"] * lik.r[None, :])

    # ── 5. distribuzioni di M e taglio ─────────────────────────────────────────────────────────
    M_s = m_values(lik, S_gen, nps, w, amp, 0.0, DIST_SEED)
    M_p = m_values(lik, S_gen, nps, w, amp, mc.T_MAX, DIST_SEED)
    M_f = m_values(lik, S_gen, nps, w, amp, 0.0, DIST_SEED + 1, fixed=(R, DT))
    cut = np.percentile(M_s, 100 * mc.ACCEPTANCE)        # M > cut -> scartato come pile-up
    rej_p, rej_f = np.mean(M_p > cut), np.mean(M_f > cut)
    rejected = d["M"] > cut

    lines = [
        f"ch {CHANNEL}  WP {WP}   amplitude A = {amp:.3g}   OF SNR = {snr:.1f}",
        f"event: pile-up with dt = {DT*1e3:.2f} ms, r = {R:.2f}   (2nd pulse carries r of the energy)",
        "",
        "1) one-pulse fit (optimum filter)",
        f"   t1 = {t1*1e3:+.3f} ms   A1 = {d['A1']:.3g}   L1 = max y(t)^2 / R(0) = {d['L1']:.1f}",
        f"2) two-pulse fits on {len(dts)} x {len(lik.r)} (dt', r')",
        f"   best: dt' = {dt_best*1e3:.2f} ms, r' = {r_best:.2f}, t = {t_best*1e3:+.3f} ms, A = {A_best:.3g}",
        f"   L_best = {d['L'][jb, rb]:.1f}   dL_max = L_best - L1 = {d['dL'][jb, rb]:.2f}",
        "3) average over the prior  pi(dt', r') = uniform(dt') x p_2b(r')",
        f"   M = log sum pi exp(dL/2) = {d['M']:.2f}",
        f"   posterior mean: dt' = {dt_mean*1e3:.2f} ms, r' = {r_mean:.2f}",
        "4) decision",
        f"   cut (accepts {mc.ACCEPTANCE:.0%} of singles): M > {cut:.2f}",
        f"   this event: {'REJECTED as pile-up' if rejected else 'ACCEPTED (looks single)'}",
        f"   pile-ups with this dt, r rejected: {rej_f:.1%}",
        f"   all pile-ups (dt < {mc.T_MAX*1e3:.1f} ms, r ~ p_2b) rejected: {rej_p:.1%}",
    ]
    print("\n".join(lines))

    # ── figura ────────────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(2, 3, figsize=(18, 10), gridspec_kw=dict(wspace=0.28, hspace=0.32))
    ms = 1e3

    # (a) l'evento e i due fit
    n0 = N // 2
    k = np.arange(n0 - 30, n0 + 150)
    tt = (k - n0) / FS * ms
    a = ax[0, 0]
    a.plot(tt, x[k], color="0.6", lw=1, label="event (signal + noise)")
    a.plot(tt, pulse(S_gen, w, amp * (1 - R), 0.0)[k], color=C_PILE, lw=1.2, ls=":", label="true 1st pulse")
    a.plot(tt, pulse(S_gen, w, amp * R, DT)[k], color=C_PILE, lw=1.2, ls="--", label="true 2nd pulse")
    a.plot(tt, pulse(S_lik, w, d["A1"], t1)[k], color=C_SINGLE, lw=2, label="best one-pulse fit")
    a.plot(tt, pulse(S_lik, w, A_best, t_best, r_best, dt_best)[k], color=C_FIXED, lw=2,
           label="best two-pulse fit")
    a.set(xlabel="time from template position [ms]", ylabel="amplitude",
          title="(a) the event and the two hypotheses")
    a.legend(fontsize=9, frameon=False)

    # (b) y(t): l'ampiezza del filtro ottimo in funzione della posizione
    a = ax[0, 1]
    a.plot(lik.t * ms, d["y"] / lik.R0, color=C_SINGLE, lw=2, label="y(t) / R(0): one-pulse amplitude at t")
    it = d["it"][jb, rb]
    cb = ((1 - r_best) * d["y"][:lik.n_t1] + r_best * d["y"][lik.d_steps[jb]:lik.d_steps[jb] + lik.n_t1])
    a.plot(lik.t[:lik.n_t1] * ms, cb / lik.normP[jb, rb], color=C_FIXED, lw=2,
           label=f"(1-r')y(t) + r'y(t+dt') / ||P||^2, best dt', r'")
    for tv, ls, lab in ((0.0, ":", "true 1st pulse"), (DT, "--", "true 2nd pulse")):
        a.axvline(tv * ms, color=C_PILE, ls=ls, lw=1.2, label=lab)
    a.plot(t1 * ms, d["A1"], "o", color=C_SINGLE, ms=8)
    a.plot(t_best * ms, A_best, "o", color=C_FIXED, ms=8)
    a.set(xlabel="position t [ms]", ylabel="amplitude", title="(b) optimum-filter correlation y(t)")
    a.legend(fontsize=9, frameon=False, loc="lower left")

    # (c) i numeri
    a = ax[0, 2]
    a.axis("off")
    a.text(0, 1, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9.2,
           transform=a.transAxes)

    dr =(lik.r[1] - lik.r[0]) / 2
    extent = [dts[0] * ms, dts[-1] * ms, lik.r[0] - dr, lik.r[-1] + dr]

    # (d) dL(dt', r')
    a = ax[1, 0]
    # scala sui valori POSITIVI (dove due impulsi spiegano di piu'): i negativi arrivano a -70 e
    # schiaccerebbero tutto; oltre -v il colore satura (freccia sulla barra)
    v = max(np.max(d["dL"]), 1.0)
    im = a.imshow(d["dL"].T, origin="lower", aspect="auto", extent=extent, cmap="RdBu_r",
                  norm=TwoSlopeNorm(0, -v, v))
    fig.colorbar(im, ax=a, extend="min", label="dL = L(dt', r') - L1   (red: two pulses fit better)")
    a.plot(DT * ms, R, "*", color="k", ms=14, label="true (dt, r)")
    a.plot(dt_best * ms, r_best, "o", mfc="none", mec="k", ms=10, mew=2, label="best fit")
    a.set(xlabel="dt' [ms]", ylabel="r'", title="(d) how much better two pulses fit than one")
    a.legend(fontsize=9, loc="upper right")

    # (e) da dove viene M: il peso di ogni (dt', r') nella somma
    a = ax[1, 1]
    im = a.imshow(d["post"].T, origin="lower", aspect="auto", extent=extent, cmap="Blues")
    fig.colorbar(im, ax=a, label="pi exp(dL/2) / sum  (share of the sum)")
    a.plot(DT * ms, R, "*", color="k", ms=14, label="true (dt, r)")
    a.plot(dt_mean * ms, r_mean, "x", color="k", ms=10, mew=2, label="weighted mean")
    a.set(xlabel="dt' [ms]", ylabel="r'", title=f"(e) where M = {d['M']:.2f} comes from")
    a.legend(fontsize=9, loc="upper right")

    # (f) distribuzioni di M: asse simmetrico-logaritmico, M va da ~-1 a centinaia
    a = ax[1, 2]
    f = lambda m: np.sign(m) * np.log10(1 + np.abs(m))
    finv = lambda u: np.sign(u) * (10 ** np.abs(u) - 1)
    allm = np.concatenate([M_s, M_p, M_f, [d["M"]]])
    bins = finv(np.linspace(f(allm.min()), f(allm.max()), 90))
    a.hist(M_s, bins=bins, histtype="step", lw=2, color=C_SINGLE, label="single pulses")
    a.hist(M_p, bins=bins, histtype="step", lw=2, color=C_PILE,
           label=f"pile-up, dt < {mc.T_MAX*1e3:.1f} ms, r ~ p_2b  ({rej_p:.0%} rejected)")
    a.hist(M_f, bins=bins, histtype="step", lw=2, color=C_FIXED,
           label=f"pile-up, dt = {DT*1e3:.2f} ms, r = {R:.2f}  ({rej_f:.0%} rejected)")
    a.axvspan(cut, bins[-1], color="0.9", zorder=0)
    a.axvline(cut, color="k", ls="--", lw=1.2, label=f"cut: {mc.ACCEPTANCE:.0%} of singles accepted")
    a.axvline(d["M"], color="k", lw=2.5, label=f"this event, M = {d['M']:.2f}")
    a.set_xscale("symlog", linthresh=1)
    a.set_yscale("log")
    a.set(xlabel="M  (large = pile-up)", ylabel="events", title="(f) where the event falls")
    a.legend(fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14))

    fig.suptitle(f"Model-averaged likelihood ratio M - ch {CHANNEL}, WP {WP}", fontsize=15)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"likelihood_explained_ch{CHANNEL}_wp{WP}_r{R:g}_dt{DT*1e3:g}ms.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
