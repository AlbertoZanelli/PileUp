"""
explain_wiener_lambda_m205.py
=============================
Perche' un lambda piccolo fa amplificare il rumore al filtro di Wiener  W = S* / (|S|^2 + lambda NPS).
Quattro pannelli, con gli spettri veri di un punto di lavoro:
  (a) spettro dell'impulso e del rumore: dove si incrociano, il filtro smette di invertire l'impulso;
  (b) guadagno del filtro: segue 1/|S| (inversione) fino all'incrocio, poi scende;
  (c) l'impulso SENZA rumore dopo il filtro: piu' lambda e' piccolo, piu' l'impulso diventa stretto
      (deconvoluzione = ricostruire l'istante del deposito di energia);
  (d) lo stesso CON il rumore: il prezzo della deconvoluzione.
I kernel sono normalizzati a guadagno 1 sul segnale; niente filtri di banda, solo il kernel.

    python explain_wiener_lambda_m205.py [canale wp]
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import simulate_BI_error_m205 as mc
import src.analysis as an

CHANNEL, WP = 34, 3
LAMBDAS = [(1.1, "#2a78d6", r"Wiener, $\lambda$ = 1.1"), (1e-4, "#eb6834", r"Wiener, $\lambda$ = 10$^{-4}$")]
C_OF, C_PULSE = "0.25", "0.6"
SEED = 3
if len(sys.argv) > 2:
    CHANNEL, WP = int(sys.argv[1]), int(sys.argv[2])


def main():
    nps = mc.load_noise(CHANNEL, WP)
    S, w, H = an.compute_H(mc.template_pulse(CHANNEL, WP, "fit"), nps, np.hanning, sampling_rate=mc.SAMPLING_RATE)
    n = len(S); fs = mc.SAMPLING_RATE
    f = np.fft.fftfreq(n, 1 / fs); pos = f > 0
    rows = csv.DictReader(open(mc.FOLDERS[0]["bi_csv"]))
    A = [float(r["signal_amp"]) for r in rows if int(r["channel"]) == CHANNEL and int(r["wp"]) == WP][0]

    # spettri nella normalizzazione del Wiener (compute_W_torch): potenze confrontabili
    P = np.abs(S) ** 2; a = np.sum(np.sqrt(P)); b = np.sum(np.sqrt(nps))
    Pn, Nn = P / a, nps * a / b ** 2
    unit = lambda K: K / np.mean((K * S).real)                     # guadagno 1 sul segnale
    kernels = [("optimum filter", C_OF, unit(H))]
    for lam, c, lab in LAMBDAS:
        W = np.conj(S) / (Pn + lam * Nn); W[0] = 0
        kernels.append((lab, c, unit(W)))

    rng = np.random.default_rng(SEED)
    noise = np.fft.ifft((rng.normal(size=n) + 1j * rng.normal(size=n)) * np.sqrt(nps)).real
    clean = np.fft.ifft(A * S).real
    tgt = n // 2
    t = (np.arange(n) - tgt) / fs * 1e3
    filt = lambda K, x: np.roll(np.fft.ifft(K * np.fft.fft(x)).real, tgt)

    fig, ax = plt.subplots(2, 2, figsize=(14, 9.5), gridspec_kw=dict(hspace=0.3, wspace=0.22))

    # (a) spettri
    a_ = ax[0, 0]
    a_.loglog(f[pos], Pn[pos], color=C_PULSE, lw=2.5, label=r"pulse  $|S|^2$")
    a_.loglog(f[pos], Nn[pos], color="k", lw=1.5, label=r"noise  NPS  ($\lambda$ = 1)")
    for lam, c, lab in LAMBDAS:
        if lam != 1.1:
            a_.loglog(f[pos], lam * Nn[pos], color=c, lw=1.5, ls="--", label=rf"{lab.split(',')[1].strip()} $\times$ NPS")
        k = np.where(Pn[pos] < lam * Nn[pos])[0][0]
        a_.axvline(f[pos][k], color=c, lw=1, ls=":")
        a_.text(f[pos][k], Pn[pos].max() * 2, f"{f[pos][k]:.0f} Hz", color=c, ha="center", fontsize=9)
    a_.set(xlabel="frequency [Hz]", ylabel="power (Wiener normalisation)", xlim=(1, fs / 2),
           title="(a) where the pulse is above the noise term, the filter inverts it")
    a_.legend(fontsize=9, frameon=False, loc="lower left")

    # (b) guadagno
    b_ = ax[0, 1]
    inv = 1 / np.abs(S); inv = inv / inv[pos][0] * np.abs(kernels[1][2])[pos][0]
    b_.loglog(f[pos], inv[pos], color="k", lw=1, ls="--", label=r"pure inversion  $1/|S|$")
    for lab, c, K in kernels:
        b_.loglog(f[pos], np.abs(K)[pos], color=c, lw=2, label=lab)
    g = np.abs(kernels[0][2])[pos]
    b_.set(xlabel="frequency [Hz]", ylabel="filter gain  |W|  (unit gain on the pulse)", xlim=(1, fs / 2),
           ylim=(g.max() * 1e-6, np.abs(kernels[-1][2])[pos].max() * 5),
           title="(b) gain: it follows 1/|S| up to the crossing, then drops")
    b_.legend(fontsize=9, frameon=False, loc="lower left")

    # (c) senza rumore, (d) con rumore
    m = (t > -6) & (t < 14)
    for a_, x, ttl in ((ax[1, 0], clean, "(c) pulse WITHOUT noise after the filter: a smaller $\\lambda$ sharpens it"),
                       (ax[1, 1], clean + noise, "(d) the same WITH noise: the price of sharpening")):
        a_.plot(t[m], (x - np.mean(x[:tgt - 100]))[m], color=C_PULSE, lw=1.2, label="pulse before the filter")
        for lab, c, K in kernels[::-1] if a_ is ax[1, 1] else kernels:
            y = filt(K, x - np.mean(x[:tgt - 100]))
            sig = np.sqrt(np.sum(np.abs(K) ** 2 * nps)) / n / A
            a_.plot(t[m], y[m], color=c, lw=1.8, label=f"{lab}   (noise / amplitude = {sig:.2g})")
        a_.axhline(0, color="0.85", lw=0.8, zorder=0)
        a_.set(xlabel="time from the pulse [ms]", ylabel="amplitude", title=ttl)
    for a_ in ax[1]:                                               # sotto gli assi: non coprono le curve
        a_.legend(fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)

    fig.suptitle(f"m205 ch{CHANNEL} WP{WP}: what $\\lambda$ does in the Wiener filter "
                 r"$W = S^* / (|S|^2 + \lambda\,\mathrm{NPS})$", fontsize=13, y=0.95)
    out = os.path.join(mc.BASE_DIR, f"explain_wiener_lambda_ch{CHANNEL}_wp{WP}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
