"""
pileup_likelihood.py
====================
Lo stimatore M: rapporto di verosimiglianza MEDIATO fra "pile-up" e "impulso singolo".

Per ogni evento risponde alla domanda: e' piu' verosimile che sia un pile-up di due impulsi -- con
un ritardo dt e una ripartizione d'energia r qualsiasi, pesati con quanto sono probabili -- oppure
un impulso singolo?  M grande = somiglia a un pile-up.

La derivazione completa e' in tesi_note/stimatore_verosimiglianza.pdf; i numeri fra parentesi
nei commenti, (eq. N), sono le equazioni di quel documento. In breve:

    y(t)     = sum_k X_k S_k* e^{i w_k t} / J_k            correlazione del filtro ottimo  (eq. 9)
    R(tau)   = sum_k |S_k|^2 cos(w_k tau) / J_k            autocorrelazione del template   (eq. 13)
    L1       = max_t y(t)^2 / R(0)                          chi^2 spiegato da UN impulso    (eq. 9)
    L(dt, r) = max_t max(0, (1-r) y(t) + r y(t+dt))^2       chi^2 spiegato dal pile-up      (eq. 14)
               / ( [(1-r)^2 + r^2] R(0) + 2 r (1-r) R(dt) )
    M        = log sum_{dt, r} pi(dt, r) exp( (L(dt, r) - L1) / 2 )                        (eq. 20)

con pi(dt, r) = uniforme in dt su [0, dt_max] x pdf_ratio2b(r). Non si addestra niente: servono
solo il template S, la NPS J e le frequenze w. La soglia su M si fissa sugli impulsi singoli
simulati, come per il rapporto dei massimi (simulate_BI_error_m205.py, LIKELIHOOD = True).

Uso:
    lik = PileupLikelihood(S, nps, w, dt_max=8e-4)     # una volta per canale e punto di lavoro
    M = lik.statistic(pulses)                          # pulses: (n_eventi, n_campioni), nel tempo

Controllo: python src/pileup_likelihood.py  (verifica y(t) contro la FFT e due casi esatti).
"""
import os
import sys

import numpy as np
from scipy.special import logsumexp

if __name__ == "__main__":      # lanciato come script: la radice del progetto non e' ancora nel path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utility.double_beta_spectrum import pdf_ratio2b


class PileupLikelihood:
    """Tutto quello che non dipende dall'evento si calcola qui, una volta sola: le frequenze
    utili, le funzioni coseno/seno per leggere y(t) su una griglia fine, R(tau), le norme dei
    template di pile-up e i pesi a priori. Per ogni evento resta solo statistic().

    Parametri
    ---------
    S, nps, w : spettro del template (finestrato come in analysis.compute_H), NPS, frequenze
                angolari; tutti lunghi quanto la finestra (10000).
    dt_max    : ritardo massimo del pile-up [s]; deve essere lo stesso T_MAX del BI e del MC.
    t_window  : l'impulso si cerca entro +-t_window [s] dalla posizione del template.
    step      : passo della griglia dei tempi [s]. Il campionamento (1e-4 s) e' troppo grosso
                rispetto ai ritardi in gioco, quindi y(t) si legge fra un campione e l'altro.
    r_values  : valori di r su cui si media (pesati con pdf_ratio2b).
    keep      : si tengono le frequenze che portano questa frazione del peso |S|^2/J; le altre
                hanno template trascurabile e non cambiano y(t).
    """

    def __init__(self, S, nps, w, dt_max=8e-4, t_window=5e-4, step=1e-5,
                 r_values=np.linspace(0.05, 0.95, 19), keep=1 - 1e-6):
        S, nps, w = np.asarray(S), np.asarray(nps, dtype=float), np.asarray(w, dtype=float)
        N = len(S)
        weight = np.abs(S) ** 2 / nps                        # |S|^2 / J: peso di ogni frequenza

        # ── 1. frequenze utili (solo positive: lo spettro di un segnale reale e' simmetrico) ──
        pos = np.arange(1, N // 2)
        order = pos[np.argsort(weight[pos])[::-1]]
        n_keep = np.searchsorted(np.cumsum(weight[order]) / weight[pos].sum(), keep) + 1
        self.k = np.sort(order[:n_keep])

        # ── 2. griglia dei tempi e funzioni per y(t)  (eq. 9) ────────────────────────────────
        # y(t) = DC + 2 sum_{k>0} Re( X_k S_k*/J_k e^{i w_k t} ): la meta' positiva, due volte.
        # t da -t_window a t_window + dt_max: serve y al tempo del primo impulso e a quello del
        # secondo, fino a dt_max dopo.
        self.t = np.round(np.arange(-t_window, t_window + dt_max + step / 2, step), 12)
        self.n_t1 = int(round(2 * t_window / step)) + 1      # tempi possibili del PRIMO impulso
        wt = np.outer(w[self.k], self.t)
        self.cos, self.sin = np.cos(wt), np.sin(wt)
        self.filt = (np.conj(S) / nps)[self.k]               # S*/J sulle frequenze tenute
        self.dc = (np.conj(S[0]) / nps[0]).real

        # ── 3. autocorrelazione del template R(tau)  (eq. 13), sulle stesse frequenze ────────
        wk, pk = w[self.k], weight[self.k]
        R = lambda tau: weight[0] + 2 * np.sum(pk * np.cos(wk * tau))
        self.R0 = R(0.0)

        # ── 4. i pile-up su cui si media: ritardi (in passi di griglia) e ripartizioni ────────
        self.d_steps = np.arange(0, int(round(dt_max / step)) + 1)
        self.r = np.asarray(r_values, dtype=float)
        # ||P||^2 per ogni (dt, r)  (eq. 13): norma del template di pile-up
        self.normP = np.array([((1 - self.r) ** 2 + self.r ** 2) * self.R0
                               + 2 * self.r * (1 - self.r) * R(d * step) for d in self.d_steps])

        # ── 5. pesi a priori log pi(dt, r)  (sez. 6.4): dt uniforme, r da pdf_ratio2b ──────────
        pr = pdf_ratio2b(self.r)
        self.log_prior = np.log(pr / pr.sum()) - np.log(len(self.d_steps))

    def correlation(self, pulses):
        """y(t) sulla griglia fine per ogni evento (eq. 9): (n_eventi, len(self.t))."""
        X = np.fft.fft(np.asarray(pulses, dtype=np.float64), axis=1)
        Z = X[:, self.k] * self.filt[None, :]
        return X[:, [0]].real * self.dc + 2 * (Z.real @ self.cos - Z.imag @ self.sin)

    def statistic(self, pulses):
        """M per ogni evento (eq. 20). pulses: (n_eventi, n_campioni), nel dominio del tempo."""
        y = self.correlation(pulses)
        y1 = y[:, :self.n_t1]                                     # y al tempo del primo impulso
        # un impulso: chi^2 spiegato, ampiezza >= 0  (eq. 9)
        L1 = np.max(np.clip(y1, 0, None) ** 2, axis=1) / self.R0
        terms = []
        for j, d in enumerate(self.d_steps):
            y2 = y[:, d:d + self.n_t1]                            # y al tempo del secondo, dt dopo
            # template di pile-up al tempo t: (1-r) y(t) + r y(t+dt)   (eq. 12), per ogni r
            c = (1 - self.r)[None, :, None] * y1[:, None, :] + self.r[None, :, None] * y2[:, None, :]
            L = np.max(np.clip(c, 0, None) ** 2, axis=2) / self.normP[j][None, :]   # (eq. 14)
            terms.append(self.log_prior[None, :] + 0.5 * (L - L1[:, None]))
        # somma pesata degli exp in forma logaritmica: L - L1 arriva a centinaia (eq. 20)
        return logsumexp(np.concatenate(terms, axis=1), axis=1)


if __name__ == "__main__":
    # Controlli: (1) y(t) sulla griglia fine coincide con la correlazione OF calcolata con la FFT
    # ai tempi dei campioni; (2) su un singolo senza rumore nessun pile-up spiega di piu', M <= 0;
    # (3) su un pile-up senza rumore con (dt, r) sulla griglia, M > 0.
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import src.analysis as an
    tpl = np.load(os.path.join(base, "residual_scan_bessel", "fits_octopus", "bestfit_ch31_wp7.npy"))
    nps = np.load(os.path.join(base, "m205_NPS_clean", "ch31", "nps_ch31_wp7.npy"))
    S, w, _ = an.compute_H(tpl, nps, np.hanning, sampling_rate=10000)
    lik = PileupLikelihood(S, nps, w)
    rng = np.random.default_rng(0)
    noise = np.fft.ifft((rng.normal(size=(4, len(S))) + 1j * rng.normal(size=(4, len(S)))) * np.sqrt(nps), axis=1).real
    x = np.fft.ifft(2e-3 * S[None, :], axis=1).real + noise
    on_samples = np.where(np.abs(lik.t * 1e4 - np.round(lik.t * 1e4)) < 1e-9)[0]
    idx = np.round(lik.t[on_samples] * 1e4).astype(int) % len(S)
    def y_fft(v):                         # correlazione OF completa, ai tempi dei campioni
        return len(S) * np.fft.ifft(np.fft.fft(v, axis=1) * np.conj(S) / nps, axis=1).real[:, idx]
    # (1a) senza rumore: le frequenze scartate portano 1e-6 del peso -> differenza ~1e-6 relativa
    clean = np.fft.ifft(2e-3 * S[None, :], axis=1).real
    rel = np.abs(lik.correlation(clean)[:, on_samples] - y_fft(clean)).max() / np.abs(y_fft(clean)).max()
    assert rel < 1e-5, f"y(t) diversa dalla correlazione OF senza rumore: {rel:.1e}"
    # (1b) con rumore: la differenza si misura in sigma del rumore di y (= sqrt(R0))
    err = np.abs(lik.correlation(x)[:, on_samples] - y_fft(x)).max() / np.sqrt(lik.R0)
    assert err < 0.01, f"y(t) diversa dalla correlazione OF: {err:.1e} sigma"
    single = np.fft.ifft(S)[None, :].real
    assert lik.statistic(single)[0] <= 1e-9, "un singolo senza rumore non deve sembrare un pile-up"
    dt, r = 3e-4, 0.5
    pile = np.fft.ifft(S * ((1 - r) + r * np.exp(-1j * w * dt)))[None, :].real
    assert lik.statistic(pile)[0] > 0, "un pile-up senza rumore deve sembrare un pile-up"
    print(f"OK  y(t) = correlazione OF: {rel:.0e} relativo senza rumore, {err:.0e} sigma con rumore;  M(singolo) = {lik.statistic(single)[0]:.3f} <= 0;"
          f"  M(pile-up 0.3 ms, r=0.5) = {lik.statistic(pile)[0]:.1f} > 0")
