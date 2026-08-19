"""
plot_AP_spectra_m205.py
=======================
Average-pulse diagnostics for the m205 load curves. Products:

  1. AP power spectra: one panel per channel (all 9 channels, CHANNELS), overlaying
     the AP power spectrum of every working point (WP), colored by V_bias. Same PSD
     definition used in the m204 study (peak-normalized AP, Hann window). The dashed
     line marks the HF cutoff (500 Hz) used for HF-power.  -> AP_power_spectra_m205.png

  1b. AP pulses (time domain): the companion to (1) — one panel per channel, all WPs
     overlaid (peak-normalized, aligned on the peak), colored by V_bias.
                                                           -> AP_pulses_m205.png

  1c. ANPS (average noise power spectra): same style as (1) but the stored one-sided
     noise PSD of every WP (read from ROOT, no FFT).       -> ANPS_m205.png

  2. AP MODEL FIT (PAUSED by default, DO_FIT; only the 5 good FIT_CHANNELS): each
     average pulse is fitted with the bolometer pole-zero model
     ported from FitPulse.C: a rational transfer function inverse-transformed to a
     residue-weighted sum of (exponential, or damped oscillation) terms,

         f(t) = baseline + tilt*(t - t0) + [t > t0] * amp * Sum(pole terms) .

     MODEL SEARCH: instead of one fixed model, every combination in
     NPOL_GRID x CC_GRID x NZER_GRID is fitted and the one with the smallest RMS is
     kept per pulse — {3,4 poles} x {real-only (fitfuncNpMz), +1 complex-conjugate
     pair (FitFakePulse)} x {0,1 zero}. Residuals are UNWEIGHTED (constant point
     errors only rescaled the least_squares cost and worsened convergence).

     By default the fit spans the WHOLE recorded window (FIT_FULL_WINDOW), so the
     result is usable as a template for the pile-up rejection. For each channel a
     GRID (a cell per WP) shows, like FitPulse.C's two-pad canvas, the AP + fit on
     top and the residuals (data - fit) below with the +-3 sigma baseline band; the
     winning model and its RMS are in each panel title.
                                                    -> AP_fit_ch<ch>_m205.png
     The best model's descriptor (npol, cc, nzer), parameters and RMS are written
     to a CSV, keyed by (channel, WP), to rebuild the template downstream with
     _model_from_theta.                             -> AP_fit_params_m205.csv

Run:
    KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/bin/python3.13 plot_AP_spectra_m205.py
"""

import os
import re
import csv
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
from scipy.optimize import least_squares
import uproot

BASE    = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(BASE, "Processed")
PATTERN = "Processed_*_000205_*.root"
OUTDIR  = os.path.join(BASE, "m205_results_octopus")
OUT_PNG    = os.path.join(OUTDIR, "AP_power_spectra_m205.png")  # AP PSD overview
OUT_AP_PNG = os.path.join(OUTDIR, "AP_pulses_m205.png")         # AP time-domain overview
OUT_ANPS_PNG = os.path.join(OUTDIR, "ANPS_m205.png")           # noise PSD overview

# All 9 m205 channels for the OVERVIEW canvases (power spectra + AP pulses): the 5
# "good" (31,34,71,83,91) and the 4 "bad" that oscillate (37,40,41,94).
CHANNELS = [31, 34, 37, 40, 41, 71, 83, 91, 94]
# Only the 5 good channels get the (paused) pole-zero model fit.
FIT_CHANNELS = [31, 34, 71, 83, 91]
# Fits active: per-channel AP model-fit grids. Set False to only produce the
# overview canvases (power spectra, AP pulses, ANPS) and skip the fits.
DO_FIT = True

VBIAS_LIST = np.array([0.6, 1.0, 1.4, 1.8, 2, 3, 4, 5, 6, 8, 10, 20, 26, 30, 40])
HF_CUT_HZ = 500.0
HIST_TMPL = "averagepulse_ap_wp{wp}_medianAP"
HIST_NPS_TMPL = "averagepowerspectrum_noise_wp{wp}_medianpower"   # one-sided ANPS

# ── AP fit (pole-zero bolometer model, ported from FitPulse.C) ─────────────────
# MODEL SEARCH: each average pulse is fitted with EVERY combination of the grids
# below and the one with the smallest RMS is kept (per (channel, WP)). The two
# FitPulse.C branches are both in the search:
#   cc=False -> fitfuncNpMz  : all npol poles are real.
#   cc=True  -> FitFakePulse : npol-2 real poles + one complex-conjugate pair
#               (sigma +- i*omega), i.e. a damped ringing term.
# Base grid = {3,4 poles} x {real-only, +CC pair} x {1,2 zeros} = 8 models.
# Adding the SECOND zero was the key improvement: it fixes the low-bias pulses
# (e.g. ch31 WP1/WP3), whose rise/peak a single zero cannot shape — their RMS drops
# ~7x (1.1e-2 -> 1.9e-3, winner 4p-z2). nzer=0 was dropped (it won only ~1/75 fits).
# The AP is peak-normalized (=1) before fitting, as in the spectra above.
NPOL_GRID = [3, 4]
CC_GRID   = [False, True]
NZER_GRID = [1, 2]
# Extra targeted (npol, cc, nzer) models appended to the search — richer than the
# base grid, aimed at the DECAY-TIME RINGING: a CC pair (damped oscillation) plus
# many real poles/zeros. 6p+cc-z4 (4 real poles + CC pair + 4 zeros) cuts the ch31
# residual oscillation on most WPs (RMS -30..-82%), though it regresses on a few and
# is slow/less stable — which is why it is a SEARCH candidate, not a fixed model:
# fit_best keeps it only where it lowers the RMS. Empty [] to disable (faster).
EXTRA_MODELS = [(6, True, 4)]
# FIT_FULL_WINDOW=True fits the ENTIRE recorded window (all samples) — the intended
# use is a fit template for the pile-up rejection, which must be valid over the
# whole record. False restricts the fit to a window around the onset (FIT_PRE_S
# before, FIT_POST_S after), useful to isolate the pulse from the flat baseline.
FIT_FULL_WINDOW = True
FIT_PRE_S  = 0.02    # (FIT_FULL_WINDOW=False) seconds of pre-onset baseline in the window
FIT_POST_S = 0.25    # (FIT_FULL_WINDOW=False) seconds after onset (decay down to <1%)

# CSV with the per-(channel, WP) best-model descriptor, fit parameters and RMS, for
# reuse by the pile-up-rejection algorithm.
FIT_CSV = os.path.join(OUTDIR, "AP_fit_params_m205.csv")


def wp_to_vbias(wp: int) -> float:
    return float(VBIAS_LIST[wp // 2])


def compute_psd(signal, sampling_rate, window_fct=np.hanning):
    signal = np.asarray(signal, dtype=float)
    signal = signal - np.mean(signal)
    xw = signal * window_fct(len(signal))
    fft_vals = np.fft.rfft(xw)
    psd = (np.abs(fft_vals) ** 2) / (sampling_rate * len(signal))
    freq = np.fft.rfftfreq(len(signal), d=1.0 / sampling_rate)
    return freq, psd


# ═════════════════════════════════════════════════════════════════════════════
# Average-pulse fit — pole-zero bolometer model (FitPulse.C, cc=0: fitfuncNpMz)
# ═════════════════════════════════════════════════════════════════════════════
def _residues(poles, zeros):
    """Residue of the rational transfer function at each pole (as in FitPulse.C):
        Res_i = Prod_j (p_i - z_j) / Prod_{k!=i} (p_i - p_k) .
    The tiny-denominator guard just keeps the model finite if the optimizer probes
    a near-degenerate pole pair; it never triggers at a good fit."""
    poles = np.asarray(poles, dtype=float)
    zeros = np.asarray(zeros, dtype=float)
    res = np.empty(len(poles))
    for i, pi in enumerate(poles):
        num = np.prod(pi - zeros) if len(zeros) else 1.0
        den = 1.0
        for k, pk in enumerate(poles):
            if k != i:
                den *= (pi - pk)
        res[i] = num / (den if abs(den) > 1e-300 else 1e-300)
    return res


def pulse_model(t, t0, amp, baseline, tilt, poles, zeros):
    """Bolometer pulse (fitfuncNpMz, PreAmp fixed to 0): baseline + tilt ramp, plus
    for t>t0 the amplitude times the residue-weighted sum of pole exponentials."""
    t = np.asarray(t, dtype=float)
    out = baseline + tilt * (t - t0)
    m = t > t0
    if np.any(m):
        res = _residues(poles, zeros)
        dt = t[m] - t0
        s = np.zeros(dt.shape)
        for r_i, p_i in zip(res, poles):
            s += r_i * np.exp(np.clip(p_i * dt, -700.0, 0.0))   # p_i<0: clip avoids underflow warnings
        out[m] += amp * s
    return out


def _cc_terms(real_poles, sigma, omega, zeros):
    """Residues of the real poles and the (magnitude 2|R|, phase arg R) of the
    complex-conjugate pole pair sigma +- i*omega, exactly as in FitPulse.C's
    FitFakePulse. The transfer function has denominator Prod_real(s-p_k) *
    ((s-sigma)^2 + omega^2)."""
    real_poles = np.asarray(real_poles, dtype=float)
    zeros = np.asarray(zeros, dtype=float)
    w2 = omega * omega if omega * omega > 1e-300 else 1e-300
    # Real-pole residues (denominator also carries the CC-pair factor at s=p_i).
    res = np.empty(len(real_poles))
    for i, pi in enumerate(real_poles):
        num = np.prod(pi - zeros) if len(zeros) else 1.0
        den = 1.0
        for k, pk in enumerate(real_poles):
            if k != i:
                den *= (pi - pk)
        den *= ((pi - sigma) ** 2 + w2)
        res[i] = num / (den if abs(den) > 1e-300 else 1e-300)
    # Complex-pair magnitude (= 2|R|) and phase (= arg R), see derivation in FitFakePulse.
    magsq = 1.0
    for z in zeros:
        magsq *= ((sigma - z) ** 2 + w2)
    for pk in real_poles:
        magsq /= ((sigma - pk) ** 2 + w2)
    magsq /= w2
    mag = np.sqrt(magsq) if magsq > 0 else 0.0
    cp = complex(sigma, omega)
    phi = complex(0.0, -1.0)
    for z in zeros:
        phi *= (cp - z)
    for pk in real_poles:
        phi /= (cp - pk)
    phi /= (2.0 * omega if abs(omega) > 1e-300 else 1e-300)
    phase = np.arctan2(phi.imag, phi.real)
    return res, mag, phase


def pulse_model_cc(t, t0, amp, baseline, tilt, real_poles, sigma, omega, zeros):
    """Bolometer pulse with a complex-conjugate pole pair (FitFakePulse, cc=1):
    baseline + tilt ramp, plus for t>t0 the real-pole exponentials and a damped
    oscillation 2|R| e^{sigma dt} cos(omega dt + arg R)."""
    t = np.asarray(t, dtype=float)
    out = baseline + tilt * (t - t0)
    m = t > t0
    if np.any(m):
        res, mag, phase = _cc_terms(real_poles, sigma, omega, zeros)
        dt = t[m] - t0
        s = np.zeros(dt.shape)
        for r_i, p_i in zip(res, real_poles):
            s += r_i * np.exp(np.clip(p_i * dt, -700.0, 0.0))
        s += mag * np.exp(np.clip(sigma * dt, -700.0, 0.0)) * np.cos(omega * dt + phase)
        out[m] += amp * s
    return out


def _model_from_theta(theta, t, nzer, npol, cc):
    """Evaluate the model on t from a flat parameter vector. Layout:
      real (cc=False): [t0, amp, baseline, tilt, zeros(nzer), poles(npol)]
      CC   (cc=True) : [t0, amp, baseline, tilt, zeros(nzer),
                        real_poles(npol-2), sigma, omega]"""
    t0, amp, baseline, tilt = theta[:4]
    zeros = theta[4:4 + nzer]
    pole_block = theta[4 + nzer:4 + nzer + npol]
    if cc:
        real_poles = pole_block[:npol - 2]
        sigma, omega = pole_block[npol - 2], pole_block[npol - 1]
        return pulse_model_cc(t, t0, amp, baseline, tilt, real_poles, sigma, omega, zeros)
    return pulse_model(t, t0, amp, baseline, tilt, pole_block, zeros)


def rise_mask_band(t, v, hi_frac):
    """Banda di ONSET (indici [i_lo, i_hi]) da ESCLUDERE dal fit: da UN PO' PRIMA del piede
    del segnale (dove lascia il rumore di baseline) fino a hi_frac*picco in salita. E' lo
    'stacco', difficile da modellare con una curva liscia. Ritorna (i_lo, i_hi) o None."""
    if hi_frac <= 0:
        return None
    v = np.asarray(v, dtype=float)
    peak = float(v.max())
    imax = int(np.argmax(v))
    up = v[:imax + 1]
    hi = np.where(up >= hi_frac * peak)[0]          # primo campione a hi_frac del picco (fine banda)
    if not len(hi):
        return None
    i_hi = int(hi[0])
    base = v[:int(0.40 * len(v))]
    thr = float(base.mean() + 3.0 * base.std())     # soglia "sopra il rumore" -> piede del segnale
    below = np.where(up[:i_hi] <= thr)[0]
    i_foot = int(below[-1]) + 1 if len(below) else max(i_hi - 1, 0)   # primo campione che cresce
    i_lo = max(i_foot - (i_hi - i_foot), 0)         # un po' prima del piede (larghezza = lo stacco)
    return i_lo, i_hi


def _ring_frequency(v, imax, dt):
    """Frequenza dominante (Hz) dell'oscillazione sulla coda post-picco, per inizializzare
    omega della coppia CC. FFT della coda a cui e' tolta la discesa liscia (media mobile);
    prende il picco dello spettro sopra ~5 Hz (salta il residuo di trend). None se non emerge."""
    seg = v[imax:]
    if len(seg) < 16:
        return None
    k = max(len(seg) // 20, 3)
    trend = np.convolve(seg, np.ones(k) / k, mode="same")   # discesa liscia (media mobile)
    x = (seg - trend) * np.hanning(len(seg))                # residuo oscillante, finestrato
    sp = np.abs(np.fft.rfft(x))
    fr = np.fft.rfftfreq(len(seg), d=dt)
    band = (fr > 5.0) & (fr < 0.25 / dt)                    # sopra il trend, sotto Nyquist/2
    if not np.any(band) or sp[band].max() <= 0:
        return None
    return float(fr[band][np.argmax(sp[band])])


def fit_average_pulse(t, v, nzer=1, npol=3, cc=False, mask_hi_frac=0.0):
    """Fitta UN average pulse (peak-normalizzato) col modello a poli/zeri del bolometro.

    Modello (_model_from_theta):  baseline + tilt*(t-t0) + [t>t0]*amp*Somma_termini_polo
      - cc=False: npol poli reali (ramo fitfuncNpMz di FitPulse.C);
      - cc=True : npol-2 poli reali + 1 coppia complessa coniugata sigma+-i*omega (FitFakePulse).
    theta = [t0, amp, baseline, tilt, zeri(nzer), poli(npol)]; per la CC gli ultimi due
    "poli" sono sigma_cc (<0) e omega_cc (>0).

    Init FISICI + MULTI-START: i poli reali si ancorano alle scale MISURATE (rise, decay),
    gli zeri alla scala di salita, e omega della coppia CC alla FREQUENZA DEL RING misurata
    (FFT della coda). Il fit parte da pochi punti (scale e omega variati) e tiene il migliore:
    cosi' il risultato non dipende da un singolo init fortunato.

    mask_hi_frac>0: toglie dal fit (e dall'RMS) la banda di onset [stacco .. mask_hi_frac*picco].

    Ritorna dict: fit, win, theta, sigma, rms, npol, cc, nzer."""
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    dt = t[1] - t[0]                                   # passo di campionamento
    peak = float(v.max())                             # ampiezza di picco (=1, AP normalizzato)
    imax = int(np.argmax(v))                          # indice del picco

    # 1) SIGMA = RMS del rumore di baseline (primo 40% del record = tutto pre-trigger).
    base = v[:int(0.40 * len(v))]
    base_mean = float(base.mean())
    sigma = float(base.std()) or 1.0                  # guardia se il baseline e' perfettamente piatto

    # 2) ONSET t0 = ultimo campione prima del picco sotto il 5% del picco (dove l'impulso "parte").
    below = np.where(v[:imax] < 0.05 * peak)[0]
    t0g = t[below[-1] if len(below) else max(imax - 10, 0)]

    # 3) SCALE TEMPORALI MISURATE: t_rise (salita 10%->90%), t_dec (discesa a 1/e) e la
    #    frequenza del ring sulla coda (per omega della coppia CC).
    up = v[:imax]
    i10 = np.where(up > 0.1 * peak)[0]
    i90 = np.where(up > 0.9 * peak)[0]
    t_rise = max((t[i90[0]] - t[i10[0]]) if (len(i10) and len(i90)) else 5 * dt, dt)
    be = np.where(v[imax:] < peak / np.e)[0]
    t_dec = (t[imax + be[0]] - t[imax]) if len(be) else 10 * dt
    f_ring = _ring_frequency(v, imax, dt)
    omega0 = 2 * np.pi * f_ring if f_ring else 2 * np.pi / max(t_dec, 5 * dt)
    nreal = npol - 2 if cc else npol                  # numero di poli REALI

    # 4-6) COSTRUTTORE dell'init: poli reali log-spaziati fra tau_fast e tau_slow (ancorati a
    #      rise/decay), zeri sulla scala di salita, coppia CC (sigma=-1/tau_slow, omega dato).
    #      theta = [t0, amp, baseline, tilt, zeri, poli]; amp autoscalato al picco dei dati.
    def build_init(tau_fast, tau_slow, omega):
        zeros0 = -1.0 / np.geomspace(t_rise / 2, t_rise * 2, nzer) if nzer else np.empty(0)
        poles0 = -1.0 / np.geomspace(tau_fast, tau_slow, nreal) if nreal else np.empty(0)
        if cc:
            poles0 = np.concatenate([poles0, [-1.0 / tau_slow, omega]])   # sigma_cc, omega_cc
            pol_lo = np.concatenate([np.full(nreal, -1e5), [-1e5, 1.0]])  # sigma_cc<0, omega_cc>0
            pol_hi = np.concatenate([np.full(nreal, -1e-3), [-1e-3, 1e4]])
        else:
            pol_lo, pol_hi = np.full(npol, -1e5), np.full(npol, -1e-3)
        theta0 = np.concatenate([[t0g, 1.0, base_mean, 0.0], zeros0, poles0])
        lo = np.concatenate([[t0g - 0.005, 1e-9, -1.0, -1e3], np.full(nzer, -1e5), pol_lo])
        hi = np.concatenate([[t0g + 0.005, 1e6, 1.0, 1e3], np.full(nzer, -1e-3), pol_hi])
        theta0 = np.clip(theta0, lo + 1e-12, hi - 1e-12)
        unit = _model_from_theta(theta0, t, nzer, npol, cc)              # autoscala amp al picco
        theta0[1] = peak / (unit.max() if unit.max() > 0 else 1.0)
        return np.clip(theta0, lo + 1e-12, hi - 1e-12), lo, hi

    # 7) FINESTRA DI FIT (uguale per tutti gli start): tutto il record o attorno all'onset,
    #    meno l'eventuale banda di onset mascherata.
    win = np.ones(len(t), bool) if FIT_FULL_WINDOW else (t > t0g - FIT_PRE_S) & (t < t0g + FIT_POST_S)
    if mask_hi_frac > 0:
        band = rise_mask_band(t, v, mask_hi_frac)
        if band is not None:
            win[band[0]:band[1] + 1] = False

    def resid(theta):
        return (_model_from_theta(theta, t[win], nzer, npol, cc) - v[win]) / sigma

    # 8) MULTI-START: pochi start FISICI (scale di rise/decay e, per la CC, omega attorno al
    #    ring misurato). Ogni start e' un least_squares; tengo il fit a RMS minimo.
    if cc:      # per la CC la ω e' l'incognita chiave -> la vario attorno al ring
        starts = [(t_rise, t_dec, omega0), (t_rise, t_dec, 0.5 * omega0),
                  (t_rise, t_dec, 2.0 * omega0), (t_rise / 2, t_dec * 2, omega0)]
    else:       # solo poli reali: vario l'estensione della scanalatura fast..slow
        starts = [(t_rise, t_dec * 3, 0.0), (t_rise / 2, t_dec, 0.0), (t_rise * 2, t_dec * 5, 0.0)]

    best = None
    for tau_fast, tau_slow, omega in starts:
        theta0, lo, hi = build_init(tau_fast, tau_slow, omega)
        r = least_squares(resid, theta0, bounds=(lo, hi), method="trf", max_nfev=5000)
        fit = _model_from_theta(r.x, t, nzer, npol, cc)
        rms = float(np.sqrt(np.mean((fit[win] - v[win]) ** 2)))
        if best is None or rms < best[0]:
            best = (rms, r.x, fit)

    # 9) RISULTATO: il migliore fra gli start.
    rms, theta, fit = best
    return {"fit": fit, "win": win, "theta": theta, "sigma": sigma, "rms": rms,
            "npol": npol, "cc": cc, "nzer": nzer}


def _model_label(npol, cc, nzer):
    """Compact model descriptor, e.g. '3p-z1' or '4p+cc-z0'. Comma-free so the CSV
    'model' column stays a single token even for naive parsers."""
    return f"{npol}p{'+cc' if cc else ''}-z{nzer}"


def fit_best(t, v):
    """Try every model in the NPOL_GRID x CC_GRID x NZER_GRID base grid PLUS the
    EXTRA_MODELS, and keep the one with the smallest RMS. Returns that fit's dict
    (see fit_average_pulse). cc=True needs npol-2 >= 1 real poles."""
    combos = [(npol, cc, nzer)
              for npol in NPOL_GRID for cc in CC_GRID for nzer in NZER_GRID
              if not (cc and npol - 2 < 1)]
    combos += [c for c in EXTRA_MODELS if c not in combos]
    best = None
    for npol, cc, nzer in combos:
        if cc and npol - 2 < 1:
            continue
        try:
            res = fit_average_pulse(t, v, nzer=nzer, npol=npol, cc=cc, mask_hi_frac=0.1)
        except Exception:
            continue
        if best is None or res["rms"] < best["rms"]:
            best = res
    return best


# CSV columns are model-agnostic: the winning model is described by (npol, cc,
# nzer) and its parameter vector is stored verbatim (theta, ';'-joined) so any
# model in the search can be logged in the same file and rebuilt downstream with
# _model_from_theta(theta, nzer, npol, cc).
FIT_CSV_HEADER = ["channel", "wp", "vbias", "model", "npol", "cc", "nzer",
                  "t0", "amp", "baseline", "tilt", "rms", "sigma_baseline", "params"]


def plot_ap_fits(files):
    """One grid per channel: a cell per WP, split into a main panel (peak-normalized
    AP + best-model fit) and a residuals sub-panel (data - fit) below, with the
    +-3 sigma noise band (sigma = baseline RMS), mirroring the two-pad residual
    canvas of FitPulse.C. For each pulse the model minimizing the RMS is chosen from
    the NPOL_GRID x CC_GRID x NZER_GRID search; the panel title reports the winning
    model and its RMS. Writes one PNG per channel (AP_fit_ch<ch>_m205.png) and a CSV
    with every best fit's model descriptor, parameters and RMS (FIT_CSV)."""
    csv_rows = []
    for ch in FIT_CHANNELS:
        fp = files.get(ch)
        if fp is None:
            print(f"Ch {ch}: file not found, skipping AP fit.")
            continue
        with uproot.open(fp) as f:
            wps = sorted(set(
                int(m.group(1)) for k in f.keys()
                for m in [re.search(r"averagepulse_ap_wp(\d+)_medianAP", k)]
                if m and (int(m.group(1)) % 2 != 0)
            ))
            ncols = 5
            nrows = int(np.ceil(len(wps) / ncols))
            fig = plt.figure(figsize=(3.4 * ncols, 3.0 * nrows))
            outer = fig.add_gridspec(nrows, ncols, left=0.06, right=0.985,
                                     top=0.90, bottom=0.07, wspace=0.28, hspace=0.42)
            n_ok = 0
            first = True
            for i, wp in enumerate(wps):
                r, c = divmod(i, ncols)
                inner = outer[r, c].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
                axm = fig.add_subplot(inner[0])
                axr = fig.add_subplot(inner[1], sharex=axm)
                try:
                    h = f[HIST_TMPL.format(wp=wp)]
                    pulse = np.asarray(h.values(), dtype=float)
                    t_s = np.asarray(h.axis().centers(), dtype=float)
                except Exception:
                    axm.axis("off"); axr.axis("off"); continue
                if pulse.max() <= 0 or len(pulse) < 2:
                    axm.axis("off"); axr.axis("off"); continue
                pulse = pulse / pulse.max()                 # peak-normalized, as the spectra
                res = fit_best(t_s, pulse)                  # search over all models, keep min-RMS
                fit = res["fit"]
                theta = res["theta"]
                t0, amp, baseline, tilt = theta[:4]
                model = _model_label(res["npol"], res["cc"], res["nzer"])
                vb = wp_to_vbias(wp)
                params_str = ";".join(f"{x:.8g}" for x in theta)
                csv_rows.append([ch, wp, vb, model, res["npol"], int(res["cc"]),
                                 res["nzer"], t0, amp, baseline, tilt, res["rms"],
                                 res["sigma"], params_str])
                t_peak = t_s[int(np.argmax(pulse))]
                sel = (t_s > t_peak - 0.005) & (t_s < t_peak + 0.12)  # zoom on the pulse
                tm = (t_s[sel] - t_peak) * 1e3                        # ms, relative to peak
                axm.plot(tm, pulse[sel], ".", ms=2.5, color="k", label="AP data")
                axm.plot(tm, fit[sel], "-", lw=1.4, color="crimson", label="fit")
                axm.set_title(f"WP {wp}  ·  {vb:g} V   {model}   rms={res['rms']:.1e}", fontsize=8)
                axm.grid(True, alpha=0.3)
                axm.tick_params(labelsize=7, labelbottom=False)
                # Residuals (data - fit), like the lower pad of FitPulse.C's c2, with
                # the +-3 sigma band (sigma = baseline RMS, FitPulse.C's SetPointError).
                s3 = 3.0 * res["sigma"]
                axr.axhspan(-s3, s3, color="#4a90d9", alpha=0.25, lw=0)
                axr.plot(tm, (pulse[sel] - fit[sel]), "-", lw=0.8, color="#c1121f")
                axr.axhline(0.0, color="gray", ls=":", lw=0.7)
                axr.grid(True, alpha=0.3)
                axr.tick_params(labelsize=6)
                axr.set_ylabel("resid", fontsize=6)
                if first:
                    from matplotlib.lines import Line2D
                    from matplotlib.patches import Patch
                    axm.legend(handles=[
                        Line2D([], [], marker=".", ls="", color="k", label="AP data"),
                        Line2D([], [], color="crimson", lw=1.4, label="fit"),
                        Patch(facecolor="#4a90d9", alpha=0.25, label=r"$\pm3\sigma$"),
                    ], fontsize=7, loc="upper right")
                    first = False
                n_ok += 1
        win_txt = "full window" if FIT_FULL_WINDOW else "onset window"
        n_models = len(NPOL_GRID) * len(CC_GRID) * len(NZER_GRID) + len(EXTRA_MODELS)
        fig.suptitle(f"Average-pulse pole-zero fit (best of {n_models} models by RMS, "
                     f"{win_txt}) — Ch {ch}  ·  Measurement 000205",
                     fontsize=14, fontweight="bold")
        fig.supxlabel("t - t_peak [ms]", fontsize=11, y=0.01)
        fig.supylabel("AP amplitude (peak-normalized)  /  residual", fontsize=11, x=0.005)
        out_png = os.path.join(OUTDIR, f"AP_fit_ch{ch}_m205.png")
        fig.savefig(out_png, dpi=180)
        plt.close(fig)
        print(f"Ch {ch}: {n_ok} WP fits  ->  {os.path.basename(out_png)}")

    # Best-model descriptor + parameters + RMS for downstream (pile-up rejection) use.
    if csv_rows:
        with open(FIT_CSV, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(FIT_CSV_HEADER)
            for row in csv_rows:
                w.writerow([f"{x:.6g}" if isinstance(x, float) else x for x in row])
        print(f"  -> {os.path.basename(FIT_CSV)}  ({len(csv_rows)} fits)")


def plot_ap_shapes(files):
    """Time-domain companion to the power-spectra canvas: one panel per channel, all
    working points overlaid (peak-normalized average pulse), colored by V_bias, with
    a shared colorbar. The pulses are aligned on their peak (t - t_peak) and zoomed
    on the pulse so rise and decay are visible. -> AP_pulses_m205.png"""
    norm = LogNorm(vmin=float(VBIAS_LIST.min()), vmax=float(VBIAS_LIST.max()))
    cmap = plt.get_cmap("viridis")
    ncols = 3
    nrows = int(np.ceil(len(CHANNELS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.6 * nrows), squeeze=False)
    axf = axes.ravel()
    fig.suptitle("Average pulses vs working point — Measurement 000205",
                 fontsize=16, fontweight="bold")
    for ax, ch in zip(axf, CHANNELS):
        fp = files.get(ch)
        if fp is None:
            ax.set_title(f"Ch {ch}  (file not found)")
            continue
        with uproot.open(fp) as f:
            wps = sorted(set(
                int(m.group(1)) for k in f.keys()
                for m in [re.search(r"averagepulse_ap_wp(\d+)_medianAP", k)]
                if m and (int(m.group(1)) % 2 != 0)
            ))
            n_ok = 0
            for wp in wps:
                try:
                    h = f[HIST_TMPL.format(wp=wp)]
                    pulse = np.asarray(h.values(), dtype=float)
                    t_s = np.asarray(h.axis().centers(), dtype=float)
                except Exception:
                    continue
                if pulse.max() <= 0 or len(pulse) < 2:
                    continue
                pulse = pulse / pulse.max()                    # peak-normalized
                t_ms = (t_s - t_s[int(np.argmax(pulse))]) * 1e3  # ms, aligned on the peak
                ax.plot(t_ms, pulse, color=cmap(norm(wp_to_vbias(wp))), lw=0.9, alpha=0.85)
                n_ok += 1
        ax.set_xlim(-5, 120)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Ch {ch}   ({n_ok} WPs)", fontsize=13)
        ax.set_xlabel("t - t_peak [ms]")
        ax.set_ylabel("AP amplitude (peak-normalized)")
        ax.grid(True, alpha=0.3)
        print(f"Ch {ch}: {n_ok} WP pulses")
    for ax in axf[len(CHANNELS):]:
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 0.94, 0.96])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cax = fig.add_axes([0.95, 0.12, 0.012, 0.76])
    fig.colorbar(sm, cax=cax, label=r"$V_{bias}$ (V)")
    fig.savefig(OUT_AP_PNG, dpi=200)
    plt.close(fig)
    print(f"\n  -> {OUT_AP_PNG}")


def plot_anps(files):
    """Average noise power spectrum (ANPS) overview, same style as the AP power
    spectra: one panel per channel, the stored one-sided noise PSD (V^2/Hz) of every
    WP overlaid, colored by V_bias, log-log, with a shared colorbar. The ANPS is read
    directly from the ROOT histograms (no FFT). -> ANPS_m205.png"""
    norm = LogNorm(vmin=float(VBIAS_LIST.min()), vmax=float(VBIAS_LIST.max()))
    cmap = plt.get_cmap("viridis")
    ncols = 3
    nrows = int(np.ceil(len(CHANNELS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.6 * nrows), squeeze=False)
    axf = axes.ravel()
    fig.suptitle("Average noise power spectra (ANPS) vs working point — Measurement 000205",
                 fontsize=16, fontweight="bold")
    for ax, ch in zip(axf, CHANNELS):
        fp = files.get(ch)
        if fp is None:
            ax.set_title(f"Ch {ch}  (file not found)")
            continue
        with uproot.open(fp) as f:
            wps = sorted(set(
                int(m.group(1)) for k in f.keys()
                for m in [re.search(r"averagepowerspectrum_noise_wp(\d+)_medianpower", k)]
                if m and (int(m.group(1)) % 2 != 0)
            ))
            n_ok = 0
            for wp in wps:
                try:
                    h = f[HIST_NPS_TMPL.format(wp=wp)]
                    nps = np.asarray(h.values(), dtype=float)
                    freq = np.asarray(h.axis().centers(), dtype=float)   # Hz, one-sided
                except Exception:
                    continue
                if len(nps) < 2 or nps.max() <= 0:
                    continue
                ax.loglog(freq, nps, color=cmap(norm(wp_to_vbias(wp))), lw=0.9, alpha=0.85)
                n_ok += 1
        ax.set_title(f"Ch {ch}   ({n_ok} WPs)", fontsize=13)
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel(r"noise PSD [V$^2$/Hz]")
        ax.grid(True, which="both", alpha=0.3)
        print(f"Ch {ch}: {n_ok} WP noise spectra")
    for ax in axf[len(CHANNELS):]:
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 0.94, 0.96])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cax = fig.add_axes([0.95, 0.12, 0.012, 0.76])
    fig.colorbar(sm, cax=cax, label=r"$V_{bias}$ (V)")
    fig.savefig(OUT_ANPS_PNG, dpi=200)
    plt.close(fig)
    print(f"\n  -> {OUT_ANPS_PNG}")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    files = {}
    for fp in glob.glob(os.path.join(DATADIR, PATTERN)):
        try:
            ch = int(os.path.basename(fp).split("_")[-1].replace(".root", ""))
        except ValueError:
            continue
        if ch in CHANNELS:
            files[ch] = fp

    norm = LogNorm(vmin=float(VBIAS_LIST.min()), vmax=float(VBIAS_LIST.max()))
    cmap = plt.get_cmap("viridis")

    ncols = 3
    nrows = int(np.ceil(len(CHANNELS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.6 * nrows), squeeze=False)
    axf = axes.ravel()
    fig.suptitle("Average-pulse power spectra vs working point — Measurement 000205",
                 fontsize=16, fontweight="bold")

    for ax, ch in zip(axf, CHANNELS):
        fp = files.get(ch)
        if fp is None:
            ax.set_title(f"Ch {ch}  (file not found)")
            continue
        with uproot.open(fp) as f:
            wps = sorted(set(
                int(m.group(1)) for k in f.keys()
                for m in [re.search(r"averagepulse_ap_wp(\d+)_medianAP", k)]
                if m and (int(m.group(1)) % 2 != 0)
            ))
            n_ok = 0
            for wp in wps:
                try:
                    h = f[HIST_TMPL.format(wp=wp)]
                    pulse = np.asarray(h.values(), dtype=float)
                    t_s = np.asarray(h.axis().centers(), dtype=float)
                except Exception:
                    continue
                if pulse.max() <= 0 or len(pulse) < 2:
                    continue
                pulse = pulse / pulse.max()
                sr = 1.0 / (t_s[1] - t_s[0])
                fr, ps = compute_psd(pulse, sr)
                ax.loglog(fr[1:], ps[1:], color=cmap(norm(wp_to_vbias(wp))),
                          lw=0.9, alpha=0.85)
                n_ok += 1
        ax.axvline(HF_CUT_HZ, color="crimson", ls="--", lw=1.0, alpha=0.7,
                   label="HF cut (500 Hz)")
        ax.set_title(f"Ch {ch}   ({n_ok} WPs)", fontsize=13)
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel("PSD [a.u.]")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9, loc="lower left")
        print(f"Ch {ch}: {n_ok} WP spectra")

    for ax in axf[len(CHANNELS):]:
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 0.94, 0.96])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cax = fig.add_axes([0.95, 0.12, 0.012, 0.76])
    fig.colorbar(sm, cax=cax, label=r"$V_{bias}$ (V)")
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)
    print(f"\n  -> {OUT_PNG}")

    # Time-domain AP overview (all WPs overlaid, one panel per channel).
    print("\nPlotting average-pulse shapes (all WPs per channel):")
    plot_ap_shapes(files)

    # Noise power spectra overview (same style as the AP spectra).
    print("\nPlotting average noise power spectra (all WPs per channel):")
    plot_anps(files)

    # AP model fit (paused by default): one grid per fit-channel (AP data + fit).
    if DO_FIT:
        print("\nFitting average pulses (pole-zero model) and plotting per-channel grids:")
        plot_ap_fits(files)


if __name__ == "__main__":
    main()
