#!/usr/bin/env python3
"""
compare_AP_noise_ch34_ch91_m205.py
----------------------------------
Per-working-point comparison of two channels (default 34 vs 91) of run m205,
to understand where they differ. For EACH working point it produces one figure
with 5 panels:

  1. AP ch34 vs ch91 in the TIME domain (average-pulse waveform, zoomed on the pulse)
  2. AP ch34 vs ch91 in the FREQUENCY domain (AP power spectrum, APPS)
  3. Noise PSD ch34 vs ch91 (ANPS)
  4. ch34: reliability diagnostic — APPS, ANPS/N, beta*ANPS/N + R(f) + soft cutoff f*
  5. ch91: reliability diagnostic (same)
  6. R(f) of both channels overlaid (soft-bandwidth comparison)

The reliability R(f) is the template-regularization factor (estimator A of the
"template_reliability_optimal_filter" note): R = Sabove/(Sabove + beta*ANPS/N + eps),
Sabove = max(APPS - beta*ANPS/N, 0). R~1 where the average-pulse power clearly exceeds
the TEMPLATE noise floor ANPS/N (N = events in the average), R~0 where the bin is
compatible with residual noise. This shows, BEFORE touching the training, where such a
regularization would act (the soft cutoff f* is where smoothed R drops below 0.5).

Physical scaling (so everything is in V^2/Hz and directly comparable):
  - APPS: the peak-normalized AP is multiplied by the amplitude used in the pile-up
    rejection algorithm (amplitude_mV from amplitudes_m205.csv, matched per
    channel & V_bias), THEN its power spectrum is taken.
  - ANPS: the stored noise PSD is multiplied by the flatt-op windowing factor.

Data source (same as plot_BI_results.py): the Octopus ROOT files in Processed/,
one per channel (Processed_*_000205_<ch>.root), each holding, per WP:
  - averagepulse_ap_wp<wp>_medianAP           (AP, time domain, peak-normalized ~1)
  - averagepowerspectrum_noise_wp<wp>_medianpower  (noise PSD, one-sided)

Run with either python (no ROOT needed):
  KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/bin/python3.13 compare_AP_noise_ch34_ch91_m205.py
"""

import os, glob, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

# ── User settings ──────────────────────────────────────────────────────────────
BASE          = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE, "Processed")
AMP_CSV       = os.path.join(BASE, "amplitudes_m205.csv")   # amplitude used in pile-up rejection
OUTDIR        = os.path.join(BASE, "AP_noise_compare_m205")
MEAS_NAME     = "000205"
SAMPLING_RATE = 10_000.0                       # m205: 10 kHz

CH_A = 34                                      # first channel (blue)
CH_B = 91                                      # second channel (red)
WP_LIST = list(range(1, 30, 2))                # odd working points 1,3,...,29 (15 WPs)

# Flat-top windowing factor by which the stored noise PSD must be multiplied to be
# in true V^2/Hz (same constant used in plot_BI_results.py: _load_ap_nps).
FLATTOP_FACTOR = 5.708

# Time-domain zoom around the pulse peak (seconds before / after the peak)
ZOOM_PRE_S  = 0.02
ZOOM_POST_S = 0.12

# ── Template-reliability diagnostic (regularization R) ────────────────────────
N_EVENTS    = 38       # numero di eventi nel medio (N): rumore del template = ANPS/N
BETA_R      = 2.0      # fattore di conservativita' beta (1.5-2 template puliti, 3-5 per N basso)
EPS_R_FRAC  = 1e-6     # epsilon (frazione del picco APPS) per stabilita' del denominatore
SMOOTH_R_HZ = 100.0    # larghezza (Hz) di smoothing di R in logit (50-150 Hz consigliati)
R_COL       = "#2ca02c"  # colore della curva R

# ── Overlay del filtro TOTALE addestrato |f_j·W_unit| (lambda SCALARE) ────────────
#   f1,f2 dai .npy salvati; lambda_wiener (scalare, addestrata) dal CSV del training;
#   W_unit ricostruito come nel training (compute_W_torch, invariante alla scala dell'nps).
SHOW_FILTER    = True
FILTER_DIR     = os.path.join(BASE, "m205_results_wiener", "trained_filters")
WIENER_CSV     = os.path.join(BASE, "m205_results_wiener", "BI_results_m205_wiener.csv")
F1_COL, F2_COL = "#9467bd", "#ff7f0e"

# V_bias look-up table (indexed by WP // 2), odd wp only
VBIAS_LIST = np.array([0.6, 1.0, 1.4, 1.8, 2, 3, 4, 5, 6, 8, 10, 20, 26, 30, 40])

COL = {CH_A: "#1f77b4", CH_B: "#d62728"}       # per-channel colors
NOISE_COL = "#555555"                          # noise color in the overlay panels


def wp_to_vbias(wp):
    return VBIAS_LIST[wp // 2]


def read_amplitudes(path):
    """Map (channel, round(V_bias,3)) -> amplitude in VOLTS (from amplitude_mV)."""
    amap = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                ch = int(float(row["channel"])); vb = float(row["vbias_V"])
                amp_mV = float(row["amplitude_mV"])
            except (TypeError, ValueError, KeyError):
                continue
            amap[(ch, round(vb, 3))] = amp_mV * 1e-3    # mV -> V
    return amap


def ap_power_spectrum(ap_phys, fs):
    """One-sided PSD (V^2/Hz) of the physical average pulse, in the SAME convention
    as the Octopus spectra: X_norm = rfft/N, power = 2*|X_norm|^2/df with df = fs/N,
    where the factor 2 (one-sided) is applied ONLY to the AC bins — DC and Nyquist
    keep factor 1, exactly like Waveform::ComputePolarCoordinates.
    No window: the AP is a localized transient (baseline at both edges), so there is
    no leakage to suppress; a rectangular periodogram gives the true pulse PSD, which
    matches the window-corrected noise ANPS (both are true V^2/Hz)."""
    N = len(ap_phys)
    X = np.fft.rfft(ap_phys)
    df = fs / N
    psd = (np.abs(X) ** 2) / (N * N) / df    # = |X_norm|^2 / df   (DC/Nyquist factor 1)
    psd[1:] *= 2.0                            # one-sided: double the AC bins ...
    if N % 2 == 0:
        psd[-1] /= 2.0                        # ... but Nyquist (even N) stays factor 1
    freq = np.fft.rfftfreq(N, d=1.0 / fs)
    return freq, psd


def reliability(apps, tnoise, beta, eps):
    """Fattore di affidabilità R(f) in [0,1] (estimatore A del PDF template-reliability):
        Sabove = max(apps - beta*tnoise, 0)
        R      = Sabove / (Sabove + beta*tnoise + eps)
    apps = potenza del segnale |S|^2 ; tnoise = ANPS/N = rumore del template."""
    thr = beta * tnoise
    s_above = np.maximum(apps - thr, 0.0)
    return s_above / (s_above + thr + eps)


def smooth_logit(R, df, width_hz, eps=1e-6):
    """Liscia R lisciando il suo logit su ~width_hz (evita di aggrapparsi ai singoli bin)."""
    n = max(1, int(round(width_hz / df)))
    z = np.log((R + eps) / (1.0 - R + eps))
    z = np.convolve(z, np.ones(n) / n, mode="same")
    return 1.0 / (1.0 + np.exp(-z))


def smooth_pos(x, n_bins):
    """Media mobile (per de-spikare le densità, che hanno le righe di rumore)."""
    n = max(1, int(n_bins))
    return np.convolve(x, np.ones(n) / n, mode="same")


def channel_reliability(fp, ps, fn, nz, N, beta, eps_frac, smooth_hz):
    """R(f) per un canale: interpola l'ANPS sulla griglia dell'APPS, divide per N
    (rumore del template), calcola R e il suo smoothing, e il cutoff soft f* (ultima
    frequenza dove R_smooth >= 0.5). Ritorna (tnoise=ANPS/N, R, R_smooth, f*)."""
    anps = np.interp(fp, fn, nz)                 # ANPS sulla griglia dell'APPS
    tnoise = anps / N                            # rumore del template = ANPS/N
    eps = eps_frac * float(np.nanmax(ps))
    R = reliability(ps, tnoise, beta, eps)
    Rs = smooth_logit(R, fp[1] - fp[0], smooth_hz)
    idx = np.where(Rs >= 0.5)[0]
    fstar = float(fp[idx.max()]) if len(idx) else np.nan
    return tnoise, R, Rs, fstar


def load_lambda_wiener(ch, wp):
    """Lambda SCALARE addestrata (colonna 'lambda_wiener') dal CSV del training wiener,
    per (ch, wp). Ritorna float o None."""
    try:
        with open(WIENER_CSV, newline="") as f:
            for row in csv.DictReader(f):
                if int(float(row["channel"])) == ch and int(float(row["wp"])) == wp:
                    return float(row["lambda_wiener"])
    except (FileNotFoundError, KeyError, ValueError):
        pass
    return None


def applied_filter(ap, nps_oneside, ch, wp):
    """Filtro TOTALE addestrato |F_j| = |f_j · W_unit| (one-sided), con la lambda SCALARE.
    W_unit ricostruito come nel training: W = conj(S)/(|S|^2_n + lam*NPS_n) con la
    normalizzazione di compute_W_torch (a=sum|S|, b=sum sqrt(NPS)); questa e' INVARIANTE
    a una scala costante di NPS, quindi va bene l'nps del compare-program. Ritorna
    (|F1|,|F2|) sui 5001 bin one-sided, oppure None se mancano i file."""
    lam = load_lambda_wiener(ch, wp)
    try:
        f1 = np.load(os.path.join(FILTER_DIR, f"f1_ch{ch}_wp{wp}.npy")).astype(float)
        f2 = np.load(os.path.join(FILTER_DIR, f"f2_ch{ch}_wp{wp}.npy")).astype(float)
    except FileNotFoundError:
        return None
    if lam is None:
        return None
    n = len(ap)
    S = np.fft.fft(ap * np.hanning(n))                        # spettro pieno del meanpulse
    nps_full = np.concatenate([nps_oneside, nps_oneside[-2:0:-1]])
    AvgPS = np.abs(S) ** 2
    a = np.sum(np.sqrt(AvgPS)); b = np.sum(np.sqrt(nps_full))
    W = np.conj(S) / (AvgPS / a + lam * nps_full * (a / b ** 2))
    W[0] = 0.0
    Wh = W[:n // 2 + 1]                                       # half-spectrum (come f1,f2)
    return f1 * Wh, f2 * Wh, S[:n // 2 + 1]                   # F1, F2 (complessi) e S (half)


def load_channel(ch):
    files = glob.glob(os.path.join(PROCESSED_DIR, f"Processed_*_{MEAS_NAME}_{ch}.root"))
    if not files:
        raise SystemExit(f"[ERROR] no ROOT file for channel {ch} in {PROCESSED_DIR}")
    return uproot.open(files[0])


def read_ap(f, wp):
    h = f[f"averagepulse_ap_wp{wp}_medianAP"]
    return np.asarray(h.axis().centers(), float), np.asarray(h.values(), float)


def read_noise(f, wp):
    """Noise PSD in V^2/Hz = stored hist × flat-top windowing factor. The frequency
    axis uses the bin LEFT EDGES (edges()[:-1]) instead of the centers, so bin k sits
    at k·binwidth ≈ k·df and aligns with the DFT frequencies of the APPS (no half-bin
    offset)."""
    h = f[f"averagepowerspectrum_noise_wp{wp}_medianpower"]
    freq = np.asarray(h.axis().edges(), float)[:-1]     # left edges, N bins
    return freq, np.asarray(h.values(), float) * FLATTOP_FACTOR


def has_wp(f, wp):
    return (f"averagepulse_ap_wp{wp}_medianAP" in f) and \
           (f"averagepowerspectrum_noise_wp{wp}_medianpower" in f)


def zoom_xlim(t, apA, apB):
    """[t_lo, t_hi] window around the pulse peak(s) of the two channels."""
    tpA = t[int(np.argmax(np.abs(apA)))]
    tpB = t[int(np.argmax(np.abs(apB)))]
    return max(t[0], min(tpA, tpB) - ZOOM_PRE_S), min(t[-1], max(tpA, tpB) + ZOOM_POST_S)


# ── Main ─────────────────────────────────────────────────────────────────────
os.makedirs(OUTDIR, exist_ok=True)
amp_of = read_amplitudes(AMP_CSV)
fA, fB = load_channel(CH_A), load_channel(CH_B)
print(f"Comparing ch{CH_A} vs ch{CH_B} — {len(WP_LIST)} working points\n")

n_done = 0
for wp in WP_LIST:
    if not (has_wp(fA, wp) and has_wp(fB, wp)):
        print(f"  [skip] WP{wp}: missing hist in ch{CH_A} or ch{CH_B}")
        continue
    vb = wp_to_vbias(wp)
    ampA = amp_of.get((CH_A, round(float(vb), 3)))
    ampB = amp_of.get((CH_B, round(float(vb), 3)))
    if ampA is None or ampB is None:
        print(f"  [skip] WP{wp} ({vb:g} V): amplitude missing for ch{CH_A} or ch{CH_B}")
        continue

    tA, apA = read_ap(fA, wp);  tB, apB = read_ap(fB, wp)
    # Scale the peak-normalized AP by the pile-up-rejection amplitude, then power spectrum
    fpA, psA = ap_power_spectrum(apA * ampA, SAMPLING_RATE)
    fpB, psB = ap_power_spectrum(apB * ampB, SAMPLING_RATE)
    fnA, nA = read_noise(fA, wp);  fnB, nB = read_noise(fB, wp)

    # Reliability R(f) per canale (ANPS/N, R, R lisciato, cutoff soft f*)
    tnoiseA, RA, RsA, fstarA = channel_reliability(fpA, psA, fnA, nA, N_EVENTS, BETA_R, EPS_R_FRAC, SMOOTH_R_HZ)
    tnoiseB, RB, RsB, fstarB = channel_reliability(fpB, psB, fnB, nB, N_EVENTS, BETA_R, EPS_R_FRAC, SMOOTH_R_HZ)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    lblA, lblB = f"Ch {CH_A}", f"Ch {CH_B}"

    # 1) AP time domain (peak-normalized, zoomed on the pulse)
    ax = axes[0, 0]
    ax.plot(tA, apA, color=COL[CH_A], lw=1.4, label=lblA)
    ax.plot(tB, apB, color=COL[CH_B], lw=1.4, label=lblB)
    ax.set_xlim(*zoom_xlim(tA, apA, apB))
    ax.set_xlabel("Time (s)"); ax.set_ylabel("AP (peak-normalized)")
    ax.set_title("Average pulse — time domain (zoom)")
    ax.grid(True, ls="--", alpha=0.5); ax.legend()

    # 2) APPS ch34 vs ch91
    ax = axes[0, 1]
    ax.loglog(fpA[1:], psA[1:], color=COL[CH_A], lw=1.1, label=lblA)
    ax.loglog(fpB[1:], psB[1:], color=COL[CH_B], lw=1.1, label=lblB)
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel(r"APPS (V$^2$/Hz)")
    ax.set_title("Average-pulse power spectrum")
    ax.grid(True, which="both", ls="--", alpha=0.4); ax.legend()

    # 3) ANPS ch34 vs ch91
    ax = axes[0, 2]
    ax.loglog(fnA, nA, color=COL[CH_A], lw=1.1, label=lblA)
    ax.loglog(fnB, nB, color=COL[CH_B], lw=1.1, label=lblB)
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel(r"ANPS (V$^2$/Hz)")
    ax.set_title("Average noise power spectrum")
    ax.grid(True, which="both", ls="--", alpha=0.4); ax.legend()

    # 4) & 5) Reliability diagnostic per channel: APPS, ANPS/N, beta*ANPS/N (left, log)
    #         + R(f) (right, linear 0..1) + cutoff soft f*
    for ax, ch, fp, ps, tnoise, R, Rs, fstar, ap_ch, nps1_ch in [
        (axes[1, 0], CH_A, fpA, psA, tnoiseA, RA, RsA, fstarA, apA, nA),
        (axes[1, 1], CH_B, fpB, psB, tnoiseB, RB, RsB, fstarB, apB, nB),
    ]:
        ax.loglog(fp[1:], ps[1:], color=COL[ch], lw=1.6, alpha=0.9, label="APPS (signal)")
        ax.loglog(fp[1:], tnoise[1:], color=NOISE_COL, lw=1.3, alpha=0.75,
                  label=f"ANPS/N  (N={N_EVENTS})")
        ax.loglog(fp[1:], (BETA_R * tnoise)[1:], color=NOISE_COL, lw=1.1, ls="--", alpha=0.75,
                  label=fr"$\beta\cdot$ANPS/N  ($\beta$={BETA_R:g})")
        ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel(r"PSD (V$^2$/Hz)")
        ax.set_title(f"Ch {ch}: reliability R  +  trained filter")
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.legend(loc="lower left", fontsize=8)
        axr = ax.twinx()                                   # R e filtri su asse lineare 0..1
        axr.semilogx(fp[1:], R[1:], color=R_COL, lw=0.7, alpha=0.30)
        axr.semilogx(fp[1:], Rs[1:], color=R_COL, lw=1.9, label="R(f)")
        # overlay: DOVE il filtro addestrato prende il SEGNALE e DOVE lascia passare RUMORE
        #   segnale in uscita  = |F·S|^2   (F = f_j·W_unit) ;  rumore in uscita = |F|^2·NPS
        #   sommati su f1,f2, lisciati e normalizzati al proprio picco (conta la posizione).
        if SHOW_FILTER:
            ff = applied_filter(ap_ch, nps1_ch, ch, wp)
            if ff is not None:
                F1c, F2c, Sh = ff
                nsm = max(1, int(SMOOTH_R_HZ / (fp[1] - fp[0])))
                sig_out = smooth_pos(np.abs(F1c * Sh) ** 2 + np.abs(F2c * Sh) ** 2, nsm)
                noi_out = smooth_pos((np.abs(F1c) ** 2 + np.abs(F2c) ** 2) * nps1_ch, nsm)
                axr.semilogx(fp[1:], (sig_out / np.nanmax(sig_out))[1:], color=F1_COL, lw=1.5,
                             alpha=0.9, label=r"signal out $|F\!\cdot\!S|^2$ (norm)")
                axr.semilogx(fp[1:], (noi_out / np.nanmax(noi_out))[1:], color=F2_COL, lw=1.5,
                             alpha=0.9, label=r"noise out $|F|^2$NPS (norm)")
        axr.set_ylim(-0.03, 1.05); axr.set_ylabel("R(f),  filter output (norm)")
        axr.legend(loc="upper right", fontsize=7.5)
        if np.isfinite(fstar):
            axr.axvline(fstar, color=R_COL, ls=":", lw=1.3)
            axr.text(fstar, 0.06, f" f*≈{fstar:.0f} Hz", color=R_COL, fontsize=8.5, ha="left")

    # 6) R(f) di entrambi i canali sovrapposte (confronto banda passante soft)
    ax = axes[1, 2]
    ax.semilogx(fpA[1:], RsA[1:], color=COL[CH_A], lw=1.9, label=f"Ch {CH_A}  (f*≈{fstarA:.0f} Hz)")
    ax.semilogx(fpB[1:], RsB[1:], color=COL[CH_B], lw=1.9, label=f"Ch {CH_B}  (f*≈{fstarB:.0f} Hz)")
    ax.axhline(0.5, color="gray", ls=":", lw=0.9)
    ax.set_xlim(1, SAMPLING_RATE / 2); ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("R(f) smoothed")
    ax.set_title(f"Soft bandwidth R(f)  —  N={N_EVENTS}, $\\beta$={BETA_R:g}")
    ax.grid(True, which="both", ls="--", alpha=0.4); ax.legend(fontsize=9)

    fig.suptitle(f"m205 — Ch {CH_A} vs Ch {CH_B} — WP {wp}  ·  $V_{{bias}}$ = {vb:g} V",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = os.path.join(OUTDIR, f"ap_noise_compare_ch{CH_A}_ch{CH_B}_wp{wp}.png")
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    n_done += 1
    print(f"  → {os.path.basename(out_png)}  (V_bias {vb:g} V)  "
          f"f*: ch{CH_A}≈{fstarA:.0f} Hz  ch{CH_B}≈{fstarB:.0f} Hz")

fA.close(); fB.close()
print(f"\n✓ {n_done} figures saved in {OUTDIR}")
