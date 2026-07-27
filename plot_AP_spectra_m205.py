"""
plot_AP_spectra_m205.py
=======================
Average-pulse diagnostics for the m205 load curves. Two products:

  1. AP power spectra: one panel per channel, overlaying the AP power spectrum of
     every working point (WP), colored by V_bias. Same PSD definition used in the
     m204 study (peak-normalized AP, Hann window). The dashed line marks the HF
     cutoff (500 Hz) used for HF-power.        -> AP_power_spectra_m205.png

  2. AP MODEL FIT: each average pulse is fitted with the bolometer pole-zero model
     ported from FitPulse.C: a rational transfer function with NPOL poles and NZER
     zeros, inverse-transformed to a residue-weighted sum of (exponential, or damped
     oscillation) terms,

         f(t) = baseline + tilt*(t - t0) + [t > t0] * amp * Sum(pole terms) .

     CC=False uses the all-real-pole branch (fitfuncNpMz); CC=True uses NPOL-2 real
     poles plus one complex-conjugate pair sigma +- i*omega (FitFakePulse), a damped
     ringing term for detectors whose pulse oscillates.

     By default the fit spans the WHOLE recorded window (FIT_FULL_WINDOW), so the
     result is usable as a template for the pile-up rejection. For each channel a
     GRID (a cell per WP) shows, like FitPulse.C's two-pad canvas, the AP + fit on
     top and the residuals (data - fit) below; the fit RMS is in each panel title.
                                                    -> AP_fit_ch<ch>_m205.png
     The per-fit parameters and RMS are also written to a CSV, keyed by (channel,
     WP), to rebuild the template downstream.       -> AP_fit_params_m205.csv

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
OUT_PNG = os.path.join(OUTDIR, "AP_power_spectra_m205.png")

CHANNELS = [31, 34, 71, 83, 91]
VBIAS_LIST = np.array([0.6, 1.0, 1.4, 1.8, 2, 3, 4, 5, 6, 8, 10, 20, 26, 30, 40])
HF_CUT_HZ = 500.0
HIST_TMPL = "averagepulse_ap_wp{wp}_medianAP"

# ── AP fit (pole-zero bolometer model, ported from FitPulse.C) ─────────────────
# NPOL poles + NZER real zeros. CC selects which FitPulse.C branch is ported:
#   CC=False -> fitfuncNpMz  : all NPOL poles are real.
#   CC=True  -> FitFakePulse : NPOL-2 real poles + one complex-conjugate pair
#               (sigma +- i*omega), i.e. a damped oscillation, for detectors whose
#               pulse rings. NPOL=4 -> 2 real poles + 1 CC pair.
# The AP is peak-normalized (=1) before fitting, as in the spectra above.
NPOL = 3
NZER = 1
CC   = False
# FIT_FULL_WINDOW=True fits the ENTIRE recorded window (all samples) — the intended
# use is a fit template for the pile-up rejection, which must be valid over the
# whole record. False restricts the fit to a window around the onset (FIT_PRE_S
# before, FIT_POST_S after), useful to isolate the pulse from the flat baseline.
FIT_FULL_WINDOW = True
FIT_PRE_S  = 0.02    # (FIT_FULL_WINDOW=False) seconds of pre-onset baseline in the window
FIT_POST_S = 0.25    # (FIT_FULL_WINDOW=False) seconds after onset (decay down to <1%)

# CSV with the per-(channel, WP) fit parameters and RMS, for reuse by the pile-up-
# rejection algorithm.
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


def fit_average_pulse(t, v, nzer=NZER, npol=NPOL, cc=CC):
    """Fit one peak-normalized average pulse with the pole-zero model (real poles,
    cc=False, or NPOL-2 real poles + one complex-conjugate pair, cc=True). Initial
    guesses/bounds follow FitPulse.C but are adapted per pulse to the m205 data
    (onset t0, rise time, 1/e decay). The fit spans the whole record when
    FIT_FULL_WINDOW is True (template for pile-up rejection), else a window around
    the onset.

    Returns a dict: fit (on the whole t axis), win (fit mask), theta (params),
    sigma (pre-onset baseline RMS), rms (fit residual RMS over the window)."""
    peak = float(v.max())
    imax = int(np.argmax(v))
    dt = t[1] - t[0]

    # Onset t0: last sample before the peak that is below 5% of the peak.
    below = np.where(v[:imax] < 0.05 * peak)[0]
    i0 = below[-1] if len(below) else max(imax - 10, 0)
    t0g = t[i0]

    # Rise (10->90%) and 1/e decay set the fast/slow timescales for the pole init.
    up = v[:imax]
    i10 = np.where(up > 0.1 * peak)[0]
    i90 = np.where(up > 0.9 * peak)[0]
    t_rise = (t[i90[0]] - t[i10[0]]) if (len(i10) and len(i90)) else 5 * dt
    t_rise = max(t_rise, dt)
    after = v[imax:]
    be = np.where(after < peak / np.e)[0]
    t_dec = (t[imax + be[0]] - t[imax]) if len(be) else 10 * dt

    zeros0 = -1.0 / np.geomspace(t_rise / 2.0, t_rise * 2.0, nzer) if nzer else np.array([])
    base0 = float(np.mean(v[:max(i0 - 5, 1)]))
    fast, slow = t_rise / 3.0, max(t_dec * 5, t_rise * 10)

    # Pole block init/bounds: real poles, or (real poles, sigma, omega) for the CC
    # pair. omega is bounded away from 0 (the CC residue formula divides by it).
    if cc:
        nreal = npol - 2
        real0 = -1.0 / np.geomspace(fast, t_dec * 2, nreal) if nreal else np.array([])
        sigma0 = -1.0 / slow                        # slow-decay real part of the CC pole
        omega0 = 2.0 * np.pi / max(t_dec, 5 * dt)   # ~one ring over the decay time
        poles0 = np.concatenate([real0, [sigma0, omega0]])
        lo_p = np.concatenate([np.full(nreal, -1e5), [-1e5, 1.0]])
        hi_p = np.concatenate([np.full(nreal, -1e-3), [-1e-3, 1e4]])
    else:
        poles0 = -1.0 / np.geomspace(fast, slow, npol)
        lo_p = np.full(npol, -1e5)
        hi_p = np.full(npol, -1e-3)

    # Per-sample uncertainty = pre-onset baseline RMS (constant, as in FitPulse.C).
    sigma = float(np.std(v[:i0])) if i0 > 10 else float(np.std(v[:max(imax, 50)]))
    if not sigma > 0:
        sigma = 1.0

    win = np.ones(len(t), dtype=bool) if FIT_FULL_WINDOW \
        else (t > t0g - FIT_PRE_S) & (t < t0g + FIT_POST_S)

    theta0 = np.concatenate([[t0g, 1.0, base0, 0.0], zeros0, poles0])
    lo = np.concatenate([[t0g - 0.005, 1e-9, -1.0, -1e3], np.full(nzer, -1e5), lo_p])
    hi = np.concatenate([[t0g + 0.005, 1e6, 1.0, 1e3], np.full(nzer, -1e-3), hi_p])
    theta0 = np.clip(theta0, lo + 1e-12, hi - 1e-12)
    # Amplitude init so the unit-amplitude model peak matches the data peak.
    unit = _model_from_theta(theta0, t, nzer, npol, cc)
    theta0[1] = peak / (unit.max() if unit.max() > 0 else 1.0)
    theta0 = np.clip(theta0, lo + 1e-12, hi - 1e-12)

    def resid(theta):
        return (_model_from_theta(theta, t[win], nzer, npol, cc) - v[win]) / sigma

    r = least_squares(resid, theta0, bounds=(lo, hi), method="trf", max_nfev=5000)
    fit = _model_from_theta(r.x, t, nzer, npol, cc)
    rms = float(np.sqrt(np.mean((fit[win] - v[win]) ** 2)))
    return {"fit": fit, "win": win, "theta": r.x, "sigma": sigma, "rms": rms}


def _fit_csv_header():
    """CSV columns, sized by NPOL/NZER/CC. For CC the pole block is NPOL-2 real
    poles followed by the complex pair (sigma_cc, omega_cc)."""
    cols = ["channel", "wp", "vbias", "t0", "amp", "baseline", "tilt"]
    if CC:
        cols += [f"pole{i}" for i in range(NPOL - 2)] + ["sigma_cc", "omega_cc"]
    else:
        cols += [f"pole{i}" for i in range(NPOL)]
    cols += [f"zero{j}" for j in range(NZER)]
    cols += ["sigma_baseline", "rms"]
    return cols


def plot_ap_fits(files):
    """One grid per channel: a cell per WP, split into a main panel (peak-normalized
    AP + pole-zero fit) and a residuals sub-panel (data - fit) below, with the
    +-3 sigma noise band (sigma = baseline RMS), mirroring the two-pad residual
    canvas of FitPulse.C. Panel title reports the fit RMS. Writes
    one PNG per channel (AP_fit_ch<ch>_m205.png) and a CSV with every fit's
    parameters + RMS (FIT_CSV)."""
    csv_rows = []
    for ch in CHANNELS:
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
                res = fit_average_pulse(t_s, pulse)
                fit = res["fit"]
                theta = res["theta"]
                t0, amp, baseline, tilt = theta[:4]
                zeros = theta[4:4 + NZER]
                pole_block = theta[4 + NZER:4 + NZER + NPOL]
                vb = wp_to_vbias(wp)
                csv_rows.append([ch, wp, vb, t0, amp, baseline, tilt,
                                 *pole_block, *zeros, res["sigma"], res["rms"]])
                t_peak = t_s[int(np.argmax(pulse))]
                sel = (t_s > t_peak - 0.005) & (t_s < t_peak + 0.12)  # zoom on the pulse
                tm = (t_s[sel] - t_peak) * 1e3                        # ms, relative to peak
                axm.plot(tm, pulse[sel], ".", ms=2.5, color="k", label="AP data")
                axm.plot(tm, fit[sel], "-", lw=1.4, color="crimson", label="fit")
                axm.set_title(f"WP {wp}  ·  {vb:g} V   rms={res['rms']:.1e}", fontsize=8)
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
        model_txt = (f"{NPOL - 2} real + 1 CC pair" if CC else f"{NPOL} real poles")
        fig.suptitle(f"Average-pulse pole-zero fit ({model_txt}, {NZER} zero"
                     f"{'s' if NZER != 1 else ''}, {win_txt}) — Ch {ch}  ·  Measurement 000205",
                     fontsize=14, fontweight="bold")
        fig.supxlabel("t - t_peak [ms]", fontsize=11, y=0.01)
        fig.supylabel("AP amplitude (peak-normalized)  /  residual", fontsize=11, x=0.005)
        out_png = os.path.join(OUTDIR, f"AP_fit_ch{ch}_m205.png")
        fig.savefig(out_png, dpi=180)
        plt.close(fig)
        print(f"Ch {ch}: {n_ok} WP fits  ->  {os.path.basename(out_png)}")

    # Fit parameters + RMS for downstream (pile-up rejection) use.
    if csv_rows:
        with open(FIT_CSV, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(_fit_csv_header())
            for row in csv_rows:
                w.writerow([f"{x:.6g}" if isinstance(x, float) else x for x in row])
        print(f"  -> {os.path.basename(FIT_CSV)}  ({len(csv_rows)} fits)")


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

    # AP model fit: one grid per channel (AP data + pole-zero fit).
    print("\nFitting average pulses (pole-zero model) and plotting per-channel grids:")
    plot_ap_fits(files)


if __name__ == "__main__":
    main()
