import numpy as np
from utility.double_beta_spectrum import inverse_cdf_ratio2b

from scipy.signal import bessel, bilinear, lfilter

def simulate_frequency_pulses(S, nps, detector_sigma, w, nsim=10000, seed=1234, signal_scale=0.002, dt_max=8e-4):
    """
    Simulate frequency-domain pulse realizations with stochastic phase and noise.

    Each realization represents a possible frequency-domain signal including:
      - A deterministic spectral shape `S`
      - Randomized energy sharing between two sub-events (via `inverse_cdf_ratio2b`)
      - Random emission time differences (Δt)
      - Additive complex Gaussian noise with baseline spectrum `nps`

    Parameters
    ----------
    S : ndarray, shape (M,)
        Deterministic signal spectrum in frequency domain.
    nps : ndarray, shape (M,)
        Baseline noise power spectrum (frequency-dependent variance).
    detector_sigma : float
        Detector-level energy fluctuation amplitude.
    w : ndarray, shape (M,)
        Frequency vector (radians per second).
    nsim : int, optional
        Number of Monte Carlo realizations (default: 10,000).
    seed : int, optional
        Random number generator seed (default: 1234).
    signal_scale : float, optional
        Base amplitude scaling factor for the deterministic signal (default: 0.002).
    dt_max : float, optional
        Maximum random time delay (seconds) between sub-events (default: 8e-4).

    Returns
    -------
    fpulses : ndarray, shape (nsim, M)
        Simulated complex frequency-domain signals for each realization.
    signal_freq : ndarray, shape (nsim, M)
        Deterministic (noise-free) signal component for each realization.
    noise_freq : ndarray, shape (nsim, M)
        Random noise component in frequency domain.
    generated_r : ndarray, shape (nsim,)
        Random energy-sharing ratios (R = E1 / E2).
    generated_dt : ndarray, shape (nsim,)
        Random inter-event time delays (seconds).
    """
    rng = np.random.default_rng(seed)
    M = S.size

    # --- complex Gaussian noise with spectral shape ---
    complex_gauss = (rng.normal(size=(nsim, M)) + 1j * rng.normal(size=(nsim, M))) * np.sqrt(2)
    noise_freq = complex_gauss * np.sqrt(nps[None, :])  # (nsim, M)

    # --- deterministic signal per realization ---
    amplitude = rng.normal(loc=signal_scale, scale=detector_sigma, size=nsim)[:, None]
    signal_freq = amplitude * S[None, :]  # (nsim, M)

    # --- random energy-sharing ratios and delays ---
    generated_r = inverse_cdf_ratio2b(rng.random(nsim))
    generated_dt = rng.uniform(0, dt_max, nsim)

    # --- phase modulation due to inter-event timing ---
    exp_term = np.exp(-1j * w * generated_dt[:, None])  # (nsim, M)
    phase = (1 - generated_r[:, None]) + generated_r[:, None] * exp_term
    # --- final frequency-domain realizations ---
    fpulses = signal_freq * phase + noise_freq
    return fpulses, signal_freq, noise_freq, generated_r, generated_dt

def simulate_frequency_pulses_fixed_dt_r(S, nps, w, dt ,r ,nsim=10000, seed=1234, signal_scale=0.002):
    rng = np.random.default_rng(seed)
    M = S.size

    # --- complex Gaussian noise with spectral shape ---
    complex_gauss = (rng.normal(size = (nsim, M)) + 1j * rng.normal(size = (nsim, M))) * np.sqrt(2)
    noise_freq = complex_gauss * np.sqrt(nps[None, :])  # (nsim, M)
    exp_term = np.exp(-1j * w * dt)  # (nsim, M)
    phase = (1 - r) + r* exp_term
    signal_freq = signal_scale * S[None, :]
    fpulses = signal_freq * phase * np.exp(-1j * w * np.random.random(nsim)[:,None]*1e-4) + noise_freq
    return fpulses, signal_freq, noise_freq


def pulse_model(t, t0, tau_rise, tau_decay1, tau_decay2, p):
    """Pulse model with 1 rise time and 2 decay times."""
    t_shifted = t - t0
    pulse = np.zeros_like(t_shifted)
    mask = t_shifted >= 0
    rise = 1 - np.exp(-t_shifted[mask]/tau_rise)
    decay = p * np.exp(-t_shifted[mask]/tau_decay1) + (1-p) * np.exp(-t_shifted[mask]/tau_decay2)
    pulse[mask] = rise * decay
    return pulse

def bessel_filter_coeffs(order, fc, fs):
    """Design a digital Bessel filter using bilinear transform."""
    # Normalize cutoff for digital design
    wc = 2 * np.pi * fc    # analog rad/s cutoff
    # Analog Bessel filter design
    b_a, a_a = bessel(order, wc, analog=True, norm='phase')
    # Convert to digital using bilinear transform
    b_d, a_d = bilinear(b_a, a_a, fs)
    return b_d, a_d


def apply_bessel_to_pulse(t, pulse, fc=450, order=2, fs=1e4):
    b_d, a_d = bessel_filter_coeffs(order, fc, fs)
    filtered_pulse = lfilter(b_d, a_d, pulse)
    return filtered_pulse

def filtered_pulse_model(t, t0, tau_rise, tau_decay1, tau_decay2, p, A, fc, order):
    pulse = pulse_model(t, t0, tau_rise, tau_decay1, tau_decay2, p)
    filtered_pulse = apply_bessel_to_pulse(t, pulse, fc=fc, order=order, fs=1e4)
    filtered_pulse *= A / np.max(filtered_pulse)
    return filtered_pulse.real

def pulse_pole_zero(t, amplitude, t0, zero, *poles):
    t = np.asarray(t, dtype=float)
    poles = np.asarray(poles, dtype=float)

    tt = t - t0
    f = np.zeros_like(tt)

    mask = tt >= 0
    tt_pos = tt[mask]

    diff = poles[:, None] - poles[None, :]
    denom = np.prod(
        np.where(np.eye(len(poles), dtype=bool), 1.0, diff),
        axis=1
    )

    k = (poles - zero) / denom

    # normalization so that f(t=0) = 1
    norm = np.sum(k * np.exp(poles * (-t0)))

    expo = np.exp(poles[:, None] * tt_pos[None, :])
    f[mask] = np.sum(k[:, None] * expo, axis=0) / norm

    return amplitude * f

import numpy as np
from scipy.signal import besselap

from scipy.optimize import minimize_scalar
def make_pulse_pole_zero_bessel_ct(
    bessel_order,
    fcut,
    zero,
    *poles
    ):
    """
    Continuous-time pole-zero pulse convolved with an
    analog Bessel low-pass filter with cutoff fcut (Hz).
    Normalized so f(0) = amplitude.
    """

    poles = np.asarray(poles, dtype=float)
    wc = 2 * np.pi * fcut   # rad/s

    # -------------------------
    # Pulse coefficients
    # -------------------------
    diff = poles[:, None] - poles[None, :]
    denom = np.prod(
        np.where(np.eye(len(poles), dtype=bool), 1.0, diff),
        axis=1
    )
    k = (poles - zero) / denom
    if bessel_order == 0:
        B = k
        exp_p = poles

        def f_raw(t):
            t = np.asarray(t, dtype = float)
            tt = t
            out = np.zeros_like(tt)

            mask = tt >= 0
            tt = tt[mask]

            out[mask] = np.sum(
                B[:, None] * np.exp(exp_p[:, None] * tt),
                axis = 0
            )
            return out

        res = minimize_scalar(
            lambda x: -f_raw(x),
            bounds = (0, 10 / min(-poles)),
            method = "bounded"
        )
        t_peak = res.x
        peak_value = f_raw(t_peak)
        def f(t):
            return f_raw(t+t_peak) / peak_value
        return f
    # -------------------------
    # Analog Bessel filter (scaled)
    # -------------------------
    _, p_norm, g_norm = besselap(bessel_order)

    p_filt = wc * p_norm
    g_filt = g_norm * wc**bessel_order

    diff_f = p_filt[:, None] - p_filt[None, :]
    denom_f = np.prod(
        np.where(np.eye(len(p_filt), dtype=bool), 1.0, diff_f),
        axis=1
    )
    A = g_filt / denom_f

    # -------------------------
    # Combine exponentials
    # -------------------------
    pi = poles[:, None]
    lj = p_filt[None, :]
    ki = k[:, None]
    Aj = A[None, :]

    coef = ki * Aj / (pi - lj)

    B = np.sum(coef, axis=1)     # exp(p_i t)
    C = -np.sum(coef, axis=0)    # exp(lambda_j t)

    # -------------------------
    # Callable
    # -------------------------
    def f_raw(t):
        tt = np.asarray(t, dtype=float)
        out = np.zeros_like(tt, dtype=np.complex128)
        mask = tt >= 0
        tt = tt[mask]

        out[mask] = (
            np.sum(B[:, None] * np.exp(poles[:, None] * tt), axis=0)
            + np.sum(C[:, None] * np.exp(p_filt[:, None] * tt), axis=0)
        )

        return out.real

    res = minimize_scalar(
        lambda x: -f_raw(x),
        bounds = (0,0.1),
        method = "bounded"
    )
    t_peak = res.x
    peak_value = f_raw(t_peak)
    def f(t):
        return f_raw(t + t_peak) / peak_value
    return f


def ROI_amp_from_sensitivity(sensitivity, area_factor=1.3, LY=0.36, gain=1104, Q_value=3.034):
    calib = 1e6 / (sensitivity * gain)  # KeV / V
    return LY * Q_value / calib * area_factor


if __name__ == "__main__":
    from torch.utils.data import TensorDataset
    import src.dataset as ds
    import src.analysis as an
    import torch
    n_deriv = 1
    channel = 15
    H_unit, S, nps, w, pulse_factor = ds.get_channel_specs(channel, n_deriv = n_deriv, window_fct = np.hanning)
    f1_opt, f2_opt = np.load(f"../outputs/Pileup_filter_functions/channel_{channel}/functions_eff90_V3.npy")
    fpulses, signal_freq, noise_freq, generated_r, generated_dt_single = simulate_frequency_pulses(S, nps,
                                                                                                       8.59898e-05, w,
                                                                                                       signal_scale = 0.001874,
                                                                                                       nsim = 10000,
                                                                                                       dt_max = 0)
    dataset_simu = TensorDataset(torch.from_numpy(np.fft.ifft(fpulses, axis = 1).real.astype(np.float32)))
    PSD_simu_single, _, _ = an.get_PSD_interpole(dataset_simu, H_unit, f1_opt, f2_opt)
    fpulses, signal_freq, noise_freq, generated_r, generated_dt = simulate_frequency_pulses(S, nps, 8.59898e-05, w,
                                                                                                signal_scale = 0.001874,
                                                                                                nsim = 10000)
    dataset_simu = TensorDataset(torch.from_numpy(np.fft.ifft(fpulses, axis = 1).real.astype(np.float32)))
    PSD_simu, Amp_1_simu, Amp_2_simu = an.get_PSD_interpole(dataset_simu, H_unit, f1_opt, f2_opt)
