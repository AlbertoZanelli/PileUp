import numpy as np
from functools import lru_cache
from utility.double_beta_spectrum import inverse_cdf_ratio2b, build_sum_energy_distribution

from scipy.signal import bessel, bilinear, lfilter

def build_event_bank(S, nps, w, signal_amp, nsim, dt_max, chunk=500, seed=1234,
                     detector_sigma=0.0, fold_ratio=False):
    """Banco di impulsi simulati nel TEMPO, (nsim, len(S)) float32.

    E' la stessa ricetta di simulate_BI_error_m205.py: generazione a blocchi di `chunk`
    (il generatore alloca sei array complessi (n, 10000), a nsim intero il processo viene
    ucciso per memoria) con UN SOLO Generator per tutto il banco, cosi' il flusso continua
    da un blocco all'altro e `seed` identifica l'intero banco. `dt_max = 0` -> impulsi
    SINGOLI, `dt_max > 0` -> PILE-UP.

    NB: gli eventi dipendono da `chunk`, perche' dentro un blocco le estrazioni vanno per
    tipo -- va tenuto uguale fra campagne che si confrontano.

    `S` e `w` sono quelli di analysis.compute_H sul template di GENERAZIONE (il fit): si
    passano, non si ricostruiscono, cosi' non si duplica la convenzione di compute_H.

    Serve a chi deve applicare piu' filtri AGLI STESSI eventi: la validazione durante il
    training (un banco, tanti passi) e i confronti appaiati.
    """
    rng = np.random.default_rng(seed)
    chunks = []
    for k in range(0, nsim, chunk):
        n = min(chunk, nsim - k)
        fp, *_ = simulate_frequency_pulses(S, nps, detector_sigma, w, nsim=n, seed=rng,
                                           signal_scale=signal_amp, dt_max=dt_max,
                                           fold_ratio=fold_ratio)
        chunks.append(np.fft.ifft(fp, axis=1).real.astype(np.float32))
    return np.concatenate(chunks)


def simulate_frequency_pulses(S, nps, detector_sigma, w, nsim=10000, seed=1234, signal_scale=0.002, dt_max=8e-4,
                              fold_ratio=False):
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
    fold_ratio : bool, optional
        If True, fold the sharing ratio onto [0, 0.5] with r -> min(r, 1-r), i.e. the convention
        of the analytical J (default: False, full range as sampled from the 2b distribution).

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
    # E|N_k|^2 = 2*nps_k: il rumore in frequenza NON e' hermitiano, quindi il .real dell'ifft
    # (piu' sotto) ne dimezza la varianza e in tempo si ottiene var = sum(nps)/M^2, cioe' la
    # convenzione in cui vale sigma_OF = 1/sqrt(sum |S|^2/nps).
    # VERIFICATO su m205 ch91 WP15 con lo stimatore OF esatto: sigma simulata / sigma_OF = 0.99.
    # Prima qui c'era un fattore sqrt(2) in piu' (E|N|^2 = 4*nps) che iniettava sqrt(2) di
    # rumore di troppo, gonfiando del ~40% le larghezze e quindi il BI da simulazione.
    complex_gauss = rng.normal(size=(nsim, M)) + 1j * rng.normal(size=(nsim, M))
    noise_freq = complex_gauss * np.sqrt(nps[None, :])  # (nsim, M)

    # --- deterministic signal per realization ---
    amplitude = rng.normal(loc=signal_scale, scale=detector_sigma, size=nsim)[:, None]
    signal_freq = amplitude * S[None, :]  # (nsim, M)

    # --- random energy-sharing ratios and delays ---
    generated_r = inverse_cdf_ratio2b(rng.random(nsim))
    if fold_ratio:
        # r e 1-r descrivono lo stesso pile-up scambiando i due eventi: ripiegando su [0, 0.5]
        # si usa la stessa convenzione del calcolo ANALITICO (r_torch = linspace(0, 0.5, N_R)).
        generated_r = np.minimum(generated_r, 1.0 - generated_r)
    generated_dt = rng.uniform(0, dt_max, nsim)

    # --- phase modulation due to inter-event timing ---
    exp_term = np.exp(-1j * w * generated_dt[:, None])  # (nsim, M)
    phase = (1 - generated_r[:, None]) + generated_r[:, None] * exp_term
    # --- final frequency-domain realizations ---
    fpulses = signal_freq * phase + noise_freq
    return fpulses, signal_freq, noise_freq, generated_r, generated_dt


# inversa della cdf dell'energia somma nella ROI: si costruisce una volta per ROI
_sum_energy_icdf = lru_cache(maxsize=None)(build_sum_energy_distribution)


def simulate_frequency_pulses_v2(S, nps, w, nsim, rng, rng_x, signal_scale, dt_max,
                                 n_photons=None, e_roi=None, q_value=3.034, shift_max=0.0):
    """
    Variante v2 di simulate_frequency_pulses (2026-10-25): gli stessi eventi, piu' le tre aggiunte
    della generazione degli eventi del paper (Berest 2026, sez. 4.2). L'ORIGINALE non si tocca:
    questa funzione la chiama e lavora solo su cio' che le passa e su cio' che ne riceve.

    Ogni aggiunta e' una riga e si spegne con il suo argomento a None / 0:

      1. ENERGIA NELLA ROI (e_roi = (E_lo, E_hi) in MeV). L'energia somma E dell'evento si estrae
         dallo spettro somma di due 2nbb ristretto alla ROI (build_sum_energy_distribution, la
         ricetta di Ahmine 2023). La luce e' proporzionale all'energia: l'ampiezza diventa
         A * E/Q. Il rapporto r resta estratto dalla f_R originale, costruita a E = Q: fra 3019 e
         3049 keV la f_R cambia dello 0.4% (distanza di variazione totale).
      2. FLUTTUAZIONE DEL NUMERO DI FOTONI (n_photons = fotoni assorbiti a E = Q). Come nel
         codice originale, con detector_sigma: l'ampiezza dell'evento e' una gaussiana di media
         A E/Q e sigma = (A E/Q) / sqrt(n_photons E/Q), cioe' Poisson nell'approssimazione
         gaussiana (~200 fotoni). E' UN fattore per evento: i due impulsi di un pile-up
         fluttuano insieme e r resta esatto.
      3. SHIFT SUB-CAMPIONE (shift_max in s; 1e-4 = un campione a 10 kHz). L'impulso arriva in un
         istante continuo, il campionamento ha una griglia fissa: l'inizio cade a caso fra due
         campioni. Tutto il SEGNALE (i due impulsi) si ritarda di tau ~ U(0, shift_max), cioe'
         si moltiplica per e^{-i w tau}; il rumore resta quello estratto. Stessa distribuzione
         dello shift di simulate_frequency_pulses_fixed_dt_r, qui con un generatore con seed.

    Due generatori:
      rng   -> quello dell'originale (rumore, ampiezza, r, dt) con le stesse estrazioni nello
               stesso ordine: con le tre aggiunte spente l'uscita e' IDENTICA all'originale.
      rng_x -> E e tau, in quest'ordine. Separato, cosi' le aggiunte non spostano le estrazioni
               di rng; con lo stesso seed, singoli e pile-up ricevono gli stessi E e tau.

    Parameters
    ----------
    S, nps, w : come in simulate_frequency_pulses.
    nsim : int
        Numero di eventi.
    rng, rng_x : numpy.random.Generator
    signal_scale : float
        Ampiezza A a E = Q (l'ampiezza di ROI).
    dt_max : float
        Ritardo massimo del pile-up [s]; 0 -> impulsi singoli.

    Returns
    -------
    fpulses : ndarray, shape (nsim, M)
        Eventi nel dominio delle frequenze, come l'originale.
    info : dict
        Per evento: "energy" [MeV], "amp" (ampiezza media, prima della fluttuazione),
        "sigma" (della fluttuazione), "tau" [s], "r", "dt" [s].
    """
    e = (_sum_energy_icdf(*e_roi)(rng_x.random(nsim)) if e_roi is not None
         else np.full(nsim, q_value))                         # 1. energia somma [MeV]
    amp = signal_scale * (e / q_value)                        #    luce proporzionale all'energia
    sigma = (amp / np.sqrt(n_photons * (e / q_value)) if n_photons
             else np.zeros(nsim))                             # 2. Poisson sui fotoni
    fpulses, _, noise, r, dt = simulate_frequency_pulses(S, nps, sigma, w, nsim=nsim, seed=rng,
                                                         signal_scale=amp, dt_max=dt_max)
    tau = rng_x.uniform(0.0, shift_max, nsim) if shift_max else np.zeros(nsim)
    if shift_max:                                             # 3. shift del solo segnale
        fpulses = (fpulses - noise) * np.exp(-1j * w[None, :] * tau[:, None]) + noise
    return fpulses, dict(energy=e, amp=amp, sigma=sigma, tau=tau, r=r, dt=dt)

def simulate_frequency_pulses_fixed_dt_r(S, nps, w, dt ,r ,nsim=10000, seed=1234, signal_scale=0.002):
    rng = np.random.default_rng(seed)
    M = S.size

    # --- complex Gaussian noise with spectral shape ---
    # stessa convenzione di simulate_frequency_pulses: E|N_k|^2 = 2*nps_k (vedi nota li')
    complex_gauss = rng.normal(size = (nsim, M)) + 1j * rng.normal(size = (nsim, M))
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
