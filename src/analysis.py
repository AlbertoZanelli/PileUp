import numpy as np
from torch.distributions.normal import Normal
from scipy.interpolate import interp1d
import torch
from torch.utils.data import DataLoader
from scipy.stats import gaussian_kde
import utility.functions as fn
from src.dataset import batch_iterator
import math
from utility.functions import cubic_interp1d_scipy, align_waveforms, gaussian_kde_torch
norm = Normal(0, 1)


def compute_H(meanpulse, nps, window_fct,sampling_rate=1e4):
    """
        Computes the Fourier transform of the mean pulse, the angular frequency array, 
        and the normalized filter response.

        Args:
            meanpulse (numpy.ndarray): The mean pulse signal in the time domain.
            nps (numpy.ndarray): The noise power spectrum.
            window_fct (function): A window function to apply to the mean pulse.

        Returns:
            tuple: A tuple containing:
                - S (numpy.ndarray): The Fourier transform of the windowed mean pulse.
                - w (numpy.ndarray): The angular frequency array.
                - H_unit (numpy.ndarray): The normalized filter response.
    """
    size = len(nps)
    S = np.fft.fft(meanpulse * window_fct(size))
    freq = np.fft.fftfreq(size, 1/sampling_rate)
    w = 2 * np.pi * freq
    H = np.conj(S) / nps
    H_unit = H / np.mean(np.abs(H * S))
    return S, w, H_unit


def compute_W(meanpulse, nps, window_fct, sampling_rate=1e4, norm_type=0, lam=1.0):
    """
        Computes the Wiener filter transfer function, faithfully porting the CUORE
        implementation (``WienerFilter`` by M. Beretta, ``QWienerFilter.cc``).

        Unlike the optimal filter (:func:`compute_H`), whose transfer function is
        ``S* / NPS``, the Wiener filter combines pulse-shape *deconvolution* with
        noise suppression:

            W(w) = S*(w) / ( |S(w)|^2 + NPS(w) )

        At low frequency (where ``|S|^2`` dominates) it tends to ``1/S`` and thus
        *sharpens* the pulse towards a delta; at high frequency (where ``NPS``
        dominates) it tends to the optimal filter ``S*/NPS``. The sharpened,
        symmetric filtered pulse makes two close-in-time events resolve into two
        separate peaks, which is what we want for pile-up discrimination.

        The relative power normalization between ``|S|^2`` and ``NPS`` is selected
        by ``norm_type`` (CUORE ``NormType``), which effectively tunes the
        regularization weight lambda balancing deconvolution vs. noise rejection.

        Args:
            meanpulse (numpy.ndarray): The mean pulse signal in the time domain.
            nps (numpy.ndarray): The noise power spectrum (same convention as
                :func:`compute_H`).
            window_fct (function): A window function to apply to the mean pulse.
            sampling_rate (float, optional): Sampling rate in Hz. Defaults to 1e4.
            norm_type (int, optional): Spectra normalization (CUORE ``NormType``):

                - 0 (default): AP and NPS normalized to equal power.
                  ``|S|^2 -> |S|^2 / sum|S|``, ``NPS -> NPS * sum|S| / (sum sqrt(NPS))^2``.
            lam (float, optional): Noise-modulation factor multiplying the NPS term
                in the denominator, ``W = S* / (|S|^2 + lam * NPS)``. Defaults to 1.0
                (standard CUORE Wiener filter). ``lam < 1`` -> more deconvolution
                (sharper, noisier); ``lam > 1`` -> closer to the optimal filter. This
                is the static counterpart of the trainable lambda optimized by
                :func:`optimize_filters_wiener_lambda`.

        Returns:
            tuple: A tuple containing:
                - S (numpy.ndarray): The Fourier transform of the windowed mean pulse.
                - w (numpy.ndarray): The angular frequency array.
                - W_unit (numpy.ndarray): The Wiener filter transfer function,
                  amplitude-normalized so that the peak of the filtered mean pulse
                  equals the mean pulse amplitude (CUORE ``conv`` convention).
    """
    size = len(nps)
    window = window_fct(size)
    avgpulse = meanpulse * window
    freq = np.fft.fftfreq(size, 1 / sampling_rate)
    w = 2 * np.pi * freq

    # nps is rescaled below; work on a copy so the caller's array is untouched.
    nps = np.asarray(nps, dtype=float).copy()

    # CUORE NormType == 0 (default): equal-power normalization of AP and NPS.
    S = np.fft.fft(avgpulse)
    AvgPS = np.abs(S) ** 2
    avg_energy = np.sum(np.sqrt(AvgPS))   # == sum |S|
    n_energy = np.sum(np.sqrt(nps))
    AvgPS = AvgPS / avg_energy
    nps = nps * (avg_energy / n_energy ** 2)

    # Transfer function: W = S* / (|S|^2 + lam * NPS). Numerator uses S (the FFT of
    # the possibly-rescaled average pulse), matching H = conj(S) in QWienerFilter.cc.
    # lam modulates the noise term (lam=1 -> standard Wiener filter).
    W = np.conj(S) / (AvgPS + lam * nps)
    # DC bin zeroed (R[0] = I[0] = 0 in the CUORE Filter()).
    W[0] = 0.0

    # Amplitude normalization (CUORE conv = max(AP) / max(filtered AP)): scale W so
    # that the peak of the filtered mean pulse reproduces the true pulse amplitude.
    filtered_avg = np.fft.ifft(W * S).real
    peak = np.max(filtered_avg)
    if peak != 0:
        W_unit = W * (np.max(meanpulse) / peak)
    else:
        W_unit = W
    return S, w, W_unit


def precompute_constants(S, H_unit, w, t, nps):
    """
    Precomputes constants for further calculations.

    Args:
        S (torch.Tensor): Input signal tensor.
        H_unit (torch.Tensor): Unit response tensor.
        w (torch.Tensor): Frequency tensor.
        t (torch.Tensor): Time tensor.
        nps (torch.Tensor): Baseline spectrum tensor.

    Returns:
        tuple: Phase tensor, weighted signal tensor, and normalized spectrum tensor.
    """
    exp_term = torch.exp(-1j * w[None, :] * t[:, None])  # (len(t), len(w))
    S_H = S * H_unit
    S2_over_nps = (S.abs() ** 2) / nps
    S_H_delayed = S_H[None, :] * exp_term
    return S_H_delayed, S_H, S2_over_nps

def compute_sigma_OF(S, nps):
    """
    Computes the optimal filter standard deviation (sigma) based on the given filter and noise power spectrum.

    Args:
        H (torch.Tensor): Filter tensor.
        nps (torch.Tensor): Noise power spectrum tensor.

    Returns:
        torch.Tensor: The computed optimal filter standard deviation (sigma).
    """
    S = torch.tensor(S, dtype=torch.cfloat)
    nps = torch.tensor(nps, dtype=torch.cfloat)
    absS2_over_nps = (S.abs() ** 2) / nps
    num = torch.sum(absS2_over_nps)
    R = torch.sum(absS2_over_nps)
    var1 = num / (R.abs() ** 2)
    sigma_OF = var1.real**0.5
    return sigma_OF

def compute_vars(S2_over_nps, f1, f2):
    """
    Computes the standard deviation (sigma) for the given inputs.

    Args:
        S2_over_nps (torch.Tensor): Normalized spectrum tensor.
        f1 (torch.Tensor): First filter tensor.
        f2 (torch.Tensor): Second filter tensor.

    Returns:
        tuple: Variances of X1, X2 and covariance of X and Y.
    """
    absS2_over_nps = S2_over_nps

    abs_f1 = f1.abs()
    abs_f2 = f2.abs()
    num1 = torch.sum(abs_f1 ** 2 * absS2_over_nps)
    num2 = torch.sum(abs_f2 ** 2 * absS2_over_nps)
    cross = torch.sum((f1 * torch.conj(f2)) * absS2_over_nps)
    R1 = torch.sum(abs_f1 * absS2_over_nps)
    R2 = torch.sum(abs_f2 * absS2_over_nps)
    var1 = num1 / (R1.abs() ** 2)
    var2 = num2 / (R2.abs() ** 2)
    cov12 = cross / (R1 * torch.conj(R2))
    return var1.real, var2.real, cov12.real


def compute_sigma_ratio(mu1, mu2, var1, var2, cov12):
    """
        Computes the standard deviation (sigma) of the ratio of two variables.

        Args:
            mu1 (array-like, torch.Tensor, or float): Mean value of the numerator variable.
            mu2 (array-like, torch.Tensor, or float): Mean value of the denominator variable.
            var1 (array-like, torch.Tensor, or float): Variance of the numerator variable.
            var2 (array-like, torch.Tensor, or float): Variance of the denominator variable.
            cov12 (array-like, torch.Tensor, or float): Covariance between the numerator and denominator variables.

        Returns:
            array-like, torch.Tensor, or float: The computed standard deviation of the ratio.
    """
    varY = var1 / mu2 ** 2 + (mu1 ** 2) * var2 / mu2 ** 4 - 2 * mu1 * cov12 / mu2 ** 3
    return varY**0.5


def compute_sigma_ratio_order4(mu1, mu2, var1, var2, cov12):
    """
        Computes the standard deviation (sigma) of the ratio of two variables using a fourth-order approximation.

        Args:
            mu1 (array-like, torch.Tensor, or float): Mean value of the numerator variable.
            mu2 (array-like, torch.Tensor, or float): Mean value of the denominator variable.
            var1 (array-like, torch.Tensor, or float): Variance of the numerator variable.
            var2 (array-like, torch.Tensor, or float): Variance of the denominator variable.
            cov12 (array-like, torch.Tensor, or float): Covariance between the numerator and denominator variables.

        Returns:
            array-like, torch.Tensor, or float: The computed standard deviation of the ratio.
    """
    sigma1 = var1**0.5
    sigma2 = var2**0.5
    # Relative errors
    eps_1 = sigma1 / mu1
    eps_2 = sigma2 / mu2

    # Convert covariance to relative form
    rho = cov12 / (sigma1 * sigma2) if sigma1 * sigma2 != 0 else 0

    # Leading order
    var_LO = eps_1 ** 2 + eps_2 ** 2 - 2 * rho * eps_1 * eps_2

    # Higher order O(sigma^4)
    var_HO = (2 * eps_1 ** 2 * eps_2 ** 2
              + 3 * eps_2 ** 4 
              - 6 * rho * eps_1 * eps_2 ** 3 
              + rho ** 2 * eps_1 ** 2 * eps_2 ** 2)

    # Total relative variance
    rel_var = var_LO + var_HO

    # Absolute variance
    var_R = (mu1 / mu2) ** 2 * rel_var

    return var_R**0.5


def compute_mu_sigma(f1, f2, S_H_delayed, r, S_H, S2_over_nps, signal_amp, pulse_center_ratio=0.5,
                     return_all=False, order=2, use_interp=False,
                     jitter_max=20, interpolation_range=5):
    """
        Computes the mean (mu) and standard deviation (sigma) of the ratio of two signals
        after applying filters and alignment.

        Args:
            f1 (torch.Tensor): First filter tensor.
            f2 (torch.Tensor): Second filter tensor.
            S_H_delayed (torch.Tensor): Weighted signal delayed tensor.
            r (torch.Tensor): Ratio tensor for pileup contributions.
            S_H (torch.Tensor): Weighted signal tensor.
            S2_over_nps (torch.Tensor): Normalized spectrum tensor.
            signal_amp (float): Amplitude of the signal.
            pulse_center_ratio (float, optional): Ratio to determine the pulse center. Defaults to 0.5.
            return_all (bool, optional): Whether to return all intermediate results. Defaults to False.
            order (int, optional): Order of the variance computation (2 or 4). Defaults to 2.
            jitter_max (int, optional): Maximum jitter for peak search. Defaults to 20.
            interpolation_range (int, optional): Range for interpolation around the maximum. Defaults to 5.

        Returns:
            tuple: If `return_all` is False, returns:
                - muY (torch.Tensor): Mean of the ratio of the two signals.
                - sigmaY (torch.Tensor): Standard deviation of the ratio.
            If `return_all` is True, returns:
                - muY (torch.Tensor): Mean of the ratio of the two signals.
                - sigmaY (torch.Tensor): Standard deviation of the ratio.
                - mu1 (torch.Tensor): Mean of the first signal.
                - mu2 (torch.Tensor): Mean of the second signal.
                - var1 (torch.Tensor): Variance of the first signal.
                - var2 (torch.Tensor): Variance of the second signal.
                - cov12 (torch.Tensor): Covariance between the two signals.
    """
    win_length = S_H.shape[-1]
    target_index = int(win_length * pulse_center_ratio)
    fine_x = torch.arange(-interpolation_range, interpolation_range + 1, 0.05, device = f1.device)
    offs = torch.arange(-interpolation_range, interpolation_range + 1, device = f1.device)
    lo = target_index - jitter_max
    hi = target_index + jitter_max

    s_H_1 = torch.fft.ifft(S_H * f1, dim = -1).real
    s_H_delayed_1 = torch.fft.ifft(S_H_delayed * f1, dim = -1).real

    s_H_2 = torch.fft.ifft(S_H * f2, dim = -1).real
    s_H_delayed_2 = torch.fft.ifft(S_H_delayed * f2, dim = -1).real
    s_pileup_1 = (1 - r[:, None, None]) * s_H_1 + r[:, None, None] * s_H_delayed_1[None, :, :]

    s_pileup_2 = (1 - r[:, None, None]) * s_H_2 + r[:, None, None] * s_H_delayed_2[None, :, :]

    if use_interp:
        s_pileup_1 = torch.roll(s_pileup_1.real, shifts = target_index, dims = -1)
        s_pileup_2 = torch.roll(s_pileup_2.real, shifts = target_index, dims = -1)

        sub1 = s_pileup_1[..., lo:hi]
        sub2 = s_pileup_2[..., lo:hi]
        max_idx1 = sub1.argmax(dim = -1) + lo
        max_idx2 = sub2.argmax(dim = -1) + lo

        idx1 = max_idx1[..., None] + offs
        idx2 = max_idx2[..., None] + offs

        y1 = s_pileup_1.gather(-1, idx1)
        y2 = s_pileup_2.gather(-1, idx2)

        interp1 = cubic_interp1d_scipy(y1, fine_x,offs)
        interp2 = cubic_interp1d_scipy(y2, fine_x,offs)

        mu1 = interp1.max(dim = -1).values
        mu2 = interp2.max(dim = -1).values
    else:
        mu1 = s_pileup_1.real.max(dim = -1).values
        mu2 = s_pileup_2.real.max(dim = -1).values
    muY = mu1 / mu2
    var1, var2, cov12 = compute_vars(S2_over_nps, f1, f2)
    if order == 4:
        sigmaY = compute_sigma_ratio_order4(mu1*signal_amp, mu2*signal_amp, var1**0.5, var2**0.5, cov12)
    else:
        sigmaY = compute_sigma_ratio(mu1*signal_amp, mu2*signal_amp, var1, var2, cov12)
    if return_all:
        return muY, sigmaY, mu1, mu2, var1, var2, cov12
    else:
        return muY, sigmaY


def compute_A(muY, sigmaY, N_sigma=1.28):
    """
    Computes the acceptance (A).

    Args:
        muY (torch.Tensor): Ratio of the maximum amplitudes of the two signals.
        sigmaY (torch.Tensor): Standard deviation of the ratio.
        N_sigma (float, optional): Sigma threshold. Defaults to 1.28.

    Returns:
        torch.Tensor: Computed probability value.
    """
    return 1 - norm.cdf((1-muY-N_sigma*sigmaY[0, 0]) / sigmaY)


def compute_J(f1, f2, S_H_delayed, r, S_H, S2_over_nps, signal_amp, ratio_distribution,
              pulse_center_ratio=0.5, N_sigma=1.28, use_interp=False ,full_output=False):
    """
    Computes the J metric for the given inputs.

    Args:
        f1 (torch.Tensor): First filter tensor.
        f2 (torch.Tensor): Second filter tensor.
        S_H_delayed (torch.Tensor): Weighted signal delayed tensor.
        r (torch.Tensor): Ratio tensor.
        S_H (torch.Tensor): Weighted signal tensor.
        S2_over_nps (torch.Tensor): Normalized spectrum tensor.
        signal_amp (float): Signal amplitude.
        ratio_distribution (torch.Tensor): Ratio distribution tensor.
        pulse_center_ratio (float, optional): Pulse center ratio. Defaults to 0.5.
        N_sigma (float, optional): Sigma threshold. Defaults to 1.28.
        full_output (bool, optional): Whether to return full output. Defaults to False.

    Returns:
            torch.Tensor: Computed J metric value if `full_output=False`.
            tuple: If `full_output=True`, returns a tuple containing:
                - torch.Tensor: Mean J metric value.
                - torch.Tensor: Acceptance values.
                - torch.Tensor: Mean Y values.
                - torch.Tensor: Sigma Y values.
    """
    muY, sigmaY = compute_mu_sigma(f1, f2, S_H_delayed, r, S_H, S2_over_nps, signal_amp,
                                   pulse_center_ratio=pulse_center_ratio, use_interp=use_interp)
    A = compute_A(muY, sigmaY, N_sigma=N_sigma)
    if full_output:
        return torch.mean(A*ratio_distribution[:, None]).real, A, muY, sigmaY
    return torch.mean(A*ratio_distribution[:, None]).real


def optimize_filters(S, H_unit, w, t, r, nps, signal_amp, ratio_distribution, N_sigma = 1.28, n_trials = 1000,
                    activation_fct=None,
                     pulse_center_ratio=0.5, f1_init=None, f2_init=None, verbose = True, use_interp = False):
    """
    Optimizes two filters (f1 and f2) to maximize the J metric using gradient-based optimization.

    Args:
        S (torch.Tensor): Input signal tensor.
        H_unit (torch.Tensor): Unit response tensor.
        w (torch.Tensor): Frequency tensor.
        t (torch.Tensor): Time tensor.
        r (torch.Tensor): Ratio tensor.
        nps (torch.Tensor): Baseline spectrum tensor.
        signal_amp (float): Signal amplitude.
        ratio_distribution (torch.Tensor): Ratio distribution tensor.
        N_sigma (float, optional): Sigma threshold for the J metric. Defaults to 1.28.
        n_trials (int, optional): Number of optimization steps. Defaults to 1000.
        pulse_center_ratio (float, optional): Pulse center ratio. Defaults to 0.5.
        f1_init (torch.Tensor or None, optional): Initial filter f1. Defaults to None.
        f2_init (torch.Tensor or None, optional): Initial filter f2. Defaults to None.
        verbose (bool, optional): Whether to print progress. Defaults to True.

    Returns:
        tuple:
            - f1 (numpy.ndarray): Optimized first filter.
            - f2 (numpy.ndarray): Optimized second filter.
            - J_values (list): List of J metric values during optimization.
    """
    # Precompute constants required for optimization
    S_H_delayed, S_H, S2_over_nps = precompute_constants(S, H_unit, w, t, nps)
    # Initialize filter parameters as complex tensors
    n = len(H_unit)
    activation_fct = torch.nn.Softplus(20,20) if activation_fct is None else activation_fct
    f1_init = activation_fct(torch.rand(n//2+1, dtype = torch.float)) if f1_init is None else f1_init.clone().abs()
    f2_init = activation_fct(torch.rand(n//2+1, dtype = torch.float)) if f2_init is None else f2_init.clone().abs()
    if len(f1_init) > n//2+1:
        f1_init = f1_init[:n//2+1]
    if len(f2_init) > n//2+1:
        f2_init = f2_init[:n//2+1]
    f1_param = torch.nn.Parameter(f1_init.to(S.device))
    f2_param = torch.nn.Parameter(f2_init.to(S.device))

    # Set up the optimizer
    optimizer = torch.optim.Adam([f1_param, f2_param], lr = 1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max = n_trials,
        eta_min = 1e-5
    )
    J_values = []  # To store J metric values during optimization
    f1 = activation_fct(f1_param)
    f2 = activation_fct(f2_param)
    f1 = torch.cat([f1, f1[1:-1].flip(0)])
    f2 = torch.cat([f2, f2[1:-1].flip(0)])
    J = torch.nan
    for step in range(n_trials):
        optimizer.zero_grad()  # Reset gradients
        # Enforce positivity of the filters
        f1 = activation_fct(f1_param)
        f2 = activation_fct(f2_param)
        f1 = torch.cat([f1, f1[1:-1].flip(0)])
        f2 = torch.cat([f2, f2[1:-1].flip(0)])
        # Normalize filters such that mean(|f_i * H * S|) = 1
        norm1 = torch.mean(torch.abs(f1 * H_unit * S))
        norm2 = torch.mean(torch.abs(f2 * H_unit * S))
        f1 = f1 / norm1
        f2 = f2 / norm2
        # Compute the J metric
        J = compute_J(f1, f2, S_H_delayed, r, S_H, S2_over_nps, signal_amp, ratio_distribution, N_sigma = N_sigma,
                      pulse_center_ratio=pulse_center_ratio, use_interp=use_interp)
        J_values.append(J.item())  # Store the current J value

        # Backpropagation and update filter parameters
        J.backward()

        optimizer.step()
        scheduler.step()
        # Uncomment the following lines to print progress every 100 steps
        if step % 10 == 0 and verbose:
            print(f"Step {step}: J = {J.item():.6f}")
    print(f"Final: J = {J.item():.6f}")
    # Convert optimized filters to numpy arrays
    f1 = f1.detach()
    f2 = f2.detach()
    return f1, f2, J_values


# =============================================================================
# Wiener-filter variants of the optimization machinery.
#
# These mirror compute_vars / compute_mu_sigma / compute_J / optimize_filters but
# replace the optimal-filter-specific noise propagation with the exact variance of
# the filtered waveform for an arbitrary applied kernel f_i * W_unit. The signal
# (mu) computation, filter normalization and the J metric are unchanged because
# they only depend on S_H = S * W_unit and are therefore filter-agnostic.
#
# Use these together with compute_W instead of compute_H.
# =============================================================================

def compute_vars_wiener(W_unit, S, nps, f1, f2):
    """
    Noise propagation for the applied Wiener kernel ``g_i = f_i * W_unit``, written
    in the same self-normalizing ``num / R^2`` style as :func:`compute_vars` so that
    it is consistent with the project's NPS convention.

    Like the optimal-filter routine, it carries **no** explicit ``1/N^2`` or ``df``
    factor: those normalizations cancel between the numerator and the response
    ``R``, so the (un-normalized by N^2/df) ``nps`` used throughout the code base can
    be passed as-is. The optimal-filter density ``|S|^2 / NPS`` is replaced by its
    Wiener analogue:

        - noise weight (numerator)  : |W|^2 * NPS
        - signal response (denom R) : Re(W * S)   [= |S|^2 / (|S|^2 + NPS) for
                                                    W = S*/(|S|^2 + NPS); W already
                                                    carries conj(S), so it is W*S]

        var_i  = sum_w |f_i|^2 * |W|^2 * NPS  /  ( sum_w |f_i| * Re(conj(W) S) )^2
        cov_12 = sum_w f1 conj(f2) |W|^2 NPS  /  ( R1 * conj(R2) )

    Any overall (e.g. amplitude / ``conv``) normalization of ``W_unit`` cancels in
    ``num / R^2``. Substituting the optimal-filter kernel ``W = c S*/NPS`` makes this
    reduce **exactly** to :func:`compute_vars` for any ``f1, f2`` (the constant ``c``
    cancels), which is the proof that it is the correct generalization.

    Args:
        W_unit (torch.Tensor): Applied transfer function (complex), e.g. from
            :func:`compute_W` converted to a torch tensor.
        S (torch.Tensor): FFT of the (windowed) mean pulse.
        nps (torch.Tensor or numpy.ndarray): Noise power spectrum (project convention).
        f1 (torch.Tensor): First band filter tensor.
        f2 (torch.Tensor): Second band filter tensor.

    Returns:
        tuple: Variances of A1, A2 and their covariance (all real-valued tensors).
    """
    if not torch.is_tensor(nps):
        nps = torch.as_tensor(nps, device=f1.device)
    nps = nps.to(device=f1.device, dtype=torch.float32)
    # Per-frequency densities, Wiener analogues of the OF's |S|^2/NPS.
    # The matched-filter kernel W already contains conj(S), so the (real, positive)
    # signal response is W*S = |S|^2/(|S|^2+NPS), not conj(W)*S.
    W_nps = (W_unit.abs() ** 2) * nps             # noise weight   |W|^2 * NPS
    resp = (W_unit * S).real                      # signal response Re(W S)

    abs_f1 = f1.abs()
    abs_f2 = f2.abs()
    num1 = torch.sum(abs_f1 ** 2 * W_nps)
    num2 = torch.sum(abs_f2 ** 2 * W_nps)
    cross = torch.sum((f1 * torch.conj(f2)) * W_nps)
    R1 = torch.sum(abs_f1 * resp)
    R2 = torch.sum(abs_f2 * resp)
    var1 = num1 / (R1.abs() ** 2)
    var2 = num2 / (R2.abs() ** 2)
    cov12 = cross / (R1 * torch.conj(R2))
    return var1.real, var2.real, cov12.real


def compute_mu_sigma_wiener(f1, f2, S_H_delayed, r, S_H, S2_over_nps, W_unit, S, nps, signal_amp,
                            pulse_center_ratio=0.5, return_all=False, order=2, use_interp=False,
                            jitter_max=20, interpolation_range=5):
    """
    Wiener-filter counterpart of :func:`compute_mu_sigma`.

    The signal means (mu1, mu2, muY) are computed by reusing :func:`compute_mu_sigma`
    (the peak-of-filtered-signal calculation is identical for any transfer function),
    while the ratio resolution ``sigmaY`` is recomputed from the exact Wiener noise
    propagation in :func:`compute_vars_wiener`.

    Args mirror :func:`compute_mu_sigma`, with two extra inputs:
        W_unit (torch.Tensor): Wiener transfer function (the same one used to build
            ``S_H = S * W_unit`` via :func:`precompute_constants`).
        nps (torch.Tensor or numpy.ndarray): Noise power spectrum.

    Returns:
        tuple: ``(muY, sigmaY)`` or, if ``return_all``, also ``mu1, mu2, var1,
        var2, cov12``.
    """
    # Signal part is filter-agnostic: reuse the optimal-filter routine (its variance
    # output is discarded and recomputed below).
    muY, _, mu1, mu2, _, _, _ = compute_mu_sigma(
        f1, f2, S_H_delayed, r, S_H, S2_over_nps, signal_amp,
        pulse_center_ratio=pulse_center_ratio, return_all=True, order=order,
        use_interp=use_interp, jitter_max=jitter_max, interpolation_range=interpolation_range)
    # Noise propagation for the actual applied Wiener kernel f_i * W_unit.
    var1, var2, cov12 = compute_vars_wiener(W_unit, S, nps, f1, f2)
    if order == 4:
        sigmaY = compute_sigma_ratio_order4(mu1 * signal_amp, mu2 * signal_amp,
                                            var1 ** 0.5, var2 ** 0.5, cov12)
    else:
        sigmaY = compute_sigma_ratio(mu1 * signal_amp, mu2 * signal_amp, var1, var2, cov12)
    if return_all:
        return muY, sigmaY, mu1, mu2, var1, var2, cov12
    return muY, sigmaY


def compute_J_wiener(f1, f2, S_H_delayed, r, S_H, S2_over_nps, W_unit, S, nps, signal_amp,
                     ratio_distribution, pulse_center_ratio=0.5, N_sigma=1.28,
                     use_interp=False, full_output=False):
    """
    Wiener-filter counterpart of :func:`compute_J`. Identical metric, but the
    resolution comes from :func:`compute_mu_sigma_wiener`.
    """
    muY, sigmaY = compute_mu_sigma_wiener(f1, f2, S_H_delayed, r, S_H, S2_over_nps, W_unit, S, nps,
                                          signal_amp, pulse_center_ratio=pulse_center_ratio,
                                          use_interp=use_interp)
    A = compute_A(muY, sigmaY, N_sigma=N_sigma)
    if full_output:
        return torch.mean(A * ratio_distribution[:, None]).real, A, muY, sigmaY
    return torch.mean(A * ratio_distribution[:, None]).real


def optimize_filters_wiener(S, W_unit, w, t, r, nps, signal_amp, ratio_distribution, N_sigma=1.28,
                            n_trials=1000, activation_fct=None, pulse_center_ratio=0.5,
                            f1_init=None, f2_init=None, verbose=True, use_interp=False):
    """
    Wiener-filter version of :func:`optimize_filters`.

    Optimizes the two band filters (f1, f2) to maximize the J metric, using the
    Wiener transfer function ``W_unit`` (from :func:`compute_W`) in place of the
    optimal-filter ``H_unit`` and the exact Wiener noise propagation
    (:func:`compute_vars_wiener`) instead of the optimal-filter formula.

    The body mirrors :func:`optimize_filters`; the differences are:
      - ``W_unit`` is used wherever ``H_unit`` was (filter normalization and S_H).
      - the J metric is :func:`compute_J_wiener`, which also needs ``nps``.

    Args:
        S (torch.Tensor): FFT of the (windowed) mean pulse.
        W_unit (torch.Tensor): Wiener transfer function from :func:`compute_W`
            (convert the numpy output to a torch tensor before calling).
        w (torch.Tensor): Angular frequency tensor.
        t (torch.Tensor): Delay grid tensor.
        r (torch.Tensor): Pile-up energy-sharing ratio tensor.
        nps (torch.Tensor or numpy.ndarray): Noise power spectrum.
        signal_amp (float): Signal amplitude.
        ratio_distribution (torch.Tensor): Ratio distribution tensor.
        N_sigma (float, optional): Sigma threshold for the J metric. Defaults to 1.28.
        n_trials (int, optional): Number of optimization steps. Defaults to 1000.
        pulse_center_ratio (float, optional): Pulse center ratio. Defaults to 0.5.
        f1_init, f2_init (torch.Tensor or None, optional): Initial filters.
        verbose (bool, optional): Whether to print progress. Defaults to True.
        use_interp (bool, optional): Whether to interpolate the peak. Defaults to False.

    Returns:
        tuple: ``(f1, f2, J_values)`` with the optimized filters and the J history.
    """
    # Precompute constants; S_H = S * W_unit is built from the Wiener kernel.
    S_H_delayed, S_H, S2_over_nps = precompute_constants(S, W_unit, w, t, nps)
    if not torch.is_tensor(nps):
        nps = torch.as_tensor(nps, device=S.device)
    nps = nps.to(device=S.device, dtype=torch.float32)

    n = len(W_unit)
    activation_fct = torch.nn.Softplus(20, 20) if activation_fct is None else activation_fct
    f1_init = activation_fct(torch.rand(n // 2 + 1, dtype=torch.float)) if f1_init is None else f1_init.clone().abs()
    f2_init = activation_fct(torch.rand(n // 2 + 1, dtype=torch.float)) if f2_init is None else f2_init.clone().abs()
    if len(f1_init) > n // 2 + 1:
        f1_init = f1_init[:n // 2 + 1]
    if len(f2_init) > n // 2 + 1:
        f2_init = f2_init[:n // 2 + 1]
    f1_param = torch.nn.Parameter(f1_init.to(S.device))
    f2_param = torch.nn.Parameter(f2_init.to(S.device))

    optimizer = torch.optim.Adam([f1_param, f2_param], lr=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_trials, eta_min=1e-5)
    J_values = []
    J = torch.nan
    for step in range(n_trials):
        optimizer.zero_grad()
        # Enforce positivity and rebuild the full (Hermitian) spectrum.
        f1 = activation_fct(f1_param)
        f2 = activation_fct(f2_param)
        f1 = torch.cat([f1, f1[1:-1].flip(0)])
        f2 = torch.cat([f2, f2[1:-1].flip(0)])
        # Normalize filters such that mean(|f_i * W_unit * S|) = 1.
        norm1 = torch.mean(torch.abs(f1 * W_unit * S))
        norm2 = torch.mean(torch.abs(f2 * W_unit * S))
        f1 = f1 / norm1
        f2 = f2 / norm2
        # Wiener J metric.
        J = compute_J_wiener(f1, f2, S_H_delayed, r, S_H, S2_over_nps, W_unit, S, nps, signal_amp,
                             ratio_distribution, N_sigma=N_sigma,
                             pulse_center_ratio=pulse_center_ratio, use_interp=use_interp)
        J_values.append(J.item())
        J.backward()
        optimizer.step()
        scheduler.step()
        if step % 10 == 0 and verbose:
            print(f"Step {step}: J = {J.item():.6f}")
    print(f"Final: J = {J.item():.6f}")
    f1 = f1.detach()
    f2 = f2.detach()
    return f1, f2, J_values


# =============================================================================
# Wiener filter with trainable noise-modulation factor lambda.
#
# The Wiener kernel is generalized to
#
#     W_lambda = S* / ( |S|^2 + lambda * NPS )        (equal-power normalized spectra)
#
# with lambda > 0 a single trainable scalar, optimized jointly with the band
# filters f1, f2. lambda = 1 reproduces the CUORE norm_type=0 Wiener filter;
# lambda -> 0 tends to pure deconvolution 1/S, lambda -> inf to the optimal
# filter shape. Since W now depends on a trainable parameter, the kernel and the
# derived quantities (S_H, S_H_delayed) are rebuilt inside the optimization loop
# in a differentiable way; only the delay phases exp(-i w t) are precomputed.
#
# Note: the amplitude (conv) normalization of compute_W is intentionally omitted
# here — compute_vars_wiener is invariant under any rescaling of W, and the band
# filters are renormalized with mean(|f_i W S|) = 1 at every step, so an overall
# scale of W cancels everywhere in J.
# =============================================================================

def compute_W_torch(S, nps, lam):
    """
    Differentiable Wiener transfer function with noise-modulation factor ``lam``.

    Builds, entirely in torch (so gradients flow through ``lam``):

        W = conj(S) / ( |S|^2_n + lam * NPS_n ),    W[0] = 0,

    where ``|S|^2_n`` and ``NPS_n`` are the equal-power-normalized spectra of
    :func:`compute_W` with ``norm_type=0`` (``|S|^2_n = |S|^2 / a``,
    ``NPS_n = NPS * a / b^2`` with ``a = sum|S|``, ``b = sum sqrt(NPS)``).
    Hence ``lam = 1`` reproduces the standard (CUORE norm_type=0) Wiener kernel.

    Args:
        S (torch.Tensor): FFT of the (windowed) mean pulse (complex tensor).
        nps (torch.Tensor): Noise power spectrum (real tensor, project convention).
        lam (torch.Tensor or float): Positive noise-modulation factor (may require grad).

    Returns:
        torch.Tensor: Complex Wiener transfer function (no amplitude normalization;
        see module note above on why it is not needed during optimization).
    """
    AvgPS = S.abs() ** 2
    a = torch.sum(torch.sqrt(AvgPS))
    b = torch.sum(torch.sqrt(nps))
    AvgPS_n = AvgPS / a
    nps_n = nps * (a / b ** 2)
    W = torch.conj(S) / (AvgPS_n + lam * nps_n)
    # Zero the DC bin without in-place ops (keeps autograd graph intact).
    dc_mask = torch.ones(len(S), device=S.device)
    dc_mask[0] = 0.0
    return W * dc_mask


def optimize_filters_wiener_lambda(S, w, t, r, nps, signal_amp, ratio_distribution, N_sigma=1.28,
                                   n_trials=500, activation_fct=None, pulse_center_ratio=0.5,
                                   f1_init=None, f2_init=None, lambda_init=1.0, lr_lambda=None,
                                   verbose=True, use_interp=False):
    """
    Wiener filter optimization with a trainable noise-modulation factor lambda.

    Like :func:`optimize_filters_wiener`, but the Wiener kernel itself carries one
    extra trainable scalar:

        W_lambda = S* / ( |S|^2 + lambda * NPS )       (see :func:`compute_W_torch`)

    optimized jointly with the band filters f1, f2 by minimizing the J metric.
    Positivity is enforced by parametrizing ``lambda = exp(log_lambda)``. Since W
    depends on lambda, S_H and S_H_delayed are rebuilt (differentiably) at every
    step; the delay phases exp(-i w t) are precomputed once.

    Args:
        S (torch.Tensor): FFT of the (windowed) mean pulse.
        w (torch.Tensor): Angular frequency tensor.
        t (torch.Tensor): Delay grid tensor.
        r (torch.Tensor): Pile-up energy-sharing ratio tensor.
        nps (torch.Tensor or numpy.ndarray): Noise power spectrum.
        signal_amp (float): Signal amplitude.
        ratio_distribution (torch.Tensor): Ratio distribution tensor.
        N_sigma (float, optional): Sigma threshold for the J metric. Defaults to 1.28.
        n_trials (int, optional): Number of optimization steps. Defaults to 1000.
        pulse_center_ratio (float, optional): Pulse center ratio. Defaults to 0.5.
        f1_init, f2_init (torch.Tensor or None, optional): Initial filters.
        lambda_init (float, optional): Initial lambda. Defaults to 1.0 (the standard
            norm_type=0 Wiener filter).
        lr_lambda (float or None, optional): Learning rate for log-lambda. Defaults
            to the filter learning rate (1e-2).
        verbose (bool, optional): Whether to print progress. Defaults to True.
        use_interp (bool, optional): Whether to interpolate the peak. Defaults to False.

    Returns:
        tuple:
            - f1 (torch.Tensor): Optimized first band filter.
            - f2 (torch.Tensor): Optimized second band filter.
            - lam (float): Optimized lambda.
            - W_unit (torch.Tensor): Final Wiener kernel built with the optimized
              lambda (detached), ready for :func:`get_PSD_interpole_torch` etc.
            - J_values (list): J metric history.
            - lambda_values (list): lambda history.
    """
    if not torch.is_tensor(nps):
        nps = torch.as_tensor(nps, device=S.device)
    nps = nps.to(device=S.device, dtype=torch.float32)

    # Delay phases do not depend on W: precompute once (cf. precompute_constants).
    exp_term = torch.exp(-1j * w[None, :] * t[:, None])  # (len(t), len(w))

    n = len(S)
    activation_fct = torch.nn.Softplus(20, 20) if activation_fct is None else activation_fct
    f1_init = activation_fct(torch.rand(n // 2 + 1, dtype=torch.float)) if f1_init is None else f1_init.clone().abs()
    f2_init = activation_fct(torch.rand(n // 2 + 1, dtype=torch.float)) if f2_init is None else f2_init.clone().abs()
    if len(f1_init) > n // 2 + 1:
        f1_init = f1_init[:n // 2 + 1]
    if len(f2_init) > n // 2 + 1:
        f2_init = f2_init[:n // 2 + 1]
    f1_param = torch.nn.Parameter(f1_init.to(S.device))
    f2_param = torch.nn.Parameter(f2_init.to(S.device))
    log_lambda = torch.nn.Parameter(torch.log(torch.tensor(float(lambda_init), device=S.device)))

    lr_lambda = 1e-1 if lr_lambda is None else lr_lambda
    optimizer = torch.optim.Adam([
        {"params": [f1_param, f2_param], "lr": 1e-2},
        {"params": [log_lambda], "lr": lr_lambda},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_trials, eta_min=1e-2)
    J_values = []
    lambda_values = []
    J = torch.nan
    for step in range(n_trials):
        optimizer.zero_grad()
        lam = torch.exp(log_lambda)
        # Rebuild the Wiener kernel and its derived spectra for the current lambda.
        W_unit = compute_W_torch(S, nps, lam)
        S_H = S * W_unit
        S_H_delayed = S_H[None, :] * exp_term
        S2_over_nps = (S.abs() ** 2) / nps  # unused by the Wiener sigma, kept for signature
        # Enforce positivity and rebuild the full (Hermitian) spectrum.
        f1 = activation_fct(f1_param)
        f2 = activation_fct(f2_param)
        f1 = torch.cat([f1, f1[1:-1].flip(0)])
        f2 = torch.cat([f2, f2[1:-1].flip(0)])
        # Normalize filters such that mean(|f_i * W_unit * S|) = 1.
        norm1 = torch.mean(torch.abs(f1 * W_unit * S))
        norm2 = torch.mean(torch.abs(f2 * W_unit * S))
        f1 = f1 / norm1
        f2 = f2 / norm2
        # Wiener J metric with the current (lambda-dependent) kernel.
        J = compute_J_wiener(f1, f2, S_H_delayed, r, S_H, S2_over_nps, W_unit, S, nps, signal_amp,
                             ratio_distribution, N_sigma=N_sigma,
                             pulse_center_ratio=pulse_center_ratio, use_interp=use_interp)
        J_values.append(J.item())
        lambda_values.append(lam.item())
        J.backward()
        optimizer.step()
        scheduler.step()
        if step % 10 == 0 and verbose:
            print(f"Step {step}: J = {J.item():.6f}  lambda = {lam.item():.4f}")
    lam = torch.exp(log_lambda).detach()
    print(f"Final: J = {J.item():.6f}  lambda = {lam.item():.4f}")
    f1 = f1.detach()
    f2 = f2.detach()
    W_unit = compute_W_torch(S, nps, lam).detach()
    return f1, f2, lam.item(), W_unit, J_values, lambda_values


def optimize_filters_wiener_lambda_freq(S, w, t, r, nps, signal_amp, ratio_distribution, N_sigma=1.28,
                                        n_trials=500, activation_fct=None, pulse_center_ratio=0.5,
                                        f1_init=None, f2_init=None, lambda_init=1.0, lr_lambda=None,
                                        lambda_smooth=0.0, verbose=True, use_interp=False):
    """
    Wiener filter optimization with a FREQUENCY-DEPENDENT noise-modulation lambda(f).

    Generalizes :func:`optimize_filters_wiener_lambda`: instead of a single scalar
    lambda, the Wiener kernel carries one trainable lambda PER FREQUENCY,

        W_lambda(f) = S*(f) / ( |S(f)|^2 + lambda(f) * NPS(f) ),

    so the deconvolution-vs-noise-rejection balance is tuned independently at each
    frequency (low-f -> more deconvolution/sharpening, high-f -> more noise
    rejection, or vice versa, as the data prefer). lambda(f) is parametrized as
    ``exp(log_lambda)`` on the independent half of the spectrum (bins 0..N/2) and
    mirrored to a symmetric full-length vector, so the kernel stays Hermitian and the
    filtered pulse real. lambda(f) is optimized jointly with the band filters f1, f2
    by minimizing the J metric. A constant lambda(f) reproduces
    :func:`optimize_filters_wiener_lambda`; lambda(f) == 1 the standard Wiener filter.

    An optional smoothness regularization ``lambda_smooth`` penalizes the squared
    difference of adjacent log-lambda bins, discouraging lambda(f) from overfitting
    the single AP+NPS realization (set to 0 to disable).

    Args:
        S (torch.Tensor): FFT of the (windowed) mean pulse.
        w (torch.Tensor): Angular frequency tensor.
        t, r (torch.Tensor): Delay grid and pile-up energy-sharing ratio tensors.
        nps (torch.Tensor or numpy.ndarray): Noise power spectrum.
        signal_amp (float): Signal amplitude.
        ratio_distribution (torch.Tensor): Ratio distribution tensor.
        N_sigma (float, optional): Sigma threshold for the J metric. Defaults to 1.28.
        n_trials (int, optional): Optimization steps. Defaults to 500.
        f1_init, f2_init (torch.Tensor or None, optional): Initial band filters.
        lambda_init (float, optional): Initial value for every lambda(f) bin. Defaults to 1.0.
        lr_lambda (float or None, optional): Learning rate for log lambda(f). Defaults to 1e-1.
        lambda_smooth (float, optional): Weight of the smoothness penalty on
            log lambda(f). Defaults to 0.0 (off).
        verbose (bool, optional): Print progress. Defaults to True.
        use_interp (bool, optional): Interpolate the peak. Defaults to False.

    Returns:
        tuple:
            - f1 (torch.Tensor): optimized first band filter (full length).
            - f2 (torch.Tensor): optimized second band filter (full length).
            - lambda_half (numpy.ndarray): optimized lambda(f) on the independent
              half (bins 0..N/2, length N//2+1). Full: concat([h, h[-2:0:-1]]).
            - W_unit (torch.Tensor): final Wiener kernel with the optimized lambda(f).
            - J_values (list): J history.
            - lambda_mean_values (list): mean of lambda(f) per step (convergence).
    """
    if not torch.is_tensor(nps):
        nps = torch.as_tensor(nps, device=S.device)
    nps = nps.to(device=S.device, dtype=torch.float32)

    # Delay phases do not depend on W: precompute once.
    exp_term = torch.exp(-1j * w[None, :] * t[:, None])  # (len(t), len(w))

    n = len(S)
    half = n // 2 + 1
    activation_fct = torch.nn.Softplus(20, 20) if activation_fct is None else activation_fct
    f1_init = activation_fct(torch.rand(half, dtype=torch.float)) if f1_init is None else f1_init.clone().abs()
    f2_init = activation_fct(torch.rand(half, dtype=torch.float)) if f2_init is None else f2_init.clone().abs()
    if len(f1_init) > half:
        f1_init = f1_init[:half]
    if len(f2_init) > half:
        f2_init = f2_init[:half]
    f1_param = torch.nn.Parameter(f1_init.to(S.device))
    f2_param = torch.nn.Parameter(f2_init.to(S.device))
    # One trainable log-lambda per independent-frequency bin (mirrored to full length).
    log_lambda = torch.nn.Parameter(
        torch.full((half,), float(np.log(lambda_init)), device=S.device))

    lr_lambda = 1e-1 if lr_lambda is None else lr_lambda
    optimizer = torch.optim.Adam([
        {"params": [f1_param, f2_param], "lr": 1e-2},
        {"params": [log_lambda], "lr": lr_lambda},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_trials, eta_min=1e-2)
    J_values = []
    lambda_mean_values = []
    J = torch.nan
    for step in range(n_trials):
        optimizer.zero_grad()
        # lambda(f): positive half, mirrored to a symmetric full-length vector.
        lam_half = torch.exp(log_lambda)
        lam = torch.cat([lam_half, lam_half[1:-1].flip(0)])
        # Rebuild the (lambda(f)-dependent) Wiener kernel and derived spectra.
        W_unit = compute_W_torch(S, nps, lam)
        S_H = S * W_unit
        S_H_delayed = S_H[None, :] * exp_term
        S2_over_nps = (S.abs() ** 2) / nps  # unused by the Wiener sigma, kept for signature
        # Enforce positivity and rebuild the full (Hermitian) band filters.
        f1 = activation_fct(f1_param)
        f2 = activation_fct(f2_param)
        f1 = torch.cat([f1, f1[1:-1].flip(0)])
        f2 = torch.cat([f2, f2[1:-1].flip(0)])
        norm1 = torch.mean(torch.abs(f1 * W_unit * S))
        norm2 = torch.mean(torch.abs(f2 * W_unit * S))
        f1 = f1 / norm1
        f2 = f2 / norm2
        J = compute_J_wiener(f1, f2, S_H_delayed, r, S_H, S2_over_nps, W_unit, S, nps, signal_amp,
                             ratio_distribution, N_sigma=N_sigma,
                             pulse_center_ratio=pulse_center_ratio, use_interp=use_interp)
        loss = J
        if lambda_smooth:
            loss = loss + lambda_smooth * torch.mean((log_lambda[1:] - log_lambda[:-1]) ** 2)
        J_values.append(J.item())
        lambda_mean_values.append(lam_half.mean().item())
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % 10 == 0 and verbose:
            lm = lam_half.detach()
            print(f"Step {step}: J = {J.item():.6f}  lambda(f) "
                  f"[min {lm.min():.3g}, med {lm.median():.3g}, max {lm.max():.3g}]")
    lam_half = torch.exp(log_lambda).detach()
    lam = torch.cat([lam_half, lam_half[1:-1].flip(0)])
    print(f"Final: J = {J.item():.6f}  lambda(f) "
          f"[min {lam_half.min():.3g}, med {lam_half.median():.3g}, max {lam_half.max():.3g}]")
    f1 = f1.detach()
    f2 = f2.detach()
    W_unit = compute_W_torch(S, nps, lam).detach()
    return f1, f2, lam_half.cpu().numpy(), W_unit, J_values, lambda_mean_values


def get_PSD(dataset, H_unit, f1, f2, batch_size=128, window_fct=np.ones):
    """
    Computes the Pulse Shape Discriminator (PSD) for the given dataset.

    Args:
        dataset (torch.utils.data.Dataset): Input dataset.
        H_unit (torch.Tensor): Unit transfer function tensor.
        f1 (torch.Tensor): First filter tensor.
        f2 (torch.Tensor): Second filter tensor.
        batch_size (int, optional): Batch size for data loading. Defaults to 128.
        window_fct (function, optional): Window function to apply. Defaults to np.ones.

    Returns:
        numpy.ndarray: Computed PSD values.
    """
    Amp_1 = []
    Amp_2 = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for pulses in loader:
        pulses = pulses.numpy()
        window = window_fct(pulses.shape[1])  # Apply Hanning window
        pulses = pulses * window
        Amp_1.append(np.max(np.fft.ifft(f1 * H_unit * np.fft.fft(pulses, axis=-1), axis=1), axis=1))
        Amp_2.append(np.max(np.fft.ifft(f2 * H_unit * np.fft.fft(pulses, axis=-1), axis=1), axis=1))
    Amp_1 = np.concatenate(Amp_1)
    Amp_2 = np.concatenate(Amp_2)
    PSD = Amp_1 / Amp_2
    return PSD


def get_PSD_interpole(dataset, H_unit, f1, f2, pulse_center_ratio=0.5, pulse_start_pos = -100,
                      batch_size=128, interpolation_range=5, jitter_max=20,
                      filter=None, n_deriv = 0, filter_prederiv=None, window_fct=np.ones):
    """
    Computes the Pulse Shape Discriminator (PSD) for the given dataset.

    Args:
        dataset (torch.utils.data.Dataset): Input dataset.
        H_unit (torch.Tensor): Unit transfer function tensor.
        f1 (torch.Tensor): First filter tensor.
        f2 (torch.Tensor): Second filter tensor.
        pulse_center_ratio (float, optional): Ratio to determine pulse center. Defaults to 0.5.
        pulse_start_pos (int, optional): Position offset for pulse start. Defaults to -100.
        batch_size (int, optional): Batch size for data loading. Defaults to 128.
        interpolation_range (int, optional): Range for interpolation around the maximum. Defaults to 5.
        jitter_max (int, optional): Maximum jitter for peak search. Defaults to 20.
        filter (np.array or None): Filter array. Default is None
        n_deriv (int, optional): Number of derivatives to apply. Defaults to 0.
        filter_prederiv (np.array or None): Pre-derivative filter array. Default is None
        window_fct (function, optional): Window function to apply. Defaults to np.ones.
    Returns:
        tuple:
            - numpy.ndarray: Computed PSD values.
            - numpy.ndarray: Amplitudes from first filter.
            - numpy.ndarray: Amplitudes from second filter.
    """
    Amp_1 = []
    Amp_2 = []
    win_length = dataset.win_length
    target_index = int(win_length * pulse_center_ratio)
    baseline_index = target_index + pulse_start_pos
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    filter = filter if filter is not None else np.ones_like(H_unit, dtype=np.float32)
    H_1_filtered = f1 * H_unit * filter
    H_2_filtered = f2 * H_unit * filter
    window = window_fct(win_length)
    for pulses in loader:
        if type(pulses) is list:
            pulses = pulses[0]
        pulses = pulses.numpy()
        pulses -= np.mean(pulses[:baseline_index], axis=-1, keepdims=True)
        if filter_prederiv is not None:
            pulses = np.fft.ifft(filter_prederiv * np.fft.fft(pulses, axis=-1), axis=-1).real
        pulses = np.diff(pulses, n=n_deriv, prepend=np.zeros((pulses.shape[0], n_deriv)), axis=-1)
        pulses *= window
        n_rows = pulses.shape[0]
        fpulse = np.fft.fft(pulses, axis=-1)
        filtered_1 = np.roll(np.fft.ifft(H_1_filtered * fpulse, axis=1).real, shift=target_index, axis=1)
        max_idx_1 = (np.argmax(filtered_1[:, target_index-jitter_max:target_index+jitter_max], axis=1)
                     + target_index - jitter_max)
        filtered_2 = np.roll(np.fft.ifft(H_2_filtered * fpulse, axis=1).real, shift=target_index, axis=1)
        max_idx_2 = (np.argmax(filtered_2[:, target_index-jitter_max:target_index+jitter_max], axis=1)
                     + target_index - jitter_max)
        # interpolate to find max
        rows = np.arange(n_rows)
        col_offsets = np.arange(-interpolation_range, interpolation_range + 1)
        cols_1 = max_idx_1[:, None] + col_offsets
        y1 = filtered_1[rows[:, None], cols_1]
        interp1d_1 = interp1d(col_offsets, y1, kind='cubic', axis=1, bounds_error=False,
                              fill_value='extrapolate')
        cols_2 = max_idx_2[:, None] + col_offsets
        y2 = filtered_2[rows[:, None], cols_2]
        interp1d_2 = interp1d(col_offsets, y2, kind='cubic', axis=1, bounds_error=False,
                              fill_value='extrapolate')
        fine_x = np.arange(-interpolation_range, interpolation_range + 1, 0.05)
        Amp_1.append(np.max(interp1d_1(fine_x), axis=1))
        Amp_2.append(np.max(interp1d_2(fine_x), axis=1))
    Amp_1 = np.concatenate(Amp_1)
    Amp_2 = np.concatenate(Amp_2)
    PSD = Amp_1 / Amp_2
    return PSD, Amp_1, Amp_2


def get_PSD_interpole_torch(dataset, H_unit, f1, f2, pulse_center_ratio=0.5, pulse_start_pos = -100,
                            batch_size=128, interpolation_range=5,
                            jitter_max=20, filter=None,
                            n_deriv=0, filter_prederiv=None,
                            window_fct=torch.ones, use_loader=False):
    """
        Computes the Pulse Shape Discriminator (PSD) for the given dataset using PyTorch.

        Args:
            dataset (torch.utils.data.Dataset): Input dataset containing pulse data.
            H_unit (torch.Tensor): Unit transfer function tensor.
            f1 (torch.Tensor): First filter tensor.
            f2 (torch.Tensor): Second filter tensor.
            pulse_center_ratio (float, optional): Ratio to determine the pulse center. Defaults to 0.5.
            pulse_start_pos (int, optional): Offset for the start of the pulse. Defaults to -100.
            batch_size (int, optional): Batch size for data loading. Defaults to 128.
            interpolation_range (int, optional): Range for interpolation around the maximum. Defaults to 5.
            jitter_max (int, optional): Maximum jitter for peak search. Defaults to 20.
            filter (torch.Tensor or None, optional): Frequency-domain filter tensor. Defaults to None.
            n_deriv (int, optional): Number of derivatives to apply to the pulses. Defaults to 0.
            filter_prederiv (torch.Tensor or None, optional): Pre-derivative filter tensor. Defaults to None.
            window_fct (function, optional): Window function to apply to the pulses. Defaults to `torch.ones`.
            use_loader (bool, optional): Whether to use a data loader for batch iteration. Defaults to False.

        Returns:
            tuple:
                - torch.Tensor: Computed PSD values.
                - torch.Tensor: Amplitudes from the first filter.
                - torch.Tensor: Amplitudes from the second filter.
    """
    device = H_unit.device
    loader = batch_iterator(dataset, batch_size=batch_size, device='cpu', use_loader=use_loader)

    win_len = dataset.win_length
    target_index = int(win_len * pulse_center_ratio)
    baseline_index = target_index + pulse_start_pos
    if filter is None:
        filter = torch.ones(win_len, device=device)

    H_1_filtered = (f1 * H_unit * filter).to(dtype=torch.complex64)
    H_2_filtered = (f2 * H_unit * filter).to(dtype=torch.complex64)

    Amp_1, Amp_2 = [], []

    # prepare fine X grid for interpolation
    fine_x = torch.arange(-interpolation_range, interpolation_range + 1, 0.05,
                          device = device, dtype = torch.float32)

    # windowing
    window = window_fct(win_len)
    if not torch.is_tensor(window):
        window = torch.tensor(window, device=device, dtype=torch.float32)

    # Peak search band
    lo = target_index - jitter_max
    hi = target_index + jitter_max

    for pulses in loader:
        if isinstance(pulses, (list, tuple)):
            pulses = pulses[0]
        pulses = pulses.to(device, dtype=torch.float32)
        pulses = pulses - pulses[:, :baseline_index].mean(dim=1, keepdim=True)

        # optional pre-filter in freq
        if filter_prederiv is not None:
            fp = torch.fft.fft(pulses, dim=-1)
            pulses = torch.fft.ifft(fp * filter_prederiv, dim=-1).real

        # derivative
        if n_deriv > 0:
            pulses = pulses.diff(n=n_deriv, dim=-1, prepend=torch.zeros((pulses.shape[0], n_deriv), device=device))

        pulses = pulses * window

        # FFT of pulses
        fpulse = torch.fft.fft(pulses, dim=-1)

        # Filtered output 1
        filt1 = torch.fft.ifft(H_1_filtered * fpulse, dim=-1).real
        filt1 = torch.roll(filt1, shifts=target_index, dims=1)

        sub1 = filt1[:, lo:hi]
        max_idx1 = sub1.argmax(dim=1) + lo

        # Same for filter 2
        filt2 = torch.fft.ifft(H_2_filtered * fpulse, dim=-1).real
        filt2 = torch.roll(filt2, shifts=target_index, dims=1)
        sub2 = filt2[:, lo:hi]
        max_idx2 = sub2.argmax(dim=1) + lo

        # interpolation samples
        offs = torch.arange(-interpolation_range, interpolation_range + 1, device=device)

        idx1 = max_idx1[:, None] + offs[None, :]
        idx2 = max_idx2[:, None] + offs[None, :]

        y1 = filt1.gather(1, idx1)
        y2 = filt2.gather(1, idx2)

        # cubic interpolation
        interp1 = cubic_interp1d_scipy(y1, fine_x,offs)
        interp2 = cubic_interp1d_scipy(y2, fine_x,offs)

        # extract amplitudes
        Amp_1.append(interp1.max(dim=1).values)
        Amp_2.append(interp2.max(dim=1).values)

    Amp_1 = torch.cat(Amp_1)
    Amp_2 = torch.cat(Amp_2)
    PSD = Amp_1 / Amp_2
    return PSD, Amp_1, Amp_2


def create_NPS(dataset, filter=None, n_deriv = 0, filter_prederiv=None,
               rms_thr = 0.0005, batch_size=128, window_fct=np.hanning):
    """
    Computes the Noise Power Spectrum (NPS) for the given dataset.

    Args:
        dataset (torch.utils.data.Dataset): Input dataset.
        filter (np.array or None): Filter array. Default is None
        n_deriv (int, optional): Number of derivatives to apply. Defaults to 0.
        filter_prederiv (np.array or None): Pre-derivative filter array. Default is None
        rms_thr (float, optional): RMS threshold for pulse selection. Defaults to 0.0005.
        batch_size (int, optional): Batch size for data loading. Defaults to 128.
        window_fct (function, optional): Window function to apply. Defaults to np.hanning.

    Returns:
        numpy.ndarray: Computed NPS values.
    """
    nps_accum = None
    count = 0
    filter = filter if filter is not None else np.ones(dataset[0].shape[0], dtype=np.float32)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for pulses in loader:
        pulses = pulses.numpy()
        if filter_prederiv is not None:
            pulses = np.fft.ifft(filter_prederiv * np.fft.fft(pulses, axis=-1), axis=-1).real
        pulses = pulses[np.std(pulses, axis=-1) < rms_thr]
        pulses = pulses - np.mean(pulses, axis = -1)[:, None]
        pulses = np.diff(pulses, n=n_deriv, prepend=np.zeros((pulses.shape[0], n_deriv)), axis=-1)
        window = window_fct(pulses.shape[1])  # Apply Hanning window
        pulses = pulses * window
        fpulse = np.fft.fft(pulses, axis=-1)
        nps_batch = np.mean(np.abs(filter * fpulse)**2, axis=0)
        if nps_accum is None:
            nps_accum = nps_batch
        else:
            nps_accum += nps_batch
        count += 1
    nps = nps_accum / count
    return nps


def create_NPS_torch(dataset,
                     filter=None,
                     n_deriv=0,
                     filter_prederiv=None,
                     rms_thr=0.0005,
                     batch_size=128,
                     window_fct=np.hanning,
                     use_loader=True,
                     device="cuda"):
    """
        Computes the Noise Power Spectrum (NPS) for the given dataset using PyTorch.

        Args:
            dataset (torch.utils.data.Dataset): Input dataset containing pulse data.
            filter (torch.Tensor or None, optional): Frequency-domain filter tensor. Defaults to None.
            n_deriv (int, optional): Number of derivatives to apply to the pulses. Defaults to 0.
            filter_prederiv (torch.Tensor or None, optional): Pre-derivative filter tensor in the frequency domain.
             Defaults to None.
            rms_thr (float, optional): RMS threshold for pulse selection. Defaults to 0.0005.
            batch_size (int, optional): Batch size for data loading. Defaults to 128.
            window_fct (function, optional): Window function to apply to the pulses. Defaults to `np.hanning`.
            use_loader (bool, optional): Whether to use a data loader for batch iteration. Defaults to True.
            device (str, optional): Device to perform computations on (e.g., "cuda" or "cpu"). Defaults to "cuda".

        Returns:
            numpy.ndarray: Computed NPS values.
    """
    # Prepare filter
    win_len = dataset.win_length
    device = torch.device(device)

    if filter is None:
        filter = torch.ones(win_len, device=device, dtype=torch.float32)
    else:
        filter = torch.tensor(filter, device=device, dtype=torch.float32)

    # Pre-derivative filter (freq-domain)
    if filter_prederiv is not None:
        filter_prederiv = torch.tensor(filter_prederiv,
                                       device=device,
                                       dtype=torch.complex64)

    # windowing
    window = window_fct(win_len)
    if not torch.is_tensor(window):
        window = torch.tensor(window, device=device, dtype=torch.float32)

    # Batch iterator (same helper as PSD)
    loader = batch_iterator(dataset,
                            batch_size=batch_size,
                            device="cpu",  # pulses come to GPU later
                            use_loader=use_loader)

    nps_accum = None
    count = 0

    for pulses in loader:
        # Allow dataset returning (pulses, label)
        if isinstance(pulses, (list, tuple)):
            pulses = pulses[0]

        pulses = pulses.to(device, dtype=torch.float32)

        # ======== FILTER BEFORE DIFF (F-domain pre-filter) ========
        if filter_prederiv is not None:
            fp = torch.fft.fft(pulses, dim=-1)
            pulses = torch.fft.ifft(fp * filter_prederiv, dim=-1).real

        # ======== RMS THRESHOLD FILTERING ========
        rms = pulses.std(dim=-1)
        mask = rms < rms_thr
        pulses = pulses[mask]
        if pulses.numel() == 0:
            continue

        # ======== REMOVE BASELINE ========
        pulses = pulses - pulses.mean(dim=-1, keepdim=True)

        # ======== N-th DERIVATIVE ========
        if n_deriv > 0:
            zeros = torch.zeros((pulses.shape[0], n_deriv),
                                device=device, dtype=pulses.dtype)
            padded = torch.cat([zeros, pulses], dim=-1)
            pulses = padded.diff(n=n_deriv, dim=-1)

        # ======== APPLY WINDOW ========
        pulses *= window

        # ======== FFT ========
        fpulse = torch.fft.fft(pulses, dim=-1)

        # ======== POWER SPECTRUM ========
        nps_batch = torch.mean(torch.abs(filter * fpulse)**2, dim=0)

        # Accumulate
        if nps_accum is None:
            nps_accum = nps_batch
        else:
            nps_accum += nps_batch

        count += 1

    # ======== FINAL AVERAGE ========
    nps = nps_accum / count
    return nps.detach().cpu().numpy()


def compute_BI(data_pileup, data_single, acceptance, H_unit, f1_opt, f2_opt,
               window_fct = np.hanning, n_deriv = 0, full_output = False):
    """
        Computes the Background Index (BI) for the given datasets.

        Args:
            data_pileup (torch.utils.data.Dataset): Dataset containing pileup pulses.
            data_single (torch.utils.data.Dataset): Dataset containing single pulses.
            acceptance (float): Acceptance level (fraction of single pulses to retain).
            H_unit (torch.Tensor): Unit transfer function tensor.
            f1_opt (torch.Tensor): Optimized first filter tensor.
            f2_opt (torch.Tensor): Optimized second filter tensor.
            window_fct (function, optional): Window function to apply to the pulses. Defaults to `np.hanning`.
            n_deriv (int, optional): Number of derivatives to apply to the pulses. Defaults to 0.
            full_output (bool, optional): Whether to return additional intermediate results. Defaults to False.

        Returns:
            tuple: If `full_output` is False, returns:
                - BI (float): Computed Background Index.
                - rp (float): Fraction of pileup pulses below the threshold.
            If `full_output` is True, returns:
                - BI (float): Computed Background Index.
                - rp (float): Fraction of pileup pulses below the threshold.
                - PSD_pileup (numpy.ndarray): PSD values for pileup pulses.
                - PSD_single (numpy.ndarray): PSD values for single pulses.
                - Amp_1_pileup (numpy.ndarray): Amplitudes from the first filter for pileup pulses.
                - Amp_2_pileup (numpy.ndarray): Amplitudes from the second filter for pileup pulses.
                - Amp_1_single (numpy.ndarray): Amplitudes from the first filter for single pulses.
                - Amp_2_single (numpy.ndarray): Amplitudes from the second filter for single pulses.
    """
    PSD_pileup, Amp_1_pileup, Amp_2_pileup = get_PSD_interpole(data_pileup, H_unit, f1_opt, f2_opt,
                                                               window_fct = window_fct, n_deriv = n_deriv)
    PSD_single, Amp_1_single, Amp_2_single = get_PSD_interpole(data_single, H_unit, f1_opt, f2_opt,
                                                               window_fct = window_fct, n_deriv = n_deriv)
    cut = np.percentile(PSD_single, 100 - acceptance * 100)
    rp = np.mean(PSD_pileup < cut)
    BI = fn.K * (1-rp)
    if full_output:
        return BI, rp, PSD_pileup, PSD_single, Amp_1_pileup, Amp_2_pileup, Amp_1_single, Amp_2_single
    else:
        return BI, rp


def compute_BI_uncertainty(PSD_single, PSD_pileup, acceptance, rp, K=fn.K):
    """
        Computes the uncertainty in the Background Index (BI) calculation.

        Args:
            PSD_single (numpy.ndarray): PSD values for single pulses.
            PSD_pileup (numpy.ndarray): PSD values for pileup pulses.
            acceptance (float): Acceptance level (fraction of single pulses to retain).
            rp (float): Fraction of pileup pulses below the threshold.
            K (float, optional): Scaling factor for the Background Index. Defaults to `fn.K`.

        Returns:
            tuple:
                - sigma_rp (float): Uncertainty in the fraction of pileup pulses below the threshold.
                - sigma_BI (float): Uncertainty in the Background Index.
    """
    N_single = len(PSD_single)
    N_pileup = len(PSD_pileup)
    sigma_rp_pileup = np.sqrt(rp * (1 - rp) / N_pileup)
    cut = np.percentile(PSD_single, 100 - acceptance*100)
    kde_single = gaussian_kde(PSD_single)
    kde_pileup = gaussian_kde(PSD_pileup)
    f_single = kde_single.evaluate(cut)[0]
    f_pileup = kde_pileup.evaluate(cut)[0]
    a = 1 - acceptance
    sigma_cut = np.sqrt(a*(1-a)) / (np.sqrt(N_single) * f_single)
    drp_dcut = -f_pileup
    sigma_rp_cut = abs(drp_dcut) * sigma_cut
    sigma_rp = np.sqrt(sigma_rp_pileup**2 + sigma_rp_cut**2)
    sigma_BI = K * sigma_rp
    return sigma_rp, sigma_BI


def compute_BI_torch(data_pileup,
                     data_single,
                     acceptance,
                     H_unit,
                     f1_opt,
                     f2_opt,
                     K=None,
                     window_fct=torch.ones,
                     n_deriv=0,
                     compute_uncertainty=False,
                     full_output=False,
                     **kwargs):
    """
       Computes the Background Index (BI) for the given datasets using PyTorch.

       Args:
           data_pileup (torch.utils.data.Dataset): Dataset containing pileup pulses.
           data_single (torch.utils.data.Dataset): Dataset containing single pulses.
           acceptance (float): Acceptance level (fraction of single pulses to retain).
           H_unit (torch.Tensor): Unit transfer function tensor.
           f1_opt (torch.Tensor): Optimized first filter tensor.
           f2_opt (torch.Tensor): Optimized second filter tensor.
           K (float, optional): Scaling factor for the Background Index. Defaults to `fn.K` if not provided.
           window_fct (function, optional): Window function to apply to the pulses. Defaults to `torch.ones`.
           n_deriv (int, optional): Number of derivatives to apply to the pulses. Defaults to 0.
           compute_uncertainty (bool, optional): Whether to compute the uncertainty in the BI calculation.
           Defaults to False.
           full_output (bool, optional): Whether to return additional intermediate results. Defaults to False.
           **kwargs: Additional keyword arguments passed to `get_PSD_interpole_torch`.

       Returns:
           tuple: If `compute_uncertainty` is False:
               - BI (torch.Tensor): Computed Background Index.
               - rp (torch.Tensor): Fraction of pileup pulses below the threshold.
               If `full_output` is True, also returns:
               - PSD_pileup (torch.Tensor): PSD values for pileup pulses.
               - PSD_single (torch.Tensor): PSD values for single pulses.
               - Amp_1_pileup (torch.Tensor): Amplitudes from the first filter for pileup pulses.
               - Amp_2_pileup (torch.Tensor): Amplitudes from the second filter for pileup pulses.
               - Amp_1_single (torch.Tensor): Amplitudes from the first filter for single pulses.
               - Amp_2_single (torch.Tensor): Amplitudes from the second filter for single pulses.
           If `compute_uncertainty` is True:
               - BI (torch.Tensor): Computed Background Index.
               - rp (torch.Tensor): Fraction of pileup pulses below the threshold.
               - sigma_rp (torch.Tensor): Uncertainty in the fraction of pileup pulses below the threshold.
               - sigma_BI (torch.Tensor): Uncertainty in the Background Index.
               If `full_output` is True, also returns:
               - PSD_pileup (torch.Tensor): PSD values for pileup pulses.
               - PSD_single (torch.Tensor): PSD values for single pulses.
               - Amp_1_pileup (torch.Tensor): Amplitudes from the first filter for pileup pulses.
               - Amp_2_pileup (torch.Tensor): Amplitudes from the second filter for pileup pulses.
               - Amp_1_single (torch.Tensor): Amplitudes from the first filter for single pulses.
               - Amp_2_single (torch.Tensor): Amplitudes from the second filter for single pulses.

       Raises:
           ValueError: If `K` is not provided and `fn.K` is not found.
    """
    if K is None:
        try:
            K = fn.K
        except NameError:
            raise ValueError("K not provided and fn.K not found. Pass K explicitly.")

    # Expect PSDs and amps as torch Tensors
    PSD_pileup, Amp_1_pileup, Amp_2_pileup = get_PSD_interpole_torch(
        data_pileup, H_unit, f1_opt, f2_opt, window_fct=window_fct, n_deriv=n_deriv, **kwargs
    )
    PSD_single, Amp_1_single, Amp_2_single = get_PSD_interpole_torch(
        data_single, H_unit, f1_opt, f2_opt, window_fct=window_fct, n_deriv=n_deriv, **kwargs
    )

    # ensure 1-D tensors
    PSD_pileup = PSD_pileup.flatten()
    PSD_single = PSD_single.flatten()

    # percentile: torch.quantile expects q in [0,1]
    # We want the value at 100 - acceptance*100 percentile => 1 - acceptance
    q = torch.tensor(1.0 - acceptance, device=PSD_single.device, dtype=PSD_single.dtype)
    cut = torch.quantile(PSD_single, q)

    rp = torch.mean((PSD_pileup < cut).to(dtype=PSD_pileup.dtype))  # fraction of pileup below cut
    BI = K * (1.0 - rp)
    if compute_uncertainty:
        sigma_rp, sigma_BI = compute_BI_uncertainty_torch(
            PSD_single, PSD_pileup, acceptance, rp, K=K
        )
        if full_output:
            return (BI, rp, sigma_rp, sigma_BI, PSD_pileup, PSD_single, Amp_1_pileup, Amp_2_pileup,
                    Amp_1_single, Amp_2_single)
        else:
            return BI, rp, sigma_rp, sigma_BI
    if full_output:
        return BI, rp, PSD_pileup, PSD_single, Amp_1_pileup, Amp_2_pileup, Amp_1_single, Amp_2_single
    else:
        return BI, rp


def compute_BI_uncertainty_torch(PSD_single: torch.Tensor,
                                 PSD_pileup: torch.Tensor,
                                 acceptance: float,
                                 rp: torch.Tensor,
                                 K=None,
                                 eps: float = 1e-12):
    """
        Computes the uncertainty in the Background Index (BI) calculation using PyTorch.

        Args:
            PSD_single (torch.Tensor): PSD values for single pulses. Shape: (N_single,).
            PSD_pileup (torch.Tensor): PSD values for pileup pulses. Shape: (N_pileup,).
            acceptance (float): Acceptance level (fraction of single pulses to retain).
            rp (torch.Tensor): Fraction of pileup pulses below the threshold.
            K (float, optional): Scaling factor for the Background Index. Defaults to `fn.K` if not provided.
            eps (float, optional): Small value to prevent division by zero. Defaults to 1e-12.

        Returns:
            tuple:
                - sigma_rp (torch.Tensor): Uncertainty in the fraction of pileup pulses below the threshold.
                - sigma_BI (torch.Tensor): Uncertainty in the Background Index.

        Raises:
            ValueError: If `PSD_single` or `PSD_pileup` is empty, or if `K` is not provided and `fn.K` is not found.
    """
    device = PSD_single.device
    dtype = PSD_single.dtype

    if K is None:
        try:
            K = fn.K
        except NameError:
            raise ValueError("K not provided and fn.K not found. Pass K explicitly.")

    N_single = PSD_single.numel()
    N_pileup = PSD_pileup.numel()

    if N_single == 0 or N_pileup == 0:
        raise ValueError("PSD_single and PSD_pileup must be non-empty tensors")

    # --- 1. Uncertainty from pileup binomial statistics ---
    # rp is a torch scalar
    sigma_rp_pileup = torch.sqrt(rp * (1.0 - rp) / N_pileup)

    # --- 2. Uncertainty of the percentile cut ---
    q = torch.tensor(1.0 - acceptance, device=device, dtype=dtype)
    cut = torch.quantile(PSD_single, q)

    # estimate PDFs for PSD_single and PSD_pileup using torch KDE
    # evaluate kde at `cut` (scalar)
    cut_tensor = cut.reshape(1)  # shape (1,)
    f_single = gaussian_kde_torch(cut_tensor, PSD_single, eps=eps)[0].clamp_min(eps)
    f_pileup = gaussian_kde_torch(cut_tensor, PSD_pileup, eps=eps)[0].clamp_min(eps)

    a = 1.0 - acceptance
    sigma_cut = math.sqrt(a * (1.0 - a)) / (math.sqrt(N_single) * f_single)

    # derivative of rp wrt cut (drp/dcut) = -pdf_pileup(cut)  (for continuous pdf)
    drp_dcut = -f_pileup
    sigma_rp_cut = torch.abs(drp_dcut) * sigma_cut

    # total
    sigma_rp = torch.sqrt(sigma_rp_pileup ** 2 + sigma_rp_cut ** 2)
    sigma_BI = K * sigma_rp

    return sigma_rp, sigma_BI


def build_mean_pulse(dataset, rms_thr, pulse_center_ratio=0.5, pulse_start_pos = -100, pulse_end_pos = 400,
                     amplitude_bounds=None,
                     batch_size=2048, device="cpu", use_loader=False):
    """
        Builds the mean pulse from a dataset by aligning and normalizing waveforms.

        Args:
            dataset (torch.utils.data.Dataset): Input dataset containing pulse data.
            rms_thr (float): RMS threshold for pulse selection.
            pulse_center_ratio (float, optional): Ratio to determine the pulse center. Defaults to 0.5.
            pulse_start_pos (int, optional): Offset for the start of the pulse. Defaults to -100.
            pulse_end_pos (int, optional): Offset for the end of the pulse. Defaults to 400.
            batch_size (int, optional): Batch size for data loading. Defaults to 2048.
            device (str, optional): Device to perform computations on (e.g., "cpu" or "cuda"). Defaults to "cpu".
            use_loader (bool, optional): Whether to use a data loader for batch iteration. Defaults to False.

        Returns:
            torch.Tensor: The computed mean pulse.
    """
    win_length = dataset.win_length
    loader = batch_iterator(dataset, batch_size=batch_size, device='cpu', use_loader=use_loader)
    target_index = int(win_length * pulse_center_ratio)
    baseline_length = target_index + pulse_start_pos

    mean_pulse = torch.zeros(win_length, dtype=torch.float32, device=device)
    total_count = 0

    for pulses in loader:
        if isinstance(pulses, (list, tuple)):
            pulses = pulses[0]
        pulses = pulses.to(device, dtype=torch.float32)

        baseline_raw = torch.cat((
            pulses[:, :baseline_length],
            pulses[:, target_index + pulse_end_pos:]
        ), dim=1)

        sel = torch.std(baseline_raw, dim=-1) < rms_thr
        pulses_sel = pulses[sel]
        if pulses_sel.shape[0] == 0:
            continue

        baseline = pulses_sel[:, :baseline_length].mean(dim=1)
        pulses_clean = pulses_sel - baseline[:, None]

        max_pos = torch.argmax(pulses_clean, dim=1)
        if amplitude_bounds is not None:
            max_val = pulses_clean[torch.arange(pulses_clean.shape[0]), max_pos]
            amp_sel = (max_val >= amplitude_bounds[0]) & (max_val <= amplitude_bounds[1])
            pulses_clean = pulses_clean[amp_sel]
            max_pos = max_pos[amp_sel]
            if pulses_clean.shape[0] == 0:
                continue
        shifts = target_index - max_pos

        pulses_aligned = align_waveforms(pulses_clean, shifts)

        max_val = pulses_aligned.max(dim=1).values
        pulses_norm = pulses_aligned / max_val[:, None]

        mean_pulse += pulses_norm.sum(dim=0)
        total_count += pulses_norm.shape[0]

    mean_pulse /= total_count
    return mean_pulse


def build_mean_pulse_filteralignement(dataset, rms_thr, H, pulse_center_ratio=0.5,
                                      pulse_start_pos=-100, pulse_end_pos=400,
                                      amplitude_bounds=None, return_amp=False, return_pulses=False,
                                      batch_size=2048, device="cpu", use_loader=False):
    """
        Builds the mean pulse from a dataset by aligning and normalizing waveforms using
        a filter-based alignment method.

        Args:
            dataset (torch.utils.data.Dataset): Input dataset containing pulse data.
            rms_thr (float): RMS threshold for pulse selection.
            H (torch.Tensor): Frequency-domain filter tensor.
            pulse_center_ratio (float, optional): Ratio to determine the pulse center. Defaults to 0.5.
            pulse_start_pos (int, optional): Offset for the start of the pulse. Defaults to -100.
            pulse_end_pos (int, optional): Offset for the end of the pulse. Defaults to 400.
            batch_size (int, optional): Batch size for data loading. Defaults to 2048.
            device (str, optional): Device to perform computations on (e.g., "cpu" or "cuda"). Defaults to "cpu".
            use_loader (bool, optional): Whether to use a data loader for batch iteration. Defaults to False.

        Returns:
            torch.Tensor: The computed mean pulse.
    """
    win_length = dataset.win_length
    loader = batch_iterator(dataset, batch_size=batch_size, device='cpu', use_loader=use_loader)
    target_index = int(win_length * pulse_center_ratio)
    baseline_length = target_index + pulse_start_pos
    jitter_max = 4

    mean_pulse = torch.zeros(win_length, dtype=torch.float32, device=device)
    total_count = 0

    interpolation_range = 5
    fine_x = torch.arange(-interpolation_range, interpolation_range + 1, 0.05, device=device)

    if return_amp:
        amplitudes = []
    if return_pulses:
        pulses_out = []
    for pulses in loader:
        if isinstance(pulses, (list, tuple)):
            pulses = pulses[0]
        pulses = pulses.to(device, dtype=torch.float32)

        baseline_raw = torch.cat((
            pulses[:, :baseline_length],
            pulses[:, target_index + pulse_end_pos:]
        ), dim=1)
        sel = torch.std(baseline_raw, dim=-1) < rms_thr
        pulses_sel = pulses[sel]
        if pulses_sel.shape[0] == 0:
            continue

        baseline = pulses_sel[:, :baseline_length].mean(dim=1)
        pulses_clean = pulses_sel - baseline[:, None]
        if amplitude_bounds is not None:
            max_val = pulses_clean.max(dim=1).values
            amp_sel = (max_val >= amplitude_bounds[0]) & (max_val <= amplitude_bounds[1])
            pulses_clean = pulses_clean[amp_sel]
            if pulses_clean.shape[0] == 0:
                continue
        fpulse = torch.fft.fft(pulses_clean, dim=-1)

        lo = target_index - jitter_max
        hi = target_index + jitter_max

        filt = torch.fft.ifft(H * fpulse, dim=-1).real
        filt = torch.roll(filt, shifts=target_index, dims=1)

        sub = filt[:, lo:hi]
        max_idx = sub.argmax(dim=1) + lo

        offs = torch.arange(-interpolation_range, interpolation_range + 1, device=device)
        idx = max_idx[:, None] + offs[None, :]
        y = filt.gather(1, idx)

        interp2 = cubic_interp1d_scipy(y, fine_x,offs)
        shift = fine_x[interp2.argmax(dim=1)]

        phase = torch.exp(-1j * 2 * np.pi * shift[:, None] / win_length)
        pulses_aligned = torch.fft.ifft(fpulse * phase, dim=-1).real

        max_val = pulses_aligned.max(dim=1).values
        if return_amp:
            amplitudes.append(max_val)
        if return_pulses:
            pulses_out.append(pulses_aligned.clone())
        pulses_norm = pulses_aligned / max_val[:, None]

        mean_pulse += pulses_norm.sum(dim=0)
        total_count += pulses_norm.shape[0]

    mean_pulse /= total_count
    if return_amp:

        amplitudes = None if len(amplitudes)==0 else torch.cat(amplitudes)
        return mean_pulse, amplitudes
    if return_pulses:
        pulses_out = torch.cat(pulses_out)
        return mean_pulse, pulses_out
    return mean_pulse


def build_mean_pulse_filteralignement_from_raw(dataset, rms_thr, nps, pulse_center_ratio=0.5,
                                      pulse_start_pos=-100, pulse_end_pos=400,
                                      amplitude_bounds=None, return_amp=False, return_pulses=False,
                                      batch_size=2048, device="cpu", use_loader=False):

    mean_pulse = build_mean_pulse(dataset, rms_thr = rms_thr,
                                     amplitude_bounds = amplitude_bounds,
                                     batch_size = batch_size, device = device,
                                     use_loader = use_loader)
    S, w, H_unit = compute_H(mean_pulse, nps, np.hanning)
    H_unit_torch = torch.tensor(H_unit, dtype = torch.cfloat, device = device)
    return build_mean_pulse_filteralignement(dataset, rms_thr, H_unit_torch,
                                                            pulse_center_ratio=pulse_center_ratio,
                                      pulse_start_pos=pulse_start_pos, pulse_end_pos=pulse_end_pos,
                                      amplitude_bounds=amplitude_bounds,return_amp=return_amp,
                                             return_pulses=return_pulses,
                                      batch_size=batch_size, device=device, use_loader=use_loader)