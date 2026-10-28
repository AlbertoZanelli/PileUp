import torch
import math
import csv
import os
from functools import lru_cache
# def calculate_K_factor(Activity = 3.09E-03,  # Activity in Bq (N/s)
#                         LMO_MASS = 0.308,  # Mass of LMO in kg
#                         second_in_year = 3600 * 24 * 365,  # Number of seconds in a year
#                         Probability = 3.41E-04,  # Given probability factor
#                         correction_factor = 1,  # Factor for corrections
#                         dt = 0.8 , # Example exposure time in seconds
#                         milliseconds_in_second = 1000) -> float:
def calculate_K_factor(Activity = 2.94E-03,  # Activity in Bq (N/s)
                        LMO_MASS = 0.28,  # Mass of LMO in kg
                        second_in_year = 3600 * 24 * 365,  # Number of seconds in a year
                        Probability = 3.41E-04,  # Given probability factor
                        correction_factor = 1,  # Factor for corrections
                        dt = 0.8 , # Example exposure time in seconds
                        milliseconds_in_second = 1000) -> float:
    """
    Calculate the background interference (BI) using given parameters.

    The formula used to calculate the BI is based on the activity of a source,
    the rejection power, the exposure time, and other constants. The formula is:

    BI = (correction_factor * Probability * (1 - rej_power) * Activity^2) /
        (LMO_MASS * second_in_year) * dt / milliseconds_in_second

    Parameters
    ----------
    rej_power : float
        The rejection power (fraction of rejection).
    dt : float
        The exposure time in seconds.

    Returns
    -------
    float
        The calculated background interference (BI) value in appropriate units.
    """
    K = (correction_factor * Probability * Activity * Activity /
            LMO_MASS * second_in_year * dt / milliseconds_in_second)
    # BI calculation formula based on provided parameters
    return K

K = calculate_K_factor()


# ── Generazione v2 degli eventi del Monte Carlo (2026-10-25) ─────────────────────────────────
# Le aggiunte della generazione degli eventi del paper (Berest 2026, sez. 4.2) e il nuovo
# ritardo massimo. Le usano src.simulation.simulate_frequency_pulses_v2,
# build_simAP_injected_m205.py e simulate_BI_error_m205.py.
DT_MAX_V2  = 2.369e-3        # [s] ritardo massimo del pile-up = rise time 10-90% massimo dei 75
                             #     punti (ch34, 0.6 V, amplitudes_m205.csv). Prima 0.8 ms (paper).
Q_BB_MEV   = 3.034           # Q-valore del 100Mo
E_ROI_MEV  = (3.019, 3.049)  # l'energia SOMMA dell'evento si estrae in questa ROI (paper, sez. 4.2)
PHOTON_EV  = 1239.84 / 670  # energia di un fotone di scintillazione [eV]: 670 nm -> 1.85 eV (relatore).
                             # Il codice del paper usava 2.30 eV, Armatol 2026 (sez. 5.3) 2.1 eV.
LY_KEV_MEV = {31: 0.17, 34: 0.18, 71: 0.17, 83: 0.20, 91: 0.22}   # light yield prima dell'NTL,
                             # LMO vicino, misurato da Denys (m92; foglio "Denys Comparison" di
                             # ResultsCaraterization.xlsx; CROSS status 16/04/2026 p.7). Errori:
                             # 0.09, 0.05, 0.05, 0.08, 0.08.

# ── Proiezione alle condizioni di CUPID (2026-10-27) ─────────────────────────────────────────
# RIMANDATA (CUPID = False nei programmi di training): G_LMO qui sotto e' un guadagno in AMPIEZZA,
# mentre G_CUPID e' un guadagno in SNR; prima di usarla va deciso G_ch (tesi_note/riscalamento_CUPID.md).
# A_CUPID = A_m205 * (G_CUPID * LY_CUPID) / (G_LMO * LY): cambiano la luce che arriva al LD e il
# guadagno NTL sulla luce di scintillazione; NTD e rumore restano quelli di m205.
LY_CUPID = 0.36              # [keV/MeV] baseline CUPID (Pageot, cap. 6: 0.3 CROSS RUN9, 0.36 CUPID)
G_CUPID  = 11.3              # guadagno NTL efficace, config. III di Armatol 2026 (sez. 5.3): eq. 5.1 con
                             # G_CROSS = 8.66 (tab. 4) e xi = 0.56 -> G_NTL = 14.7; eq. 5.3 con xi = 0.75
                             # (tab. 5) -> 0.25 + 0.75 * 14.7. Non e' scritto nell'articolo.
# Guadagno NTL sulla luce del LMO vicino in m205. Misurato solo col LED (m199, tutti a 120 V,
# status p.16); il rapporto LED/LMO viene da m96 (status p.11), scalato su G - 1 (che va come V).
_G_LED_M199 = {31: 85, 34: 73, 71: 72, 83: 63, 91: 10}
_G_M96      = {31: (65, 30), 34: (38, 18), 71: (37, 18), 83: (17, 8), 91: (7, 5)}   # (LED, LMO)
G_LMO = {ch: 1 + (_G_LED_M199[ch] - 1) * (lmo - 1) / (led - 1) for ch, (led, lmo) in _G_M96.items()}


def cupid_factor(channel):
    """Fattore che porta l'ampiezza di m205 del canale alle condizioni di CUPID."""
    return G_CUPID * LY_CUPID / (G_LMO[channel] * LY_KEV_MEV[channel])


def K_for(dt_max_s):
    """K del BI per un ritardo massimo dt_max_s [s]. K e' proporzionale alla finestra dei ritardi:
    K_for(8e-4) e' il K di sempre."""
    return calculate_K_factor(dt=dt_max_s * 1e3)


@lru_cache(maxsize=None)
def _risetimes():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "amplitudes_m205.csv")
    with open(path, newline="") as f:
        return {(int(r["channel"]), round(float(r["vbias_V"]), 3)): float(r["risetime_ms"]) * 1e-3
                for r in csv.DictReader(f) if r["risetime_ms"]}


def dt_max_for(t_max, channel, vbias):
    """Ritardo massimo del pile-up [s] di un punto: t_max, oppure, se t_max == "rise", il rise
    time 10-90% dell'AP di Octopus di (canale, V_bias) (colonna risetime_ms di amplitudes_m205.csv)."""
    return _risetimes()[(int(channel), round(float(vbias), 3))] if t_max == "rise" else t_max


def dt_tag(t_max):
    """Suffisso delle cartelle per il ritardo massimo: "" per 8e-4 (tesi), "_dtrise" per il rise
    time del punto, "_dt<ms>ms" per un valore fisso."""
    return "" if t_max == 8e-4 else "_dtrise" if t_max == "rise" else f"_dt{t_max * 1e3:g}ms"


def n_photons(channel, energy_mev=Q_BB_MEV, cupid=False):
    """Fotoni di scintillazione assorbiti dal light detector per un evento di energy_mev:
        N = LY [keV/MeV] * E [MeV] * 1000 [eV/keV] / PHOTON_EV [eV],
    con LY del canale, o LY_CUPID se cupid. Statistica di Poisson -> sigma_A / A = 1 / sqrt(N)."""
    return (LY_CUPID if cupid else LY_KEV_MEV[channel]) * energy_mev * 1e3 / PHOTON_EV

def cubic_interp1d_scipy(y, fine_x, x=None):
    """
        Performs cubic interpolation on the input data `y` using the Catmull-Rom spline method.

        Args:
            y (torch.Tensor): Input tensor of shape (..., S), where S is the number of samples.
            fine_x (torch.Tensor): Tensor of interpolation points. Must be within the range [0, S-1].

        Returns:
            torch.Tensor: Interpolated values at the specified `fine_x` points. The output shape is
                          the same as the input `fine_x` with additional batch dimensions from `y`.
    """
    if not torch.is_tensor(fine_x):
        fine_x = torch.as_tensor(fine_x, device=y.device, dtype=y.dtype)

    *batch, S = y.shape
    if S < 2:
        raise ValueError("Need at least 2 samples for interpolation")

    y = y.reshape(-1, S)   # (B, S)
    B = y.shape[0]
    N = fine_x.numel()

    # ----------------------------------
    # x grid
    # ----------------------------------
    if x is None:
        x = torch.arange(S, device=y.device, dtype=y.dtype)
    else:
        if x.ndim != 1 or x.numel() != S:
            raise ValueError("x must have shape (S,)")
        x = x.to(device=y.device, dtype=y.dtype)

    # ----------------------------------
    # Tangents (Catmull–Rom)
    # ----------------------------------
    m = torch.zeros_like(y)

    dx = x[2:] - x[:-2]
    m[:, 1:-1] = (y[:, 2:] - y[:, :-2]) / dx

    m[:, 0]  = (y[:, 1]  - y[:, 0])  / (x[1]  - x[0])
    m[:, -1] = (y[:, -1] - y[:, -2]) / (x[-1] - x[-2])

    # ----------------------------------
    # Interval search
    # ----------------------------------
    fine_x = fine_x.clamp(x[0], x[-1] - 1e-12)
    i = torch.searchsorted(x, fine_x, right=True) - 1
    i = i.clamp(0, S - 2)

    x_i  = x[i]
    x_ip = x[i + 1]

    t = ((fine_x - x_i) / (x_ip - x_i))[None, :]  # (1, N)

    y_i  = y.gather(1, i.expand(B, -1))
    y_ip = y.gather(1, (i + 1).expand(B, -1))

    m_i  = m.gather(1, i.expand(B, -1))
    m_ip = m.gather(1, (i + 1).expand(B, -1))

    # ----------------------------------
    # Hermite basis
    # ----------------------------------
    t2 = t * t
    t3 = t2 * t

    h00 =  2*t3 - 3*t2 + 1
    h10 =      t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 =      t3 -    t2

    # ----------------------------------
    # Cubic spline
    # ----------------------------------
    out = (
        h00 * y_i +
        h10 * (x_ip - x_i)[None, :] * m_i +
        h01 * y_ip +
        h11 * (x_ip - x_i)[None, :] * m_ip
    )

    return out.view(*batch, N)


def gaussian_kde_torch(x: torch.Tensor, points: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
        Estimates the probability density function (PDF) of a given set of points using Gaussian Kernel Density Estimation (KDE).

        Args:
            x (torch.Tensor): Tensor of points where the PDF is to be evaluated. Shape: (...,).
            points (torch.Tensor): Tensor of data points used to estimate the PDF. Shape: (N,).
            eps (float, optional): Small value to prevent division by zero. Defaults to 1e-12.

        Returns:
            torch.Tensor: Estimated PDF values at the points in `x`. Shape: (...,).
    """
    points = points.to(x.device)
    N = points.numel()
    if N == 0:
        return torch.zeros_like(x)
    std = torch.std(points, unbiased=True)
    std = std.clamp_min(eps)
    h = 1.06 * std * (N ** (-1/5))
    h = h.clamp_min(eps)
    diff = (x.unsqueeze(-1) - points.unsqueeze(0)) / h  # (..., N)
    exponent = -0.5 * diff * diff
    pdf = torch.exp(exponent).sum(dim=-1) / (N * h * math.sqrt(2 * math.pi))
    return pdf


def align_waveforms(pulses, shift):
    """
        Aligns waveforms by shifting them along the time axis without circular wrap-around.

        Args:
            pulses (torch.Tensor): A tensor of shape (N, L) representing N waveforms, each of length L.
            shift (torch.Tensor): A tensor of shape (N,) containing integer shifts for each waveform
                                  (positive values shift to the right).

        Returns:
            torch.Tensor: A tensor of the same shape as `pulses` with the waveforms aligned based on the specified shifts.
    """
    N, L = pulses.shape
    aligned = torch.zeros_like(pulses)

    # Create index grid
    idx = torch.arange(L, device = pulses.device).unsqueeze(0).repeat(N, 1)  # (N, L)
    # Compute shifted indices
    shifted_idx = idx - shift.unsqueeze(1)  # (N, L)

    # Mask valid indices
    mask = (shifted_idx >= 0) & (shifted_idx < L)
    shifted_idx_clipped = torch.clamp(shifted_idx, 0, L - 1)

    # Fill aligned pulses
    aligned[mask] = pulses[torch.arange(N, device = pulses.device).unsqueeze(1).repeat(1, L)[mask],
    shifted_idx_clipped[mask]]
    return aligned


import numpy as np
def asymmetric_hann(N, p):
    n = np.arange(N)
    n_p = p * (N - 1)
    w = np.zeros(N)

    left = n <= n_p
    right = n > n_p

    w[left] = 0.5 * (1 - np.cos(np.pi * n[left] / n_p))
    w[right] = 0.5 * (1 + np.cos(np.pi * (n[right] - n_p) / ((N - 1) - n_p)))

    return w

def compute_rt_bin(arr,sampling_f):
    arg_max = np.argmax(arr)
    btm, top = np.interp([0.1,0.9],arr[:arg_max],np.arange(len(arr[:arg_max])))
    return (top-btm)/sampling_f