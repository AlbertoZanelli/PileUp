import torch
import math
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