import numpy as np
from scipy import integrate, interpolate


# Total available decay energy (MeV)
L = 3.034


def energy_polynomial(E):
    """Polynomial part of the double beta decay energy spectrum."""
    return E**4 + 10*E**3 + 40*E**2 + 60*E + 30


def g(E):
    """
    Unnormalized double beta decay energy spectrum.

    Represents the probability density (up to normalization)
    for one emitted electron to have energy E, assuming a
    total available decay energy of L.

    g(E) ∝ E * (L - E)^5 * P(E)

    Parameters
    ----------
    E : array_like
        Electron energy (0 < E < L).

    Returns
    -------
    spectrum : ndarray
        Unnormalized PDF values of the single-electron spectrum.
    """
    E = np.asarray(E)
    spectrum = E * (L - E)**5 * energy_polynomial(E)
    spectrum[(E <= 0) | (E >= L)] = 0
    return spectrum


def unnormalized_ratio_pdf(ratio):
    """
    Unnormalized probability density for the energy ratio R = E1 / E2,
    given that E1 + E2 = L and E1 < E2.

    Context
    -------
    - E1, E2: energies of two event in double beta decay.
    - g(E) = E * (L - E)^5 * P(E) is the (unnormalized) single-event
      energy spectrum.
    - R = E1 / E2 describes how asymmetrically the total decay energy L
      is shared between the two event.

    Parameters
    ----------
    ratio : array_like
        Ratio values (0 < ratio < 1).

    Returns
    -------
    pdf : ndarray
        Unnormalized PDF values for R.
    """
    ratio = np.asarray(ratio)
    pdf = np.zeros_like(ratio)
    valid = (ratio > 0) & (ratio < 1)

    if np.any(valid):
        r = ratio[valid]
        pdf[valid] = (
            (L**13)
            * (r**6)
            / (1 + r)**14
            * energy_polynomial(L * r / (1 + r))
            * energy_polynomial(L / (1 + r))
        )

    return pdf


def build_ratio_distribution():
    """
    Constructs callable normalized PDF, CDF, and inverse CDF for the
    ratio R = E1 / (E1 + E2) directly from the continuous definition.

    Uses numerical integration to normalize and to define the CDF.

    Returns
    -------
    pdf : function
        Callable normalized PDF, pdf(r).
    cdf : function
        Callable normalized CDF, cdf(r).
    inverse_cdf : function
        Callable inverse CDF, inverse_cdf(u), with u ∈ [0, 1].
    """
    # Compute normalization constant by integrating over (0,1)
    def unnormalized_ratio_pdf(r):
        """
        Unnormalized probability density for the energy ratio R = E1 / (E1 + E2),
        given that E1 + E2 = L.

        Parameters
        ----------
        r : array_like
            Ratio values (0 < r < 1).

        Returns
        -------
        pdf : ndarray
            Unnormalized PDF values for R.
        """
        r = np.asarray(r)
        pdf = np.zeros_like(r)
        valid = (r > 0) & (r < 1)

        if np.any(valid):
            r_valid = r[valid]
            E1 = r_valid * L
            E2 = (1 - r_valid) * L
            pdf[valid] = (
                (L**13)
                * (r_valid**6)
                * ((1 - r_valid)**6)
                * energy_polynomial(E1)
                * energy_polynomial(E2)
            )

        return pdf

    norm, _ = integrate.quad(lambda r: unnormalized_ratio_pdf(r), 0, 1)

    # Normalized PDF
    def pdf(r):
        return unnormalized_ratio_pdf(r) / norm

    # CDF via numerical integration
    def cdf(r):
        if np.any(r <= 0):
            return 0.0
        if np.any(r >= 1):
            return 1.0
        result, _ = integrate.quad(pdf, 0, r)
        return result

    # Build inverse CDF by interpolating CDF samples
    r_grid = np.linspace(0, 1, 2000)
    cdf_grid = np.array([cdf(r) for r in r_grid])
    inverse_cdf_fn = interpolate.interp1d(
        cdf_grid,
        r_grid,
        kind='linear',
        bounds_error=False,
        fill_value=(r_grid[0], r_grid[-1])
    )

    def inverse_cdf(u):
        """Inverse CDF (quantile function), maps u ∈ [0,1] to ratio r."""
        return inverse_cdf_fn(u)

    return pdf, cdf, inverse_cdf


pdf_ratio2b, cdf_ratio2b, inverse_cdf_ratio2b = build_ratio_distribution()
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # n_samples = 10000
    # samples = inverse_cdf_ratio2b(np.random.rand(n_samples))
    # plt.hist(samples, bins = 100, density = True, alpha = 0.5, label = "Samples")
    z_grid = np.linspace(0, 1, 1000)
    plt.plot(z_grid, pdf_ratio2b(z_grid)/np.trapz(pdf_ratio2b(z_grid), z_grid), 'r-', label="PDF")
    plt.xlabel("ratio r")
    plt.ylabel("Density")
    plt.legend()
    plt.show()
    # E = np.linspace(0, L, 1000)
    # plt.plot(E, g(E)/np.trapz(g(E), E))
    # plt.xlabel("Energy E (MeV)")
    # plt.ylabel("Normalized Spectrum")
    # plt.show()


