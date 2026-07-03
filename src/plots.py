import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic
from matplotlib.ticker import LogLocator
import pandas as pd
def plot_psd_vs_dt(PSD_single, PSD_pileup, dt_pileup, cut=0.9, bins=40,PSD_min=-0.01,PSD_max =0.02):
    """
    Plot the Pulse Shape Discriminator (PSD) as a function of time difference (dt).

    This function generates a plot showing the relationship between PSD and dt for
    pile-up simulated pulses, along with a histogram of PSD values for single and
    pile-up pulses.

    Args:
        PSD_single (numpy.ndarray): PSD values for single simulated pulses.
        PSD_pileup (numpy.ndarray): PSD values for pile-up simulated pulses.
        dt_pileup (numpy.ndarray): Time differences for pile-up simulated pulses.
        cut (float, optional): Acceptance cut value for PSD. Defaults to 0.9.
        bins (int, optional): Number of bins for the histogram. Defaults to 40.

    Returns:
        tuple: A tuple containing the figure, main axis, and inset axis objects.
    """
    dt_pileup_sel = dt_pileup[np.abs(PSD_pileup - 1) < 0.1]
    PSD_double_sel = PSD_pileup[np.abs(PSD_pileup - 1) < 0.1]
    bin_means, _, _ = binned_statistic(
        dt_pileup_sel * 1e-1, PSD_double_sel, np.mean, bins
    )
    bin_high, bin_edges, binnumber = binned_statistic(
        dt_pileup_sel * 1e-1, PSD_double_sel, lambda x: np.percentile(x, 84.1), bins
    )
    bin_low, bin_edges, binnumber = binned_statistic(
        dt_pileup_sel * 1e-1, PSD_double_sel, lambda x:np.percentile(x,15.9), bins
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hlines(cut, 0, 0.8, colors="#fc342a", label="90% acceptance cut")
    ax.errorbar(
        bin_edges[:-1],
        bin_means,
        yerr = [bin_means - bin_low, bin_high - bin_means],
        fmt = "o",
        label = "Pile-up Simulated Pulses",
        color = "#D55E00",
    )
    ax.set_xlim(-0.01, 0.8)
    ax.set_xlabel("$\Delta$t [ms]")
    ax.set_ylabel("PSD")
    ax2 = ax.inset_axes([1.0, 0, 1, 1], sharey=ax)
    hist,_,_ = ax2.hist(
        PSD_single,
        bins=100,
        range=(PSD_min,PSD_max),
        alpha=0.9,
        label="Single Simulated Pulses",
        color="#0072B2",
        orientation="horizontal",
    )
    ax2.hist(
        PSD_pileup,
        bins=100,
        range=(PSD_min,PSD_max),
        alpha=0.9,
        label="Pile-up Simulated Pulses",
        color="#D55E00",
        orientation="horizontal",
    )

    ax2.hlines(cut, 0, hist.max(), colors="#fc342a", label="90% acceptance cut")
    ax2.tick_params(left=False, labelleft=False)
    # ax2.set_xticks([50, 100, 150, 200, 250, 300])
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax2.set_xlabel("Counts")
    ax.set_xticks(np.arange(0, 0.8, 0.1))
    ax2.legend()
    ax.legend(loc="lower right")

    return fig, ax, ax2

def plot_biais_vs_dt(PSD_single, PSD_pileup, dt_pileup, cut=0.9, bins=40):
    """
    Plot the bias (1 - PSD) as a function of time difference (dt).

    This function generates a plot showing the relationship between the bias
    (1 - PSD) and dt for pile-up simulated pulses, along with a histogram of
    bias values for single and pile-up pulses.

    Args:
        PSD_single (numpy.ndarray): PSD values for single simulated pulses.
        PSD_pileup (numpy.ndarray): PSD values for pile-up simulated pulses.
        dt_pileup (numpy.ndarray): Time differences for pile-up simulated pulses.
        cut (float, optional): Acceptance cut value for the bias. Defaults to 0.9.
        bins (int, optional): Number of bins for the histogram. Defaults to 40.

    Returns:
        tuple: A tuple containing the figure, main axis, and inset axis objects.
    """
    dt_pileup_sel = dt_pileup[np.abs(PSD_pileup - 1) < 0.05]
    PSD_double_sel = PSD_pileup[np.abs(PSD_pileup - 1) < 0.05]
    Biais = 1 - PSD_double_sel

    bin_means, bin_edges, binnumber = binned_statistic(
        dt_pileup_sel * 1e-1, Biais, np.mean, bins
    )
    bin_high, bin_edges, binnumber = binned_statistic(
        dt_pileup_sel * 1e-1, Biais, lambda x: np.percentile(x, 84.1), bins
    )
    bin_low, bin_edges, binnumber = binned_statistic(
        dt_pileup_sel * 1e-1, Biais, lambda x: np.percentile(x, 15.9), bins
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hlines(cut, 0, 0.8, colors="#fc342a", label="90% acceptance cut")
    ax.errorbar(
        bin_edges[:-1],
        bin_means,
        yerr=[bin_means - bin_low, bin_high - bin_means],
        fmt="o",
        label="Pile-up Simulated Pulses",
        color="#2d508c",
    )
    ax.set_xlim(0, 0.8)
    ax.set_xlabel("dt [ms]")
    ax.set_ylabel("Pileup Bias (1 - PSD)")

    ax2 = ax.inset_axes([1.0, 0, 1, 1], sharey=ax)
    hist,_,_ = ax2.hist(
        1 - PSD_single,
        bins=100,
        range=(-0.008, 0.04),
        density=True,
        alpha=0.7,
        label="Single Simulated Pulses",
        color="#ff9200",
        orientation="horizontal",
    )
    ax2.hist(
        1 - PSD_pileup,
        bins=100,
        range=(-0.008, 0.04),
        density=True,
        alpha=0.7,
        label="Pile-up Simulated Pulses",
        color="#2d508c",
        orientation="horizontal",
    )
    ax2.hlines(cut, 0, hist.max(), colors="#fc342a", label="90% acceptance cut")
    ax2.tick_params(left=False, labelleft=False)
    ax2.set_xlabel("Counts")
    ax.set_xticks(np.arange(0, 0.8, 0.1))
    ax2.legend()
    ax.legend(loc="lower right")

    return fig, ax, ax2
def plot_compted_mu_sigma():
    """
    Plot a 3D scatter plot of computed values (mu, sigma) for a given channel.

    This function visualizes the computed mu and sigma values for a specific
    channel as a 3D scatter plot, using the acceptance level and precomputed
    data.
    """
    import torch
    from scipy.stats import norm
    from utility.double_beta_spectrum import pdf_ratio2b

    channel = 3
    acceptance = 0.9
    muY, sigmaY = np.load(f"../outputs/mu_sigma_channel_{channel}_eff{int(acceptance * 100):d}.npy")

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    t_torch = torch.linspace(-8e-4, 8e-4, 200, dtype=torch.cfloat)
    r_torch = torch.linspace(0, 1, 200, dtype=torch.cfloat)

    A = (1 - norm.cdf((1 - muY - sigmaY[0, 0]) / sigmaY)) * pdf_ratio2b(t_torch.numpy()[:, None])
    r_plot, t_plot = np.meshgrid(r_torch[::5].numpy(), t_torch[::5].numpy())

    ax.scatter(r_plot.reshape(-1), t_plot.reshape(-1), A[::5, ::5].reshape(-1))
    plt.show()


def plot_3d():
    """
    Plot a 3D heatmap of the mean PSD values as a function of time difference (dt)
    and energy-sharing ratio (r) for pile-up simulated pulses.

    This function computes a 2D binned statistic (mean) for the PSD values and
    visualizes the result as a 3D heatmap.
    """
    import src.dataset as ds
    from scipy.stats import binned_statistic_2d

    channel = 11
    data_single, data_pileup, dt_pileup, r_pileup, E_pileups = ds.create_data_sets(
        channel, win_length=2000, shift=-2, pos_prefix="_bis", differenciate=True
    )

    PSD_pileup = np.load(f"../outputs/PSDs/PSD_pileup_c5_a90.npy")
    dt_pileup_sel = dt_pileup[np.abs(PSD_pileup - 1) < 0.02]
    r_pileup_sel = r_pileup[np.abs(PSD_pileup - 1) < 0.02]
    PSD_double_sel = PSD_pileup[np.abs(PSD_pileup - 1) < 0.02]

    dt_bins = np.linspace(min(dt_pileup_sel), max(dt_pileup_sel), 30)
    r_bins = np.linspace(min(r_pileup_sel), max(r_pileup_sel), 2)

    # Compute 2D binned statistic — for example, the mean
    statistic, dt_edges, r_edges, binnumber = binned_statistic_2d(
        dt_pileup_sel,
        r_pileup_sel,
        PSD_double_sel,
        statistic='mean',  # or 'median', 'count', 'sum', 'std'
        bins=[dt_bins, r_bins]
    )

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(dt_edges, r_edges, statistic.T, shading='auto')
    plt.xlabel('dt_pileup_sel')
    plt.ylabel('r_pileup_sel')
    plt.title('Mean PSD_double_sel per bin')
    plt.colorbar(label='Mean PSD_double_sel')
    plt.show()

def J_vs_rp():
    import pandas as pd
    out_data = pd.read_csv('../outputs/BI_results_interp_v2.csv')
    plt.scatter(out_data['J'], 1 - out_data['rp'], c = out_data['channel_ID'], cmap = 'tab10')
    plt.plot([0.08, 0.4], [0.08, 0.4], 'k--')
    plt.xlabel('J')
    plt.ylabel('1 - $r_p$')
    plt.show()

def plot_Trial_vs_J():
    fig, ax = plt.subplots(figsize=(8.4, 7))

    channels = [3,5,9,11,13,15,17,19]
    all_J = []
    colors = [
        "#000000",  # black
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#009E73",  # bluish green
        "#F0E442",  # yellow
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#CC79A7"  # reddish purple
    ]
    channel_IDs = [5, 9, 11, 10, 4, 2, 12, 3]
    for i, idx in enumerate(np.argsort(channel_IDs)):
        channel = channels[idx]
        J_values = np.load(f"../outputs/training_Js/channel_{channel}/J_90_base.npy")
        all_J.append(J_values)
        ax.plot(J_values*100, color=colors[i], alpha=0.7)
        ax.plot([], [], color=colors[i], label=f"Detector {channel_IDs[idx]}",marker="s", markersize=10, linestyle="")

    mean_J = np.mean(all_J, axis=0)
    ax.plot(mean_J*100, color="black", lw=2.5)

    ax.plot([], color="black", lw=2.5, label="Mean",marker="s", markersize=10, linestyle="")

    ax.set_xlabel("Training iteration", fontsize=20)
    ax.set_ylabel(
        r"Global pile-up misidentification rate $\langle \mathcal{M} \rangle$ [%]",
        fontsize = 18
    )


    ax.tick_params(axis='both', labelsize=20)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=20, ncol=2)

    fig.tight_layout()
    fig.savefig("/local/home/mp274748/Documents/paper/pile_up_cupid/figures/J_convergence_all_channels_eff90.pdf", dpi=300, bbox_inches="tight")
    plt.show()

def plot_m1_2_all_channel():
    channels = [3, 5, 9, 11, 13, 15, 17, 19]

    f1_all, f2_all = [], []

    fig, ax = plt.subplots(figsize = (8, 6))

    for ch in channels:
        f1, f2 = np.load(f"../outputs/Pileup_filter_functions/channel_{ch}/functions_eff90_base_true_mp.npy")
        f1_all.append(np.abs(f1))
        f2_all.append(np.abs(f2))

        freq = np.fft.fftfreq(len(f1), d = 1e-4)
        pos_freq = freq > 0
        ax.plot(freq[pos_freq], f1[pos_freq], alpha = 0.1, color = "k")
        ax.plot(freq[pos_freq], f2[pos_freq], alpha = 0.1, color = "k")

    # Convert to arrays
    f1_all = np.array(f1_all)
    f2_all = np.array(f2_all)

    # Median values
    f1_med = np.median(f1_all, axis = 0)
    f2_med = np.median(f2_all, axis = 0)

    # Positive frequencies only
    pos_freq = freq > 0

    ax.plot(freq[pos_freq], f1_med[pos_freq],
            color = "#2d508c", linewidth = 2, label = "Median $m_1$")
    ax.plot(freq[pos_freq], f2_med[pos_freq],
            color = "#fc342a", linewidth = 2, label = "Median $m_2$")

    # Axis styling
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency [Hz]", fontsize = 20)
    ax.set_ylabel("Gain", fontsize = 20)
    ax.tick_params(axis = "both", which = "major", labelsize = 20)
    ax.legend(fontsize = 20)
    ax.grid(True, which = "both", ls = "--", lw = 0.5)
    fig.tight_layout()
    ax.set_ylim(1e-2, 30)
    # fig.savefig("../outputs/plots/Median_filters_channel_all_eff90.pdf", dpi=300)
    fig.savefig("/local/home/mp274748/Documents/paper/pile_up_cupid/figures/Median_filters_channel_all_eff90.pdf", dpi=300)
    plt.show()
def plot_m1_2_old_all_channel():
    import src.dataset as ds
    channels = [3, 5, 9, 11, 13, 15, 17, 19]

    f1_all, f2_all = [], []
    f1_all_old, f2_all_old = [], []
    fig, ax = plt.subplots(figsize = (8, 6))

    for ch in channels:
        f1, f2 = np.load(f"../outputs/Pileup_filter_functions/channel_{ch}/functions_eff90_V2.npy")
        f1_all.append(np.abs(f1))
        f2_all.append(np.abs(f2))

        freq = np.fft.fftfreq(len(f1), d = 1e-4)
        ax.plot(freq[:1000], f1[:1000], alpha = 0.1, color = "k")
        ax.plot(freq[:1000], f2[:1000], alpha = 0.1, color = "k")
        H_unit, S, nps, w, pulse_factor = ds.get_channel_specs(ch, n_deriv = 1, window_fct = np.hanning)
        f1, f2 = np.ones(2000), np.abs(H_unit * S)
        f1, f2  = f1 / np.mean(np.abs(S * f1 * H_unit)), f2 / np.mean(np.abs(S * f2 * H_unit))
        f1_all_old.append(np.abs(f1))
        f2_all_old.append(np.abs(f2))

        freq = np.fft.fftfreq(len(f1), d = 1e-4)
        ax.plot(freq[:1000], f1[:1000], alpha = 0.1, color = "k")
        ax.plot(freq[:1000], f2[:1000], alpha = 0.1, color = "k")
    f1_all_old = np.array(f1_all_old )
    f2_all_old = np.array(f2_all_old)
    f1_med_old = np.median(f1_all_old, axis = 0)
    f2_med_old = np.median(f2_all_old, axis = 0)

    # Convert to arrays
    f1_all = np.array(f1_all)
    f2_all = np.array(f2_all)

    # Median values
    f1_med = np.median(f1_all, axis = 0)
    f2_med = np.median(f2_all, axis = 0)

    # Positive frequencies only
    pos_freq = freq > 0

    ax.plot(freq[pos_freq], f1_med[pos_freq],
            color = "#2d508c", linewidth = 2, label = "Median $m_1$")
    ax.plot(freq[pos_freq], f2_med[pos_freq],
            color = "#fc342a", linewidth = 2, label = "Median $m_2$")

    ax.plot(freq[pos_freq], f1_med_old[pos_freq],
            color = "deepskyblue", linewidth = 2, label = "Median $m_{1,old}$ ")
    ax.plot(freq[pos_freq], f2_med_old[pos_freq],
            color = "coral", linewidth = 2, label = "Median $m_{2,old}$")

    # Axis styling
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency [Hz]", fontsize = 20)
    ax.set_ylabel("Gain", fontsize = 20)
    ax.set_title("Frequency vs $m_{1,opt}$ & $m_{2,opt}$", fontsize = 20)
    ax.tick_params(axis = "both", which = "major", labelsize = 20)
    ax.legend(fontsize = 20)
    ax.yaxis.set_major_locator(LogLocator(base = 10.0))
    ax.set_ylim(1e-2, 1e2)
    # Minor ticks: base-10 minor ticks (1–9)
    ax.yaxis.set_minor_locator(
        LogLocator(base = 10.0, subs = np.arange(1.0, 10.0), numticks = 1000)
    )
    ax.grid(True, which = "both", ls = "--", lw = 0.5)
    fig.tight_layout()
    fig.show()
    fig.savefig("../outputs/plots/Median_filters_channel_all_eff90_with_old.pdf", dpi=300)


def plot_H1_2_all_channel():
    import src.dataset as ds

    channels = [3, 5, 9, 11, 13, 15, 17, 19]

    f1_all, f2_all = [], []

    fig, ax = plt.subplots(figsize = (8, 6))

    for ch in channels:
        f1, f2 = np.load(f"../outputs/Pileup_filter_functions/channel_{ch}/functions_eff90_V2.npy")
        H_unit, S, nps, w, pulse_factor = ds.get_channel_specs(ch, n_deriv = 1, window_fct = np.hanning)
        f1,f2 = f1 * H_unit * S, f2 * H_unit * S
        f1_all.append(np.abs(f1))
        f2_all.append(np.abs(f2))

        freq = np.fft.fftfreq(len(f1), d = 1e-4)
        ax.plot(freq[:1000], f1[:1000], alpha = 0.1, color = "k")
        ax.plot(freq[:1000], f2[:1000], alpha = 0.1, color = "k")

    # Convert to arrays
    f1_all = np.array(f1_all)
    f2_all = np.array(f2_all)

    # Median values
    f1_med = np.median(f1_all, axis = 0)
    f2_med = np.median(f2_all, axis = 0)

    # Positive frequencies only
    pos_freq = freq > 0

    ax.plot(freq[pos_freq], f1_med[pos_freq],
            color = "#2d508c", linewidth = 2, label = "Median $|H_1 \hat{s}|$")
    ax.plot(freq[pos_freq], f2_med[pos_freq],
            color = "#fc342a", linewidth = 2, label = "Median $|H_2 \hat{s}|$")

    # Axis styling
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency [Hz]", fontsize = 16)
    ax.set_ylabel("Gain", fontsize = 16)
    ax.set_title("Frequency vs $|H_{1,opt}\hat{s}|$ & $|H_{2,opt}\hat{s}|$", fontsize = 18)
    ax.tick_params(axis = "both", which = "major", labelsize = 14)
    ax.legend(fontsize = 14)
    ax.grid(True, which = "both", ls = "--", lw = 0.5)
    fig.tight_layout()
    fig.show()
    fig.savefig("../outputs/plots/Median_filters_channel_all_eff90.pdf", dpi=300)
def plot_H1_2_old_all_channel():
    import src.dataset as ds

    channels = [3, 5, 9, 11, 13, 15, 17, 19]

    f1_all, f2_all = [], []
    f1_all_old, f2_all_old = [], []
    fig, ax = plt.subplots(figsize = (8, 6))

    for ch in channels:
        f1, f2 = np.load(f"../outputs/Pileup_filter_functions/channel_{ch}/functions_eff90_V2.npy")
        H_unit, S, nps, w, pulse_factor = ds.get_channel_specs(ch, n_deriv = 1, window_fct = np.hanning)
        f1,f2 = f1 * H_unit * S, f2 * H_unit * S
        f1_all.append(np.abs(f1))
        f2_all.append(np.abs(f2))

        freq = np.fft.fftfreq(len(f1), d = 1e-4)
        ax.plot(freq[:1000], f1[:1000], alpha = 0.1, color = "k")
        ax.plot(freq[:1000], f2[:1000], alpha = 0.1, color = "k")
        f1, f2 = np.ones(2000), np.abs(H_unit * S)
        f1, f2 = f1 / np.mean(np.abs(S * f1 * H_unit)), f2 / np.mean(np.abs(S * f2 * H_unit))
        f1,f2 = f1 * H_unit * S, f2 * H_unit * S
        f1_all_old.append(np.abs(f1))
        f2_all_old.append(np.abs(f2))

        freq = np.fft.fftfreq(len(f1), d = 1e-4)
        ax.plot(freq[:1000], f1[:1000], alpha = 0.1, color = "k")
        ax.plot(freq[:1000], f2[:1000], alpha = 0.1, color = "k")
    f1_all_old = np.array(f1_all_old)
    f2_all_old = np.array(f2_all_old)
    f1_med_old = np.median(f1_all_old, axis = 0)
    f2_med_old = np.median(f2_all_old, axis = 0)

    # Convert to arrays
    f1_all = np.array(f1_all)
    f2_all = np.array(f2_all)

    # Median values
    f1_med = np.median(f1_all, axis = 0)
    f2_med = np.median(f2_all, axis = 0)

    # Positive frequencies only
    pos_freq = freq > 0

    ax.plot(freq[pos_freq], f1_med[pos_freq],
            color = "#2d508c", linewidth = 2, label = "Median $|H_1 \hat{s}|$")
    ax.plot(freq[pos_freq], f2_med[pos_freq],
            color = "#fc342a", linewidth = 2, label = "Median $|H_2 \hat{s}|$")

    ax.plot(freq[pos_freq], f1_med_old[pos_freq],
            color = "deepskyblue", linewidth = 2, label = "Median $|H_{1,old} \hat{s}|$")
    ax.plot(freq[pos_freq], f2_med_old[pos_freq],
            color = "coral", linewidth = 2, label ="Median $|H_{1,old} \hat{s}|$")
    # Axis styling
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency [Hz]", fontsize = 16)
    ax.set_ylabel("Gain", fontsize = 16)
    ax.set_title("Frequency vs $|H_{1,opt}\hat{s}|$ & $|H_{2,opt}\hat{s}|$", fontsize = 18)
    ax.tick_params(axis = "both", which = "major", labelsize = 14)
    ax.legend(fontsize = 14)
    ax.grid(True, which = "both", ls = "--", lw = 0.5)
    fig.tight_layout()
    fig.show()
    # fig.savefig("../outputs/plots/Median_filters_channel_all_eff90.pdf", dpi=300)

    fig.savefig("/local/home/mp274748/Documents/paper/pile_up_cupid/figures/Median_filters_channel_all_eff90.pdf", dpi=300)

def Y_distrib_fixe_r_dt():
    from torch.utils.data import TensorDataset
    import src.dataset as ds
    import src.simulation as sim
    import src.analysis as an
    from torch import from_numpy
    n_deriv = 1
    channel = 15
    f1_opt, f2_opt = np.load(f"../outputs/Pileup_filter_functions/channel_{channel}/functions_eff90_V3.npy")

    H_unit, S, nps, w, pulse_factor = ds.get_channel_specs(channel, n_deriv = n_deriv, window_fct = np.hanning)
    fpulses, signal_freq, noise_freq = sim.simulate_frequency_pulses_fixed_dt_r(S, nps, w, 3e-4, 0.5, nsim = 10000,
                                                                                seed = 15, signal_scale = 0.001874)
    dataset_simu = TensorDataset(from_numpy(np.fft.ifft(fpulses, axis = 1).real.astype(np.float32)))
    PSD_simu, _, _ = an.get_PSD_interpole(dataset_simu, H_unit, f1_opt, f2_opt)
    plt.hist(PSD_simu, bins = 100, zorder = 3, color = "#2d508c")
    plt.xlim(0.9955, 0.9998)
    plt.xlabel("PSD")
    ax = plt.gca()
    ax.grid(True, zorder = 0, which = 'both', linestyle = '--', linewidth = 0.5)
    plt.ylabel("Number of events")
    plt.title("PSD Distribution for simulated pile-up events r=0.5, dt=0.3 ms")


from matplotlib.colors import LogNorm
def plot_covariance_matrix(result):
    plt.figure(figsize=(8, 6))
    plt.imshow(result.covar, cmap='viridis', aspect='auto',norm = LogNorm(vmin=1e-12, vmax=np.max(result_2.covar)))
    plt.colorbar(label='Covariance')
    plt.title('Covariance Matrix of Fit Parameters (Bessel order 2)')
    plt.xticks(range(len(result.var_names)), result.var_names, rotation=45)
    plt.yticks(range(len(result.var_names)), result.var_names)
    plt.show()

import matplotlib as mpl

def plot_bessel_filters(channel,fcs):
    fcts = []
    for fc in fcs:
        f1_f2 = np.load(f"../outputs/Pileup_filter_functions/channel_{channel}/bessel_study/functions_fc_{fc}.npy")
        fcts.append((f1_f2[0], f1_f2[1]))
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('viridis')
    freq = np.fft.rfftfreq(len(fcts[0][0]), d=1e-4)
    for idx,(f1,f2 )in enumerate(fcts):
        ax.plot(freq,f1[:1001],c=cmap(idx/len(fcts)),alpha=0.7)
        ax.plot(freq,f2[:1001],c=cmap(idx/len(fcts)),alpha=0.7)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=min(fcs), vmax=max(fcs)))
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Frequency [Hz]')
    cbar.set_label('Bessel Filter Cutoff Frequency [Hz]')
    ax.set_ylabel('Filter Amplitude')
    plt.show()

def plot_J_vs_bessel_fc():
    channels = [3, 5, 9, 11, 13, 15, 17, 19]
    channel_ID = [5, 9, 11, 10, 4, 2, 12, 3]
    for index_channel in np.argsort(channel_ID):
        channel = channels[index_channel]
        Js = np.loadtxt(f"../outputs/Pileup_filter_functions/channel_{channel}/bessel_study/Js_vs_fc")
        fcs = [500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
        plt.plot(fcs, 1 - np.array(Js), marker = 'o', label = f'detector ID {channel_ID[index_channel]}')
    plt.xlabel('Bessel Filter Cutoff Frequency [Hz]',fontdict = {'size':14})
    plt.ylabel('1-J aka "rejection power"',fontdict = {'size':14})
    plt.legend(fontsize = 12)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 14)
    plt.show()

def plot_mp_freq_all_channel():
    import src.dataset as ds
    channels = [3, 5, 9, 11, 13, 15, 17, 19]
    channel_ID = [5, 9, 11, 10, 4, 2, 12, 3]
    plt.figure(figsize = (8, 6))
    for index_channel in np.argsort(channel_ID):
        channel = channels[index_channel]
        H_unit, S, nps, w, pulse_factor = ds.get_channel_specs(channel, n_deriv = 0, window_fct = np.hanning)
        plt.plot(w[:1000] / (2 * np.pi), np.abs(S)[:1000], label = f"detector ID {channel_ID[index_channel]}")
    plt.legend(fontsize = 14)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 14)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Frequency [Hz]', fontdict = {'size': 14})
    plt.ylabel('Pulse Amplitude Spectrum |S(w)|', fontdict = {'size': 14})
    plt.show()
def plot_meanpulse_fit(meanpulse, y, t):
    fig,(ax1,ax2) = plt.subplots(2,1, figsize=(10,8), sharex=True, gridspec_kw={'height_ratios': [3, 1],'hspace':0})
    ax1.plot(t, y, label="Transfer function model")
    ax1.plot(t, meanpulse)
    ax2.plot(t, meanpulse - y, label="Fit Residuals",marker='o',markersize=2,ls="")
    plt.legend()
def plot_meanpulse_fit_freq(meanpulse, y,axs=None, labels=None):
    labels = ["Original mean pulse", "Fitted mean pulse"] if labels is None else labels
    (ax1,ax2) = plt.subplots(2,1, figsize=(10,8), sharex=True, gridspec_kw={'height_ratios': [3, 1],'hspace':0})[1] if axs is None else axs
    freq = np.fft.rfftfreq(len(meanpulse),1e-4)
    window = np.hanning(len(meanpulse))
    fmp = np.abs(np.fft.rfft(window*meanpulse))
    ax1.loglog(freq,fmp, label=labels[0],alpha=0.7)
    ffmp = np.abs(np.fft.rfft(window*y))
    ax1.loglog(freq,ffmp, label=labels[1],alpha=0.7)
    ax2.semilogx(freq,ffmp - fmp, label="Residuals",marker='o',markersize=2,ls="")
    ax2.set_ylim(-0.1*np.max(fmp),0.1*np.max(fmp))
    ax1.legend()
    ax1.set_ylim(1e-4,1e2)
    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel("Amplitude")
def plot_PSD_distribution(
    PSD_single, PSD_pileup,
    muY, sigmaY, ratio_distribution,
    N_sigma=3
):
    from scipy.stats import norm

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(16, 6),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0}
    )

    # -----------------------
    # Single events
    # -----------------------
    cut = np.percentile(PSD_single.real, 10)

    hist_single, bins_single = np.histogram(
        PSD_single, bins=200, range=(0.99, 1.01), density=True
    )
    A_single = 0.5 * (bins_single[1:] + bins_single[:-1])

    mu = muY[0, 0].numpy()
    sigma = sigmaY[0, 0].numpy()
    pdf_single = norm.pdf(A_single, mu, sigma)

    ax1.plot(
        A_single, hist_single,
        drawstyle='steps-mid',
        color='tab:blue',
        alpha=0.6,
        lw=1.5,
        label='Single events (MC)'
    )
    ax1.plot(
        A_single, pdf_single,
        color='tab:blue',
        lw=2.5,
        label='Single events (model)'
    )


    # -----------------------
    # Pileup events
    # -----------------------
    hist_pu, bins_pu = np.histogram(
        PSD_pileup, bins=200, range=(0.96, 1.01), density=True
    )
    A_pu = 0.5 * (bins_pu[1:] + bins_pu[:-1])

    ax1.plot(
        A_pu, hist_pu,
        drawstyle='steps-mid',
        color='tab:orange',
        alpha=0.6,
        lw=1.5,
        label='Pileup events (MC)'
    )

    pdf_pu = np.sum(
        norm.pdf(A_pu[None, None, :], muY[:, :, None], sigmaY[:, :, None])
        * ratio_distribution[:, None, None],
        axis=(0, 1)
    ).real
    pdf_pu /= np.trapz(pdf_pu, A_pu)

    ax1.plot(
        A_pu, pdf_pu,
        color='tab:orange',
        lw=2.5,
        label='Pileup events (model)'
    )

    ax1.axvline(
        cut, color='tab:red', ls='--', lw=1.5,
        label='10th percentile cut'
    )
    ax1.axvline(
        mu - N_sigma * sigma,
        color='tab:green', ls='-.', lw=1.5,
        label=rf'$\mu - {N_sigma:.2f}\sigma$'
    )

    # -----------------------
    # Residuals
    # -----------------------
    ax2.axhline(0, color='k', lw=1, alpha=0.6)

    ax2.plot(
        A_single, pdf_single - hist_single,
        color='tab:blue',
        lw=1.5,
        label='Model − MC (single)'
    )
    ax2.plot(
        A_pu, pdf_pu - hist_pu,
        color='tab:orange',
        lw=1.5,
        label='Model − MC (pileup)'
    )

    res_max = max(
        np.abs(pdf_single - hist_single).max(),
        np.abs(pdf_pu - hist_pu).max()
    )
    ax2.set_ylim(-1.1 * res_max, 1.1 * res_max)

    # -----------------------
    # Final formatting
    # -----------------------
    ax1.set_xlim(0.96, 1.01)
    ax1.set_ylabel("Probability density")
    ax2.set_ylabel("Residual")
    ax2.set_xlabel("PSD")

    ax1.legend(frameon=False, ncol=3)
    ax2.legend(frameon=False)

    ax1.grid(axis='y', alpha=0.3)
    ax2.grid(axis='y', alpha=0.3)

    plt.show()


def pulse_template():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    df_1 = pd.read_csv("../outputs/Systematics/BI_meanpulse_test.csv")
    df_2 = pd.read_csv("../outputs/Systematics/BI_meanpulses_test_2.csv")
    df = pd.concat((df_1, df_2))
    df = df[df["meanpulse_type"] != "fit"]
    energy_raw = [35.47, 87.59, 39.87, 29.4, 36.1, 101.5, 73.23, 122.69]
    types = ['single_40Q', 'single_70Q', 'single_100Q', 'raw', 'fit_bessel', 'mean']
    offsets = np.linspace(-0.3, 0.3, len(types))
    df["energy"] = df.apply(
        lambda x: energy_raw[channels.index(x["channel"])] if x["meanpulse_type"] == "raw"
        else float(x["meanpulse_type"].split("_")[1][:-1]) if "single" in x["meanpulse_type"]
        else 150 if x["meanpulse_type"] == "fit_bessel"
        else 200,
        axis = 1
    )

    def assign_offsets(group):
        group = group.sort_values("energy")
        group["offset"] = offsets
        return group

    df = df.groupby("channel", group_keys = False).apply(assign_offsets)

    # Colorblind-friendly palette
    color_map = {
        'raw': "#D55E00",
        'single_40Q': "#0072B2",
        'single_70Q': "#0072B2",
        'single_100Q': "#0072B2",
        'fit_bessel': "#E69F00",
        'mean': "#CC79A7"
    }

    markers = {
        'raw': 'X',
        'single_40Q':'P',
        'single_70Q': 'P',
        'single_100Q': 'P',
        'fit_bessel': 'D',
        'mean': 'o'
    }

    # Categorical x positions (0,1,2,3,...)
    unique_channels = sorted(
        df["channel"].unique(),
        key=lambda x: channel_ID[channels.index(x)]
    )

    x_positions = {ch: i for i, ch in enumerate(unique_channels)}

    # Jitter offsets
    df["x_offset"] = df.apply(lambda row: row["offset"] + x_positions[row["channel"]], axis=1)

    plt.figure(figsize=(8, 6))
    for t in types:
        subset = df[df["meanpulse_type"] == t].copy()
        subset["x"] = subset["channel"].map(x_positions)

        subset.sort_values(by="x", inplace=True)

        x = subset["x_offset"]

        plt.errorbar(
            x,
            subset["BI"] * 1e5,
            yerr=subset["BI_uncertainty"] * 1e5,
            ls="",
            marker=markers[t],
            color=color_map[t],
            markersize=6,
            capthick=2,
            elinewidth=2,
            capsize=5
        )

    # Grid BETWEEN channels
    plt.grid(axis='x', which='major', alpha=0)
    plt.grid(axis='y', alpha=0.5)

    for i in range(len(unique_channels) - 1):
        plt.axvline(i + 0.5, color='black', lw=0.8, zorder=0)
    plt.xlim(-0.5,len(unique_channels)-0.5)
    # X ticks = real detector IDs
    plt.xticks(
        range(len(unique_channels)),
        [str(channel_ID[channels.index(ch)]) for ch in unique_channels]
    )

    # Legend
    plt.plot([], [], marker = "X", ls = "", ms = 12, color = "#D55E00",
             label = "single high energy pulses\nused in [49]")
    plt.plot([], [], marker="P", ls="", ms=12, color="#0072B2",
             label="single high energy pulses\nselected at different amplitudes")
    plt.plot([], [], marker="D", ls="", ms=12, color="#E69F00",
             label="1 zero 4 poles + bessel fit")
    plt.plot([], [], marker="o", ls="", ms=12, color="#CC79A7",
             label="average pulse")

    plt.xlabel("Detector ID", fontsize=20)
    plt.ylabel("BI [1e-5 cts/keV/kg/year]", fontsize=20)
    plt.legend(title="Injected Pulse template", fontsize=16, title_fontsize=18, ncol=1)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tight_layout()
    plt.subplots_adjust(left = 0.12)
    plt.savefig("/local/home/mp274748/Documents/paper/pile_up_cupid/figures/pulse_template.pdf", dpi=300)
    plt.show()
channel_ID = [5, 9, 11, 10, 4, 2, 12, 3]
channels = [3, 5, 9, 11, 13, 15, 17, 19]
def RT_A_fit_plot():
    import utility.functions as fn
    df = pd.read_csv("../outputs/Systematics/BI_pulsetimeamp_test.csv")
    from scipy.optimize import curve_fit
    fig, (ax, ax_res) = plt.subplots(2, 1, figsize = (12, 8), sharex = True,
                                     gridspec_kw = {'height_ratios': [3, 1], 'hspace': 0})
    for index_channel in np.argsort(channel_ID):
        channel = channels[index_channel]
        data_channel = df[df["channel"] == channel]
        l, = ax.plot(data_channel["risetime"] ** 0.7514 / data_channel["amp_gain"] ** 0.5773,
                     data_channel["J"] * fn.K * 1e5, marker = 'o', ls = "", ms = 1)
        linear_f = lambda x, a, b: a * x + b
        popt, pcov = curve_fit(linear_f, data_channel["risetime"] ** 0.7514 / data_channel["amp_gain"] ** 0.5773,
                               data_channel["J"] * fn.K * 1e5)
        x_fit = np.linspace(1.5, 8, 100)
        ax.plot(x_fit, popt[0] * x_fit + popt[1], ls = "--", lw=0.7,
                c = l.get_color())
        residuals = data_channel["J"] * fn.K * 1e5 - linear_f(
            data_channel["risetime"] ** 0.7514 / data_channel["amp_gain"] ** 0.5773, *popt)
        ax_res.plot(data_channel["risetime"] ** 0.7514 / data_channel["amp_gain"] ** 0.5773, residuals, marker = 'o',
                    ls = "", c = l.get_color(), ms = 1)
        ax.plot([], [], c = l.get_color(), label = f"detector ID {channel_ID[index_channel]}",marker='s',ls="")

    ax.legend(fontsize = 12)
    ax.set_ylabel("Estimated BI [1e-5 cts/keV/kg/year]",fontdict = {'size': 14})
    ax_res.axhline(0, color = 'black', ls = '--')
    ax_res.set_xlabel("$RT^{n}/A^{m}$ with n=0.751 and m=0.577",fontdict = {'size': 14})
    ax_res.set_ylabel("Residuals [1e-5 cts/keV/kg/year]",fontdict = {'size': 14})
    ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
    ax_res.tick_params(axis = 'both', which = 'major', labelsize = 14)
    fig.savefig("/local/home/mp274748/Documents/paper/pile_up_cupid/figures/RT_A_fit.pdf", dpi = 300)

    plt.show()
def grid_size_systematics():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import utility.functions as fn

    df = pd.read_csv("../outputs/Systematics/BI_varying_grid_size.csv")

    # Colorblind-safe palette (Okabe-Ito)
    colors = [
        "#000000",  # black
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#009E73",  # bluish green
        "#F0E442",  # yellow
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#CC79A7"  # reddish purple
    ]

    markers = ['o', 's', 'D', '^', 'v', 'P', 'X']

    plt.figure(figsize=(8.5, 6.5))

    for i, index_channel in enumerate(np.argsort(channel_ID)):
        channel = channels[index_channel]
        data_channel = df[df["channel"] == channel]
        data_channel = data_channel[data_channel["grid_N_t"] > 9]  # Focus on N ≤ 200
        grid_N = data_channel["grid_N_t"].to_numpy()
        BI = data_channel["BI"].to_numpy()

        rel_diff = (data_channel["BI"] - data_channel["J"] * fn.K) / (data_channel["BI"] + data_channel["J"] * fn.K) * 200
        plt.plot(
            grid_N,
            rel_diff,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            ls="-",
            lw=0.4,
            markersize=6,
            label=f"Detector ID {channel_ID[index_channel]}"
        )
        # plt.errorbar(
        #     grid_N,
        #     rel_diff,
        #     data_channel["BI_uncertainty"] / (data_channel["BI"] + data_channel["J"] * fn.K) * 200,
        #     marker = markers[i % len(markers)],
        #     color = colors[i % len(colors)],
        #     ls = "-",
        #     lw = 0.4,
        #     label = f"Detector ID {channel_ID[index_channel]}",
        #     markersize = 6,
        #     capthick = 2,
        #     elinewidth = 2,
        #     capsize = 5
        # )
        print(grid_N[len(grid_N)-np.argmax(np.abs(rel_diff-np.max(rel_diff))[::-1]>= 1)])

    # Threshold line (neutral color)
    plt.axvline(
        100,
        color="black",
        ls="--",
        linewidth=1.5,
        label="Stability threshold \n(<1% variation)"
    )

    plt.xlabel("Grid size N", fontsize=20)
    plt.ylabel("Relative difference\n(simulated − analytical BI) [%]", fontsize=20)

    plt.xticks(grid_N[::2])
    plt.xticks(grid_N[1::2], minor=True)

    #plt.yticks(np.arange(0, 14, 2))
    #plt.yticks(np.arange(1, 14, 2), minor=True)

    plt.tick_params(axis='both', which='major', labelsize=20)

    # Grid styling
    plt.grid(which='major', alpha=0.4)
    plt.grid(which='minor', ls='--', alpha=0.3)

    plt.legend(fontsize=16, ncol=2,framealpha=1)
    plt.tight_layout()

    plt.savefig("/local/home/mp274748/Documents/paper/pile_up_cupid/figures/grid_size_systematics.pdf", dpi=300)
    plt.show()
def pulse_pos_systematic():
    df = pd.read_csv("../outputs/Systematics/BI_frac_4Pole_Bessel.csv")
    for index_channel in np.argsort(channel_ID):
        channel = channels[index_channel]
        data_channel = df[df["channel"] == channel]
        data_channel.loc[data_channel["BI_uncertainty"] > 3e-6, "BI_uncertainty"] = 0
        plt.errorbar(data_channel["pulse_center_ratio"], data_channel["BI"] * 1e5,
                     yerr = data_channel["BI_uncertainty"] * 1e5, label = f"detector ID {channel_ID[index_channel]}",
                     marker = 'o', ls = "")
        # print((np.max(data_channel["J"]) - np.min(data_channel["J"]))/np.mean(data_channel["J"]))
    plt.xlabel("Pulse center position within window")
    plt.ylabel("BI [1e-5 cts/keV/kg/year]")
    plt.legend(ncols = 3, bbox_to_anchor = (0.49, 0.19, 0.5, 0.5), shadow = True)
    plt.savefig("/local/home/mp274748/Documents/paper/pile_up_cupid/figures/pulse_position_systematics.pdf", dpi = 300)
    plt.grid()
    plt.show()
def window_size_systematic():
    df = pd.read_csv("../outputs/Systematics/BI_window_size.csv")
    for index_channel in np.argsort(channel_ID):
        channel = channels[index_channel]
        data_channel = df[df["channel"] == channel]
        data_channel.loc[data_channel["BI_uncertainty"] > 2e-6, "BI_uncertainty"] = 0
        plt.errorbar(data_channel["window_size"]*1e-4, data_channel["BI"] * 1e5,
                     yerr = data_channel["BI_uncertainty"] * 1e5, label = f"detector ID {channel_ID[index_channel]}",
                     marker = 'o', ls = "")
        # print((np.max(data_channel["J"]) - np.min(data_channel["J"]))/np.mean(data_channel["J"]))
    plt.xlabel("Window size [s]")
    plt.ylabel("BI [1e-5 cts/keV/kg/year]")
    plt.legend(ncols = 3, shadow = True)
    plt.grid()
    plt.ylim(4.1, 7)
    #plt.savefig("/local/home/mp274748/Documents/paper/pile_up_cupid/figures/window_size_systematics.pdf", dpi = 300)
    plt.show()
def compare_simulated_analytical_BI():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import utility.functions as fn

    # Load data
    #df1 = pd.read_csv("../outputs/Systematics/BI_meanpulse_test.csv", engine="python")
    #df2 = pd.read_csv("../outputs/Systematics/BI_meanpulses_test_2.csv", engine="python")
    #df = pd.concat((df1, df2), ignore_index=True)
    df = pd.read_csv("../outputs/BI_results_base_bis.csv")
    #df = pd.read_csv("../outputs/Systematics/BI_varying_grid_size.csv")
    #df = df[df["grid_N"] == 100]
    # Add detector ID and sort
    df["detector_id"] = df["channel"].apply(lambda x: channel_ID[channels.index(x)])
    df.sort_values("detector_id", inplace=True)

    # Select mean pulse only (can be extended later)
    # pulse_type = "mean"
    # df_sub = df[df["meanpulse_type"] == pulse_type].copy()
    df_sub = df
    # Compute relative difference [%]
    analytical_BI = df_sub["J"] * fn.K
    simulated_BI = df_sub["BI"]

    rel_diff = (simulated_BI - analytical_BI) / (simulated_BI + analytical_BI) * 200
    rel_diff_unc = df_sub["BI_uncertainty"] / (simulated_BI + analytical_BI) * 200

    # Categorical x positions
    detector_ids = df_sub["detector_id"].values
    x = np.arange(len(detector_ids))

    # Colorblind-friendly color
    color = "#0072B2"  # Okabe-Ito blue

    plt.figure(figsize=(8, 5))

    plt.errorbar(
        x,
        rel_diff,
        yerr=rel_diff_unc,
        marker='_',
        ls='',
        capsize=3*2,
        markersize=12,
        elinewidth=2,
        capthick=2,
        markeredgewidth = 2,
        color=color
    )

    # Reference line
    plt.axhline(0, color='black', ls='--', linewidth=1)

    # Axes labels
    plt.xlabel("Detector ID", fontsize=20)
    plt.ylabel("Relative difference\n(simulated − analytical BI) [%]", fontsize=20)

    # Ticks
    plt.xticks(x, detector_ids.astype(str))
    plt.tick_params(axis='both', which='major', labelsize=20)

    # Grid
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(left = 0.2)
    plt.savefig(
        "/local/home/mp274748/Documents/paper/pile_up_cupid/figures/simulated_analytical_BI.pdf",
        dpi=300
    )

    plt.show()


def WP_test():
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(f"../outputs/Systematics/BI_WP.csv")

    fig, ax1 = plt.subplots(1, figsize=(8.4, 7))

    # Colorblind-friendly palette (Okabe-Ito)
    color_ch1 = "#0072B2"  # blue
    color_ch2 = "#CC79A7"  # purple

    # Channel 3 (left axis)
    channel = 3
    data_channel = df[df["channel"] == channel]
    data_channel.sort_values("biais_current",inplace=True)
    p1, = ax1.plot(
        data_channel["biais_current"],
        data_channel["J_value"] * 100,
        marker='o',
        ls='-',
        lw=1,
        markersize=10,
        color=color_ch1,
        label=f"Detector ID {channel}"
    )

    # Channel 5 (right axis)
    channel = 5
    data_channel = df[df["channel"] == channel]
    data_channel.sort_values("biais_current",inplace=True)
    p2, = ax1.plot(
        data_channel["biais_current"],
        data_channel["J_value"] * 100,
        marker='s',
        ls='-',
        lw=1,
        markersize=10,
        color=color_ch2,
        label=f"Detector ID {channel}"
    )

    # Labels
    ax1.set_xlabel("Bias current [nA]", fontsize=20)

    ax1.set_ylabel(
        r"Global pile-up misidentification rate $\langle \mathcal{M} \rangle$ [%]",
        fontsize=20
    )

    ax1.tick_params(axis='both', which='major', labelsize=20)
    # Grid only on x (neutral)
    ax1.grid( alpha=0.3)

    # Offset text size
    ax1.yaxis.get_offset_text().set_fontsize(14)

    fig.tight_layout()
    ax1.legend(fontsize=20)
    plt.savefig("/local/home/mp274748/Documents/paper/pile_up_cupid/figures/WP_test.pdf", dpi=300)
    plt.show()


def meanpulse_with_dt_range():
    from scipy.interpolate import interp1d
    mp = np.load("../outputs/meanpulses_build/channel3_meanpulse_raw.npy")
    mp = mp / np.max(mp)
    mp_interp = interp1d(np.arange(len(mp)), mp, kind = 'cubic')
    mp = mp_interp(np.linspace(0, len(mp) - 1, 10000))
    fig, ax = plt.subplots(1, 1, figsize = (8, 6))
    ax.plot(np.arange(- np.argwhere(mp > 1e-2)[0], len(mp) - np.argwhere(mp > 1e-2)[0]) * 4096 / 10000 * 1e-1, mp,
            label = 'Pulse impulse response s(t)', color = "black")
    ax.fill_betweenx(np.array([-0.11, 1.1]), -0e-1, 8e-1, color = 'green', alpha = 0.3,
                     label = '$\Delta t$ window: 0.8 ms')
    ax.set_xlim(-5, 20)
    ax.set_ylim(-0.04, 1.1)
    plt.yticks(fontsize = 14)
    plt.legend(fontsize = 16)
    ax.set_xlabel("Time (ms)", fontsize = 16)
    plt.xticks(fontsize = 14)
    plt.xticks(np.arange(-5, 20, 1), minor = True)
    plt.yticks(np.arange(-0.1, 1, 0.1), minor = True)
    ax.grid(linestyle = '--', linewidth = 1)
    ax.grid(which = 'minor', linestyle = ':', linewidth = 0.5)
    fig.set_tight_layout(True)
    fig.savefig("/local/home/mp274748/Documents/paper/pile_up_cupid/figures/meanpulse_with_dt_range.pdf", dpi = 300)
    plt.show()


def r_PDF():
    from utility.double_beta_spectrum import pdf_ratio2b
    fig, ax = plt.subplots(figsize = (8, 6))
    r = np.linspace(0, 1, 1000)
    PDF = pdf_ratio2b(r)
    PDF_norm = np.sum(PDF) * (r[1] - r[0])
    PDF /= PDF_norm
    ax.plot(r, PDF, color = 'red')
    ax.fill_between(r, np.zeros_like(r), PDF, color = 'red', alpha = 0.2)
    ax.set_xlabel('Relative amplitude r', fontsize = 20)
    ax.set_ylabel('Probability Density', fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    ax.grid(linestyle = '--', linewidth = 1)
    fig.set_tight_layout(True)
    fig.savefig("/local/home/mp274748/Documents/paper/pile_up_cupid/figures/r_pdf.pdf", dpi = 300)
    plt.show()

def bb_spectrum():
    from utility.double_beta_spectrum import g
    fig, ax = plt.subplots(figsize = (8, 6))
    E = np.linspace(0, 3.034, 1000)
    spectrum = g(E)
    ax.plot(E, spectrum, color = 'blue')
    ax.fill_between(E, np.zeros_like(E), spectrum, color = 'blue', alpha = 0.2)
    ax.set_xlabel('Energy [MeV]', fontsize = 20)
    ax.set_ylabel('Energy distribution [Counts/keV]', fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    ax.grid(linestyle = '--', linewidth = 1)
    fig.set_tight_layout(True)
    fig.savefig("/local/home/mp274748/Documents/paper/pile_up_cupid/figures/bb_spectrum.pdf", dpi = 300)
    plt.show()

def denoise_NPS_comparaison():
    channel_list = [3, 5, 9, 11, 13, 15, 17, 19]
    channel_ID = [5, 9, 11, 10, 4, 2, 12, 3]
    RMS_reductions = [0.96, 0.78, 0.43, 0.61, 0.3, 0.5, 0.88, 0.34]
    StdCuts = [0.0003, 0.0004, 0.0008, 0.0006, 0.0012, 0.0007, 0.00035, 0.0025]

    fig, axes = plt.subplots(2, 4, figsize = (18, 9), sharex = True, sharey = True)

    freq = np.fft.rfftfreq(2048, d = 1 / 10000)
    window_size = 2048
    meas_name = "000813_20230628T161508"
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20
    })
    print(np.argsort(channel_ID))
    for idx, channel_arg in enumerate(np.argsort(channel_ID)):
        ax = axes.flatten()[idx ]

        channel = channel_list[channel_arg]
        RMS_reduction = RMS_reductions[channel_arg]

        nps_raw = np.load(
            f"/local/home/mp274748/PycharmProjects/Pileup_Analysis/outputs/NPS_study/"
            f"NPS_channel{channel}_win{window_size}_{meas_name}.npy"
        )[:1025]

        nps_denoised = np.load(
            f"/local/home/mp274748/PycharmProjects/Pileup_Analysis/outputs/NPS_study/"
            f"NPS_channel{channel}_win{window_size}_{meas_name}_denoised.npy"
        )[:1025]

        # Plot styling
        ax.plot(freq, nps_raw, label = "Raw", lw = 1.8, alpha = 0.8, color = "#D55E00")
        ax.plot(freq, nps_denoised, label = "Denoised", lw = 2.2, alpha = 0.9, color = "#0072B2")

        ax.set_xscale("log")
        ax.set_yscale("log")
        # Channel title instead of legend duplication
        ax.set_title(f"Detector ID {channel_ID[channel_arg]}", fontsize = 20)
        RMS_reduction = (np.sum(nps_raw) - np.sum(nps_denoised)) / np.sum(nps_raw)
        # Subtle annotation
        ax.text(
            0.05, 0.1,
            f"ΔRMS: {RMS_reduction * 100:.1f}%",
            transform = ax.transAxes,
            fontsize = 20,
            bbox = dict(facecolor = 'white', alpha = 0.7, edgecolor = 'none')
        )

        # Grid (important for log plots)
        ax.grid(True, which = "both", ls = "--", alpha = 0.3)

    # Global labels
    fig.supxlabel("Frequency [Hz]", fontsize = 20)
    fig.supylabel(r"NPS [ADU/$\sqrt{\mathrm{Hz}}$]", fontsize = 20)

    # Single legend for all subplots
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc = "upper center", ncol = 2, fontsize = 20)

    plt.tight_layout(rect = [0, 0, 1, 0.95])
    plt.savefig("/local/home/mp274748/Documents/these/figures/NPS_comparison_denoising.pdf", dpi = 300)


def meanpulse_fig():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d

    mp = np.load("../outputs/meanpulses_build/channel3_meanpulse_raw.npy")
    mp = mp / np.max(mp)

    mp_interp = interp1d(np.arange(len(mp)), mp, kind='cubic')
    mp = mp_interp(np.linspace(0, len(mp) - 1, 10000))

    # Time axis
    t = np.arange(len(mp))
    t0 = np.argwhere(mp > 1e-2)[0][0]
    t = (t - t0) * 4096 / 10000 * 1e-1  # ms

    # --- Thresholds ---
    y10, y90, y30 = 0.1, 0.9, 0.3

    # --- Find indices ---
    i10 = np.argmax(mp > y10)
    i90 = np.argmax(mp > y90)

    peak_idx = np.argmax(mp)

    # decay: after peak
    decay_region = mp[peak_idx:]
    i90_decay = peak_idx + np.argmax(decay_region < y90)
    i30_decay = peak_idx + np.argmax(decay_region < y30)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Pulse
    ax.plot(t, mp, label='Pulse impulse response s(t)', color="black")

    # --- Horizontal dashed lines ---
    ax.hlines(y10, xmin=t[0], xmax=t[np.argmax(mp > y10)],
              linestyles='dashed', colors='blue', linewidth=1)


    ax.hlines(y30, xmin=t[0], xmax=t[i30_decay],
              linestyles='dashed', colors='red', linewidth=1)
    ax.hlines(y90, xmin=t[0], xmax=t[i90_decay],
              linestyles='dashed', colors='black', linewidth=1)
    ax.text(-4, 0.91, '90% ', color = 'black', fontsize = 12)
    ax.text(-4, 0.31, '30%', color = 'red', fontsize = 12)
    ax.text(-4, 0.11, '10%', color = 'blue', fontsize = 12)
    ax.hlines(1, xmin=t[0], xmax=t[np.argmax(mp)],
              linestyles='dashed', colors='black', linewidth=1)
    ax.text(-4, 1.01, 'Amplitude', color = 'black', fontsize = 12)

    # --- Fill rise area (10% → 90%) ---
    ax.fill_between(t[i10:i90], mp[i10:i90],
                    color='blue', alpha=0.3, label='Rise region')

    # --- Fill decay area (90% → 30%) ---
    ax.fill_between(t[i90_decay:i30_decay], mp[i90_decay:i30_decay],
                    color='red', alpha=0.3, label='Decay region')

    # --- Label rise region ---
    t_rise_center = 0.5 * (t[i10] + t[i90])
    y_rise_center = -0.05

    ax.text(t_rise_center, y_rise_center,
            "Rise time",
            color='blue', fontsize=13,
            ha='center', va='center',
            alpha=0.9)

    # --- Label decay region ---
    t_decay_center = 0.5 * (t[i90_decay] + t[i30_decay])
    y_decay_center = -0.05

    ax.text(t_decay_center, y_decay_center,
            "Decay time",
            color='red', fontsize=13,
            ha='center', va='center',
            alpha=0.9)
    # --- Formatting ---
    ax.set_xlim(-5, 25)
    ax.set_ylim(-0.1, 1.1)

    ax.set_xlabel("Time (ms)", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # --- Remove box, keep only axes ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Optional: make axes slightly thicker (nicer for thesis)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)


    fig.set_tight_layout(True)

    fig.savefig("/local/home/mp274748/Documents/these/figures/meanpulse.pdf", dpi=300)
    plt.show()

def NPS_figure():
    nps = np.load("../outputs/NPS_study/NPS_channel3_win4096_000813_20230628T161508.npy")
    freq = np.fft.rfftfreq(len(nps), d=1/10000)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(freq, nps[:len(nps)//2+1], color = 'k')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency [Hz]", fontsize=20)
    ax.set_ylabel(r"NPS [ADU/$\sqrt{\mathrm{Hz}}$]", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    # --- Remove box, keep only axes ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Optional: make axes slightly thicker (nicer for thesis)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    fig.set_tight_layout(True)
    fig.savefig("/local/home/mp274748/Documents/these/figures/NPS.pdf", dpi=300)


def plot_mu_sigma_M(t_torch, mu_raw, sigma_raw, mu_corr, sigma_corr):
    from src.analysis import compute_A
    fig, ax1 = plt.subplots(figsize = (9, 7))
    ax2 = ax1.twinx()
    mu_raw = mu_raw.real
    sigma_raw = sigma_raw.real
    mu_corr = mu_corr.real
    sigma_corr = sigma_corr.real
    t_torch = t_torch.real
    # --- AXIS 1 (left) ---
    l1, = ax1.plot(t_torch * 1e3, mu_raw[-1], color = "#0072B2", label = "Raw μ")
    f1 = ax1.fill_between(
        t_torch * 1e3,
        (mu_raw[-1] - sigma_raw[-1]).cpu(),
        (mu_raw[-1] + sigma_raw[-1]).cpu(),
        alpha = 0.3, color = "#0072B2", label = "Raw ±σ"
    )

    l2, = ax1.plot(t_torch * 1e3, mu_corr[-1], color = "#D55E00", label = "Denoised μ")
    f2 = ax1.fill_between(
        t_torch * 1e3,
        (mu_corr[-1] - sigma_corr[-1]).cpu(),
        (mu_corr[-1] + sigma_corr[-1]).cpu(),
        alpha = 0.3, color = "#D55E00", label = "Denoised ±σ"
    )

    ax1.set_xlabel("Δt [s]", fontsize = 14)
    ax1.set_ylabel("Y", fontsize = 14)

    # --- AXIS 2 (right) ---
    l3, = ax2.plot(
        t_torch * 1e3,
        compute_A(mu_raw, sigma_raw, N_sigma = 1.28)[-1],
        color = "#0072B2", ls = "--", label = "Raw $\mathcal{M}$"
    )

    l4, = ax2.plot(
        t_torch * 1e3,
        compute_A(mu_corr, sigma_corr, N_sigma = 1.28)[-1],
        color = "#D55E00", ls = "-.", label = "Denoised $\mathcal{M}$"
    )

    ax2.set_ylabel(r"$\mathcal{M}$", fontsize = 14)
    ax2.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax1.tick_params(axis = 'both', which = 'major', labelsize = 12)
    # --- Separate legends ---
    # --- SHARED LEGEND ---
    handles = [l1, f1, l2, f2, l3, l4]
    labels = [h.get_label() for h in handles]
    ticks2 = [0, 0.2, 0.4, 0.6, 0.8, 1]
    ticks1 = np.linspace(0.988, 1.0, len(ticks2))
    ax1.set_yticks(ticks1)
    ax2.set_yticks(ticks2)
    ax1.set_ylim(0.986, 1.002)
    # map limits to ax2
    t1_min, t1_max = ticks1[0], ticks1[-1]
    t2_min, t2_max = ticks2[0], ticks2[-1]
    y1_min, y1_max = ax1.get_ylim()
    y2_min = t2_min + (y1_min - t1_min) * (t2_max - t2_min) / (t1_max - t1_min)
    y2_max = t2_min + (y1_max - t1_min) * (t2_max - t2_min) / (t1_max - t1_min)

    ax2.set_ylim(y2_min, y2_max)
    ax1.grid(linestyle = '--', linewidth = 0.5)
    ax1.legend(handles, labels, loc = "upper right", fontsize = 12)
    fig.savefig("/local/home/mp274748/Documents/these/figures/mu_sigma_M_raw_denoise.pdf", dpi = 300)


if __name__ == "__main__":
    pulse_template()
    plt.show()
