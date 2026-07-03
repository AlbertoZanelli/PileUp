import sys
from pathlib import Path
# Add src to Python path dynamically
sys.path.append(str(Path(__file__).resolve().parent.parent))
import time
import numpy as np
import src.analysis as an
import src.dataset as ds
import src.simulation as sim
from utility.double_beta_spectrum import pdf_ratio2b
import torch
from torch.utils.data import DataLoader
from scipy.interpolate import interp1d
from scipy.stats import norm
from scipy.optimize import curve_fit
import os
# Ensure we have GPU available
BASE = Path(__file__).resolve().parent.parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
StdCuts = [0.0003, 0.0004, 0.0008, 0.0006, 0.0012, 0.0007, 0.00035, 0.0025]
channels = [3, 5, 9, 11, 13, 15, 17, 19]
window_size_data = []
channel_data = []

BI_uncertainties, rp_uncertainties = [], []
BIs = []
rps = []
Js = []
acceptance = 0.9
n_deriv = 0
meas_name = "000813_20230628T161508"
for channel, StdCut in zip(channels, StdCuts):
    window_sizes = [2 ** 13 - 6 * 2 ** 9 - i * 4 * 2 ** 8 + 5*2 ** 10 for i in range(10)]
    # window_sizes = [2 ** 13 - 6 * 2 ** 9 - i * 2 ** 10 + 5*2 ** 10 for i in range(10)]
    # fitted_pulse_function = np.fromfile(ds.find_file(f"ch{channel}_fit*.bin",
    #                                                  specific_subdir="RUN9_pulse_injected_new/"))
    # fitted_pulse_function = np.pad(fitted_pulse_function, (4000, 4000), 'constant', constant_values=0)
    # meanpulse = np.fromfile(ds.find_file(f"{meas_name}_{channel:03}_???.bin_edmean.bin",
    #                                      specific_subdir="RUN9_pulse_injected_new/"))
    # t = np.arange(-len(meanpulse) // 2, len(meanpulse) // 2) * 1e-4
    # p0 = [1, t[np.argmax(meanpulse)], -1000, -1339.1, -237.0, -100, -1000.0]
    fit_fct_pole = lambda x, *params: sim.apply_bessel_to_pulse(x, sim.pulse_pole_zero(x, *params), fc=2500, order=6,
                                                                fs=1e4)
    # popt, pcov = curve_fit(fit_fct_pole, t, meanpulse, p0=p0,
    #                        bounds=([0, -0.5, -5000, -5000, -5000, -5000, -5000], [10, 0.5, 0, 0, 0, 0, 0]))
    # fitted_pulse_function = lambda t: fit_fct_pole(t*1e-4, *popt)
    # dataset_single_name = "pup_n1-d0_000813_20230628T161508"
    # file_path_single, path_pos_single = ds.find_files(meas_name, dataset_single_name, channel,
    #                                                   specific_subdir="RUN9_pulse_injected_new/")
    # data_single = ds.CachedBinaryDataset_withgenerated(file_path_single, path_pos_single, window_sizes[0],
    #                                                    pulse=fitted_pulse_function,
    #                                                    n_windows=12288, win_shift=window_sizes[0]//2)
    #
    # dataset_pileup_name = "pup_n1-d8_000813_20230628T161508"
    # file_path_pileup, path_pos_pileup = ds.find_files(meas_name, dataset_pileup_name, channel,
    #                                                   specific_subdir="RUN9_pulse_injected_new/")
    # data_pileup = ds.CachedBinaryDataset_withgenerated(file_path_pileup, path_pos_pileup, window_sizes[0],
    #                                                    pulse=fitted_pulse_function,
    #                                                    n_windows=12288, win_shift=window_sizes[0]//2)

    _, _, _, _, pulse_factor, _ = ds.get_channel_specs(channel, n_deriv=n_deriv, window_fct=np.hanning)
    signal_amp = ds.get_amp_Q_val(channel) * pulse_factor
    t_min, t_max, N_t = 0, 8e-4, 100
    r_min, r_max, N_r = 0., .5, 25

    ratio_distribution = pdf_ratio2b(np.linspace(r_min, r_max, N_r))
    ratio_distribution /= np.mean(ratio_distribution)

    # meanpulse = np.fromfile(ds.find_file(f"{meas_name}_{channel:03}_???.bin_edmean.bin",
    #                                      specific_subdir="RUN9_pulse_injected_new/"))
    # meanpulse = np.pad(meanpulse, (4000, 4000), 'constant', constant_values=0)
    # mean_pulse = an.build_mean_pulse(data_single, rms_thr=StdCut, batch_size=2048, device=device,
    #                                  use_loader=False).cpu().numpy()
    # S, w, H_unit = ds.compute_H(mean_pulse, nps, np.hanning)
    # H_unit_torch = torch.tensor(H_unit, dtype=torch.cfloat, device=device)
    # meanpulse = an.build_mean_pulse_filteralignement(data_single, rms_thr=StdCut,
    #                                                            H=H_unit_torch, batch_size=2048,
    #                                                            device=device, use_loader=False).cpu().numpy()
    # directory_path = f"{BASE}/outputs/meanpulses_build/"
    # os.makedirs(directory_path, exist_ok=True)
    # np.save(f"{directory_path}/channel{channel}_meanpulse_build_from_fit_4Pole_Bessel.npy"
    #         ,meanpulse)
    meanpulse = np.load(f"{BASE}/outputs/meanpulses_build/channel{channel}_meanpulse_build_from_fit_4Pole_Bessel.npy")
    meanpulse /= np.max(meanpulse)
    t = np.arange(-len(meanpulse) // 2, len(meanpulse) // 2) * 1e-4
    p0 = [1, t[np.argmax(meanpulse)], -1000, -1339.1, -237.0, -100, -1000.0]
    # meanpulse = np.pad(meanpulse, (4000, 4000), 'constant', constant_values=0)
    popt, pcov = curve_fit(fit_fct_pole, t, meanpulse, p0=p0,
                           bounds=([0, -0.5, -5000, -5000, -5000, -5000, -5000], [10, 0.5, 0, 0, 0, 0, 0]))
    fitted_pulse_function_on_data = lambda t: fit_fct_pole(t * 1e-4, *popt)

    t_torch = torch.linspace(t_min, t_max, N_t, dtype=torch.cfloat, device=device)
    r_torch = torch.linspace(r_min, r_max, N_r, dtype=torch.cfloat, device=device)
    ratio_distribution_torch = torch.tensor(ratio_distribution, dtype=torch.cfloat, device=device)

    for window_size in window_sizes:
        # data_single.set_win_length(window_size, win_shift=window_size // 2)
        # data_pileup.set_win_length(window_size, win_shift=window_size // 2)
        nps_bis = np.load(f"{BASE}/results/NPS_study/NPS_channel{channel}_win{window_size}_cut{StdCut}_{meas_name}.npy")
        nps_bis *= (8./3.)  # Adjust for Hann window effect
        # meanpulse_bis = meanpulse[np.argmax(meanpulse) - window_size // 2: np.argmax(meanpulse) + window_size // 2]
        meanpulse_bis = fitted_pulse_function_on_data( np.arange(-window_size // 2, window_size // 2))
        S, w, H_unit = ds.compute_H(meanpulse_bis, nps_bis, np.hanning)
        # Convert all arrays to torch tensors on GPU
        N_sigma = norm.ppf(1 - (1 - acceptance) * 100 / 100)
        S_torch = torch.tensor(S, dtype=torch.cfloat, device=device)
        H_unit_torch = torch.tensor(H_unit, dtype=torch.cfloat, device=device)
        w_torch = torch.tensor(w, dtype=torch.cfloat, device=device)
        nps_torch = torch.tensor(nps_bis, dtype=torch.cfloat, device=device)
        signal_amp_torch = torch.tensor(signal_amp, dtype=torch.float32, device=device)

        # f1_opt, f2_opt, J_values = an.optimize_filters(S_torch, H_unit_torch, w_torch, t_torch, r_torch, nps_torch,
        #                                                signal_amp_torch, ratio_distribution_torch, N_sigma=N_sigma,
        #                                                n_trials=500, verbose=False)
        # directory_path = f"{BASE}/outputs/Pileup_filter_functions/channel_{channel}/window_size/"
        # os.makedirs(directory_path, exist_ok=True)
        # np.save(directory_path +
        #         f"functions_eff{int(acceptance * 100):d}_window_size_{int(window_size):d}_4Pole_Bessel_fit_data",
        #         np.array([f1_opt, f2_opt]))
        # np.save(directory_path +
        #         f"J_{int(acceptance * 100):d}_window_size_{int(window_size):d}_4Pole_Bessel_fit_data",
        #         np.array(J_values))
        f1_opt, f2_opt = np.load(
            f"{BASE}/outputs/Pileup_filter_functions/channel_{channel}/window_size/"
            f"functions_eff{int(acceptance * 100):d}_window_size_{int(window_size):d}_4Pole_Bessel_fit_data.npy"
        )
        f1_t = torch.tensor(f1_opt, dtype=torch.float32, device=device)
        f2_t = torch.tensor(f2_opt, dtype=torch.float32, device=device)
        # f1_t = torch.tensor(f1_opt_inter(w), dtype=torch.float32, device=device)
        # f2_t = torch.tensor(f2_opt_inter(w), dtype=torch.float32, device=device)

        S_H_delayed, S_H, S2_over_nps = an.precompute_constants(S_torch, H_unit_torch, w_torch, t_torch, nps_torch)

        f1_t /= torch.mean(f1_t * S_H).real
        f2_t /= torch.mean(f2_t * S_H).real

        J = an.compute_J(f1_t, f2_t, S_H_delayed, r_torch, S_H, S2_over_nps, signal_amp_torch, ratio_distribution_torch,
                         N_sigma=N_sigma)

        # BI, rp, sigma_rp, sigma_BI = an.compute_BI_torch(
        #     data_pileup,
        #     data_single,
        #     acceptance,
        #     H_unit_torch,
        #     f1_t,
        #     f2_t,
        #     window_fct=np.hanning,
        #     n_deriv=0,
        #     compute_uncertainty=True,
        #     batch_size=2048 * 2,
        #     use_loader=False
        # )
        # BIs.append(float(BI))
        # BI_uncertainties.append(float(sigma_BI))
        # rps.append(float(rp))
        # rp_uncertainties.append(float(sigma_rp))
        Js.append(float(J))
        channel_data.append(channel)
        window_size_data.append(window_size)
# BIs = np.array(BIs)
# BI_uncertainties = np.array(BI_uncertainties)
# rps = np.array(rps)
# rp_uncertainties = np.array(rp_uncertainties)
Js = np.array(Js)
channel_data = np.array(channel_data)
window_size_data = np.array(window_size_data)

# data = np.column_stack([channel_data, window_size_data, BIs,BI_uncertainties, 1 - rps,rp_uncertainties, Js])
# header = "channel,window_size,BI,BI_err,1-rp,rp_err,J"

data = np.column_stack([channel_data, window_size_data,Js])
header = "channel,window_size,J"

np.savetxt(
    f"{BASE}/outputs/Systematics/BI_window_size_4Pole_Bessel_fit_data_bis.csv",
    data,
    delimiter=",",
    header=header,
    comments=""
)
print("Results saved to outputs/Systematics/BI_window_size_4Pole_Bessel_fit_data.csv")
