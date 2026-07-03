import sys
from pathlib import Path
# Add src to Python path dynamically
BASE = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE))
import numpy as np
import src.analysis as an
import src.dataset as ds
import src.simulation as sim
import utility.functions as fn
from utility.double_beta_spectrum import pdf_ratio2b
import torch
from scipy.stats import norm
from scipy.optimize import curve_fit
import os
# Ensure we have GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
StdCuts = [0.0003, 0.0004, 0.0008, 0.0006, 0.0012, 0.0007, 0.00035, 0.0025]
channels = [3, 5, 9, 11, 13, 15, 17, 19]
BIs = []
BI_uncertainties = []
rps = []
rp_uncertainties = []
Js = []
channel_data = []
pulse_center_ratios_data = []
pulse_center_ratios = [i*0.05 + 0.2 for i in range(0,10)]
data_fit = np.loadtxt(f"{BASE}/outputs/fits/pulse_fit_params_bessel.txt")
for channel, StdCut in zip(channels, StdCuts):
    window_size = 4096
    meas_name = "000813_20230628T161508"
    # fitted_pulse_function = np.fromfile(ds.find_file(f"ch{channel}_fit*.bin",
    #                                                  specific_subdir = "RUN9_pulse_injected_new/"))
    popt = data_fit[data_fit[:, 0] == channel, 1:][0]
    fitted_pulse_function = lambda x: sim.apply_bessel_to_pulse(x * 1e-4, sim.pulse_pole_zero(x * 1e-4, *popt),
                                                                fc = 2500, order = 6, fs = 1e4)
    dataset_single_name = "pup_n1-d0_000813_20230628T161508"
    file_path_single, path_pos_single = ds.find_files(meas_name, dataset_single_name, channel,
                                                      specific_subdir = "RUN9_pulse_injected_new/")
    data_single = ds.CachedBinaryDataset_withgenerated(file_path_single, path_pos_single, window_size*2,
                                                       pulse = fitted_pulse_function,
                                                       n_windows = 12288, win_shift = 0)
    data_single.set_win_length(4096)
    dataset_pileup_name = "pup_n1-d8_000813_20230628T161508"
    file_path_pileup, path_pos_pileup = ds.find_files(meas_name, dataset_pileup_name, channel,
                                                      specific_subdir = "RUN9_pulse_injected_new/")
    data_pileup = ds.CachedBinaryDataset_withgenerated(file_path_pileup, path_pos_pileup, window_size*2,
                                                       pulse = fitted_pulse_function,
                                                       n_windows = 12288, win_shift = 0)
    data_pileup.set_win_length(4096)
    nps = np.load(f"{BASE}/outputs/NPS_study/NPS_channel{channel}_win{window_size}_{meas_name}.npy")
    nps *= (8./3.)  # Adjust for Hann window effect
    acceptance = 0.9
    n_deriv = 0
    signal_amp = ds.get_amp_Q_val(channel)

    t_min, t_max, N_t = 0, 8e-4, 100
    r_min, r_max, N_r = 0., .5, 25

    ratio_distribution = pdf_ratio2b(np.linspace(r_min, r_max, N_r))
    ratio_distribution /= np.mean(ratio_distribution)

    t_torch = torch.linspace(t_min, t_max, N_t, dtype = torch.cfloat, device = device)
    r_torch = torch.linspace(r_min, r_max, N_r, dtype = torch.cfloat, device = device)
    ratio_distribution_torch = torch.tensor(ratio_distribution, dtype = torch.cfloat, device = device)

    N_sigma = norm.ppf(1 - (1 - acceptance) * 100 / 100)
    signal_amp_torch = torch.tensor(signal_amp, dtype = torch.float32, device = device)

    for pulse_center_ratio in pulse_center_ratios:
        data_single.set_fracshift(0.5-pulse_center_ratio)
        data_pileup.set_fracshift(0.5-pulse_center_ratio)
        meanpulse = an.build_mean_pulse(data_single, rms_thr=StdCut, batch_size=2048, device=device,
                                        pulse_center_ratio=pulse_center_ratio,
                                         use_loader=False).cpu().numpy()
        _, _, H_unit = an.compute_H(meanpulse, nps, lambda x: fn.asymmetric_hann(x, pulse_center_ratio))
        H_unit_torch = torch.tensor(H_unit, dtype=torch.cfloat, device=device)
        meanpulse = an.build_mean_pulse_filteralignement(data_single, rms_thr=StdCut,
                                                         pulse_center_ratio=pulse_center_ratio,
                                                         H=H_unit_torch, batch_size=2048,
                                                         device=device, use_loader=False).cpu().numpy()
        meanpulse /= np.max(meanpulse)

        S, w, H_unit = an.compute_H(meanpulse, nps, lambda x: fn.asymmetric_hann(x, pulse_center_ratio))

        S_torch = torch.tensor(S, dtype = torch.cfloat, device = device)
        H_unit_torch = torch.tensor(H_unit, dtype = torch.cfloat, device = device)
        w_torch = torch.tensor(w, dtype = torch.cfloat, device = device)
        nps_torch = torch.tensor(nps, dtype = torch.cfloat, device = device)
        f1_opt, f2_opt = np.load(
            f"{BASE}/outputs/Pileup_filter_functions/channel_{channel}/window_size/"
            f"functions_eff{int(acceptance * 100):d}_window_size_{int(window_size):d}_4Pole_Bessel.npy"
        )
        f1_t = torch.tensor(f1_opt, dtype = torch.float32, device = device)
        f2_t = torch.tensor(f2_opt, dtype = torch.float32, device = device)
        f1_opt, f2_opt, J_values = an.optimize_filters(S_torch, H_unit_torch, w_torch, t_torch, r_torch, nps_torch,
                                                       signal_amp_torch, ratio_distribution_torch, N_sigma=N_sigma,
                                                       n_trials=500, verbose=False,f1_init = f1_t, f2_init = f2_t,
                                                       use_interp = True)
        f1_t = torch.tensor(f1_opt, dtype = torch.float32, device = device)
        f2_t = torch.tensor(f2_opt, dtype = torch.float32, device = device)

        directory_path = f"{BASE}/outputs/Pileup_filter_functions/channel_{channel}/frac_test/"
        os.makedirs(directory_path, exist_ok=True)
        np.save(directory_path +
                f"functions_eff{int(acceptance * 100):d}_frac{pulse_center_ratio}_4Pole_Bessel_interp_train",
                np.array([f1_opt, f2_opt]))
        np.save(directory_path +
                f"J_{int(acceptance * 100):d}_frac{pulse_center_ratio}_4Pole_Bessel_interp_train",
                np.array(J_values))

        S_H_delayed, S_H, S2_over_nps = an.precompute_constants(S_torch, H_unit_torch, w_torch, t_torch, nps_torch)
        J = an.compute_J(f1_t, f2_t, S_H_delayed, r_torch, S_H, S2_over_nps, signal_amp_torch, ratio_distribution_torch,
                         N_sigma=N_sigma, full_output=False, use_interp = True)
        BI, rp, sigma_rp, sigma_BI = an.compute_BI_torch(
            data_pileup,
            data_single,
            acceptance,
            H_unit_torch,
            f1_t,
            f2_t,
            window_fct= lambda x: fn.asymmetric_hann(x, pulse_center_ratio),
            n_deriv=0,
            compute_uncertainty=True,
            batch_size=2048 * 2,
            use_loader=False
        )
        BIs.append(float(BI))
        BI_uncertainties.append(float(sigma_BI))
        rps.append(float(rp))
        rp_uncertainties.append(float(sigma_rp))
        Js.append(float(J))
        channel_data.append(channel)
        pulse_center_ratios_data.append(pulse_center_ratio)
        print(f"Channel {channel}, window size {window_size}, pulse center ratio {pulse_center_ratio}, "
              f"J: {J:.4e}, 1-rp: {1 - rp:.4f}, BI: {BI:.4e} ± {sigma_BI:.4e}")
BIs = np.array(BIs)
BI_uncertainties = np.array(BI_uncertainties)
rps = np.array(rps)
rp_uncertainties = np.array(rp_uncertainties)
Js = np.array(Js)
channel_data = np.array(channel_data)
pulse_center_ratios_data = np.array(pulse_center_ratios_data)
data = np.column_stack([channel_data, pulse_center_ratios_data, BIs, BI_uncertainties,
                        1 - rps, rp_uncertainties, Js])
header = "channel,pulse_center_ratio,BI,BI_uncertainty,1-rp,rp_uncertainty,J"
np.savetxt(
    f"{BASE}/outputs/Systematics/BI_frac_4Pole_Bessel_interp_train.csv",
    data,
    delimiter=",",
    header=header,
    comments=""
)

