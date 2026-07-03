import sys
from pathlib import Path
# Add src to Python path dynamically
BASE = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE))
import numpy as np
import src.analysis as an
import src.dataset as ds
import src.simulation as sim
from utility.double_beta_spectrum import pdf_ratio2b
import torch
from scipy.optimize import root_scalar
from scipy.stats import norm
from scipy.optimize import curve_fit
import os
import json
# Ensure we have GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
StdCuts = [0.0003, 0.0004, 0.0008, 0.0006, 0.0012, 0.0007, 0.00035, 0.0025]
channels = [3, 5, 9, 11, 13, 15, 17, 19]
fp_model = lambda *params: sim.filtered_pulse_model(*params, fc = 2500, order = 6)

BIs = []
BI_uncertainties = []
rps = []
rp_uncertainties = []
Js = []
channel_data = []
risetimes = []
polemultipliers = []
amp_gains = []
meanpulse_type_list = []
popt_list = []
single_pulse_indexes =  [[126, 47, 77],
                         [394, 390, 240],
                         [33, 64, 322],
                         [71, 312, 199],
                         [101, 233, 71],
                         [253, 52, 147],
                         [238, 103, 6],
                         [295, 67, 199]]

fit_parameters = np.loadtxt(f"{BASE}/outputs/fits/pulse_fit_params.csv", delimiter = ',', skiprows = 1)

for channel, StdCut in zip(channels, StdCuts):
    window_size = 4096
    meas_name = "000813_20230628T161508"

    # bessel_order = 6
    # zero, *poles = fit_parameters[(fit_parameters[:, 0] == channel) & (fit_parameters[:, 1] == bessel_order), 3:][0]
    # poles = np.sort(poles)
    # fitted_pulse_function = lambda x: sim.make_pulse_pole_zero_bessel_ct(bessel_order, 2500, zero, *poles)(x * 1e-4)
    fitted_pulse_function = np.fromfile(f"{BASE}/outputs/meanpulses_build/pos_pulses_rawdata/average_pulse_channel_{channel}.bin")
    pos_pulses = np.load(f"{BASE}/outputs/meanpulses_build/pos_pulses_rawdata/pos_pulses_channel_{channel}.npy")
    file_name = ds.find_file(f"000813_20230628T161508_{channel:03}_000.bin")
    dataset_pulse = ds.CachedBinaryDataset(file_name, window_size, positions = pos_pulses,
                                           win_shift_start = -window_size // 2)
    dataset_single_name = "pup_n1-d0_000813_20230628T161508"
    file_path_single, path_pos_single = ds.find_files(meas_name, dataset_single_name, channel,
                                                      specific_subdir = "RUN9_pulse_injected_new/")
    data_single = ds.CachedBinaryDataset_withgenerated(file_path_single, path_pos_single, 4096,
                                                       pulse = fitted_pulse_function,
                                                       n_windows = 12288, win_shift = 0)

    dataset_pileup_name = "pup_n1-d8_000813_20230628T161508"
    file_path_pileup, path_pos_pileup = ds.find_files(meas_name, dataset_pileup_name, channel,
                                                      specific_subdir = "RUN9_pulse_injected_new/")
    data_pileup = ds.CachedBinaryDataset_withgenerated(file_path_pileup, path_pos_pileup, 4096,
                                                       pulse = fitted_pulse_function,
                                                       n_windows = 12288, win_shift = 0)

    nps_bis = np.load(f"{BASE}/outputs/NPS_study/NPS_channel{channel}_win{window_size}_{meas_name}.npy")
    nps_bis *= (8./3.)  # Adjust for Hann window effect
    acceptance = 0.9
    n_deriv = 0

    _, _, _, _, pulse_factor, _ = ds.get_channel_specs(channel, n_deriv = n_deriv,
                                                       window_fct = np.hanning)  # , nps_name="denoised_noise"
    signal_amp = ds.get_amp_Q_val(channel) * pulse_factor

    # meanpulse_bis = fitted_pulse_function(np.arange(-window_size//2,window_size//2))
    mean_pulse = an.build_mean_pulse(data_single, rms_thr = StdCut, batch_size = 2048, device = device,
                                     use_loader = False).cpu().numpy()
    S, w, H_unit = an.compute_H(mean_pulse, nps_bis, np.hanning)
    H_unit_torch = torch.tensor(H_unit, dtype = torch.cfloat, device = device)
    mean_pulse_filtered = an.build_mean_pulse_filteralignement(data_single, rms_thr = StdCut,
                                                               H = H_unit_torch, batch_size = 2048,
                                                               pulse_start_pos=-100, pulse_end_pos=4096,
                                                               device = device, use_loader = False).cpu().numpy()
    for meanpulse_type in ["raw", "fitted"]:
        if meanpulse_type == "fitted":
            t = np.arange(-len(mean_pulse_filtered) // 2, len(mean_pulse_filtered) // 2) * 1e-4
            p0 = [-1000, -1339.1, -1000.0, -237.0, -100]
            fit_fct_pole = lambda x, *params: sim.make_pulse_pole_zero_bessel_ct(6, 2500, *params)(x)
            popt, pcov = curve_fit(fit_fct_pole, t, mean_pulse_filtered, p0=p0,
                                   bounds=([-5000, -5000, -5000, -5000, -5000], [0, 0, 0, 0, 0]))
            mean_pulse_filtered = fit_fct_pole(t, *popt)
            popt_list.append(popt)
        else:
            popt_list.append([])
        S, w, H_unit = an.compute_H(mean_pulse_filtered, nps_bis, np.hanning)

        # Convert all arrays to torch tensors on GPU
        N_sigma = norm.ppf(1 - (1 - acceptance) * 100 / 100)
        S_torch = torch.tensor(S, dtype = torch.cfloat, device = device)
        H_unit_torch = torch.tensor(H_unit, dtype = torch.cfloat, device = device)
        w_torch = torch.tensor(w, dtype = torch.cfloat, device = device)
        nps_torch = torch.tensor(nps_bis, dtype = torch.cfloat, device = device)
        signal_amp_torch = torch.tensor(signal_amp, dtype = torch.float32, device = device)

        t_min, t_max, N_t = 0, 8e-4, 50
        r_min, r_max, N_r = 0., .5, 50

        ratio_distribution = pdf_ratio2b(np.linspace(r_min, r_max, N_r))
        ratio_distribution /= np.mean(ratio_distribution)

        t_torch = torch.linspace(t_min, t_max, N_t, dtype = torch.cfloat, device = device)
        r_torch = torch.linspace(r_min, r_max, N_r, dtype = torch.cfloat, device = device)
        ratio_distribution_torch = torch.tensor(ratio_distribution, dtype = torch.cfloat, device = device)

        f1_opt, f2_opt, J_values = an.optimize_filters(S_torch, H_unit_torch, w_torch, t_torch, r_torch, nps_torch,
                                                       signal_amp_torch, ratio_distribution_torch, N_sigma = N_sigma,
                                                       n_trials = 500, verbose = False)
        directory_path = f"{BASE}/outputs/Pileup_filter_functions/channel_{channel}/meanpulse_test/"
        os.makedirs(directory_path, exist_ok = True)
        np.save(directory_path +
                f"channel{channel}_functions_eff{int(acceptance * 100):d}_meanpulses_analysis_test_{meanpulse_type}",
                np.array([f1_opt, f2_opt]))
        np.save(directory_path +
                f"channel{channel}_J_{int(acceptance * 100):d}_meanpulses_analysis_test_{meanpulse_type}",
                np.array(J_values))
        # f1_opt, f2_opt = np.load(
        #     f"{BASE}/outputs/Pileup_filter_functions/channel_{channel}/meanpulse_test/" +
        #     f"channel{channel}_functions_eff{int(acceptance * 100):d}_meanpulse_test_{meanpulse_type}.npy"
        # )
        f1_t = torch.tensor(f1_opt, dtype = torch.float32, device = device)
        f2_t = torch.tensor(f2_opt, dtype = torch.float32, device = device)

        S_H_delayed, S_H, S2_over_nps = an.precompute_constants(S_torch, H_unit_torch, w_torch, t_torch, nps_torch)

        f1_t /= torch.mean(f1_t * S_H).real
        f2_t /= torch.mean(f2_t * S_H).real

        S_H_delayed, S_H, S2_over_nps = an.precompute_constants(S_torch, H_unit_torch, w_torch, t_torch, nps_torch)
        J = an.compute_J(f1_t, f2_t, S_H_delayed, r_torch, S_H, S2_over_nps, signal_amp_torch, ratio_distribution_torch,
                         N_sigma = N_sigma, full_output = False, use_interp = True)
        BI, rp, sigma_rp, sigma_BI = an.compute_BI_torch(
            data_pileup,
            data_single,
            acceptance,
            H_unit_torch,
            f1_t,
            f2_t,
            window_fct = np.hanning,
            n_deriv = 0,
            compute_uncertainty = True,
            batch_size = 2048 * 2,
            use_loader = False
        )
        print(f"Channel {channel}, window size {window_size}, J: {J:.4e}, 1-rp: {1 - rp:.4f}, BI: {BI:.4e}")
        BIs.append(float(BI))
        BI_uncertainties.append(float(sigma_BI))
        rps.append(float(rp))
        rp_uncertainties.append(float(sigma_rp))
        Js.append(float(J))
        channel_data.append(channel)
        # polemultipliers.append(poles_bis[0] / poles[0])
        # risetimes.append(risetime)
        meanpulse_type_list.append(meanpulse_type)

        BIs_np = np.array(BIs)
        BI_uncertainties_np = np.array(BI_uncertainties)
        rps_np = np.array(rps)
        rp_uncertainties_np = np.array(rp_uncertainties)
        Js_np = np.array(Js)
        channel_data_np = np.array(channel_data)
        polemultipliers_np = np.array(polemultipliers)
        risetimes_np = np.array(risetimes)
        amp_gains_np = np.array(amp_gains)
        meanpulse_type_np = np.array(meanpulse_type_list)


        popt_list_np = np.array([json.dumps(list(lst)) for lst in popt_list])
        # Create a structured array
        data = np.zeros(BIs_np.shape[0], dtype=[
            ('channel', 'U32'),
            ("meanpulse_type", 'U32'),
            ('BI', float),
            ('BI_uncertainty', float),
            ('1-rp', float),
            ('rp_uncertainty', float),
            ('J', float),
            ('popt', 'U256')
        ])

        data['channel'] = channel_data_np
        data['meanpulse_type'] = meanpulse_type_np
        data['BI'] = BIs_np
        data['BI_uncertainty'] = BI_uncertainties_np
        data['1-rp'] = 1 - rps_np
        data['rp_uncertainty'] = rp_uncertainties_np
        data['J'] = Js_np
        data['popt'] = popt_list_np
        header = "channel,meanpulse_type,BI,BI_uncertainty,1-rp,rp_uncertainty,J,popt"
        import csv
        np.savetxt(
            f"{BASE}/outputs/Systematics/BI_meanpulses_analysis_test.csv",
            data,
            delimiter=",",
            header=header,
            comments="",
            fmt="%s,%s,%.18e,%.18e,%.18e,%.18e,%.18e,%s",
            quoting = csv.QUOTE_MINIMAL
        )

        print(f"BI saved at {BASE}/outputs/Systematics/BI_meanpulses_analysis_test.csv")