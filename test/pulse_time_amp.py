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
import os
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
fit_parameters = np.loadtxt(f"{BASE}/outputs/fits/pulse_fit_params.csv", delimiter = ',', skiprows = 1)

for channel, StdCut in zip(channels, StdCuts):
    window_size = 4096
    meas_name = "000813_20230628T161508"
    bessel_order = 6
    zero, *poles = fit_parameters[(fit_parameters[:, 0] == channel) & (fit_parameters[:, 1] == bessel_order), 3:][0]
    poles = np.sort(poles)
    poles_bis = np.copy(poles)
    for i in range(-5, 10):
        poles_bis[:3] = poles[:3] * (1 + i * 0.05)
        fitted_pulse_function_base = lambda x: sim.make_pulse_pole_zero_bessel_ct(bessel_order, 2500, zero, *poles_bis)(x*1e-4)
        t10 = root_scalar(lambda x: fitted_pulse_function_base(x) - 0.1, bracket = [-50, 0]).root
        t90 = root_scalar(lambda x: fitted_pulse_function_base(x) - 0.9, bracket = [-50, 0]).root
        risetime = t90 - t10
        nps_bis = np.load(f"{BASE}/outputs/NPS_study/NPS_channel{channel}_win{window_size}_{meas_name}.npy")
        nps_bis *= (8. / 3.)  # Adjust for Hann window effect
        acceptance = 0.9
        n_deriv = 0

        _, _, _, _, pulse_factor, _ = ds.get_channel_specs(channel, n_deriv=n_deriv,
                                                           window_fct=np.hanning)  # , nps_name="denoised_noise"
        signal_amp = ds.get_amp_Q_val(channel) * pulse_factor
        if os.path.isfile(f"{BASE}/outputs/meanpulses_build/channel{channel}_meanpulse_time_{i}.npy"):
            mean_pulse_filtered = np.load(f"{BASE}/outputs/meanpulses_build/channel{channel}_meanpulse_time_{i}.npy")
        else:
            dataset_single_name = "pup_n1-d0_000813_20230628T161508"
            file_path_single, path_pos_single = ds.find_files(meas_name, dataset_single_name, channel,
                                                              specific_subdir="RUN9_pulse_injected_new/")
            data_single = ds.CachedBinaryDataset_withgenerated(file_path_single, path_pos_single, 4096,
                                                               pulse=fitted_pulse_function_base,
                                                               n_windows=12288, win_shift=0)
            mean_pulse = an.build_mean_pulse(data_single, rms_thr=StdCut, batch_size=2048, device=device,
                                             use_loader=False).cpu().numpy()
            S, w, H_unit = an.compute_H(mean_pulse, nps_bis, np.hanning)
            H_unit_torch = torch.tensor(H_unit, dtype=torch.cfloat, device=device)
            mean_pulse_filtered = an.build_mean_pulse_filteralignement(data_single, rms_thr=StdCut,
                                                                       H=H_unit_torch, batch_size=2048,
                                                                       device=device, use_loader=False).cpu().numpy()
            np.save(f"{BASE}/outputs/meanpulses_build/channel{channel}_meanpulse_time_{i}", mean_pulse_filtered)
        for amp_gain in np.arange(0.5, 2.1, 0.1):
            fitted_pulse_function = lambda x : amp_gain * fitted_pulse_function_base(x)
            S, w, H_unit = an.compute_H(mean_pulse_filtered, nps_bis, np.hanning)

            # Convert all arrays to torch tensors on GPU
            N_sigma = norm.ppf(1 - (1 - acceptance) * 100 / 100)
            S_torch = torch.tensor(S, dtype = torch.cfloat, device = device)
            H_unit_torch = torch.tensor(H_unit, dtype = torch.cfloat, device = device)
            w_torch = torch.tensor(w, dtype = torch.cfloat, device = device)
            nps_torch = torch.tensor(nps_bis, dtype = torch.cfloat, device = device)
            signal_amp_torch = torch.tensor(signal_amp*amp_gain, dtype = torch.float32, device = device)

            t_min, t_max, N_t = 0, 8e-4, 50
            r_min, r_max, N_r = 0., .5, 50

            ratio_distribution = pdf_ratio2b(np.linspace(r_min, r_max, N_r))
            ratio_distribution /= np.mean(ratio_distribution)

            t_torch = torch.linspace(t_min, t_max, N_t, dtype = torch.cfloat, device = device)
            r_torch = torch.linspace(r_min, r_max, N_r, dtype = torch.cfloat, device = device)
            ratio_distribution_torch = torch.tensor(ratio_distribution, dtype = torch.cfloat, device = device)
            directory_path = f"{BASE}/outputs/Pileup_filter_functions/channel_{channel}/meanpulse_test/"
            if os.path.isfile(f"channel{channel}_functions_eff{int(acceptance * 100):d}_pulsetimeamp_test_{i}_{int(amp_gain*10):d}.npy"):
                f1_opt, f2_opt = np.load(
                    f"{BASE}/outputs/Pileup_filter_functions/channel_{channel}/meanpulse_test/" +
                    f"channel{channel}_functions_eff{int(acceptance * 100):d}_meanpulse_test_{meanpulse_type}.npy"
                )
            else :
                f1_opt, f2_opt, J_values = an.optimize_filters(S_torch, H_unit_torch, w_torch, t_torch, r_torch, nps_torch,
                                                               signal_amp_torch, ratio_distribution_torch, N_sigma = N_sigma,
                                                               n_trials = 500, verbose = False)

                os.makedirs(directory_path, exist_ok = True)
                np.save(directory_path +
                        f"channel{channel}_functions_eff{int(acceptance * 100):d}_pulsetimeamp_test_{i}_{int(amp_gain*10):d}",
                        np.array([f1_opt, f2_opt]))
                np.save(directory_path +
                        f"channel{channel}_J_{int(acceptance * 100):d}_pulsetimeamp_test_{i}_{int(amp_gain*10):d}",
                        np.array(J_values))

            f1_t = torch.tensor(f1_opt, dtype = torch.float32, device = device)
            f2_t = torch.tensor(f2_opt, dtype = torch.float32, device = device)

            S_H_delayed, S_H, S2_over_nps = an.precompute_constants(S_torch, H_unit_torch, w_torch, t_torch, nps_torch)

            f1_t /= torch.mean(f1_t * S_H).real
            f2_t /= torch.mean(f2_t * S_H).real

            S_H_delayed, S_H, S2_over_nps = an.precompute_constants(S_torch, H_unit_torch, w_torch, t_torch, nps_torch)
            J = an.compute_J(f1_t, f2_t, S_H_delayed, r_torch, S_H, S2_over_nps, signal_amp_torch, ratio_distribution_torch,
                             N_sigma = N_sigma, full_output = False, use_interp = True)

            Js.append(float(J))
            channel_data.append(channel)
            risetimes.append(risetime)
            amp_gains.append(amp_gain)
            polemultipliers.append( (1 + i * 0.05) )

            Js_np = np.array(Js)
            channel_data_np = np.array(channel_data)
            polemultipliers_np = np.array(polemultipliers)
            risetimes_np = np.array(risetimes)
            amp_gains_np = np.array(amp_gains)

            # Create a structured array
            data = np.zeros(Js_np.shape[0], dtype=[
                ('channel', 'U32'),
                ('polemultiplier', float),
                ('risetime', float),
                ("amp_gain", float),
                ('J', float)
            ])

            data['channel'] = channel_data_np
            data['polemultiplier'] = polemultipliers_np
            data['risetime'] = risetimes_np
            data['amp_gain'] = amp_gains_np
            data['J'] = Js_np

            header = "channel,polemultipliers,risetime,amp_gain"

            np.savetxt(
                f"{BASE}/outputs/Systematics/BI_pulsetimeamp_test.csv",
                data,
                delimiter=",",
                header=header,
                comments="",
                fmt="%s,%.18e,%.18e,%.18e,%.18e"
            )

            print(f"BI saved at {BASE}/outputs/Systematics/BI_pulsetimeamp_test.csv")