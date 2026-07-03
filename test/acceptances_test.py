import sys
from pathlib import Path
# Add src to Python path dynamically
BASE = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE))
import time
import numpy as np
import src.analysis as an
import src.dataset as ds
import src.simulation as sim
from utility.double_beta_spectrum import pdf_ratio2b
import torch
from scipy.stats import norm
n_deriv = 0
window_size = 4096
meas_name = "000813_20230628T161508"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
BIs = []
BI_uncertainties = []
channel_list = []
J_list = []
acceptance_list = []
channels = [3, 5, 9, 11, 13, 15, 17, 19]
channel_ID = [5, 9, 11, 10, 4, 2, 12, 3]
StdCuts = [0.0003, 0.0004, 0.0008, 0.0006, 0.0012, 0.0007, 0.00035, 0.0025]



for channel, StdCut in zip(channels, StdCuts):
    fitted_pulse_function = np.load(f"{BASE}/outputs/meanpulses_build/pos_pulses_rawdata/average_pulse_channel{channel}.npy")
    # Load datasets and channel specifications
    file_path = ds.find_file(f"000813_20230628T161508_{channel:03}_000.bin")
    dataset_single_name = "pup_n1-d0_000813_20230628T161508"
    file_path_single, path_pos_single = ds.find_files(meas_name, dataset_single_name, channel,
                                                      specific_subdir="RUN9_pulse_injected_new/")
    data_single = ds.CachedBinaryDataset_withgenerated(file_path_single, path_pos_single, window_size,
                                                       pulse=fitted_pulse_function,
                                                       n_windows=15999, win_shift=0)

    dataset_pileup_name = "pup_n1-d8_000813_20230628T161508"
    file_path_pileup, path_pos_pileup = ds.find_files(meas_name, dataset_pileup_name, channel,
                                                      specific_subdir="RUN9_pulse_injected_new/")
    data_pileup = ds.CachedBinaryDataset_withgenerated(file_path_pileup, path_pos_pileup, window_size,
                                                       pulse=fitted_pulse_function,
                                                       n_windows=15999, win_shift=0)
    # data_single, data_pileup = ds.create_data_sets_denoise(channel, win_length=2000, shift=0, pos_prefix="_bis")
    nps_bis = np.load(f"{BASE}/outputs/NPS_study/NPS_channel{channel}_win{window_size}_{meas_name}.npy")
    nps_bis *= (8. / 3.)  # Adjust for Hann window effect
    mean_pulse = an.build_mean_pulse(data_single, rms_thr=StdCut, batch_size=2048, device=device,
                                     use_loader=False).cpu().numpy()
    S, w, H_unit = an.compute_H(mean_pulse, nps_bis, np.hanning)
    H_unit_torch = torch.tensor(H_unit, dtype=torch.cfloat, device=device)
    meanpulse_bis = an.build_mean_pulse_filteralignement(data_single, rms_thr=StdCut,
                                                               H=H_unit_torch, batch_size=2048,
                                                               pulse_start_pos=-100, pulse_end_pos=4096,
                                                               device=device, use_loader=False).cpu().numpy()
    # meanpulse_bis = np.load(f"{BASE}/outputs/meanpulses_build/channel{channel}_meanpulse_mean.npy")
    np.save(f"{BASE}/outputs/meanpulses_build/channel{channel}_meanpulse_mean.npy", meanpulse_bis)
    signal_amp = ds.get_amp_Q_val(channel)
    S, w, H_unit = an.compute_H(meanpulse_bis, nps_bis, np.hanning)
    # Convert all arrays to torch tensors on GPU
    S_torch = torch.tensor(S, dtype=torch.cfloat, device=device)
    H_unit_torch = torch.tensor(H_unit, dtype=torch.cfloat, device=device)
    w_torch = torch.tensor(w, dtype=torch.cfloat, device=device)
    nps_torch = torch.tensor(nps_bis, dtype=torch.cfloat, device=device)
    signal_amp_torch = torch.tensor(signal_amp, dtype=torch.float32, device=device)

    t_min, t_max, N_t = 0, 8e-4, 100
    r_min, r_max, N_r = 0., .5, 100
    ratio_distribution = pdf_ratio2b(np.linspace(r_min, r_max, N_r))
    ratio_distribution /= np.mean(ratio_distribution)

    t_torch = torch.linspace(t_min, t_max, N_t, dtype=torch.cfloat, device=device)
    r_torch = torch.linspace(r_min, r_max, N_r, dtype=torch.cfloat, device=device)
    ratio_distribution_torch = torch.tensor(ratio_distribution, dtype=torch.cfloat, device=device)
    # Loop over acceptance levels
    acceptances = [0.7, 0.8, 0.84, 0.87, 0.9, 0.92, 0.93, 0.94, 0.95, 0.96, 0.99]
    for acceptance in acceptances:
        N_sigma = norm.ppf(1 - (1 - acceptance) * 100 / 100)
        torch.cuda.synchronize()
        t0 = time.time()
        f1_opt, f2_opt, J_values = an.optimize_filters(S_torch, H_unit_torch, w_torch, t_torch, r_torch, nps_torch,
                                                       signal_amp_torch, ratio_distribution_torch, N_sigma=N_sigma,
                                                       n_trials=1000, verbose=False, use_interp=False)
        torch.cuda.synchronize()
        t1 = time.time()
        print("Time:", t1 - t0)
        J = float(J_values[-1])
        # f1_opt, f2_opt = np.load(f"{BASE}/outputs/Pileup_filter_functions/channel_{channel}/functions_eff{int(acceptance*100):d}_good_mp_no_interp.npy")
        # print(torch.tensor(f1_0,dtype=torch.float).size())
        # save_filter_function
        #np.save(f"{BASE}/outputs/training_Js/channel_{channel}/J_{int(acceptance * 100):d}_base_true_mp_2048", np.array(J_values))
        np.save(f"{BASE}/outputs/Pileup_filter_functions/channel_{channel}/functions_eff{int(acceptance * 100):d}.npy",
                np.array([f1_opt, f2_opt]))
        f1_t = torch.tensor(f1_opt, dtype=torch.float32, device=device)
        f2_t = torch.tensor(f2_opt, dtype=torch.float32, device=device)

        S_H_delayed, S_H, S2_over_nps = an.precompute_constants(S_torch, H_unit_torch, w_torch, t_torch, nps_torch)

        f1_t /= torch.mean(f1_t * S_H).real
        f2_t /= torch.mean(f2_t * S_H).real

        # J = an.compute_J(f1_t, f2_t, S_H_delayed, r_torch, S_H, S2_over_nps, signal_amp_torch, ratio_distribution_torch,
        #                  N_sigma=N_sigma, full_output=False, use_interp=True)
        # J = float(J)

        # Compute PSD and BI
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
            batch_size = 2048,
            use_loader = False
        )
        BIs.append(float(BI))
        BI_uncertainties.append(float(sigma_BI))
        channel_list.append(channel)
        J_list.append(J)
        acceptance_list.append(acceptance)

        print("max memory: ",torch.cuda.max_memory_allocated() / 1024**2)
        print(f"Channel {channel}, Acceptance {acceptance}: J={J}, 1-rp={1 - rp}, BI={BI}")


        data = np.column_stack([channel_list,acceptance_list, BIs, BI_uncertainties,J_list])
        header = "channel,acceptance,BI,BI_uncertainty,J"

        np.savetxt(
            f"{BASE}/outputs/BI_acceptance_study.csv",
            data,
            delimiter = ",",
            header = header,
            comments = ""
        )
