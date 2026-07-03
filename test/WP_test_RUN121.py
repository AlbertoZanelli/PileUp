import sys
from pathlib import Path
# Add src to Python path dynamically
BASE = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE))
import numpy as np
import src.analysis as an
import src.dataset as ds
import utility.functions as fn
import src.simulation as sim
from utility.double_beta_spectrum import pdf_ratio2b
import torch
from scipy.optimize import root_scalar
from scipy.stats import norm
import os

# Ensure we have GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.utils.data import DataLoader
def build_MP_and_NPS():
    path = "/media/mp274748/Transcend/data/Ulysse/RUN121"
    file_name_list = []
    rms_thr_list = []
    amp_max_list = []
    rt_list = []
    channel_list = []
    meas_list = []
    sigma_list = []
    SNR_list = []
    for file_name in os.listdir(path):
        file_name = "000007_20260204T131521_006_000.bin"
        file_path = ds.find_file(os.path.join(path,file_name))
        if file_path[-7:]!="000.bin":
            continue
        dataset = ds.BinaryDataset(file_path, 4096)
        dl =DataLoader(dataset, batch_size=4096, shuffle=False)
        #fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,6))
        for pulses in dl:
            std = pulses.std(dim=1)
        rms_thr_list.append(np.percentile(std,70))
        file_name_list.append(file_name)
        nps = an.create_NPS_torch(dataset,n_deriv=0,rms_thr=np.percentile(std,70), batch_size=2048, window_fct=np.hanning,
                                              use_loader=True, device='cpu').astype(np.float64)
        np.save(f"/media/mp274748/Transcend/data/Ulysse/RUN121/NPS_RUN121_{file_name.split('.bin')[0]}.npy",nps)
        #ax1.loglog(nps)
        #nps = np.load(f"/media/mp274748/Transcend/data/Ulysse/RUN121/NPS_RUN121_{file_name.split('.bin')[0]}.npy")
        # plt.show()
        arg_max = torch.argmax(pulses[std.numpy()>np.percentile(std,70)*4],dim=1).numpy()
        pos_pulses = dataset.positions[std.numpy()>np.percentile(std,70)*4]
        pos_pulses = pos_pulses[np.diff(pos_pulses,prepend =torch.tensor([-torch.inf]))>5000]+arg_max[np.diff(pos_pulses,prepend =torch.tensor([-torch.inf]))>5000]
        pos_pulses = pos_pulses[dataset.positions[-1]-pos_pulses>2048]
        dataset = ds.BinaryDataset(file_path, 4096,positions=pos_pulses,win_shift = 2048)
        if len(dataset)==0:
            amp_max_list.append(0)
            continue
        meanpulse,amp = an.build_mean_pulse_filteralignement_from_raw(dataset, rms_thr = np.percentile(std,70),
                                                                  nps = nps, batch_size = 2048,
                                                                  pulse_start_pos=-40, pulse_end_pos=300,
                                                                  amplitude_bounds=(0.1,5),
                                                                  return_amp = True,return_pulses = False,
                                                                  device = 'cpu', use_loader = False)
        #ax2.plot(pulse.T/torch.max(pulse,dim=1).values,color='gray',alpha=0.3)
        try:
            hist, bin = np.histogram(amp,bins=50,range=(np.percentile(amp,10),np.percentile(amp,90)))
            amp_max = bin[np.argmax(hist)]
        except:
            amp_max = 0
        amp_max_list.append(amp_max)
        try:
            rt = fn.compute_rt_bin(meanpulse, 10000)
        except:
            rt = 0
        rt_list.append(rt)
        channel_list.append(int(file_name.split("_")[2]))
        meas_list.append(int(file_name.split("_")[0]))
        try:
            S, w, H_unit = an.compute_H(meanpulse, nps, np.hanning)
            sigma = np.sqrt(np.mean(np.abs(H_unit) ** 2 * nps))
            SNR = amp / sigma
        except:
            sigma = 0
            SNR = 0
        sigma_list.append(sigma)
        SNR_list.append(SNR)
        np.save(f"/media/mp274748/Transcend/data/Ulysse/RUN121/MP_RUN121_{file_name.split('.bin')[0]}.npy",meanpulse)
        print(file_name_list,rms_thr_list,amp_max_list,rt_list)
        np.savetxt("/media/mp274748/Transcend/data/Ulysse/RUN121/rms_amp_RUN121.txt",np.column_stack([file_name_list,channel_list,meas_list,amp_max_list,sigma_list,SNR_list,rt_list]),fmt="%s")
def main():
    J_values_all = []
    meas_values = []
    channel_list = []
    directory = f"/feynman/work/projets/cupid/data/Ulisse/RUN121/"
    rms_amp_data = np.loadtxt(directory+"rms_amp_RUN121.csv",delimiter=",",dtype=str)[1:]
    data = np.loadtxt(
        f"{BASE}/outputs/Systematics/BI_WP_RUN121_Ulysse.csv",
        delimiter=",",
        skiprows=1   # skip header
    )

    channel_list    = data[:, 0].astype(int).tolist()
    meas_values     = data[:, 1].astype(int).tolist()
    J_values_all    = data[:, 2].astype(float).tolist()
    for data in rms_amp_data:
        file_name = data[0]
        meas = int(data[4])
        channel = int(data[3])
        amp = float(data[2])
        if meas<16:
            continue
        try:
            mean_pulse_filtered = np.load(directory+f"MP_RUN121_{file_name.split('.bin')[0]}.npy")
            nps = np.load(directory+f"NPS_RUN121_{file_name.split('.bin')[0]}.npy")
        except:
            continue
        S, w, H_unit = an.compute_H(mean_pulse_filtered, nps, np.hanning)
        signal_amp = amp*0.1
        if signal_amp==0:
            continue
        acceptance = 0.9
        # Convert all arrays to torch tensors on GPU
        N_sigma = norm.ppf(1 - (1 - acceptance) * 100 / 100)
        S_torch = torch.tensor(S, dtype = torch.cfloat, device = device)
        H_unit_torch = torch.tensor(H_unit, dtype = torch.cfloat, device = device)
        w_torch = torch.tensor(w, dtype = torch.cfloat, device = device)
        nps_torch = torch.tensor(nps, dtype = torch.cfloat, device = device)
        signal_amp_torch = torch.tensor(signal_amp, dtype = torch.float32, device = device)

        t_min, t_max, N_t = 0, 8e-4, 100
        r_min, r_max, N_r = 0., .5, 100

        ratio_distribution = pdf_ratio2b(np.linspace(r_min, r_max, N_r))
        ratio_distribution /= np.mean(ratio_distribution)

        t_torch = torch.linspace(t_min, t_max, N_t, dtype = torch.cfloat, device = device)
        r_torch = torch.linspace(r_min, r_max, N_r, dtype = torch.cfloat, device = device)
        ratio_distribution_torch = torch.tensor(ratio_distribution, dtype = torch.cfloat, device = device)
        try:
            f1_opt, f2_opt, J_values = an.optimize_filters(S_torch, H_unit_torch, w_torch, t_torch, r_torch, nps_torch,
                                                           signal_amp_torch, ratio_distribution_torch, N_sigma = N_sigma,
                                                           n_trials = 500, verbose = False)
        except:
            continue
        # directory_path = f"{BASE}/outputs/Pileup_filter_functions/channel_{channel}/WP_test/"
        # os.makedirs(directory_path, exist_ok = True)
        # np.save(directory_path +
        #         f"channel{channel}_functions_eff{int(acceptance * 100):d}_WP_test_{meas}",
        #         np.array([f1_opt, f2_opt]))
        # np.save(directory_path +
        #         f"channel{channel}_J_{int(acceptance * 100):d}_WP_test_{meas}",
        #         np.array(J_values))
        f1_t = torch.tensor(f1_opt, dtype = torch.float32, device = device)
        f2_t = torch.tensor(f2_opt, dtype = torch.float32, device = device)

        S_H_delayed, S_H, S2_over_nps = an.precompute_constants(S_torch, H_unit_torch, w_torch, t_torch, nps_torch)

        f1_t /= torch.mean(f1_t * S_H).real
        f2_t /= torch.mean(f2_t * S_H).real

        S_H_delayed, S_H, S2_over_nps = an.precompute_constants(S_torch, H_unit_torch, w_torch, t_torch, nps_torch)
        J = an.compute_J(f1_t, f2_t, S_H_delayed, r_torch, S_H, S2_over_nps, signal_amp_torch, ratio_distribution_torch,
                         N_sigma = N_sigma, full_output = False, use_interp = True)
        J_values_all.append(J.cpu().numpy())
        meas_values.append(meas)
        channel_list.append(channel)
        data =np.column_stack([np.array(channel_list),np.array(meas_values), np.array(J_values_all)])
        print(f"Measurement {meas}: J = {J:.6e}")
        header = "channel,measurement,J_value"
        np.savetxt(
            f"{BASE}/outputs/Systematics/BI_WP_RUN121_Ulysse.csv",
            data,
            comments="",
            delimiter = ",",
            header = header
        )
    print("Done.")