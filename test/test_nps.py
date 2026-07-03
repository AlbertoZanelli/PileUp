# import numpy as np
# import matplotlib.pyplot as plt
#
# BASE = "../"
# channels = [3, 5, 9, 11, 13, 15, 17, 19]
# channel_ID = [5, 9, 11, 10, 4, 2, 12, 3]
#
# window_size = 4096
# meas_name = "000813_20230628T161508"
# fig,ax = plt.subplots(figsize=(10,6))
# for index_channel in np.argsort(channel_ID):
#     channel = channels[index_channel]
#     nps_bis = np.load(f"{BASE}/outputs/NPS_study/NPS_channel{channel}_win{window_size}_{meas_name}.npy")
#     nps_bis[nps_bis>4e-4]=np.nan  # limit to better see the differences
#     l,=plt.loglog(nps_bis[:window_size//2],lw=0.7)
#     plt.plot([],[],label=f"ID {channel_ID[index_channel]}",ls='',marker='s',c=l.get_color())
#
#
# plt.legend()
# plt.show()
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
from scipy.optimize import root_scalar,curve_fit
from scipy.stats import norm
import os
import matplotlib.pyplot as plt

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# meas_name = "000813_20230628T161508"
# BASE = "../"
# channels = [3, 5, 9, 11, 13, 15, 17, 19]
# channel_ID = [5, 9, 11, 10, 4, 2, 12, 3]
# StdCuts = [0.0003, 0.0004, 0.0008, 0.0006, 0.0012, 0.0007, 0.00035, 0.0025]
#
# channel=channels[channel_ID.index(11)]
# print(channel)
# StdCut = StdCuts[channels.index(channel)]
#
# pos_pulses = np.load(f"{BASE}/outputs/meanpulses_build/pos_pulses_rawdata/pos_pulses_channel_{channel}.npy")
# file_name = ds.find_file(f"000813_20230628T161508_{channel:03}_000.bin")
# window_size = 4096
# dataset_pulse = ds.CachedBinaryDataset(file_name, window_size, positions = pos_pulses,
#                                        win_shift_start = -window_size // 2)
# single_pulse_indexes =  [[126, 47, 77],
#                          [394, 390, 240],
#                          [33, 334, 322],
#                          [71, 312, 199],
#                          [101, 233, 71],
#                          [253, 52, 147],
#                          [238, 103, 6],
#                          [295, 67, 199]]
# for meanpulse_type in ["mean","single_40Q","single_70Q","single_100Q"]:
#     if meanpulse_type == "mean":
#         pulses_min_baseline = (dataset_pulse.data - dataset_pulse.data[:, :window_size // 2 - 48].mean(dim = 1,
#                                                                                                        keepdim = True))
#         amp_pulse = torch.max(pulses_min_baseline, dim = 1).values
#         pulses_min_baseline_norm = pulses_min_baseline / amp_pulse[:, None]
#         meanpulse = torch.mean(pulses_min_baseline_norm, dim = 0)
#         meanpulse /= torch.max(meanpulse)
#         nps_bis = np.load(f"{BASE}/outputs/NPS_study/NPS_channel{channel}_win{window_size}_{meas_name}.npy")
#         nps_bis *= (8. / 3.)  # Adjust for Hann window effect
#         S, w, H_unit = an.compute_H(meanpulse, nps_bis, np.hanning)
#         H_unit_torch = torch.tensor(H_unit, dtype = torch.cfloat, device = device)
#         mean_pulse_filtered = an.build_mean_pulse_filteralignement(dataset_pulse, rms_thr = StdCut,
#                                                                    H = H_unit_torch,
#                                                                    batch_size = len(dataset_pulse),
#                                                                    pulse_start_pos = -100, pulse_end_pos = 4096,
#                                                                    device = device,
#                                                                    use_loader = False).cpu().numpy()
#         fitted_pulse_function = mean_pulse_filtered
#         t = np.arange(-len(mean_pulse_filtered) // 2, len(mean_pulse_filtered) // 2) * 1e-4
#         p0 = [-1000, -1339.1, -1000.0, -237.0, -100]
#         fit_fct_pole = lambda x, *params: sim.make_pulse_pole_zero_bessel_ct(6, 2500, *params)(x)
#         popt, pcov = curve_fit(fit_fct_pole, t, mean_pulse_filtered, p0 = p0,
#                                bounds = ([-5000, -5000, -5000, -5000, -5000], [0, 0, 0, 0, 0]))
#
#         base = fitted_pulse_function
#     else :
#         if meanpulse_type == "single_40Q":
#             index = single_pulse_indexes[channels.index(channel)][0]
#         elif meanpulse_type == "single_70Q":
#             index = single_pulse_indexes[channels.index(channel)][1]
#         elif meanpulse_type == "single_100Q":
#             index = single_pulse_indexes[channels.index(channel)][2]
#         fitted_pulse_function = dataset_pulse.data[index].cpu().numpy()
#         fitted_pulse_function -= np.mean(fitted_pulse_function[:window_size // 2 - 48])
#         fitted_pulse_function /= np.max(fitted_pulse_function)
#     l,=plt.loglog(np.fft.fftfreq(4096,1e-4)[:2048],np.abs(np.fft.fft(fitted_pulse_function))[:2048],lw=0.1)
#     # l,=plt.plot(fitted_pulse_function-base)
#     plt.plot([],[],label=meanpulse_type,c=l.get_c())
#
# fit_parameters = np.loadtxt(f"{BASE}/outputs/fits/pulse_fit_params.csv", delimiter = ',', skiprows = 1)
#
# bessel_order = 6
# zero, *poles = fit_parameters[(fit_parameters[:, 0] == channel) & (fit_parameters[:, 1] == bessel_order), 3:][0]
# poles = np.sort(poles)
# fitted_pulse_function = lambda x: sim.make_pulse_pole_zero_bessel_ct(bessel_order, 2500, zero, *poles)(x * 1e-4)
#
# l,=plt.loglog(np.fft.fftfreq(4096,1e-4)[:2048],np.abs(np.fft.fft(fitted_pulse_function(np.arange(-2048,2048))))[:2048],lw=1)
# plt.plot([],[],label="fit",c=l.get_c())
# l,=plt.loglog(np.fft.fftfreq(4096,1e-4)[:2048],np.abs(np.fft.fft(fit_fct_pole(t, *popt)))[:2048],lw=1)
#
# # l,=plt.plot(fitted_pulse_function-base)
# plt.plot([],[],label="fit meanpulse",c=l.get_c())
# plt.legend()
# plt.show()
# measurements = [779,    780,    781,    782,    783,    784,    785,    786,    787]
# biais_current = [2.89,2.036,1.57,1.0,0.549,0.279,4.8383,7.696,10]
# channel = 3
# meas = 787
# for meas in measurements:
#     l,=plt.loglog(np.fromfile(f"{BASE}/outputs/NPS_study/NPS_measurement_{meas}_channel{channel}.bin"),label = biais_current[measurements.index(meas)])
#     mean_pulse_filtered = np.fromfile(f"{BASE}/outputs/meanpulses_build/meanpulse_measurement_{meas}_channel{channel}.bin")
#     plt.loglog(np.abs(np.fft.fft(mean_pulse_filtered)),c=l.get_color())
# plt.legend()
# plt.show()


rms_s = np.array([29, 35, 40, 50, 70, 95, 23, 18, 18]) * 1e-5
signal_amps = [0.028, 0.033, 0.035, 0.04, 0.04, 0.031, 0.022, 0.018, 0.015]
measurements = [779, 780, 781, 782, 783, 784, 785, 786, 787]
channel = 3
rms_s = np.array([29,35,40,50,70,95,23,18,16-1.3])*1e-5
signal_amps =  [0.028,  0.033,  0.035,  0.04,   0.04,  0.031,  0.022,  0.018,  0.015]
measurements = [779,    780,    781,    782,    783,    784,    785,    786,    787]
biais_current = [2.89,2.036,1.57,1.0,0.549,0.279,4.8383,7.696,10]

channel = 3
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,6))
for idx in np.argsort(biais_current):
    meas = measurements[idx]
    nps = np.fromfile(f"../outputs/NPS_study/NPS_measurement_{meas}_channel{channel}.bin")
    meanpulse = np.fromfile(f"../outputs/meanpulses_build/meanpulse_measurement_{meas}_channel{channel}.bin")
    l,=ax1.loglog(nps[:1025],label=f"{biais_current[idx]}")
    ax2.loglog(np.abs(np.fft.fft(meanpulse)[:1025]),c=l.get_color())
ax1.legend()
plt.show()
