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

for channel, StdCut in zip(channels, StdCuts):
    window_size = 4096
    meas_name = "000813_20230628T161508"
    pos_pulses = np.load(f"{BASE}/outputs/meanpulses_build/pos_pulses_rawdata/pos_pulses_channel_{channel}.npy")
    file_name = ds.find_file(f"000813_20230628T161508_{channel:03}_000.bin")
    dataset_pulse = ds.CachedBinaryDataset(file_name, window_size, positions = pos_pulses,
                                           win_shift_start = -window_size // 2)
    meanpulse = an.build_mean_pulse(dataset_pulse, rms_thr=StdCut, batch_size=2048, device=device,
                                     use_loader=False).cpu().numpy()
    nps_bis = np.load(f"{BASE}/outputs/NPS_study/NPS_channel{channel}_win{window_size}_{meas_name}.npy")
    nps_bis *= (8. / 3.)  # Adjust for Hann window effect
    S, w, H_unit = an.compute_H(meanpulse, nps_bis, np.hanning)
    H_unit_torch = torch.tensor(H_unit, dtype = torch.cfloat, device = device)
    mean_pulse_filtered = an.build_mean_pulse_filteralignement(dataset_pulse, rms_thr = StdCut,
                                                               H = H_unit_torch,
                                                               batch_size = len(dataset_pulse),
                                                               pulse_start_pos = -100, pulse_end_pos = 4096,
                                                               device = device,
                                                               use_loader = False).cpu().numpy()
    np.save(f"{BASE}/outputs/meanpulses_build/pos_pulses_rawdata/average_pulse_channel{channel}.npy",mean_pulse_filtered)