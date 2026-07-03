import sys
from pathlib import Path
# Add src to Python path dynamically
BASE = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE))
import numpy as np
import src.analysis as an
import src.dataset as ds
import torch
import os
# Ensure we have GPU available

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
StdCuts = [0.0003, 0.0004, 0.0008, 0.0006, 0.0012, 0.0007, 0.00035, 0.0025]
channels = [3, 5, 9, 11, 13, 15, 17, 19]
n_deriv = 0
meas_name = "000813_20230628T161508"
for channel, StdCut in zip(channels, StdCuts):
    window_sizes = [2 ** 13 - 6 * 2 ** 9 - i * 4 * 2 ** 8 + 5*2 ** 10 for i in range(10)]
    data_noise, _ = ds.create_data_sets(channel, {"data_name": ["000813_20230628T161508"],
                                                  "pos_name": ["pup_n1-d0_000813_20230628T161508"],
                                                  "specific_subdir": [""],
                                                  "full_info": [False]},
                                        win_length=window_sizes[0], shift=0, pos_prefix="_bis",
                                        len_data=2 ** 13 + 2 ** 12, cached=True)

    for window_size in window_sizes:
        data_noise.set_win_length(window_size, win_shift=window_size // 2)
        nps = an.create_NPS_torch(data_noise,n_deriv=0,rms_thr=StdCut, batch_size=2048, window_fct=np.hanning,
                                      use_loader=False,device=device)
        directory_path = f"{BASE}/outputs/NPS_study/"
        os.makedirs(directory_path, exist_ok=True)
        np.save(f"{BASE}/outputs/NPS_study/NPS_channel{channel}_win{window_size}_{meas_name}_denoised.npy",
                nps)