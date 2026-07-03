from torch.utils.data import DataLoader
import sys
from pathlib import Path
# Add src to Python path dynamically
BASE = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE))
import numpy as np
import src.dataset as ds
import torch
import os
# Ensure we have GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
channels = [3, 5, 9, 11, 13, 15, 17, 19]


len_file = 268435459
position_data = np.arange(10240*4, len_file, 10240)*4
np.savetxt(ds.main_dir + "/RUN9/000813_20230628T161508.pos", position_data[None,:].T.astype(int), fmt="%d")
noise_pos_file = np.loadtxt(ds.main_dir + "/RUN9/000813_20230628T161508.pos", ndmin = 2)
StdCuts = [0.0003, 0.0004, 0.0008, 0.0006, 0.0012, 0.0007, 0.00035, 0.0025]
meas_name = "000813_20230628T161508"
for channel, StdCut in zip(channels, StdCuts):
    data_noise, _ = ds.create_data_sets(channel, {"data_name": ["000813_20230628T161508"],
                                                  "pos_name": ["000813_20230628T161508"],
                                                  "specific_subdir": [""],
                                                  "full_info": [False]},
                                        win_length = 10240, shift = 0, pos_prefix = "",
                                        len_data = 26200, cached = False)
    dataset = DataLoader(data_noise, batch_size = 1024, shuffle = False, num_workers = 0)
    selected = None
    N_data = 16000
    for pulse in dataset:
        s = torch.std(pulse, dim = 1)
        if selected is None:
            selected = s < StdCut
        else:
            selected = torch.cat((selected, s < StdCut), dim = 0)
        if selected.sum() >= N_data:
            print("break")
            break
    arg_selected = torch.argwhere(selected)[:, 0]
    dataset_pileup_name = "pup_n1-d8_000813_20230628T161508"
    file_path_pileup, path_pos_pileup = ds.find_files(meas_name, dataset_pileup_name, channel, pos_prefix = "",
                                                      specific_subdir = "RUN9_pulse_injected_new/")
    pos_file = np.loadtxt(path_pos_pileup)
    pos_file[:, 0] = noise_pos_file[arg_selected[:len(pos_file)]].T
    np.savetxt(path_pos_pileup[:-4] + f"_stdcut.pos", pos_file, fmt = "%d %.8f %.8f %.5f %.5f %.7f %.5f")
    dataset_single_name = "pup_n1-d0_000813_20230628T161508"
    file_path_single, path_pos_single = ds.find_files(meas_name, dataset_single_name, channel, pos_prefix = "",
                                                      specific_subdir = "RUN9_pulse_injected_new/")
    pos_file = np.loadtxt(path_pos_single)
    pos_file[:, 0] = noise_pos_file[arg_selected[:len(pos_file)]].T
    np.savetxt(path_pos_single[:-4] + f"_stdcut.pos", pos_file, fmt = "%d %.8f %.7f")
