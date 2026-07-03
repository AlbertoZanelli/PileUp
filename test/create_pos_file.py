import numpy as np
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE))
import src.dataset as ds
signal_amp = 11.182/1000.
window_size = 4096
std_cut= 0.00023
sampling_rate = 2000
rawdata_path = f"/local/home/mp274748/Documents/data/RUN14/000204_20260405T115934_071_000.bin"
position_file ='/media/mp274748/Transcend1/data/LSC/RUN9_pulse_injected_new/channel5/pup_n1-d0_000813_20230628T161508_005_000_stdcut.pos'
pos_single = np.loadtxt(position_file, ndmin = 2)
dataset = ds.CachedBinaryDataset(rawdata_path, window_size,np.arange(0,24000)*window_size)
std= dataset.data.std(dim=1)
position = dataset.positions[std<std_cut]
pos_single[:,0] = position[:len(pos_single)]*4
mean_amp = np.mean(pos_single[:,1])
pos_single[:,1] = pos_single[:,1]/mean_amp*signal_amp
pos_single[:, -1] = pos_single[:, -1]
print(np.mean(pos_single[:,1]))
pos_pileup = np.loadtxt('/media/mp274748/Transcend1/data/LSC/RUN9_pulse_injected_new/channel5/pup_n1-d8_000813_20230628T161508_005_000_stdcut.pos')
pos_pileup[:,0] = position[:len(pos_pileup)]*4
pos_pileup[:,1] = pos_pileup[:,1]/mean_amp*signal_amp
pos_pileup[:, 2] = pos_pileup[:,2]/mean_amp*signal_amp
pos_pileup[:, -2] = pos_pileup[:, -2]
pos_pileup[:, -1] = pos_pileup[:, -1]
np.savetxt("/local/home/mp274748/Documents/data/RUN14/000204_20260405T115934_071_000.bin_pos_single.pos", pos_single)
np.savetxt("/local/home/mp274748/Documents/data/RUN14/000204_20260405T115934_071_000.bin_pos_pileup.pos", pos_pileup)