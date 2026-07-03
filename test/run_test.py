import os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
BASE = "../"
sys.path.append(BASE)
import matplotlib.pyplot as plt
import utility.functions
import src.analysis
import src.dataset
import src.plots
import src.simulation
import numpy as np
import src.analysis as an
import src.dataset as ds
import torch
from scipy.stats import norm
from utility.double_beta_spectrum import pdf_ratio2b
import utility.functions as fn

channels = [3, 5, 9, 11, 13, 15, 17, 19]
channel_ID = [5, 9, 11, 10, 4, 2, 12, 3]
StdCuts = [0.0003, 0.0004, 0.0008, 0.0006, 0.0012, 0.0007, 0.00035, 0.0025]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

meas_name = "000813_20230628T161508"
channel = 31
sampling_rate = 10000
acceptance = 0.9
N_sigma = norm.ppf(1 - (1 - acceptance) * 100 / 100) #number of sigma corresponding to the desired acceptance
signal_amp = 1.81e-3

#define the time and ratio grid for the analysis
t_min, t_max, N_t = 0, 8e-4, 100
r_min, r_max, N_r = 0., .5, 100
#get the corresponding ratio distribution 
ratio_distribution = pdf_ratio2b(np.linspace(r_min, r_max, N_r))
ratio_distribution /= np.mean(ratio_distribution)

window_size = 10000 #size of the window used for the analysis, should be the same as the one used for the NPS computation
n_trials=30

#load the meanpulse
meanpulse = np.fromfile(f"{BASE}/m202data/m202_ch{channel}_combined.bin_edmean.bin", dtype=np.float64)
if len(meanpulse) > window_size:
    meanpulse = meanpulse[np.argmax(meanpulse) - window_size//2 : np.argmax(meanpulse) + window_size//2]
#load the NPS
nps = np.fromfile(f"{BASE}/m202data/m202_ch{channel}_combined.bin_spec.bin", dtype=np.float64)
nps *= (8. / 3.)  # Adjust for Hann window effect
nps *= 1e-12

S, w, H_unit = an.compute_H(meanpulse, nps, np.hanning,sampling_rate=sampling_rate)

S_torch = torch.tensor(S, dtype=torch.cfloat, device=device)
H_unit_torch = torch.tensor(H_unit, dtype=torch.cfloat, device=device)
w_torch = torch.tensor(w, dtype=torch.cfloat, device=device)
nps_torch = torch.tensor(nps, dtype=torch.cfloat, device=device)
signal_amp_torch = torch.tensor(signal_amp, dtype=torch.float32, device=device)
t_torch = torch.linspace(t_min, t_max, N_t, dtype=torch.cfloat, device=device)
r_torch = torch.linspace(r_min, r_max, N_r, dtype=torch.cfloat, device=device)
ratio_distribution_torch = torch.tensor(ratio_distribution, dtype=torch.cfloat, device=device)

print("Starting optimization...")
try:
    f1_t, f2_t, J_values = an.optimize_filters(S_torch, H_unit_torch, w_torch, t_torch, r_torch, nps_torch,
                                               signal_amp_torch, ratio_distribution_torch, N_sigma=N_sigma, 
                                               activation_fct = torch.abs,
                                               f1_init = None, f2_init = None,
                                               n_trials=n_trials, use_interp = True, verbose=True)
    print("Optimization successful!")
except Exception as e:
    print("Optimization failed with error:", e)
