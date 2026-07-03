import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE))
import numpy as np
import src.simulation as sim
import src.dataset as ds
meas_name = "000813_20230628T161508"
channel = 5
channel_data = []
order_data = []
params_data = []
channels = [3, 5, 9, 11, 13, 15, 17, 19]
for channel in channels:
    meanpulse = np.fromfile(ds.find_file(f"{meas_name}_{channel:03}_???.bin_edmean.bin",
                                         specific_subdir="RUN9_pulse_injected_new/"))
    t = np.arange(-len(meanpulse) // 2, len(meanpulse) // 2) * 1e-4
    p0 = [t[np.argmax(meanpulse)], -1000, -1339.1, -237.0, -100, -1000.0]
    # fit_fct_pole = lambda x, *params: sim.apply_bessel_to_pulse(x, sim.pulse_pole_zero(x, *params), fc=2500, order=6,
    #                                                             fs=1e4)
    fit_fct_pole = lambda x, *params: sim.make_pulse_pole_zero_bessel_ct(1, 6, 2500, *params)(x)
    popt, pcov = curve_fit(fit_fct_pole, t, meanpulse, p0=p0,
                           bounds=([-0.5, -5000, -5000, -5000, -5000, -5000], [0.5, 0, 0, 0, 0, 0]))
    channel_data.append(channel)
    order_data.append(6)
    params_data.append(popt)
    # fitted_pulse_function = sim.make_pulse_pole_zero_bessel_ct(1, 6, 2500, *popt)
    # plt.plot(t, meanpulse, label="Meanpulse")
    # plt.plot(t, fitted_pulse_function(t), label="Meanpulse")
    fit_fct_pole = lambda x, *params: sim.make_pulse_pole_zero_bessel_ct(1, 0, 2500, *params)(x)
    popt, pcov = curve_fit(fit_fct_pole, t, meanpulse, p0=p0,
                           bounds=([-0.5, -5000, -5000, -5000, -5000, -5000], [0.5, 0, 0, 0, 0, 0]))
    channel_data.append(channel)
    order_data.append(0)
    params_data.append(popt)
    # fitted_pulse_function = sim.make_pulse_pole_zero_bessel_ct(1, 0, 2500, *popt)
    # plt.plot(t, fitted_pulse_function(t), label="Meanpulse")
    # plt.xlim(-0.005,0.005)
    # plt.show()
params_array = np.array(params_data)
df = pd.DataFrame({
    "channel": channel_data,
    "bessel_order": order_data,
    "t0":params_array[:,0],"z":params_array[:,1],"p1":params_array[:,2],"p2":params_array[:,3],"p3":params_array[:,4],"p4":params_array[:,5]
})
df.to_csv(f"{BASE}/outputs/fits/pulse_fit_params.csv", index=False)