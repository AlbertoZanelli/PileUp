import pandas as pd
import numpy as np
import sys
from pathlib import Path
# Add src to Python path dynamically
BASE = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE))
import utility.functions as fn
channels = [3, 5, 9, 11, 13, 15, 17, 19]
channel_ID = [5, 9, 11, 10, 4, 2, 12, 3]
df_w = pd.read_csv("../outputs/Systematics/BI_window_size.csv")
df_w = df_w.loc[df_w['window_size']>=2048]
w = df_w.groupby("channel").apply(lambda x: (x["BI"].max()-x["BI"].min())/2*1e5*0.67)
df_pos = pd.read_csv("../outputs/Systematics/BI_frac_4Pole_Bessel.csv")
pos = df_pos.groupby("channel").apply(lambda x: (x["BI"].max()-x["BI"].min())/2*1e5*0.67)
df_grid = pd.read_csv("../outputs/Systematics/BI_varying_grid_size.csv")
df_grid = df_grid.loc[df_grid['grid_N_t']>=100]
grid = df_grid.groupby("channel").apply(lambda x: (x["J"].max()-x["J"].min())/2*1e5*fn.K*0.67)

df_1 = pd.read_csv("../outputs/Systematics/BI_meanpulse_test.csv")
df_2 = pd.read_csv("../outputs/Systematics/BI_meanpulses_test_2.csv")
df_injection = pd.concat((df_1, df_2))
df_injection = df_injection.loc[df_injection["meanpulse_type"]!="fit"]

meanpulse_type = "mean" # "mean" or "raw"
mp_sys_min = df_injection.groupby("channel").apply(lambda x: (x.loc[x["meanpulse_type"]==meanpulse_type,"BI"]-x["BI"].min())*1e5*0.67)
mp_sys_max = df_injection.groupby("channel").apply(lambda x: np.abs(x.loc[x["meanpulse_type"]==meanpulse_type,"BI"]-x["BI"].max())*1e5*0.67)


df_sys = pd.DataFrame({"channel":w.index,
                       "window_size_sys":w.values,
                       "pole_filter_sys":pos.values,
                       "grid_size_sys":grid.values,
                       "meanpulse_sys_max":mp_sys_max.values,
                       "meanpulse_sys_min":mp_sys_min.values})
df_sys["total_sys_max"] = np.sqrt(df_sys["window_size_sys"]**2+
                               df_sys["pole_filter_sys"]**2+
                               df_sys["grid_size_sys"]**2+
                               df_sys["meanpulse_sys_max"]**2)
df_sys["total_sys_min"] = np.sqrt(df_sys["window_size_sys"]**2+
                               df_sys["pole_filter_sys"]**2+
                               df_sys["grid_size_sys"]**2+
                               df_sys["meanpulse_sys_min"]**2)
df_sys["detector_id"] = df_sys["channel"].apply(lambda x: channel_ID[channels.index(x)])

df_sys = df_sys.sort_values("detector_id")
print(df_sys.loc[:,["detector_id", "total_sys_min", "total_sys_max"]])
