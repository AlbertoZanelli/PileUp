#!/usr/bin/env python3
"""
risetime_and_amplitude.py
-----------------------
Analyzes the channel signals.

Per-pulse observables: Risetime (10->90%), Decaytime (90->10%) and AP HF-power
(power spectrum integrated above HF_CUT_HZ, a proxy for residual noise in the
template — the metric adopted in the m204 study). Freq-sigma has been retired.

Available modes (ANALYSIS_MODE):
  - "LOAD_CURVE"          : Reads the load-curve ROOT files and plots
                            Risetime/Decaytime/Amplitude/HF-power vs V_bias.
                            CSV saved in LOAD_CURVE_RESULTS_DIR.
  - "AVERAGE_PULSES_BIN"  : Reads the average-pulse .bin files and plots
                            Risetime/Decaytime/HF-power vs Channel + overlaid waveforms.
  - "AVERAGE_PULSES_ROOT" : Reads SINGLE-RUN (no load curve) Octopus ROOT files,
                            one per channel, each holding the 'averagepulse_ap_medianAP'
                            histogram (same files handled by analyze_BI_singlerun.py).
                            Plots Risetime/Decaytime/HF-power vs Channel + overlaid
                            waveforms, exactly like AVERAGE_PULSES_BIN but from ROOT.
"""

import uproot
import numpy as np
import os, glob, sys, re
from array import array
import csv

try:
    import ROOT
    ROOT.gErrorIgnoreLevel = ROOT.kWarning   # suppress ROOT info messages
except ImportError:
    sys.exit("PyROOT is required to use interactive ROOT plots.")


# ── User settings ──────────────────────────────────────────────────────────────
ANALYSIS_MODE = "LOAD_CURVE"   # "LOAD_CURVE" | "AVERAGE_PULSES_BIN" | "AVERAGE_PULSES_ROOT"
BASE          = os.path.dirname(os.path.abspath(__file__))

# ── Input/output locations (folders are relative to this script) ───────────────
#   LOAD_CURVE — reads ROOT files
LOAD_CURVE_DIR     = "Processed"                      # folder containing the .root files
LOAD_CURVE_PATTERN = "Processed_*_000205_*.root"      # glob pattern of the files to read
LOAD_CURVE_RESULTS_DIR    = "m205_results_octopus"                 # folder to save the results
AMP_CSV_NAME       = "amplitudes_m205.csv"            # output CSV (amplitudes)

#   AVERAGE_PULSES_BIN — reads average-pulse .bin files
BIN_DIR     = "NuoveAnalisiArgonauts_m204"                              # folder containing the .bin files
BIN_PATTERN = "000204_20260405T115934_*_000.bin_edmean.bin"      # glob pattern of the files to read
RT_CSV_NAME = "risetime_m204_argonauts.csv"                        # output CSV (risetimes)
# Regex (with ONE capture group = channel number) to pull the channel from the
# filename. Argonauts m204: '..._031_000.bin_edmean.bin' -> 31.
# For the old m202 files use instead:  r"m202_ch(\d+)_combined".
BIN_CH_REGEX = r"_(\d+)_000\.bin_edmean\.bin$"

# ── AVERAGE_PULSES_ROOT settings ─────────────────────────────────────────────────
#   Single-run Octopus ROOT files (one per channel). Unlike the .bin files these
#   carry a real time axis (hist.axis().centers()), so no sampling period is needed.
AP_ROOT_DIR      = "Processed/m204_AP"                 # folder with the .root files
AP_ROOT_PATTERN  = "Processed_*_000204_*_new.root"     # glob pattern of the files to read
AP_ROOT_HIST     = "averagepulse_ap_medianAP"          # histogram name inside each file
AP_ROOT_CSV_NAME = "risetime_m204_octopus.csv"         # output CSV (risetimes)

# ── LOAD_CURVE settings ──────────────────────────────────────────────────────────
TARGET_CH    = "71"   # channel whose average pulse is drawn (single-pulse canvas)
TARGET_VBIAS = 8      # V_bias at which the target pulse is drawn

APPLY_AMP_SCALE = True    # if True, multiply amplitudes by the per-channel factor
AMP_SCALE_FACTORS = {     # {channel_number: factor} — fill in by hand
    31: 0.004785946463,
    34: 0.008110341755,
    37: 0.005293546845,
    40: 0.01075334608,
    41: 0.008774844546,
    71: 0.01033197024,
    83: 0.01389222099,
    91: 0.002320458891,
    94: 0.008851234945,
}
AMP_SCALE_DEFAULT = 1.0    # factor used for channels not present in the dictionary

# ── V_bias look-up table (indexed by WP // 2) ─────────────────────────────────
VBIAS_LIST = np.array([0.6, 1.0, 1.4, 1.8, 2, 3, 4, 5, 6, 8, 10, 20, 26, 30, 40])

SHOW_PLOTS_LOAD_CURVE = False   # enable/disable interactive plots in LOAD_CURVE

# ── AVERAGE_PULSES_BIN settings ──────────────────────────────────────────────────
# The .bin files have no time axis: set here the acquisition sampling rate of the RUN.
# It MUST match the run so that risetime (ms) and freq_sigma (Hz) are comparable to the
# Octopus ROOT results. m204 is sampled at 2000 Hz (window 800 -> 0.4 s, dt = 0.5 ms).
SAMPLING_FREQUENCY_HZ = 2000
SAMPLING_PERIOD_S = 1 / SAMPLING_FREQUENCY_HZ

SHOW_PLOTS_BIN        = False  # enable/disable interactive plots in AVERAGE_PULSES_BIN



SHOW_PLOTS_AP_ROOT    = False  # enable/disable interactive plots in AVERAGE_PULSES_ROOT

# ── HF-power settings ─────────────────────────────────────────────────────────
# Cutoff (Hz) above which the average pulse (low-pass) has no signal left, so any
# residual power is noise leaked into the template. Integrating the AP power
# spectrum above it gives one number for that noise (same metric used on m204).
HF_CUT_HZ = 500.0


def wp_to_vbias(wp_idx: int) -> float:
    return VBIAS_LIST[wp_idx // 2]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – WAVEFORM ANALYSIS (TIME & FREQUENCY DOMAIN)
# ══════════════════════════════════════════════════════════════════════════════

def risetime_cubic(pulse: np.ndarray, t_s: np.ndarray,
                   low: float = 0.10, high: float = 0.90) -> float:
    """Risetime 10%->90% on the rising edge with cubic interpolation."""
    peak_idx = int(np.argmax(np.abs(pulse)))
    t_seg = np.asarray(t_s[:peak_idx + 1], dtype=float)
    p_seg = np.asarray(pulse[:peak_idx + 1], dtype=float)
    if len(p_seg) < 2:
        return np.nan

    peak_val = p_seg[-1]
    if peak_val == 0:
        return np.nan

    p_work   = p_seg * np.sign(peak_val)
    peak_abs = p_work[-1]

    m = np.zeros_like(p_work, dtype=float)
    m[1:-1] = (p_work[2:] - p_work[:-2]) / (t_seg[2:] - t_seg[:-2])
    m[0]    = (p_work[1]  - p_work[0])   / (t_seg[1]  - t_seg[0])
    m[-1]   = (p_work[-1] - p_work[-2])  / (t_seg[-1] - t_seg[-2])

    def _cross(thr):
        idxs = np.where(np.diff((p_work >= thr).astype(int)) == 1)[0]
        if len(idxs) == 0: return np.nan
        i  = int(idxs[0])
        dx = float(t_seg[i + 1] - t_seg[i])
        if dx == 0.0: return np.nan
        y0, y1 = float(p_work[i]), float(p_work[i + 1])
        m0, m1 = float(m[i]) * dx, float(m[i + 1]) * dx
        a =  2.0 * (y0 - y1) + m0 + m1
        b = -3.0 * (y0 - y1) - 2.0 * m0 - m1
        c =  m0; d =  y0 - thr
        roots = np.roots([a, b, c, d])
        real_roots = roots[np.abs(roots.imag) < 1e-8].real
        valid = real_roots[(real_roots >= -1e-9) & (real_roots <= 1.0 + 1e-9)]
        if len(valid) == 0:
            return t_seg[i] + (thr - y0) * dx / (y1 - y0) if y1 != y0 else t_seg[i]
        return t_seg[i] + float(np.clip(np.min(valid), 0.0, 1.0)) * dx

    t_10 = _cross(low  * peak_abs)
    t_90 = _cross(high * peak_abs)

    if np.isnan(t_10) or np.isnan(t_90) or t_90 <= t_10: return np.nan
    return (t_90 - t_10) * 1e3


def decaytime_cubic(pulse: np.ndarray, t_s: np.ndarray,
                    high: float = 0.90, low: float = 0.10) -> float:
    """Decay time 90%->10% on the FALLING edge with cubic interpolation.

    Mirror of :func:`risetime_cubic`, but works on the segment after the peak and
    detects downward threshold crossings (from >=thr to <thr)."""
    peak_idx = int(np.argmax(np.abs(pulse)))
    t_seg = np.asarray(t_s[peak_idx:], dtype=float)
    p_seg = np.asarray(pulse[peak_idx:], dtype=float)
    if len(p_seg) < 2:
        return np.nan

    peak_val = p_seg[0]
    if peak_val == 0:
        return np.nan

    p_work   = p_seg * np.sign(peak_val)
    peak_abs = p_work[0]

    m = np.zeros_like(p_work, dtype=float)
    m[1:-1] = (p_work[2:] - p_work[:-2]) / (t_seg[2:] - t_seg[:-2])
    m[0]    = (p_work[1]  - p_work[0])   / (t_seg[1]  - t_seg[0])
    m[-1]   = (p_work[-1] - p_work[-2])  / (t_seg[-1] - t_seg[-2])

    def _cross(thr):
        idxs = np.where(np.diff((p_work >= thr).astype(int)) == -1)[0]
        if len(idxs) == 0: return np.nan
        i  = int(idxs[0])
        dx = float(t_seg[i + 1] - t_seg[i])
        if dx == 0.0: return np.nan
        y0, y1 = float(p_work[i]), float(p_work[i + 1])
        m0, m1 = float(m[i]) * dx, float(m[i + 1]) * dx
        a =  2.0 * (y0 - y1) + m0 + m1
        b = -3.0 * (y0 - y1) - 2.0 * m0 - m1
        c =  m0; d =  y0 - thr
        roots = np.roots([a, b, c, d])
        real_roots = roots[np.abs(roots.imag) < 1e-8].real
        valid = real_roots[(real_roots >= -1e-9) & (real_roots <= 1.0 + 1e-9)]
        if len(valid) == 0:
            return t_seg[i] + (thr - y0) * dx / (y1 - y0) if y1 != y0 else t_seg[i]
        return t_seg[i] + float(np.clip(np.min(valid), 0.0, 1.0)) * dx

    t_90 = _cross(high * peak_abs)
    t_10 = _cross(low  * peak_abs)

    if np.isnan(t_90) or np.isnan(t_10) or t_10 <= t_90: return np.nan
    return (t_10 - t_90) * 1e3


def hf_power(pulse: np.ndarray, t_s: np.ndarray, f_cut: float = HF_CUT_HZ) -> float:
    """
    High-frequency power of the (peak-normalized) average pulse: the AP power
    spectrum integrated above ``f_cut``. There the low-pass pulse has no signal,
    so this is a single-number proxy for the residual noise in the template
    (same metric adopted in the m204 Octopus-vs-Argonauts study).
    """
    pulse = np.asarray(pulse, dtype=float)
    if len(pulse) < 2: return np.nan
    dt = float(t_s[1] - t_s[0])
    if dt <= 0: return np.nan
    peak = np.max(np.abs(pulse))
    if peak <= 0: return np.nan

    sig = pulse / peak
    sig = sig - np.mean(sig)
    win = np.hanning(len(sig))
    xw = sig * win
    fft_vals = np.fft.rfft(xw)
    sr = 1.0 / dt
    psd = (np.abs(fft_vals) ** 2) / (sr * len(sig))
    freq = np.fft.rfftfreq(len(sig), d=dt)
    return float(np.sum(psd[freq >= f_cut]))


def export_amplitudes_csv(out_path):
    """Write LOAD_CURVE results to CSV (Risetime, Decaytime, Amp, HF-power)."""
    def scale_for(ch_int):
        if not APPLY_AMP_SCALE: return 1.0
        if ch_int not in AMP_SCALE_FACTORS:
            print(f"  ⚠ Channel {ch_int}: no factor defined, using {AMP_SCALE_DEFAULT}")
        return AMP_SCALE_FACTORS.get(ch_int, AMP_SCALE_DEFAULT)

    channels = sorted(set(plot_data_risetime) | set(plot_data_decaytime) |
                      set(plot_data_amplitude) | set(plot_data_hfpower), key=int)

    rows = []
    for ch in channels:
        ch_int = int(ch)
        k = scale_for(ch_int)

        rt_map  = dict(zip(plot_data_risetime.get(ch, {"x":[]})["x"],  plot_data_risetime.get(ch, {"y":[]})["y"]))
        dec_map = dict(zip(plot_data_decaytime.get(ch, {"x":[]})["x"], plot_data_decaytime.get(ch, {"y":[]})["y"]))
        amp_map = dict(zip(plot_data_amplitude.get(ch, {"x":[]})["x"], plot_data_amplitude.get(ch, {"y":[]})["y"]))
        hf_map  = dict(zip(plot_data_hfpower.get(ch, {"x":[]})["x"],   plot_data_hfpower.get(ch, {"y":[]})["y"]))

        for vb in sorted(set(rt_map) | set(dec_map) | set(amp_map) | set(hf_map)):
            rt  = rt_map.get(vb)
            dec = dec_map.get(vb)
            amp = amp_map.get(vb)
            hf  = hf_map.get(vb)
            rows.append((
                ch_int, float(vb), rt, dec,
                amp * k * 1000 if amp is not None else None,
                hf
            ))

    rows.sort(key=lambda r: (r[0], r[1]))
    with open(out_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["channel", "vbias_V", "risetime_ms", "decaytime_ms", "amplitude_mV", "hf_power"])
        for ch, vb, rt, dec, amp, hf in rows:
            writer.writerow([
                ch, f"{vb:.3f}",
                f"{rt:.6f}"  if rt  is not None else "",
                f"{dec:.6f}" if dec is not None else "",
                f"{amp:.6f}" if amp is not None else "",
                f"{hf:.6e}"  if hf  is not None else "",
            ])

    print(f"✓ Data (incl. decaytime & HF-power) saved to: {out_path}")


def export_risetime_csv(out_path):
    """Write AVERAGE_PULSES results to CSV (Risetime, Decaytime, HF-power)."""
    with open(out_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["channel", "risetime_ms", "decaytime_ms", "hf_power"])
        for ch in sorted(set(plot_data_risetime) | set(plot_data_decaytime) | set(plot_data_hfpower), key=int):
            rt  = plot_data_risetime.get(ch, np.nan)
            dec = plot_data_decaytime.get(ch, np.nan)
            hf  = plot_data_hfpower.get(ch, np.nan)

            rt_str  = f"{rt:.6f}"  if not np.isnan(rt)  else ""
            dec_str = f"{dec:.6f}" if not np.isnan(dec) else ""
            hf_str  = f"{hf:.6e}"  if not np.isnan(hf)  else ""

            writer.writerow([int(ch), rt_str, dec_str, hf_str])
    print(f"✓ Data (incl. decaytime & HF-power) saved to: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

plot_data_risetime   = {}
plot_data_decaytime  = {}
plot_data_amplitude  = {}
plot_data_hfpower    = {}
plot_data_pulses     = {}

target_pulse_t = None
target_pulse_v = None

print(f"Active mode: {ANALYSIS_MODE}\n")

if ANALYSIS_MODE == "LOAD_CURVE":
    data_dir = os.path.join(BASE, LOAD_CURVE_DIR)
    FILES    = sorted(glob.glob(os.path.join(data_dir, LOAD_CURVE_PATTERN)))

    if not FILES: sys.exit(f"No ROOT file found in {data_dir} with pattern {LOAD_CURVE_PATTERN}")

    print(f"Analyzing {len(FILES)} channels (positive polarity only)...\n")

    for filepath in FILES:
        channel = os.path.basename(filepath).split("_")[-1].replace(".root", "")
        vbias_rt  = []; rt_arr  = []
        vbias_dec = []; dec_arr = []
        vbias_amp = []; amp_arr = []
        vbias_hf  = []; hf_arr  = []

        with uproot.open(filepath) as f:
            wp_indices = sorted(set(
                int(m.group(1)) for k in f.keys()
                for m in [re.search(r'averagepulse_ap_wp(\d+)_medianAP', k)]
                if m and (int(m.group(1)) % 2 != 0)
            ))

            for wp in wp_indices:
                vb = wp_to_vbias(wp)

                # Waveform (Risetime, Decaytime & HF-power)
                try:
                    hist  = f[f"averagepulse_ap_wp{wp}_medianAP"]
                    pulse = np.asarray(hist.values(), dtype=float)
                    t_s   = np.asarray(hist.axis().centers(), dtype=float)

                    rt_ms  = risetime_cubic(pulse, t_s)
                    dec_ms = decaytime_cubic(pulse, t_s)
                    hf_val = hf_power(pulse, t_s)

                    if not np.isnan(rt_ms):
                        vbias_rt.append(vb); rt_arr.append(rt_ms)
                    if not np.isnan(dec_ms):
                        vbias_dec.append(vb); dec_arr.append(dec_ms)
                    if not np.isnan(hf_val):
                        vbias_hf.append(vb); hf_arr.append(hf_val)

                    if channel == TARGET_CH and abs(vb - TARGET_VBIAS) < 0.1:
                        target_pulse_t = t_s; target_pulse_v = pulse
                except (uproot.exceptions.KeyInFileError, Exception): pass

                # Amplitude
                try:
                    tree       = f[f"optimumfilter__wp{wp}"]
                    amplitudes = tree["amplitude"].array(library="np")
                    good_flags = tree["good"].array(library="np")
                    amp_good   = amplitudes[good_flags == 1]

                    if len(amp_good) > 0:
                        vbias_amp.append(vb); amp_arr.append(float(np.median(amp_good)))
                except (uproot.exceptions.KeyInFileError, Exception): pass

        plot_data_risetime[channel]   = {"x": vbias_rt,  "y": rt_arr}
        plot_data_decaytime[channel]  = {"x": vbias_dec, "y": dec_arr}
        plot_data_amplitude[channel]  = {"x": vbias_amp, "y": amp_arr}
        plot_data_hfpower[channel]    = {"x": vbias_hf,  "y": hf_arr}
        print(f"✓ Channel {channel} | {len(rt_arr)} RT, {len(dec_arr)} DT, {len(amp_arr)} Amp, {len(hf_arr)} HF")

elif ANALYSIS_MODE == "AVERAGE_PULSES_BIN":
    bin_dir  = os.path.join(BASE, BIN_DIR)
    bin_glob = os.path.join(bin_dir, BIN_PATTERN)
    FILES    = sorted(glob.glob(bin_glob))

    if not FILES: sys.exit(f"No .bin file found in {bin_dir} with pattern {BIN_PATTERN}")

    print(f"Analyzing {len(FILES)} Average Pulse binary files...\n")

    for filepath in FILES:
        match = re.search(BIN_CH_REGEX, os.path.basename(filepath))
        if not match:
            print(f"  [warn] cannot parse channel from {os.path.basename(filepath)} - skipped")
            continue
        channel = match.group(1); ch_int = int(channel)

        meanpulse = np.fromfile(filepath)
        t_s = np.arange(len(meanpulse)) * SAMPLING_PERIOD_S

        rt_ms  = risetime_cubic(meanpulse, t_s)
        dec_ms = decaytime_cubic(meanpulse, t_s)
        hf_val = hf_power(meanpulse, t_s)

        plot_data_pulses[ch_int] = (t_s, meanpulse)
        if not np.isnan(rt_ms):  plot_data_risetime[ch_int]  = rt_ms
        if not np.isnan(dec_ms): plot_data_decaytime[ch_int] = dec_ms
        if not np.isnan(hf_val): plot_data_hfpower[ch_int]   = hf_val

        rt_str  = f"{rt_ms:.4f} ms" if not np.isnan(rt_ms) else "N/A"
        dec_str = f"{dec_ms:.4f} ms" if not np.isnan(dec_ms) else "N/A"
        hf_str  = f"{hf_val:.3e}" if not np.isnan(hf_val) else "N/A"
        print(f"✓ Channel {channel:<3} | RT: {rt_str:>10} | DT: {dec_str:>10} | HF: {hf_str:>12}")

elif ANALYSIS_MODE == "AVERAGE_PULSES_ROOT":
    root_dir  = os.path.join(BASE, AP_ROOT_DIR)
    root_glob = os.path.join(root_dir, AP_ROOT_PATTERN)
    FILES     = sorted(glob.glob(root_glob))

    if not FILES: sys.exit(f"No ROOT file found in {root_dir} with pattern {AP_ROOT_PATTERN}")

    print(f"Analyzing {len(FILES)} single-run Average Pulse ROOT files...\n")

    for filepath in FILES:
        # Channel = the number just before the trailing '_new.root'
        match = re.search(r'_(\d+)_new\.root$', os.path.basename(filepath))
        if not match: continue
        channel = match.group(1); ch_int = int(channel)

        try:
            with uproot.open(filepath) as f:
                hist      = f[AP_ROOT_HIST]
                meanpulse = np.asarray(hist.values(), dtype=float)
                t_s       = np.asarray(hist.axis().centers(), dtype=float)  # real time axis
        except (uproot.exceptions.KeyInFileError, Exception) as e:
            print(f"  [warn] Channel {channel}: cannot read '{AP_ROOT_HIST}' ({e}) - skipped")
            continue

        rt_ms  = risetime_cubic(meanpulse, t_s)
        dec_ms = decaytime_cubic(meanpulse, t_s)
        hf_val = hf_power(meanpulse, t_s)

        plot_data_pulses[ch_int] = (t_s, meanpulse)
        if not np.isnan(rt_ms):  plot_data_risetime[ch_int]  = rt_ms
        if not np.isnan(dec_ms): plot_data_decaytime[ch_int] = dec_ms
        if not np.isnan(hf_val): plot_data_hfpower[ch_int]   = hf_val

        rt_str  = f"{rt_ms:.4f} ms" if not np.isnan(rt_ms) else "N/A"
        dec_str = f"{dec_ms:.4f} ms" if not np.isnan(dec_ms) else "N/A"
        hf_str  = f"{hf_val:.3e}" if not np.isnan(hf_val) else "N/A"
        print(f"[ok] Channel {channel:<3} | RT: {rt_str:>10} | DT: {dec_str:>10} | HF: {hf_str:>12}")

print("-" * 65)
if ANALYSIS_MODE == "LOAD_CURVE":
    results_dir = os.path.join(BASE, LOAD_CURVE_RESULTS_DIR)
    os.makedirs(results_dir, exist_ok=True)
    export_amplitudes_csv(os.path.join(results_dir, AMP_CSV_NAME))
elif ANALYSIS_MODE == "AVERAGE_PULSES_BIN":
    export_risetime_csv(os.path.join(BASE + "/" + BIN_DIR, RT_CSV_NAME))
elif ANALYSIS_MODE == "AVERAGE_PULSES_ROOT":
    export_risetime_csv(os.path.join(BASE, AP_ROOT_DIR, AP_ROOT_CSV_NAME))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – INTERACTIVE ROOT PLOTS
# ══════════════════════════════════════════════════════════════════════════════

# Resolve the show-plots flag for the active mode (both pulse modes share plots).
SHOW_PLOTS = {
    "LOAD_CURVE":          SHOW_PLOTS_LOAD_CURVE,
    "AVERAGE_PULSES_BIN":  SHOW_PLOTS_BIN,
    "AVERAGE_PULSES_ROOT": SHOW_PLOTS_AP_ROOT,
}.get(ANALYSIS_MODE, False)

app = ROOT.gROOT.GetApplication()
if not app: app = ROOT.TApplication("app", sys.argv, sys.argv)

graphs, canvases, keepalive = [], [], []

COLORS  = [ROOT.kRed+1, ROOT.kBlue+1, ROOT.kGreen+2, ROOT.kMagenta+1, ROOT.kCyan+1, ROOT.kOrange+1, ROOT.kBlack, ROOT.kYellow+2, ROOT.kPink+2]
MARKERS = [20, 21, 22, 23, 29, 33, 34, 43, 45]

def _make_multigraph_canvas(canvas_name, canvas_title, data_dict, y_title, x_min=7, x_max=45):
    canvas = ROOT.TCanvas(canvas_name, canvas_title, 1000, 700)
    canvas.SetGrid()
    mg = ROOT.TMultiGraph()
    mg.SetTitle(f"{canvas_title};V_{{bias}} (V);{y_title}")
    legend = ROOT.TLegend(0.75, 0.65, 0.88, 0.88)

    local_graphs = []; local_y_max = 0.0
    for idx, (ch, data) in enumerate(data_dict.items()):
        if len(data["x"]) == 0: continue
        local_y_max = max(local_y_max, max([y for x, y in zip(data["x"], data["y"]) if x_min <= x <= x_max] + [0]))
        gr = ROOT.TGraph(len(data["x"]), array('d', data["x"]), array('d', data["y"]))
        gr.SetTitle(f"Ch {ch}")
        gr.SetMarkerStyle(MARKERS[idx % len(MARKERS)])
        gr.SetMarkerSize(1.5)
        gr.SetMarkerColor(COLORS[idx % len(COLORS)]); gr.SetLineColor(COLORS[idx % len(COLORS)])
        mg.Add(gr, "PL")
        legend.AddEntry(gr, f"Channel {ch}", "p")
        local_graphs.append(gr)

    canvas.cd()
    mg.Draw("AP")
    legend.Draw()
    mg.GetYaxis().SetRangeUser(0.0, local_y_max * 1.1 if local_y_max > 0 else 1.0)
    mg.GetXaxis().SetLimits(x_min, x_max)
    canvas.Update()
    return canvas, mg, legend, local_graphs

def _make_channel_graph(canvas_name, canvas_title, data_dict, y_title):
    canvas = ROOT.TCanvas(canvas_name, canvas_title, 800, 600)
    canvas.SetGrid()

    sorted_chs = sorted(data_dict.keys())
    x_arr = array('d', sorted_chs)
    y_arr = array('d', [data_dict[ch] for ch in sorted_chs])

    gr = ROOT.TGraph(len(x_arr), x_arr, y_arr)
    gr.SetTitle(f"{canvas_title};Channel Number;{y_title}")
    gr.SetMarkerStyle(20); gr.SetMarkerSize(1.5)
    gr.SetMarkerColor(ROOT.kAzure+1); gr.SetLineColor(ROOT.kAzure+1); gr.SetLineWidth(2)

    canvas.cd(); gr.Draw("APL")
    if len(x_arr) > 1: gr.GetXaxis().SetLimits(min(x_arr)-1, max(x_arr)+1)
    gr.GetYaxis().SetRangeUser(0.0, max(y_arr) * 1.1 if len(y_arr)>0 else 1.0)
    canvas.Update()
    return canvas, gr

if ANALYSIS_MODE == "LOAD_CURVE" and SHOW_PLOTS:
    c_rt, mg_rt, leg_rt, gr_rt = _make_multigraph_canvas("c_risetime", "Risetime vs Bias Voltage", plot_data_risetime, "Risetime 10%#rightarrow90% (ms)")
    c_dec, mg_dec, leg_dec, gr_dec = _make_multigraph_canvas("c_decaytime", "Decaytime vs Bias Voltage", plot_data_decaytime, "Decaytime 90%#rightarrow10% (ms)")
    c_amp, mg_amp, leg_amp, gr_amp = _make_multigraph_canvas("c_amplitude", "Median Amplitude vs Bias Voltage", plot_data_amplitude, "Median Amplitude (V)", x_min=0)
    c_hf, mg_hf, leg_hf, gr_hf = _make_multigraph_canvas("c_hfpower", "AP HF-power vs Bias Voltage", plot_data_hfpower, "HF-power ( >500 Hz )", x_min=0)

    canvases.extend([c_rt, c_dec, c_amp, c_hf]); graphs.extend(gr_rt + gr_dec + gr_amp + gr_hf)
    keepalive.extend([mg_rt, leg_rt, mg_dec, leg_dec, mg_amp, leg_amp, mg_hf, leg_hf])

    if target_pulse_t is not None:
        c_p = ROOT.TCanvas("c_pulse", f"Pulse Ch {TARGET_CH} @ {TARGET_VBIAS}V", 800, 600)
        c_p.SetGrid()
        gr_p = ROOT.TGraph(len(target_pulse_t), array('d', target_pulse_t), array('d', target_pulse_v))
        gr_p.SetTitle(f"Average Pulse - Ch {TARGET_CH} @ {TARGET_VBIAS}V;Time (s);Amplitude")
        gr_p.SetLineColor(ROOT.kAzure + 1)
        c_p.cd(); gr_p.Draw("AL")
        canvases.append(c_p); graphs.append(gr_p); c_p.Update()

elif ANALYSIS_MODE in ("AVERAGE_PULSES_BIN", "AVERAGE_PULSES_ROOT") and SHOW_PLOTS:
    c_rt, gr_rt   = _make_channel_graph("c_rt_single", "Risetime vs Channel", plot_data_risetime, "Risetime (ms)")
    c_dec, gr_dec = _make_channel_graph("c_dec_single", "Decaytime vs Channel", plot_data_decaytime, "Decaytime (ms)")
    c_hf, gr_hf   = _make_channel_graph("c_hf_single", "AP HF-power vs Channel", plot_data_hfpower, "HF-power ( >500 Hz )")
    canvases.extend([c_rt, c_dec, c_hf]); graphs.extend([gr_rt, gr_dec, gr_hf])

    if plot_data_pulses:
        _src = ".bin" if ANALYSIS_MODE == "AVERAGE_PULSES_BIN" else "ROOT"
        c_p = ROOT.TCanvas("c_pulses_all", f"All average pulses ({_src})", 1000, 700)
        c_p.SetGrid()
        mg_p = ROOT.TMultiGraph()
        mg_p.SetTitle("Average Pulses;Time (s);Amplitude")
        leg_p = ROOT.TLegend(0.75, 0.65, 0.88, 0.88)

        for idx, (ch, (t_s, pulse)) in enumerate(sorted(plot_data_pulses.items(), key=lambda x: int(x[0]))):
            gr = ROOT.TGraph(len(t_s), array('d', t_s), array('d', pulse))
            gr.SetTitle(f"Ch {ch}")
            gr.SetLineColor(COLORS[idx % len(COLORS)]); gr.SetLineWidth(2)
            mg_p.Add(gr, "L"); leg_p.AddEntry(gr, f"Ch {ch}", "l")
            graphs.append(gr)

        c_p.cd(); mg_p.Draw("A"); leg_p.Draw(); c_p.Update()
        canvases.append(c_p); keepalive.extend([mg_p, leg_p])

if SHOW_PLOTS:
    print("\n" + "=" * 65 + "\n  Interactive ROOT plots ready.\n  Close a canvas window or press Ctrl+C to exit.\n" + "=" * 65 + "\n")
    app.Run()
else:
    print("\nPlotting disabled for this mode — done.\n")