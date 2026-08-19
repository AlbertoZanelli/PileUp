"""
fit_pulses_m205.py
==================
Versione essenziale di test/fit_pulses.py adattata alla run 205: fitta l'AVERAGE PULSE
di UN canale a UN working point con il modello pole-zero + filtro BESSEL del DAQ
(src.simulation.make_pulse_pole_zero_bessel_ct). Il Bessel modella la salita/picco che il
pole-zero puro doveva "falsificare" con tanti poli/zeri -> qui bastano pochi poli.

Dati m205: file ROOT Processed_*_000205_<ch>.root, istogramma averagepulse_ap_wp<wp>_medianAP
(peak-normalizzato ~1, 10 kHz, picco a ~meta' finestra). Il modello lavora in SECONDI e ha
il picco a t=0, quindi l'asse tempi viene centrato sul picco.

Parametri di fit = [t0, zero, p1, p2, p3, p4]  (1 zero + 4 poli reali), come in fit_pulses.py
(nessun amp/baseline: l'AP e' gia' peak-normalizzato). GUESS e BOUND identici a fit_pulses.py.

NB: in fit_pulses.py il modello e' chiamato con gli argomenti nell'ordine SBAGLIATO
(make_pulse_pole_zero_bessel_ct(1, 6, 2500, ...) -> order=1, fcut=6). Qui e' corretto:
make_pulse_pole_zero_bessel_ct(BESSEL_ORDER, FCUT, zero, *poli).

Run:  KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/bin/python3.13 test/fit_pulses_m205.py
"""

import sys
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import uproot

BASE = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE))
import src.simulation as sim

# ═════════════════════════════════════════════════════════════════════════════
# Config
# ═════════════════════════════════════════════════════════════════════════════
CHANNEL      = 91
WP           = 1
BESSEL_ORDER = 6         # ordine del Bessel del DAQ (0 = nessun filtro, pole-zero puro)
FCUT         = 2500      # Hz, taglio del Bessel
MEAS         = "000205"
DATA_DIR     = BASE / "Processed"
HIST_TMPL    = "averagepulse_ap_wp{wp}_medianAP"
OUT_DIR      = BASE / "test" / "m205_bessel_fit"
VBIAS_LIST   = np.array([0.6, 1.0, 1.4, 1.8, 2, 3, 4, 5, 6, 8, 10, 20, 26, 30, 40])

# Guess e bound IDENTICI a fit_pulses.py: params = [t0, zero, p1, p2, p3, p4].
#   p0[0] = istante del picco (dati centrati -> ~0); poi zero e 4 poli (1/s, negativi).
P0_POLES = [-1000.0, -1339.1, -237.0, -100.0, -1000.0]     # [zero, p1, p2, p3, p4]
BOUND_LO = [-0.5, -5000, -5000, -5000, -5000, -5000]
BOUND_HI = [0.5, 0, 0, 0, 0, 0]


def load_ap(channel, wp):
    """AP peak-normalizzato e asse tempi (s) CENTRATO sul picco, dal file ROOT di m205."""
    fp = glob.glob(str(DATA_DIR / f"Processed_*_{MEAS}_{channel}.root"))
    if not fp:
        sys.exit(f"[ERROR] ROOT del canale {channel} non trovato in {DATA_DIR}")
    with uproot.open(fp[0]) as f:
        h = f[HIST_TMPL.format(wp=wp)]
        v = np.asarray(h.values(), dtype=float)
        t = np.asarray(h.axis().centers(), dtype=float)
    v = v / v.max()
    return t - t[int(np.argmax(v))], v          # picco a t=0


def model(t, t0, zero, *poles):
    """Impulso pole-zero + Bessel (picco 1 a t=0), traslato di t0. f_norm dalla libreria."""
    f = sim.make_pulse_pole_zero_bessel_ct(BESSEL_ORDER, FCUT, zero, *poles)
    return f(t - t0)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t, v = load_ap(CHANNEL, WP)
    sigma = float(v[:int(0.40 * len(v))].std()) or 1.0          # RMS del baseline pre-trigger

    p0 = [t[int(np.argmax(v))], *P0_POLES]                       # come fit_pulses.py
    popt, _ = curve_fit(model, t, v, p0=p0, bounds=(BOUND_LO, BOUND_HI), maxfev=20000)

    fit = model(t, *popt)
    rms = float(np.sqrt(np.mean((v - fit) ** 2)))
    t0, zero, poles = popt[0], popt[1], np.sort(popt[2:])
    vbias = VBIAS_LIST[WP // 2]
    print(f"Ch {CHANNEL} · WP {WP} ({vbias:g} V) · Bessel order {BESSEL_ORDER} @ {FCUT} Hz")
    print(f"  t0={t0*1e3:+.3f} ms   zero={zero:.1f} 1/s   poli={np.array2string(poles, precision=1)}")
    print(f"  RMS={rms:.3e}   RMS/sigma={rms/sigma:.2f}")

    # CSV parametri (una riga)
    csv = OUT_DIR / f"fit_params_ch{CHANNEL}_wp{WP}_m205.csv"
    header = "channel,wp,bessel_order,fcut,t0,zero,p1,p2,p3,p4"
    row = [CHANNEL, WP, BESSEL_ORDER, FCUT, t0, zero, *poles]
    np.savetxt(csv, [row], delimiter=",", header=header, comments="",
               fmt=["%d", "%d", "%d", "%g"] + ["%.6e"] * 6)

    # Plot: AP + fit (sopra) e residuo/sigma (sotto), full e zoom sulla salita
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex="col",
                             gridspec_kw={"height_ratios": [3, 1]})
    for col, (lo_ms, hi_ms) in enumerate([(-5, 60), (-1.5, 2.0)]):
        sel = (t * 1e3 >= lo_ms) & (t * 1e3 <= hi_ms)
        tm = t[sel] * 1e3
        axes[0, col].plot(tm, v[sel], ".", ms=3, color="k", label="AP data")
        axes[0, col].plot(tm, fit[sel], "-", lw=1.5, color="#c1121f", label="pole-zero + Bessel fit")
        axes[0, col].grid(True, alpha=0.3)
        axes[1, col].axhspan(-3, 3, color="#4a90d9", alpha=0.2, lw=0)
        axes[1, col].plot(tm, (v[sel] - fit[sel]) / sigma, "-", lw=0.9, color="#c1121f")
        axes[1, col].axhline(0, color="gray", ls=":", lw=0.7)
        axes[1, col].grid(True, alpha=0.3)
        axes[1, col].set_xlabel("t - t_peak [ms]")
        axes[0, col].set_title("full pulse" if col == 0 else "zoom on the rise", fontsize=10)
    axes[0, 0].legend(fontsize=9, loc="upper right")
    axes[0, 0].set_ylabel("AP amplitude (peak-norm.)")
    axes[1, 0].set_ylabel("residual / σ")
    fig.suptitle(f"AP pole-zero + Bessel fit — Ch {CHANNEL} · WP {WP} ({vbias:g} V) · "
                 f"Bessel {BESSEL_ORDER}@{FCUT}Hz · 4 poles · RMS/σ={rms/sigma:.1f}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT_DIR / f"fit_ch{CHANNEL}_wp{WP}_m205.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out}")
    print(f"  -> {csv}")


if __name__ == "__main__":
    main()
