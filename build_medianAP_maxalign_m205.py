#!/usr/bin/env python3
"""
build_medianAP_maxalign_m205.py
===============================
Costruisce l'average pulse dei .npy di extract_AP_pulses_m205.py allineando gli
impulsi sul MASSIMO, come fa test/build_meanpulse.py (src.analysis.build_mean_pulse),
ma prendendo la MEDIANA invece della media.

Differenza con l'AP di Octopus (`averagepulse_ap_wp<wp>_medianAP`): Octopus allinea
sul punto di MEZZA SALITA (triggerdelay.midsample -> pretrigger), qui si allinea sul
massimo (argmax -> pulse_center_ratio*window). Il jitter residuo campione-per-campione
e' diverso, quindi il picco puo' risultare leggermente piu' stretto/alto.

Ricetta di build_mean_pulse, passo per passo:
  1. baseline = media dei primi `target + PULSE_START_POS` campioni (= 4900) e sottrazione
     -> gia' fatto in extract_AP_pulses_m205.py, con la stessa finestra di Octopus;
  2. shift = target - argmax, applicato SENZA wrap-around (bordi riempiti di zero,
     cfr. utility.functions.align_waveforms);
  3. normalizzazione di ogni impulso al suo massimo;
  4. media -> qui MEDIANA sugli eventi.
Il taglio in RMS di build_mean_pulse NON viene rifatto: gli impulsi salvati sono gia'
quelli selezionati da Octopus per l'AP, e ritagliarli cambierebbe N (che e' proprio la
quantita' sotto studio per la regolarizzazione R).

Uso:
    KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/bin/python3.13 build_medianAP_maxalign_m205.py
"""

import os, glob
import numpy as np
import uproot

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "Processed")
PULSE_DIR  = os.path.join(BASE_DIR, "m205_AP_pulses")
MEAS_NAME  = "000205"

CHANNEL    = 91
WPS        = list(range(1, 30, 2))
CENTER_RATIO    = 0.5     # build_mean_pulse: target_index = win_length * pulse_center_ratio
PLOT       = True


def shift_pad(p, s):
    """Trasla `p` di `s` campioni (positivo = verso destra) riempiendo di zeri, senza
    wrap-around: equivalente a utility.functions.align_waveforms su una singola traccia
    (qui in numpy, cosi' il programma non ha bisogno di torch)."""
    out = np.zeros_like(p)
    if s >= 0:
        out[s:] = p[:len(p) - s] if s else p
    else:
        out[:s] = p[-s:]
    return out


def align_on_max(pulses, center_ratio=CENTER_RATIO):
    """Impulsi allineati sul massimo (portato a center_ratio*window) e normalizzati al
    massimo: passi 2-3 di build_mean_pulse. Serve anche fuori di qui, per calcolare
    l'errore per time-bin dalla dispersione degli impulsi (test/fit_one_pulse_m205.py)."""
    target = int(pulses.shape[1] * center_ratio)
    aligned = np.array([shift_pad(p, target - int(np.argmax(p))) for p in pulses])
    return aligned / aligned.max(axis=1, keepdims=True)


def median_ap_maxalign(pulses, center_ratio=CENTER_RATIO):
    """Median AP con gli impulsi allineati sul massimo (passi 2-4 di build_mean_pulse)."""
    return np.median(align_on_max(pulses, center_ratio), axis=0)


def main():
    root = glob.glob(os.path.join(DATA_DIR, f"Processed_*_{MEAS_NAME}_{CHANNEL}.root"))[0]
    for wp in WPS:
        pulses = np.load(os.path.join(PULSE_DIR, f"pulses_ch{CHANNEL}_wp{wp}.npy"))
        ap_max = median_ap_maxalign(pulses)
        out = os.path.join(PULSE_DIR, f"medianAP_maxalign_ch{CHANNEL}_wp{wp}.npy")
        np.save(out, ap_max)

        with uproot.open(root) as f:
            ap_oct = np.asarray(f[f"averagepulse_ap_wp{wp}_medianAP"].values(), float)
        shifts = np.array([int(pulses.shape[1] * CENTER_RATIO) - int(np.argmax(p)) for p in pulses])
        print(f"wp{wp:<3d} N={len(pulses):3d}  shift sul massimo: {shifts.min()}..{shifts.max()} campioni"
              f"   picco {ap_max.argmax()} (Octopus {ap_oct.argmax()})"
              f"   -> {os.path.basename(out)}")

        if PLOT:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            n = len(ap_max)
            # nel confronto l'AP di Octopus e' riportata sullo stesso picco: cosi' si
            # vede la differenza di FORMA, non l'offset dovuto al riferimento diverso
            ap_oct_c = shift_pad(ap_oct, ap_max.argmax() - ap_oct.argmax())
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
            for ax, (lo, hi) in zip(axes, [(4900, 5200), (4970, 5030)]):
                x = np.arange(lo, hi)
                ax.plot(x, ap_max[lo:hi], label="median AP, max-aligned")
                ax.plot(x, ap_oct_c[lo:hi], "--", label="Octopus medianAP (peak-matched)")
                ax.set_xlabel("sample")
            axes[0].set_ylabel("normalized")
            axes[0].legend()
            axes[1].set_title(f"peak, zoom  —  max|diff| = {np.abs(ap_max - ap_oct_c).max():.3f}")
            fig.suptitle(f"m205 Ch{CHANNEL} WP{wp} — {len(pulses)} pulses")
            fig.tight_layout()
            fig.savefig(os.path.join(PULSE_DIR, f"medianAP_maxalign_ch{CHANNEL}_wp{wp}.png"), dpi=130)
            plt.close(fig)
    print(f"\nTutto in {PULSE_DIR}")


if __name__ == "__main__":
    # self-check di shift_pad: traslazione senza wrap, in entrambi i versi
    v = np.array([1., 2., 3., 4.])
    assert np.array_equal(shift_pad(v,  1), [0., 1., 2., 3.])
    assert np.array_equal(shift_pad(v, -1), [2., 3., 4., 0.])
    assert np.array_equal(shift_pad(v,  0), v)
    assert np.argmax(median_ap_maxalign(np.array([shift_pad(v, k) for k in (0, 1, 2)]))) == 2
    main()
