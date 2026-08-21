#!/usr/bin/env python3
"""
extract_AP_pulses_m205.py
=========================
Estrae i SINGOLI impulsi che formano l'average pulse (AP) di Octopus, andando a
riprendere le waveform dal file binario RAW a partire dal `triggersample` degli
eventi selezionati (quelli che Octopus ha effettivamente mediato).

Ricetta (ricostruita e verificata su m205 ch91 WP15, vedi sotto):
  1. ROOT: gli eventi usati nell'AP del WP sono quelli con
     `crosscorr_signal_wp<wp>.pass == True` (e' il Select del modulo
     averagepulse nel configModule). Per ognuno servono:
       - module.triggersample  -> posizione ASSOLUTA del trigger nello stream raw
       - triggerdelay.midsample -> posizione del punto di mezza salita nella finestra
  2. BIN: finestra di WINDOW campioni che parte da
        triggersample - PRETRIGGER*WINDOW + (midsample - PRETRIGGER*WINDOW)
     cioe' la finestra gia' corretta per il ritardo di trigger. Si legge direttamente
     spostati invece di fare np.roll sulla finestra: cosi' i campioni di bordo sono
     dati veri e non il wrap-around (che da solo valeva ~1e-3 sull'ultimo 0.1% AP).
  3. Sottrazione della baseline: Octopus la calcola sulla finestra NON corretta (e' il
     modulo `baseline`, che gira prima della triggerdelaycorrection) e poi sottrae quel
     valore. Vale la media dei primi 4900 campioni (49% = pretrigger - 1%) della finestra
     originale; qui si legge direttamente dal ramo `baseline.baseline`. Ricalcolarla sulla
     finestra gia' spostata sembra equivalente ma lascia un residuo di ~3e-6 sull'AP.
  4. (opzionale) normalizzazione al massimo -> la mediana dei 38 impulsi riproduce
     `averagepulse_ap_wp<wp>_medianAP`.

Formato del file binario (rawType = "Cupid"), dedotto dal file di prova:
  - header di 12 byte:  uint32 (ignoto, 27680) | float32 sampling rate (10000.0 Hz)
                        | float32 fondo scala (10.069444 V)
  - poi un campione ogni 4 byte, uint32 little-endian, ADC a 24 bit nei 3 byte alti
    (il byte basso e' sempre 0). Conversione in volt:
        V = (u32/256 / 2**23 - 1) * FULLSCALE       (offset binary, centro 2**31)
  - i file sono segmenti da 1 GiB: <run>_<prefix>_<ch:03d>_<seg:03d>.bin

VERIFICA (ch91 WP15, 38 eventi):
  - maxvalue e baseline degli eventi riprodotti dal ROOT a 5e-15 V e 7e-16 V;
  - la TH2D averagepulse_ap_wp15_APdistro ricostruita ha le STESSE 380000 entries e
    9965/10000 colonne identiche bin per bin (le altre differiscono di 1 conteggio,
    campioni esattamente sul bordo di un bin);
  - la mediana normalizzata coincide con medianAP entro 2e-16, cioe' alla precisione
    macchina: la ricostruzione e' esatta, non approssimata.

Uso:
    KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/bin/python3.13 extract_AP_pulses_m205.py
"""

import os, glob, struct
import numpy as np
import uproot

# ═════════════════════════════════════════════════════════════════════════════
# Config
# ═════════════════════════════════════════════════════════════════════════════
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "Processed")
BIN_DIR    = "/data2/LSC/DATA/RUN14/000205"
OUT_DIR    = os.path.join(BASE_DIR, "m205_AP_pulses")
MEAS_NAME  = "000205"
PREFIX     = "20260406T161631"      # filenamePrefix del run (vedi cfg/configMerger_0.toml)

CHANNELS   = [91]                   # uno o piu' canali: serve il .bin di ciascuno in BIN_DIR
WPS        = list(range(1, 30, 2))  # WP dispari (convenzione del progetto); [15] per uno solo

WINDOW     = 10_000                 # module.module.light windowlength 1.0 s @ 10 kHz
PRETRIGGER = 0.5                    # module.module.light pretrigger
HDR_BYTES  = 12
NORMALIZE  = True                   # ogni impulso diviso per il suo massimo (come Octopus)
PLOT       = True


# ═════════════════════════════════════════════════════════════════════════════
# Raw binary
# ═════════════════════════════════════════════════════════════════════════════
def bin_path(channel, seg=0):
    return os.path.join(BIN_DIR, f"{MEAS_NAME}_{PREFIX}_{channel:03d}_{seg:03d}.bin")


def bin_header(channel, seg=0):
    """(campo ignoto, sampling rate [Hz], fondo scala [V]) dai primi 12 byte."""
    with open(bin_path(channel, seg), "rb") as f:
        return struct.unpack("<Iff", f.read(HDR_BYTES))


def read_samples(channel, start, count, seg_len):
    """`count` campioni RAW (uint32) dallo stream continuo, a partire dal campione
    assoluto `start`, attraversando i segmenti da 1 GiB se serve.
    ponytail: si assume che ogni segmento abbia lo stesso header di 12 byte e la
    stessa lunghezza del segmento 000 (unico file disponibile per il test)."""
    out = []
    while count > 0:
        seg, off = divmod(start, seg_len)
        k = min(count, seg_len - off)
        chunk = np.fromfile(bin_path(channel, seg), dtype="<u4",
                            count=k, offset=HDR_BYTES + 4 * off)
        if len(chunk) < k:
            raise RuntimeError(f"segmento {seg} troppo corto o mancante: {bin_path(channel, seg)}")
        out.append(chunk)
        start += k
        count -= k
    return np.concatenate(out) if len(out) > 1 else out[0]


def to_volt(u32, fullscale):
    """ADC 24 bit in offset binary (centro 2**31) -> volt."""
    return (u32.astype(np.float64) / 256.0 / 2 ** 23 - 1.0) * fullscale


# ═════════════════════════════════════════════════════════════════════════════
# Estrazione
# ═════════════════════════════════════════════════════════════════════════════
def load_ap_pulses(channel, wp, normalize=NORMALIZE):
    """Impulsi (n_eventi, WINDOW) che formano l'AP di (channel, wp).

    Allineati col ritardo di trigger, baseline sottratta, in volt (o normalizzati
    al massimo se `normalize`). La loro mediana e' l'average pulse."""
    root = glob.glob(os.path.join(DATA_DIR, f"Processed_*_{MEAS_NAME}_{channel}.root"))
    if not root:
        raise RuntimeError(f"file ROOT non trovato per il canale {channel}")
    with uproot.open(root[0]) as f:
        sel = f[f"crosscorr_signal_wp{wp}"]["pass"].array(library="np")
        tsample = f["module"]["triggersample"].array(library="np")[sel]
        midsamp = f["triggerdelay"]["midsample"].array(library="np")[sel]
        # baseline calcolata da Octopus sulla finestra NON corretta: e' quella che sottrae
        baseline = f["baseline"]["baseline"].array(library="np")[sel]

    _, fs, fullscale = bin_header(channel)
    seg_len = (os.path.getsize(bin_path(channel)) - HDR_BYTES) // 4
    pre = int(PRETRIGGER * WINDOW)

    pulses = np.empty((len(tsample), WINDOW))
    for i, (ts, mid, b) in enumerate(zip(tsample, midsamp, baseline)):
        start = int(ts) - pre + (int(mid) - pre)   # finestra gia' corretta per il ritardo
        pulses[i] = to_volt(read_samples(channel, start, WINDOW, seg_len), fullscale) - b
    if normalize:
        pulses /= pulses.max(axis=1, keepdims=True)
    return pulses, fs


def main():
    channel_out_dir = os.path.join(OUT_DIR, f"ch{CHANNEL}")
    os.makedirs(channel_out_dir, exist_ok=True)
    root = glob.glob(os.path.join(DATA_DIR, f"Processed_*_{MEAS_NAME}_{CHANNEL}.root"))[0]
    for wp in WPS:
        pulses, fs = load_ap_pulses(CHANNEL, wp)
        out = os.path.join(channel_out_dir, f"pulses_ch{CHANNEL}_wp{wp}.npy")
        np.save(out, pulses)

            # ── controllo: la mediana deve riprodurre l'AP di Octopus ─────────
            with uproot.open(root[0]) as f:
                ap = np.asarray(f[f"averagepulse_ap_wp{wp}_medianAP"].values(), float)
            med = np.median(pulses, axis=0)
            med = med / med.max() if NORMALIZE else med
            print(f"ch{channel} wp{wp:<3d} {len(pulses):3d} impulsi x {pulses.shape[1]} @ {fs:.0f} Hz"
                  f"   max|median - medianAP| = {np.abs(med - ap).max():.1e}   -> {os.path.basename(out)}")

        if PLOT:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            t = np.arange(WINDOW) / fs
            fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
            for p in pulses:
                axes[0].plot(t, p, lw=0.4, alpha=0.5)
            axes[0].plot(t, med, "k", lw=1.5, label="median of the pulses")
            axes[0].set_title(f"m205 Ch{CHANNEL} WP{wp} — {len(pulses)} pulses forming the AP")
            axes[0].legend()
            axes[1].plot(t, med, label="median (from raw .bin)")
            axes[1].plot(t, ap, "--", label="Octopus medianAP")
            axes[1].legend()
            axes[2].plot(t, med - ap, lw=0.8)
            axes[2].set_ylabel("residual")
            axes[2].set_xlabel("time [s]")
            for a in axes[:2]:
                a.set_ylabel("normalized" if NORMALIZE else "V")
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, f"pulses_ch{CHANNEL}_wp{wp}.png"), dpi=130)
            plt.close(fig)
    print(f"\nTutto in {OUT_DIR}")


if __name__ == "__main__":
    main()
