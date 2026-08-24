#!/usr/bin/env python3
"""
build_NPS_clean_m205.py
=======================
NPS "pulita" ricalcolata dalle FINESTRE DI RUMORE VERE del binario, come media del periodogramma
dopo un taglio sulla RMS.

Perche' non basta quella del ROOT:
  - `averagepowerspectrum_noise_wp<wp>_medianpower` e' una MEDIANA sugli eventi. Il simulatore
    vuole la MEDIA. Sul continuo mediana = ln2 x media, sulle righe mediana = media: non c'e' un
    fattore unico. In piu' l'array e' one-sided e viene specchiato, il che raddoppia la potenza.
    Netto misurato su ch91: la nps usata oggi vale 1.84 volte la potenza vera.
  - `..._power` e' gia' la media, ma e' calcolata anche sulle finestre contaminate: la potenza
    totale risulta 8-650 volte quella vera. Inutilizzabile.
Qui si rifa' il conto sulle finestre vere, con lo stesso taglio RMS del paper
(`test/select_noise_traces.py`) e la stessa ricetta di `src/analysis.py: create_NPS_torch`:
RMS cut sulla finestra intera -> sottrai la media -> finestra di Hanning -> |FFT|^2 -> MEDIA
sugli eventi -> x N/sum(w^2). Il risultato e' nella convenzione del simulatore, cioe'
E|FFT(x)|^2, e si passa a `simulate_frequency_pulses` cosi' com'e'.

Le finestre disponibili per WP sono 365-672: con ~200 l'errore sulla media e' ~7% per bin, e
scende come 1/sqrt(N).

Va eseguito dove stanno i binari. `--check` valida su finestre DIVERSE da quelle usate per
costruire la NPS (meta' e meta').

Uso:
    KMP_DUPLICATE_LIB_OK=TRUE python3 build_NPS_clean_m205.py [--check]
"""

import os
import numpy as np

import build_simAP_injected_m205 as inj      # lettura finestre + taglio RMS, gia' verificati

BASE_DIR = inj.BASE_DIR
OUT_DIR  = os.path.join(BASE_DIR, "m205_NPS_clean")
CHANNELS = [91]                  # sul server: [31, 34, 71, 83, 91]
WPS      = list(range(1, 30, 2))
N_WIN    = None                  # finestre usate per WP; None = tutte quelle disponibili
                                 # (356-648 dopo il taglio -> errore ~4.6% per bin)
WINDOW   = inj.WINDOW


def clean_nps(channel, wp, n_win=N_WIN, skip=0):
    """NPS come media del periodogramma su `n_win` finestre vere, nella convenzione del
    simulatore: E|FFT(x)|^2. `skip` salta le prime finestre, per validare su un campione
    indipendente da quello di costruzione."""
    amps, tsample, baseline = inj.wp_selection(channel, wp)
    if skip:
        tsample, baseline = tsample[skip:], baseline[skip:]
    w = inj.noise_windows(channel, tsample, baseline, n_win)
    w = w - w.mean(axis=1, keepdims=True)
    han = np.hanning(WINDOW)
    p = np.abs(np.fft.fft(w * han, axis=1)) ** 2 * WINDOW / np.sum(han ** 2)
    # la MEDIA e' quella che serve: e' la PSD vera, ed e' cio' che il simulatore e il filtro
    # ottimo assumono. La MEDIANA si salva solo come controllo di robustezza: sulle finestre
    # pulite le due stanno a 1.084 in potenza totale (1.44 sul bin tipico, 1.00 sulle righe).
    return p.mean(axis=0), np.median(p, axis=0), w


def main(check=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"NPS pulita da {N_WIN} finestre vere per WP -> {OUT_DIR}\n")
    hdr = f"{'ch':>4s} {'wp':>3s} {'RMS da NPS':>11s} {'RMS vera':>10s} {'rapporto':>9s}"
    print(hdr + (f" | {"Octopus":>8s}" if check else ""))
    for ch in CHANNELS:
        for wp in WPS:
            try:
                nps, nps_med, _ = clean_nps(ch, wp)
                path = os.path.join(OUT_DIR, f"ch{ch}", f"nps_ch{ch}_wp{wp}.npy")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                np.save(path, nps)
                np.save(path.replace("nps_ch", "npsmedian_ch"), nps_med)
                line = f"{ch:>4d} {wp:>3d} {np.sqrt(nps.sum())/WINDOW:11.3e}"
                if check:
                    # validazione su finestre DIVERSE (le ultime della lista)
                    amps, ts, bl = inj.wp_selection(ch, wp)
                    val = inj.noise_windows(ch, ts[len(ts)//2:], bl[len(ts)//2:], 120)
                    rms_true = float(np.median(val.std(axis=1)))
                    from scipy.signal.windows import flattop
                    import uproot, glob
                    with uproot.open(inj.root_file(ch)) as f:
                        md = np.asarray(f[f"averagepowerspectrum_noise_wp{wp}_medianpower"].values(), float)
                    oct_nps = (np.concatenate([md, md[-2:0:-1]]) * WINDOW / np.sum(flattop(WINDOW)**2)
                               * WINDOW ** 2 / (WINDOW / 10_000))
                    line += (f" {rms_true:10.3e} {np.sqrt(nps.sum())/WINDOW/rms_true:9.3f}"
                             f" | {np.sqrt(oct_nps.sum())/WINDOW/rms_true:8.3f}")
                else:
                    line += f" {'-':>10s} {'-':>9s}"
                print(line)
            except Exception as e:
                print(f"{ch:>4d} {wp:>3d}   [ERROR] {e}")


if __name__ == "__main__":
    import sys
    main(check="--check" in sys.argv)
