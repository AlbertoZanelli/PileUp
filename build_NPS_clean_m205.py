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
PLOT_WPS = [15]                  # WP per cui produrre le figure di diagnostica; [] = nessuna
PLOT_DIR = os.path.join(OUT_DIR, "plots")
N_SHOW   = 6                     # finestre accettate da mostrare nella figura
N_WIN    = None                  # finestre usate per WP; None = tutte quelle disponibili
                                 # (356-648 dopo il taglio -> errore ~4.6% per bin)
WINDOW   = inj.WINDOW


def octopus_nps(channel, wp):
    """medianpower dal ROOT nella STESSA convenzione della NPS pulita (E|FFT(x)|^2), cioe'
    specchiata e corretta per la flattop: e' l'array che i programmi usano oggi."""
    import uproot
    from scipy.signal.windows import flattop
    with uproot.open(inj.root_file(channel)) as f:
        md = np.asarray(f[f"averagepowerspectrum_noise_wp{wp}_medianpower"].values(), dtype=float)
    md = np.concatenate([md, md[-2:0:-1]])
    return md * (WINDOW / np.sum(flattop(WINDOW) ** 2)) * WINDOW ** 2 / (WINDOW / 10_000)


def plot_windows(channel, wp, kept, rejected, thr):
    """Le finestre scartate dal taglio RMS e un campione di quelle tenute."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(WINDOW) / 10_000.0
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2),
                           gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for x in kept[:N_SHOW]:
        ax[0].plot(t, x * 1e3, lw=0.6, alpha=0.8)
    ax[0].set_title(f"accepted (sample of {N_SHOW} out of {len(kept)})")
    for x in rejected:
        ax[1].plot(t, x * 1e3, lw=0.6, alpha=0.8)
    ax[1].set_title(f"rejected by the RMS cut ({len(rejected)})")
    for a in ax[:2]:
        a.set_xlabel("time [s]"); a.set_ylabel("amplitude [mV]"); a.grid(True, ls="--", alpha=0.4)
    sd = np.concatenate([kept.std(axis=1), rejected.std(axis=1)]) * 1e3
    ax[2].hist(sd, bins=np.geomspace(sd.min(), sd.max(), 40), color="tab:blue")
    ax[2].axvline(thr * 1e3, color="r", ls="--", label=f"cut = median + {inj.STD_CUT_MAD:g} MAD")
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    ax[2].set_xlabel("window RMS [mV]"); ax[2].set_ylabel("windows")
    ax[2].set_title("RMS distribution"); ax[2].legend(fontsize=8)
    ax[2].grid(True, ls="--", alpha=0.4)
    fig.suptitle(f"m205 Ch{channel} WP{wp} - noise windows")
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, f"windows_ch{channel}_wp{wp}.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


def plot_nps(channel, wp, mean, median):
    """Le tre NPS sovrapposte, nella stessa convenzione, piu' il rapporto con la media pulita."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    oc = octopus_nps(channel, wp)
    n = WINDOW // 2
    fr = np.fft.fftfreq(WINDOW, 1 / 10_000)[1:n]
    sl = slice(1, n)
    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                           gridspec_kw={"height_ratios": [2.2, 1]})
    # il rapporto bin-per-bin e' illeggibile sopra i ~200 Hz: Octopus stima con una finestra
    # FLATTOP e qui si usa una HANNING, che spalmano le righe su un numero diverso di bin.
    # La mediana mobile toglie quell'effetto e lascia vedere il fattore, che e' quello che conta.
    from scipy.ndimage import median_filter
    for y, lab, c in [(oc, "Octopus medianpower (as used today)", "tab:red"),
                      (median, "median of cleaned windows", "tab:orange"),
                      (mean, "mean of cleaned windows (used for simulation)", "tab:blue")]:
        ax[0].loglog(fr, y[sl], lw=0.7, alpha=0.85, label=lab, color=c)
        rat = y[sl] / mean[sl]
        ax[1].semilogx(fr, rat, lw=0.5, alpha=0.15, color=c)
        ax[1].semilogx(fr, median_filter(rat, size=51, mode="nearest"), lw=1.4, color=c)
    ax[0].set_ylabel(r"E$|$FFT$(x)|^2$  [a.u.]")
    ax[0].legend(fontsize=9); ax[0].grid(True, which="both", ls="--", alpha=0.3)
    ax[1].axhline(1, color="k", lw=0.8)
    ax[1].axhline(2, color="0.5", lw=0.8, ls=":")
    ax[1].axhline(2 * np.log(2), color="0.5", lw=0.8, ls="--")
    ax[1].text(fr[-1], 2, " 2", va="center", fontsize=8, color="0.4")
    ax[1].text(fr[-1], 2 * np.log(2), r" 2ln2", va="center", fontsize=8, color="0.4")
    ax[1].set_ylim(0.3, 3.5); ax[1].set_xlabel("frequency [Hz]")
    ax[1].set_ylabel("ratio to clean mean")
    ax[1].grid(True, which="both", ls="--", alpha=0.3)
    ax[0].set_title(f"m205 Ch{channel} WP{wp} - noise power spectra")
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, f"nps_ch{channel}_wp{wp}.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


def clean_nps(channel, wp, n_win=N_WIN, skip=0):
    """NPS come media del periodogramma su `n_win` finestre vere, nella convenzione del
    simulatore: E|FFT(x)|^2. `skip` salta le prime finestre, per validare su un campione
    indipendente da quello di costruzione."""
    amps, tsample, baseline = inj.wp_selection(channel, wp)
    if skip:
        tsample, baseline = tsample[skip:], baseline[skip:]
    w, rejected, thr = inj.noise_windows(channel, tsample, baseline, n_win, with_rejected=True)
    if wp in PLOT_WPS:
        os.makedirs(PLOT_DIR, exist_ok=True)
        print(f"   -> {os.path.relpath(plot_windows(channel, wp, w, rejected, thr), BASE_DIR)}")
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
                if wp in PLOT_WPS:
                    print(f"   -> {os.path.relpath(plot_nps(ch, wp, nps, nps_med), BASE_DIR)}")
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
