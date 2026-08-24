#!/usr/bin/env python3
"""
build_simAP_injected_m205.py
============================
AP SIMULATI iniettando il template su FINESTRE DI RUMORE VERE del file binario, invece di
generare rumore gaussiano dalla NPS (che e' quello che fa `simulate_BI_error_m205.py --make-ap`).
Con NOISE_SOURCE = "clean_nps" si possono generare le tracce dalla NPS misurata invece di
leggerle, per quantificare quanto pesa la scelta (misurato su ch91: 1.08 contro 1.17).

Perche': il rumore generato dalla `medianpower` di Octopus e' ~1.28 volte troppo grande (il
fattore 2*ln2 della NPS) ed e' gaussiano per costruzione, mentre il rumore vero e' dominato da
righe di ampiezza fissa e fase casuale, che la mediana attenua meglio di una gaussiana (altro
~1.19). Insieme fanno un AP simulato 1.5 volte piu' rumoroso di quello vero. Prendendo il rumore
dai dati, entrambi i fattori spariscono per costruzione.

E' il procedimento del paper: `test/analysis_meanpulse_test.py` non genera rumore, inietta gli
impulsi su finestre vere del binario (`src/dataset.py: CachedBinaryDataset_withgenerated`,
`self.data += e1*pulse(...)`), e le finestre di rumore le sceglie `test/select_noise_traces.py`
con un taglio sulla RMS della finestra (`std < StdCut`, una soglia per canale).

Qui quel taglio non serve rifarlo a mano: Octopus lo ha gia' fatto, per ogni WP, con
`cuts_noise_rms_wp<wp>`, `cuts_noise_slope_wp<wp>` e `cuts_noise_amplitude_wp<wp>` (~510 eventi
per WP). Sono esattamente le finestre con cui e' costruita `averagepowerspectrum_noise_wp<wp>`,
quindi il rumore iniettato e' per costruzione la stessa popolazione della NPS del confronto.

Differenze volute rispetto al paper: N = quello dell'AP vero (36-39, non 12288), perche' la
domanda e' proprio quanto pesa il rumore di un template fatto con pochi impulsi; mediana e non
media, per restare sulla convenzione di Octopus.

Va eseguito DOVE STANNO I BINARI (server): BIN_DIR non esiste in locale. Poi si sincronizzano
gli .npy, come per extract_AP_pulses_m205.py.
`--selftest` gira in locale: sostituisce il lettore del binario con uno stream sintetico.

Uso:
    KMP_DUPLICATE_LIB_OK=TRUE python3 build_simAP_injected_m205.py
"""

import os, glob
import numpy as np
import uproot

import extract_AP_pulses_m205 as ex        # lettore del binario, gia' verificato bit-a-bit
import src.simulation as sim               # solo per NOISE_SOURCE = "clean_nps"

# ═════════════════════════════════════════════════════════════════════════════
# Config
# ═════════════════════════════════════════════════════════════════════════════
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "Processed")
PULSE_DIR  = os.path.join(BASE_DIR, "m205_AP_pulses")
FIT_DIR    = os.path.join(BASE_DIR, "residual_scan_bessel", "fits_octopus")
OUT_DIR    = os.path.join(BASE_DIR, "m205_AP_sim")
MEAS_NAME  = "000205"
BIN_DIR    = os.path.join(DATA_DIR, "Bin file")   # sul server: "/data2/LSC/DATA/RUN14/000205"
ex.BIN_DIR = BIN_DIR

CHANNELS   = [91]           # in locale c'e' solo il binario di ch91; sul server [31, 34, 71, 83, 91]
WPS        = list(range(1, 30, 2))
TEMPLATES  = ["root", "fit"]           # template iniettato -> file con tag "rootinj"/"fitinj"

WINDOW     = 10_000
PRETRIGGER = 0.5
# ── Da dove viene il rumore delle tracce su cui si costruisce l'AP simulato ──────────
# "real": finestre vere lette dal binario (default). E' la strada del paper, ed e' la piu'
#   fedele: su ch91 l'AP simulato ha rumore 1.08 volte quello dell'AP vero.
# "clean_nps": rumore GENERATO dalla NPS misurata (build_NPS_clean_m205.py). Coerente con il
#   rumore degli eventi del Monte Carlo, che e' generato anch'esso, ma un po' peggiore come
#   template: 1.17 invece di 1.08, perche' resta gaussiano mentre il rumore vero e' dominato
#   da righe e la mediana di 38 tracce le attenua meglio. Serve a quantificare quella
#   differenza come sistematica, non a sostituire l'iniezione.
# I due set finiscono in file diversi ("...inj..." e "...gen..."), non si mescolano.
NOISE_SOURCE = "real"                  # "real" | "clean_nps"
NPS_DIR     = os.path.join(BASE_DIR, "m205_NPS_clean")
NPS_PATTERN = os.path.join("ch{ch}", "nps_ch{ch}_wp{wp}.npy")

STD_CUT_MAD = 5.0                      # taglio sulla RMS della finestra INTERA, come il
                                       # `std < StdCut` di test/select_noise_traces.py: scarta
                                       # le finestre oltre mediana + STD_CUT_MAD*MAD. Serve:
                                       # i tagli di Octopus valutano la RMS sul pretrigger, e
                                       # qualche finestra con un impulso nella seconda meta'
                                       # passa (misurate std fino a 70 volte la mediana).
POOL_FACTOR = 3                        # finestre lette per ognuna che serve, prima del taglio
AMP_MODE   = "real"                    # "real": le ampiezze dei 38 eventi veri, con la loro
                                       # dispersione; "median": tutte uguali alla mediana
JITTER     = 0                         # spostamento casuale del template [campioni], come lo
                                       # `smearing` del pos file del paper. 0 = allineamento
                                       # perfetto (gli impulsi veri hanno un residuo di jitter)
SEED       = 1234


def root_file(channel):
    return glob.glob(os.path.join(DATA_DIR, f"Processed_*_{MEAS_NAME}_{channel}.root"))[0]


def seg_length(channel):
    return (os.path.getsize(ex.bin_path(channel)) - ex.HDR_BYTES) // 4


def template(channel, wp, source):
    if source == "fit":
        path = os.path.join(FIT_DIR, f"bestfit_ch{channel}_wp{wp}.npy")
        if not os.path.exists(path):
            raise RuntimeError(f"template 'fit' non trovato: {path}")
        v = np.load(path)
    else:
        with uproot.open(root_file(channel)) as f:
            v = np.asarray(f[f"averagepulse_ap_wp{wp}_medianAP"].values(), dtype=float)
    return np.asarray(v, dtype=float) / np.max(v)


def wp_selection(channel, wp):
    """(ampiezze LED del WP, posizioni delle finestre di rumore del WP, loro baseline).

    Le finestre di rumore sono ESATTAMENTE quelle con cui Octopus costruisce la NPS del WP:
    il modulo `averagepowerspectrum_noise_wp<wp>` seleziona `cuts_noise_amplitude_wp<wp>.pass`,
    ultimo di una catena che chiede isnoise=1, numberoftriggers=1, isLED=0, l'intervallo
    temporale del WP (timestamp.timefromstartrun) e i tagli su pendenza, RMS e ampiezza.
    Sono il 47-88% dei trigger che cadono nell'intervallo del WP."""
    with uproot.open(root_file(channel)) as f:
        sig = f[f"crosscorr_signal_wp{wp}"]["pass"].array(library="np")
        amp = f["maxminusbaseline"]["amplitude"].array(library="np")[sig]
        noi = f[f"cuts_noise_amplitude_wp{wp}"]["pass"].array(library="np")
        tsample = f["module"]["triggersample"].array(library="np")[noi]
        baseline = f["baseline"]["baseline"].array(library="np")[noi]
    return np.asarray(amp, float), tsample.astype(np.int64), np.asarray(baseline, float)


def noise_windows(channel, tsample, baseline, n_needed=None, with_rejected=False):
    """`n_needed` finestre di rumore vere (None = tutte quelle che passano il taglio), in volt e con la baseline di Octopus sottratta.
    Prese distribuite uniformemente sull'intervallo del WP, non le prime N, cosi' coprono lo
    stesso arco temporale degli eventi che formano l'AP vero."""
    if n_needed is not None and len(tsample) < n_needed:
        raise RuntimeError(f"solo {len(tsample)} finestre di rumore, ne servono {n_needed}")
    pool = len(tsample) if n_needed is None else min(len(tsample), POOL_FACTOR * n_needed)
    idx = np.linspace(0, len(tsample) - 1, pool).astype(int)
    _, _, fullscale = ex.bin_header(channel)
    seg_len = seg_length(channel)
    pre = int(PRETRIGGER * WINDOW)
    win = np.empty((pool, WINDOW))
    for i, k in enumerate(idx):
        raw = ex.read_samples(channel, int(tsample[k]) - pre, WINDOW, seg_len)
        win[i] = ex.to_volt(raw, fullscale) - baseline[k]

    sd = win.std(axis=1)
    thr = np.median(sd) + STD_CUT_MAD * np.median(np.abs(sd - np.median(sd))) / 0.6745
    keep = np.flatnonzero(sd < thr)
    if n_needed is None:
        out = win[keep]
    else:
        if len(keep) < n_needed:
            raise RuntimeError(f"dopo il taglio RMS restano {len(keep)} finestre su {pool}, "
                               f"ne servono {n_needed}")
        out = win[keep[np.linspace(0, len(keep) - 1, n_needed).astype(int)]]
    return (out, win[np.flatnonzero(sd >= thr)], thr) if with_rejected else out


def noise_traces(channel, wp, n, tsample, baseline):
    """`n` tracce di rumore secondo NOISE_SOURCE: vere dal binario, oppure generate dalla
    NPS misurata (che e' gia' nella convenzione E|FFT(x)|^2 del simulatore)."""
    if NOISE_SOURCE == "real":
        return noise_windows(channel, tsample, baseline, n)
    path = os.path.join(NPS_DIR, NPS_PATTERN.format(ch=channel, wp=wp))
    if not os.path.exists(path):
        raise RuntimeError(f"NPS 'clean' non trovata: {path}")
    nps = np.asarray(np.load(path), dtype=float)
    w = 2 * np.pi * np.fft.fftfreq(WINDOW, 1.0 / 10_000)
    fp, *_ = sim.simulate_frequency_pulses(np.zeros(WINDOW), nps, 0.0, w, nsim=n,
                                           seed=SEED + 1000 * channel + wp,
                                           signal_scale=0.0, dt_max=0.0)
    return np.fft.ifft(fp, axis=1).real


def build(channel, wp, sources):
    amps, tsample, baseline = wp_selection(channel, wp)
    N = len(amps)
    noise = noise_traces(channel, wp, N, tsample, baseline)

    rng = np.random.default_rng(SEED + 1000 * channel + wp)
    a = rng.permutation(amps if AMP_MODE == "real" else np.full(N, np.median(amps)))
    shifts = rng.integers(-JITTER, JITTER + 1, N) if JITTER else np.zeros(N, int)

    out = {}
    for src in sources:
        tpl = template(channel, wp, src)
        p = noise + a[:, None] * np.array([np.roll(tpl, int(s)) for s in shifts])
        p /= p.max(axis=1, keepdims=True)
        ap = np.median(p, axis=0)
        ap /= ap.max()
        tag = f"{src}{'inj' if NOISE_SOURCE == 'real' else 'gen'}"
        path = os.path.join(OUT_DIR, f"ch{channel}", f"simAP_{tag}_ch{channel}_wp{wp}.npy")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, ap)
        out[src] = float(ap[:4000].std())
    return out, noise, N, len(tsample)


def main():
    print(f"AP simulati per iniezione su rumore vero -> {OUT_DIR}")
    print(f"rumore '{NOISE_SOURCE}'" + (", tagli di Octopus (cuts_noise_amplitude) del WP + taglio RMS"
          if NOISE_SOURCE == "real" else ", generato dalla NPS misurata") +
          f", ampiezze '{AMP_MODE}', jitter {JITTER}\n")
    print(f"{'ch':>4s} {'wp':>3s} {'N':>3s} {'finestre':>9s} | {'rms rumore':>11s} {'rms veri':>10s} "
          f"{'rapporto':>9s} | " + " ".join(f"{'AP ' + s:>10s}" for s in TEMPLATES) + f" {'AP vero':>10s}")
    for ch in CHANNELS:
        for wp in WPS:
            try:
                src = [s for s in TEMPLATES if s != "fit" or
                       os.path.exists(os.path.join(FIT_DIR, f"bestfit_ch{ch}_wp{wp}.npy"))]
                res, noise, N, n_avail = build(ch, wp, src)
                # controllo: il rumore delle finestre deve valere quanto quello degli impulsi veri
                pul = np.load(os.path.join(PULSE_DIR, f"ch{ch}", f"pulses_ch{ch}_wp{wp}.npy"))
                with uproot.open(root_file(ch)) as f:
                    apv = np.asarray(f[f"averagepulse_ap_wp{wp}_medianAP"].values(), float)
                apv /= apv.max()
                rms_n = noise[:, :4000].std(axis=1).mean() / np.median(wp_selection(ch, wp)[0])
                rms_r = float(np.median(pul[:, :4000].std(axis=1)))
                print(f"{ch:>4d} {wp:>3d} {N:>3d} {n_avail:>9d} | {rms_n:11.3e} {rms_r:10.3e} "
                      f"{rms_n / rms_r:9.2f} | " +
                      " ".join(f"{res.get(s, float('nan')):10.2e}" for s in TEMPLATES) +
                      f" {apv[:4000].std():10.2e}")
            except Exception as e:
                tag = "INFO salto" if "non trovato" in str(e) else "ERROR"
                print(f"{ch:>4d} {wp:>3d}   [{tag}] {e}")


def selftest():
    """Autotest locale: sostituisce il lettore del binario con uno stream sintetico, cosi' la
    selezione delle finestre e la costruzione dell'AP si verificano senza avere i .bin.
    Controlla che l'RMS dell'AP simulato sia quello atteso per una mediana di N impulsi."""
    ch, wp, FS, SIGMA = CHANNELS[-1], 15, 10.069444, 5e-4

    def fake_read(channel, start, count, seg_len):
        v = np.random.default_rng(int(start)).normal(0, SIGMA, count)
        return ((v / FS + 1.0) * 2 ** 23 * 256).astype(np.uint64)

    real_sel = wp_selection
    # lo stream sintetico ha media zero: la baseline vera di Octopus qui non va sottratta
    globals()["wp_selection"] = lambda c, w: (real_sel(c, w)[0], real_sel(c, w)[1],
                                              np.zeros_like(real_sel(c, w)[2]))
    ex.bin_header = lambda c, seg=0: (0, 10000.0, FS)
    ex.read_samples = fake_read
    globals()["seg_length"] = lambda c: 1 << 40
    globals()["OUT_DIR"] = os.path.join(BASE_DIR, "m205_AP_sim", "_selftest")

    # col template "fit" (liscio) l'AP simulato ha solo il rumore nuovo: si confronta con la
    # formula della mediana. Col template "root" ci sarebbe in quadratura il rumore del template.
    res, noise, N, n_avail = build(ch, wp, ["fit"])
    amps = wp_selection(ch, wp)[0]
    exp = 1.2533 * SIGMA / np.median(amps) / np.sqrt(N)
    got = res["fit"]
    assert abs(noise[:, :4000].std(axis=1).mean() / SIGMA - 1) < 0.05, "rumore delle finestre non letto bene"
    assert 0.75 < got / exp < 1.25, f"RMS dell'AP simulato {got:.2e}, atteso {exp:.2e}"
    print(f"selftest ok: {n_avail} finestre disponibili, N={N}, "
          f"RMS AP simulato {got:.2e} vs atteso {exp:.2e} ({got / exp:.2f})")


if __name__ == "__main__":
    import sys
    selftest() if "--selftest" in sys.argv else main()
