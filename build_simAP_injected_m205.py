#!/usr/bin/env python3
"""
build_simAP_injected_m205.py
============================
Costruisce gli AVERAGE PULSE SIMULATI usati come TEMPLATE DI ADDESTRAMENTO.

Due modi, scelti da MODE:

  MODE = "mc"  (consigliato)
    L'AP e' la mediana di N_PULSES impulsi SINGOLI generati dallo STESSO simulatore del Monte
    Carlo: stesso template, stessa NPS misurata, ampiezza di ROI. E' la struttura del paper
    (sez. 4.5): il template dell'analisi si ri-misura dagli eventi simulati, invece di essere
    lo stesso oggetto che li ha generati. Con N grande il rumore del template diventa
    trascurabile (va come 1/sqrt(N)), che e' il punto: si vuole addestrare su una FORMA, non
    su una realizzazione di rumore. Il paper ne usa 12288.
    Non serve il file binario: gira in locale.

  MODE = "realnoise"  (vecchio, per la sistematica)
    Il template viene iniettato su FINESTRE DI RUMORE VERE lette dal binario, tante quanti
    sono gli impulsi dell'AP vero (36-39), con le ampiezze LED vere. Serviva a misurare quanto
    pesa il rumore di un template fatto con pochi impulsi, e quanto il rumore vero (righe di
    ampiezza fissa e fase casuale) differisca da quello gaussiano generato dalla NPS.
    Va eseguito DOVE STANNO I BINARI (server): BIN_DIR non esiste in locale.

NOMENCLATURA dei file prodotti (m205_AP_sim/ch<ch>/):
    simAP_APsim<template><N>_ch<ch>_wp<wp>.npy    MODE="mc",        es. APsimfit5000
    simAP_APreal<template><N>_ch<ch>_wp<wp>.npy   MODE="realnoise", es. APrealfit38
Il tag dice le tre cose che servono: e' un AP simulato, da quale template, da quanti impulsi.
I vecchi tag "fitinj"/"rootinj"/"fitgen"/"rootgen" NON vengono piu' prodotti: il suffisso
inj/gen si riferiva alla sorgente del RUMORE, non al template, e si prestava a confusione.
I file gia' su disco restano leggibili, cambia solo cosa si scrive da qui in avanti.

Il tag va poi messo in TEMPLATE_SOURCE dei programmi di analisi, ed e' quello che finisce nel
nome della cartella dei risultati (m205_results_wiener_APsimfit5000_npsclean).

`--selftest` gira in locale e verifica il rumore residuo dell'AP contro la formula.

Uso:
    KMP_DUPLICATE_LIB_OK=TRUE python3 build_simAP_injected_m205.py
"""

import os, csv, glob
import numpy as np
import uproot

import extract_AP_pulses_m205 as ex        # lettore del binario, gia' verificato bit-a-bit
import src.simulation as sim               # generatore degli impulsi, lo stesso del MC

# ═════════════════════════════════════════════════════════════════════════════
# Config
# ═════════════════════════════════════════════════════════════════════════════
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "Processed")
PULSE_DIR  = os.path.join(BASE_DIR, "m205_AP_pulses")
FIT_DIR    = os.path.join(BASE_DIR, "residual_scan_bessel", "fits_octopus")
OUT_DIR    = os.path.join(BASE_DIR, "m205_AP_sim")
MEAS_NAME  = "000205"
BIN_DIR    = "/data2/LSC/DATA/RUN14/000205" #os.path.join(DATA_DIR, "Bin file")   # sul server: "/data2/LSC/DATA/RUN14/000205"
ex.BIN_DIR = BIN_DIR

CHANNELS   = [31, 34, 71, 83, 91]      # in "mc" non serve il binario: tutti i canali buoni
WPS        = list(range(1, 30, 2))
TEMPLATES  = ["fit"]                   # template da cui costruire l'AP: "fit" e/o "root"

WINDOW     = 10_000
PRETRIGGER = 0.5
SAMPLING_RATE = 10_000

# ── Come si costruisce l'AP ─────────────────────────────────────────────────
MODE     = "mc"          # "mc" = impulsi generati come nel Monte Carlo | "realnoise" = iniezione
                         #        su finestre vere dal binario (serve il .bin, solo sul server)
N_PULSES = 10000         # solo per MODE="mc": quanti impulsi mediare. Il rumore del template
                         # e' sigma_raw / (A * sqrt(N)). Con la media si accumula, quindi N non
                         # ha limiti di memoria: costa solo CPU (~4.6 s ogni 5000 impulsi).
                         # Misurato su ch91 WP15 all'ampiezza di ROI (sigma_raw/A = 0.364):
                         #        N        rumore del template     vs AP vero (1.74e-4)
                         #     5000            5.1e-3                    30x
                         #    12288            3.3e-3                    19x   (come il paper)
                         #      1e6            3.6e-4                     2x
                         #    4.4e6            1.7e-4                     1x
# A che AMPIEZZA si generano gli impulsi da mediare.
#   "roi" (default): l'ampiezza di amplitudes_m205.csv, la STESSA che il Monte Carlo inietta
#       negli eventi. E' la scelta coerente: l'AP dev'essere la media di QUEGLI impulsi.
#       Il rumore del template e' sigma_raw/(A*sqrt(N)) e lo si sceglie con N (vedi tabella
#       nel commento di N_PULSES). Attenzione a non confondere i due SNR: quello del CSV e'
#       A/sigma_OF = 10-150 (risoluzione del filtro ottimo, integra su tutto l'impulso), mentre
#       la costruzione dell'AP vede A/sigma_raw = 1.4-13.6 (RMS della traccia non filtrata).
#       Sono diversi di un fattore 5-15, ed e' il secondo a fissare il rumore del template.
#   "led": l'ampiezza LED vera del WP, quella a cui l'AP vero e' misurato davvero (~0.5 V,
#       SNR di picco ~1200). Da' un template ~10 volte piu' pulito di quello vero con N=5000,
#       ma NON e' la media degli eventi del Monte Carlo: e' un altro oggetto. Il tag del file
#       prende il suffisso "led" per non confonderli.
AP_AMPLITUDE = "roi"     # "led" | "roi"
GEN_CHUNK = 500          # impulsi generati per volta: il simulatore alloca sei array complessi
                         # (n, 10000), a 5000 in un colpo sono ~4 GB l'uno.

NPS_DIR     = os.path.join(BASE_DIR, "m205_NPS_clean")
NPS_PATTERN = os.path.join("ch{ch}", "nps_ch{ch}_wp{wp}.npy")
AMP_CSV     = os.path.join(BASE_DIR, "amplitudes_m205.csv")   # ampiezze di ROI, in mV
VBIAS_LIST  = np.array([0.6, 1.0, 1.4, 1.8, 2, 3, 4, 5, 6, 8, 10, 20, 26, 30, 40])

# ── Solo per MODE = "realnoise" ─────────────────────────────────────────────
STD_CUT_MAD = 5.0                      # taglio sulla RMS della finestra INTERA, come il
                                       # `std < StdCut` di test/select_noise_traces.py: scarta
                                       # le finestre oltre mediana + STD_CUT_MAD*MAD. Serve:
                                       # i tagli di Octopus valutano la RMS sul pretrigger, e
                                       # qualche finestra con un impulso nella seconda meta'
                                       # passa (misurate std fino a 70 volte la mediana).
POOL_FACTOR = 3                        # finestre lette per ognuna che serve, prima del taglio
AMP_MODE   = "real"                    # "real": le ampiezze LED dei 38 eventi veri, con la loro
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


def load_nps(channel, wp):
    """NPS misurata, gia' nella convenzione E|FFT(x)|^2 usata da tutto il progetto."""
    path = os.path.join(NPS_DIR, NPS_PATTERN.format(ch=channel, wp=wp))
    if not os.path.exists(path):
        raise RuntimeError(f"NPS 'clean' non trovata: {path}")
    return np.asarray(np.load(path), dtype=float)


_AMPS = None
def signal_amp(channel, wp):
    """Ampiezza di ROI [V] di (canale, WP), dal CSV delle ampiezze. E' la STESSA che il Monte
    Carlo inietta negli eventi: l'AP dev'essere la mediana di QUEGLI impulsi, non di altri."""
    global _AMPS
    if _AMPS is None:
        _AMPS = {}
        with open(AMP_CSV, newline="") as f:
            for r in csv.DictReader(f):
                a = (r.get("amplitude_mV") or "").strip()
                if a:
                    _AMPS[(int(r["channel"]), round(float(r["vbias_V"]), 3))] = float(a) * 1e-3
    key = (channel, round(float(VBIAS_LIST[wp // 2]), 3))
    if key not in _AMPS:
        raise RuntimeError(f"ampiezza di ROI non trovata in {os.path.basename(AMP_CSV)} per {key}")
    return _AMPS[key]


def save_ap(channel, wp, tag, ap):
    """Salva l'AP e restituisce la RMS del suo pretrigger, cioe' il rumore residuo."""
    path = os.path.join(OUT_DIR, f"ch{channel}", f"simAP_{tag}_ch{channel}_wp{wp}.npy")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, ap)
    return float(ap[:4000].std())


def led_amp(channel, wp):
    """Ampiezza LED mediana del WP [V]: quella degli impulsi con cui l'AP vero e' costruito."""
    with uproot.open(root_file(channel)) as f:
        sig = f[f"crosscorr_signal_wp{wp}"]["pass"].array(library="np")
        return float(np.median(f["maxminusbaseline"]["amplitude"].array(library="np")[sig]))


def build_mc(channel, wp, sources):
    """AP = MEDIA di N_PULSES impulsi SINGOLI generati come li genera il Monte Carlo.

    Tre scelte, tutte dovute al fatto che qui gli impulsi sono SIMULATI e non veri:

    - NIENTE allineamento: il generatore li produce gia' allineati (il Monte Carlo non simula
      il jitter di trigger), quindi allinearli aggiungerebbe solo l'errore dell'allineamento.
    - NIENTE normalizzazione al massimo del singolo impulso: hanno tutti la STESSA ampiezza per
      costruzione, quindi normalizzare non toglie una dispersione che non c'e', e in compenso
      divide ogni traccia per un massimo gonfiato dal rumore. All'ampiezza di ROI il SNR di
      PICCO della traccia grezza (A/sigma_raw) e' 1.4-13.6 -- il massimo non e' il picco -- e
      misurato costa: RMS del template 6.3e-3 con la normalizzazione contro 5.6e-3 senza.
      Sugli impulsi VERI la normalizzazione ha senso (sono LED, SNR di picco ~1200) ed e'
      infatti quello che fa Octopus; qui no.
    - MEDIA e non mediana: la mediana serve a reggere gli outlier dei dati veri, che in
      simulazione non esistono, costa un fattore 1.2533 di rumore e soprattutto obbliga a
      tenere in memoria tutti gli N impulsi. Con la media si accumula e basta, quindi N non ha
      limite di memoria: il rumore del template scende come sigma_raw/(A*sqrt(N)).

    Il template si passa al generatore NON finestrato (`np.fft.fft(tpl)`): compute_H
    restituirebbe FFT(tpl * hanning), e la finestra verrebbe applicata una seconda volta
    dall'addestramento."""
    nps = load_nps(channel, wp)
    amp = led_amp(channel, wp) if AP_AMPLITUDE == "led" else signal_amp(channel, wp)
    w = 2 * np.pi * np.fft.fftfreq(WINDOW, 1.0 / SAMPLING_RATE)
    out = {}
    for src in sources:
        tpl = template(channel, wp, src)
        S_raw = np.fft.fft(tpl)
        acc = np.zeros(WINDOW)
        for k in range(0, N_PULSES, GEN_CHUNK):
            n = min(GEN_CHUNK, N_PULSES - k)
            fp, *_ = sim.simulate_frequency_pulses(S_raw, nps, 0.0, w, nsim=n,
                                                   seed=SEED + 1000 * channel + wp + k,
                                                   signal_scale=amp, dt_max=0.0)
            acc += np.fft.ifft(fp, axis=1).real.sum(axis=0)
            del fp
        ap = acc / N_PULSES
        ap /= ap.max()
        tag = f"APsim{src}{N_PULSES}" + ("led" if AP_AMPLITUDE == "led" else "")
        out[src] = save_ap(channel, wp, tag, ap)
    return out, N_PULSES, amp


def build_realnoise(channel, wp, sources):
    """AP iniettando il template su finestre di rumore VERE dal binario, N = quello dell'AP
    vero e ampiezze LED. E' la sistematica sul rumore di template, non il caso nominale."""
    amps, tsample, baseline = wp_selection(channel, wp)
    N = len(amps)
    noise = noise_windows(channel, tsample, baseline, N)
    rng = np.random.default_rng(SEED + 1000 * channel + wp)
    a = rng.permutation(amps if AMP_MODE == "real" else np.full(N, np.median(amps)))
    shifts = rng.integers(-JITTER, JITTER + 1, N) if JITTER else np.zeros(N, int)
    out = {}
    for src in sources:
        tpl = template(channel, wp, src)
        p = noise + a[:, None] * np.array([np.roll(tpl, int(sh)) for sh in shifts])
        p /= p.max(axis=1, keepdims=True)
        ap = np.median(p, axis=0)
        ap /= ap.max()
        out[src] = save_ap(channel, wp, f"APreal{src}{N}", ap)
    return out, N, float(np.median(amps))


def main():
    if MODE not in ("mc", "realnoise"):
        raise SystemExit(f"[ERROR] MODE='{MODE}' non valido: 'mc' o 'realnoise'.")
    print(f"AP simulati -> {OUT_DIR}")
    if MODE == "mc":
        print(f"  media di {N_PULSES} impulsi GENERATI: template + rumore dalla NPS misurata, "
              f"ampiezza '{AP_AMPLITUDE}'")
        print(f"  tag dei file: APsim<template>{N_PULSES}\n")
    else:
        print(f"  iniezione su finestre di rumore VERE dal binario, N = quello dell'AP vero, "
              f"ampiezze LED '{AMP_MODE}', jitter {JITTER}")
        print(f"  tag dei file: APreal<template><N>\n")
    print(f"{'ch':>4s} {'wp':>3s} {'N':>5s} {'amp [V]':>9s} | "
          + " ".join(f"{'RMS AP ' + t:>13s}" for t in TEMPLATES)
          + f" {'RMS AP vero':>12s} {'rapporto':>9s}")
    for ch in CHANNELS:
        for wp in WPS:
            try:
                src = [t for t in TEMPLATES if t != "fit" or
                       os.path.exists(os.path.join(FIT_DIR, f"bestfit_ch{ch}_wp{wp}.npy"))]
                if not src:
                    raise RuntimeError("template 'fit' non trovato")
                if MODE == "mc":
                    res, N, amp = build_mc(ch, wp, src)
                else:
                    res, N, amp = build_realnoise(ch, wp, src)
                # riferimento: il rumore residuo dell'AP VERO, quello che si sta sostituendo
                with uproot.open(root_file(ch)) as f:
                    apv = np.asarray(f[f"averagepulse_ap_wp{wp}_medianAP"].values(), float)
                apv /= apv.max()
                rms_true = float(apv[:4000].std())
                first = res.get(TEMPLATES[0], float("nan"))
                print(f"{ch:>4d} {wp:>3d} {N:>5d} {amp:9.2e} | "
                      + " ".join(f"{res.get(t, float('nan')):13.2e}" for t in TEMPLATES)
                      + f" {rms_true:12.2e} {first / rms_true:9.2f}")
            except Exception as e:
                tag = "INFO salto" if "non trovat" in str(e) else "ERROR"
                print(f"{ch:>4d} {wp:>3d}   [{tag}] {e}")


def selftest():
    """Verifica il rumore residuo dell'AP costruito in MODE='mc' contro la formula.

    Col template 'fit' (liscio) l'AP simulato ha SOLO il rumore nuovo, quindi la RMS del suo
    pretrigger dev'essere quella della mediana di N impulsi normalizzati:
        sigma_traccia / ampiezza / sqrt(N)
    dove sigma_traccia = sqrt(sum(nps))/M e' la convenzione del generatore (verificata a parte).
    Gira in locale: servono solo la NPS misurata, il bestfit e il CSV delle ampiezze."""
    ch, wp, n = 91, 15, 2000
    globals()["N_PULSES"] = n
    globals()["OUT_DIR"] = os.path.join(BASE_DIR, "m205_AP_sim", "_selftest")
    res, N, amp = build_mc(ch, wp, ["fit"])
    nps = load_nps(ch, wp)
    sigma = float(np.sqrt(nps.sum()) / WINDOW)
    exp = sigma / amp / np.sqrt(N)
    got = res["fit"]
    assert 0.85 < got / exp < 1.15, f"RMS dell'AP simulato {got:.2e}, attesa {exp:.2e}"
    print(f"[OK] selftest: ch{ch} WP{wp}, N={N}, ampiezza {amp:.2e} V, SNR per impulso "
          f"{amp / sigma:.0f}\n     RMS pretrigger dell'AP {got:.2e} contro l'attesa "
          f"{exp:.2e}  ({got / exp:.2f})")


if __name__ == "__main__":
    import sys
    selftest() if "--selftest" in sys.argv else main()
