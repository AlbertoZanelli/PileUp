"""
simulate_BI_error_m205.py
=========================
Calcola il BI per MONTE CARLO e la sua INCERTEZZA, applicando i filtri GIA' ADDESTRATI da
analyse_BI_m205.py (filtro ottimo) a impulsi simulati. Non riaddestra niente: legge i .npy dei
filtri e il CSV dei risultati, quindi gira in pochi secondi per coppia (canale, WP).

A cosa serve, e cosa NON e':
  - il BI del CSV e' ANALITICO, BI = J_final * K, un integrale sulla distribuzione dei rapporti;
    non ha campioni, quindi non ha un errore statistico da propagare.
  - qui si generano DUE popolazioni di eventi (impulsi singoli e di pile-up) dagli STESSI S e
    nps del calcolo analitico, si passano per i filtri addestrati e si misura la frazione di
    pile-up che sopravvive al taglio di accettanza. Da quella frazione escono BI_mc e sigma_BI
    (compute_BI_uncertainty: termine binomiale + incertezza sulla posizione del taglio).
  - ATTENZIONE: sigma_BI e' l'errore STATISTICO DEL MONTE CARLO, si abbassa a piacere alzando
    NSIM. NON e' l'incertezza fisica sul BI (per quella serve il bootstrap sui singoli impulsi
    che formano il template). Il valore vero di questo programma e' un altro: BI_mc deve
    coincidere con BI_analitico. Se non lo fa, c'e' un problema in uno dei due.

Ingredienti (tutti gia' disponibili, nessun rifit):
  - f1, f2, kernel da <RESULTS_DIR>/trained_filters/*.npy (meta' indipendente dello spettro);
  - S, w, NPS ricalcolati dal ROOT come nel worker (medianAP + medianpower, finestra di Hanning);
  - signal_amp e BI analitico dal CSV dei risultati.

Uso:
    KMP_DUPLICATE_LIB_OK=TRUE python simulate_BI_error_m205.py
"""

import os
import sys
import re
import csv
import glob
import time
import fcntl
import tempfile
import subprocess

import numpy as np
import uproot
import torch

# BASE_DIR va definito QUI, prima degli import di src/utility: eseguendo la copia congelata
# da freeze_script(), che sta nella cartella dei risultati, sys.path[0] e' quella cartella e
# `import src.analysis` fallirebbe. E' anche l'UNICA riga che freeze_script riscrive, quindi
# non va duplicata piu' in basso.
BASE_DIR = '/mnt/disk1/data/users/azanelli/PileUp'   # congelato da freeze_script(): la copia sta altrove
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import src.analysis as an
import src.dataset as ds
import src.simulation as sim
import utility.functions as fn

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE
# ═════════════════════════════════════════════════════════════════════════════
SCRIPT_PATH = os.path.abspath(__file__)      # BASE_DIR e' definito in cima, con gli import
DATA_DIR    = os.path.join(BASE_DIR, "Processed")
MEAS_NAME   = "000205"

# ═════════════════════════════════════════════════════════════════════════════
# COSA SIMULARE: due sole scelte
# ═════════════════════════════════════════════════════════════════════════════
# 1) RESULTS_NAME: il NOME DELLA CARTELLA prodotta da analyse_BI_m205.py o dai due programmi
#    Wiener. Da quel nome si deducono tipo di filtro, template del training e sorgente della
#    NPS: sono quattro cose che DEVONO combaciare con il training, quindi e' meglio leggerle
#    dal nome che riscriverle a mano. Il controllo sul kernel verifica comunque il risultato.
#      m205_results_octopus                    -> filtro ottimo, template root, NPS Octopus
#      m205_results_octopus_npsclean           -> idem, NPS misurata
#      m205_results_octopus_fit                -> filtro ottimo, template fit
#      m205_results_octopus_sim_fitinj         -> filtro ottimo, template = AP simulato fitinj
#      m205_results_wiener                     -> Wiener (lambda scalare), template root
#      m205_results_wiener_fit                 -> Wiener, template fit
#      m205_results_wiener_root_R              -> Wiener + R(f), template root
#      m205_results_wiener_sim_fitinj_R_npsclean -> Wiener + R(f), template simulato, NPS pulita
RESULTS_NAME = "m205_results_wiener_sim_rootinj_npsclean"

# 2) GEN_TEMPLATE: il template che GENERA gli eventi simulati, cioe' cosa consideri la verita'.
#    "root" -> medianAP di Octopus dal ROOT (la scelta normale: e' l'impulso vero);
#    "fit"  -> il bestfit dello scan (template liscio, senza rumore finito-N).
#    E' l'UNICA scelta libera: se coincide con il template del training il conto e'
#    auto-consistente e serve solo da validazione; se differisce, misuri quanto costa
#    addestrare sul template sbagliato.
GEN_TEMPLATE = "root"       # "root" | "fit"

# Il rumore degli eventi e' SEMPRE generato (400-600 finestre vere per WP non bastano per
# NSIM eventi): la sua sorgente e' la NPS, che viene dedotta da RESULTS_NAME insieme al resto.


def _parse_results_name(name):
    """(filter_type, train_template, sim_ap_from, nps_source) dedotti dal nome della cartella."""
    if name.startswith("m205_results_octopus"):
        base, tag = "optimum", name[len("m205_results_octopus"):]
    elif name.startswith("m205_results_wiener"):
        base, tag = "wiener", name[len("m205_results_wiener"):]
    else:
        raise SystemExit(f"[ERROR] RESULTS_NAME='{name}' non riconosciuto: deve iniziare per "
                         "'m205_results_octopus' o 'm205_results_wiener'.")
    nps = "clean" if tag.endswith("_npsclean") else "octopus"
    tag = tag[:-len("_npsclean")] if nps == "clean" else tag
    if base == "wiener" and tag.endswith("_R"):
        base, tag = "wiener_R", tag[:-2]
    if tag in ("", "_root"):
        train, sim = "root", None
    elif tag == "_fit":
        train, sim = "fit", None
    elif tag.startswith("_sim_"):
        train, sim = "sim", tag[len("_sim_"):]
    else:
        raise SystemExit(f"[ERROR] non so dedurre il template dal nome '{name}' (resto: '{tag}'). "
                         "Cartelle come m205_results_wiener_freq non sono supportate.")
    return base, train, sim, nps


FILTER_TYPE, TRAIN_TEMPLATE, SIM_AP_FROM, NPS_SOURCE = _parse_results_name(RESULTS_NAME)

RESULTS_DIR = os.path.join(BASE_DIR, RESULTS_NAME)
if not os.path.isdir(RESULTS_DIR):
    raise SystemExit(f"[ERROR] cartella non trovata: {RESULTS_DIR}")
# analyse_BI_* scrive "BI_results_*<tag>.csv": si prende quello che c'e' nella cartella.
_csv = glob.glob(os.path.join(RESULTS_DIR, "BI_results_*.csv"))
if not _csv:
    raise SystemExit(f"[ERROR] nessun CSV dei risultati in {RESULTS_DIR}")
BI_CSV      = _csv[0]
FILTERS_DIR = os.path.join(RESULTS_DIR, "trained_filters")
OUT_CSV     = os.path.join(RESULTS_DIR, "BI_mc_error_m205.csv")

FIT_DIR     = os.path.join(BASE_DIR, "residual_scan_bessel", "fits_octopus")
FIT_PATTERN = "bestfit_ch{ch}_wp{wp}.npy"
SIM_AP_DIR     = os.path.join(BASE_DIR, "m205_AP_sim")
SIM_AP_PATTERN = os.path.join("ch{ch}", "simAP_{gen}_ch{ch}_wp{wp}.npy")
NPS_DIR     = os.path.join(BASE_DIR, "m205_NPS_clean")
NPS_PATTERN = os.path.join("ch{ch}", "nps_ch{ch}_wp{wp}.npy")

if GEN_TEMPLATE not in ("root", "fit"):
    raise SystemExit(f"[ERROR] GEN_TEMPLATE='{GEN_TEMPLATE}' non valido: 'root' o 'fit'.")

ONLY_CHANNELS = None        # lista, oppure None/[] per tutti i canali del CSV
ONLY_WPS      = None        # lista, oppure None/[] per tutti i WP

# Parametri della simulazione (gli stessi del calcolo analitico in analyse_BI_m205.py)
NSIM        = 20_000        # eventi per popolazione; l'errore MC scala come 1/sqrt(NSIM)
CHUNK       = 2_000         # eventi generati per volta. simulate_frequency_pulses alloca sei
                            # array (n, 10000) COMPLESSI: a n=20000 sono 3.2 GB l'uno, ~19 GB in
                            # tutto, e il processo viene ucciso dall'OOM killer. Generando a
                            # blocchi il picco scende come CHUNK/NSIM (a 1000: ~1 GB).
SEED        = 1234
ACCEPTANCE  = 0.9
T_MAX       = 8e-4          # ritardo massimo del pile-up [s] (= T_MAX del BI analitico)
DETECTOR_SIGMA = 0.0        # spread di ampiezza AGGIUNTIVO al rumore. 0.0 = solo rumore, che e'
                            # l'ipotesi del calcolo analitico: metterlo > 0 rende i due non
                            # confrontabili (il rumore e' gia' incluso a parte dal simulatore).
PAIRED_NOISE = True         # True: singoli e pile-up generati con lo STESSO seed, quindi stessa
                            # traccia di rumore, stessa ampiezza e stesso rapporto r; l'unica
                            # differenza e' il ritardo del secondo impulso. E' la struttura del
                            # paper, dove i due file di posizioni condividono le posizioni delle
                            # finestre di rumore (test/create_pos_file.py). MISURATO su ch91
                            # WP15, 12 seed: NON migliora la precisione (dispersione 2.24% contro
                            # 1.87% indipendenti, compatibili) e non sposta la media (-0.1%).
                            # E' una scelta di fedelta' al paper, non un guadagno statistico.
FOLD_RATIO  = False         # True: r -> min(r, 1-r), la convenzione del J analitico (r in [0,0.5]).
                            # False: r sull'intero range della distribuzione 2beta, come il
                            # simulatore fa di default. E' la prima cosa da provare se BI_mc e
                            # BI_analitico non tornano.

WINDOW_SIZE   = 10_000
SAMPLING_RATE = 10_000
SAMPLING_TIME = WINDOW_SIZE / SAMPLING_RATE

PLOT = [(91, 15)]           # coppie (canale, WP) per cui disegnare le distribuzioni; [] = nessuna

# ═════════════════════════════════════════════════════════════════════════════
# Esecuzione: un job per coppia (canale, WP), come in analyse_BI_m205.py
# ═════════════════════════════════════════════════════════════════════════════
# Orchestratore (default): enumera le coppie, sottomette un job per ciascuna e termina.
# Worker (--worker --channel C --wp W): esegue UNA coppia e appende la sua riga al CSV.
SUBMIT_MODE       = "qsub"    # "qsub" = un job per coppia ; "local" = in sequenza (debug)
QUEUE             = "cupid"
WALLTIME          = "24:00:00"
RAM_GB            = 4         # con CHUNK=1000 il picco misurato e' ~1.5 GB
MAX_PARALLEL_JOBS = 100
SLEEP_INTERVAL    = 20        # s tra un controllo di slot e l'altro
JOB_NAME_PREFIX   = "MC"      # nome job / throttling via qstat
EXPORT_ENV        = True      # "-V" al qsub: esporta l'ambiente corrente
RESET_CSV         = True      # l'orchestratore riparte da un CSV pulito (solo header)
ENV_SETUP_LINES   = ["source /home/zanelli/LoadOctopus.sh"]
LOG_DIR           = os.path.join(RESULTS_DIR, "logs_mc")
JOBS_DIR          = os.path.join(RESULTS_DIR, "jobs_mc")

CSV_FIELDNAMES = ["channel", "wp", "vbias", "gen", "train", "nps", "filter", "BI_analytic",
                  "BI_mc", "sigma_BI", "rp", "sigma_rp", "nsim", "ratio"]


def flattop_power_factor(n: int) -> float:
    """N/sum(w^2) della finestra flattop: stessa correzione applicata dal worker alla NPS."""
    from scipy.signal.windows import flattop
    return float(n / np.sum(flattop(n) ** 2))


def full_spectrum(half):
    """Ricostruisce lo spettro completo dalla meta' indipendente salvata (DC..Nyquist).
    Per uno spettro COMPLESSO (il kernel) la meta' speculare va CONIUGATA: e' la simmetria
    hermitiana di una funzione del tempo reale. Per f1/f2, che sono reali, il coniugio non
    cambia nulla."""
    half = np.asarray(half).ravel()
    mirror = half[-2:0:-1]
    return np.concatenate([half, np.conj(mirror) if np.iscomplexobj(half) else mirror])


def root_file(channel):
    files = glob.glob(os.path.join(DATA_DIR, f"Processed_*_{MEAS_NAME}_{channel}.root"))
    if not files:
        raise RuntimeError(f"ROOT del canale {channel} non trovato")
    return files[0]


def template_pulse(channel, wp, source):
    """Template peak-normalizzato secondo `source`: "root" (medianAP di Octopus), "fit"
    (bestfit dello scan) o "sim" (AP simulato prodotto da --make-ap)."""
    if source == "fit":
        path = os.path.join(FIT_DIR, FIT_PATTERN.format(ch=channel, wp=wp))
    elif source == "sim":
        path = os.path.join(SIM_AP_DIR, SIM_AP_PATTERN.format(ch=channel, wp=wp, gen=SIM_AP_FROM))
    else:
        with uproot.open(root_file(channel)) as f:
            v = np.asarray(f[f"averagepulse_ap_wp{wp}_medianAP"].values(), dtype=float)
        return v / v.max()
    if not os.path.exists(path):
        raise RuntimeError(f"template '{source}' non trovato: {path}")
    v = np.asarray(np.load(path), dtype=float)
    return v / v.max()


def load_noise(channel, wp):
    """NPS nella convenzione del codice, dalla sorgente scelta da NPS_SOURCE. Stesse
    normalizzazioni del worker di analyse_BI_m205."""
    if NPS_SOURCE == "clean":
        path = os.path.join(NPS_DIR, NPS_PATTERN.format(ch=channel, wp=wp))
        if not os.path.exists(path):
            raise RuntimeError(f"NPS 'clean' non trovata: {path}")
        return np.asarray(np.load(path), dtype=float)    # gia' corretta per la finestra
    with uproot.open(root_file(channel)) as f:
        nps = np.asarray(f[f"averagepowerspectrum_noise_wp{wp}_medianpower"].values(), dtype=float)
    nps = np.concatenate([nps, nps[-2:0:-1]])
    return nps * flattop_power_factor(WINDOW_SIZE) * WINDOW_SIZE ** 2 / SAMPLING_TIME


def ap_statistics(channel, wp):
    """(N, ampiezza) degli impulsi che formano l'AP vero: N dalle entries della TH2D APdistro,
    l'ampiezza come mediana di maxminusbaseline.amplitude sugli eventi selezionati
    (crosscorr_signal_wp<wp>.pass). NB: sono impulsi LED di calibrazione, con ampiezza di ORDINI
    DI GRANDEZZA maggiore del segnale di ROI: simulare l'AP con signal_amp darebbe impulsi
    dominati dal rumore, e una mediana peak-normalizzata senza senso."""
    with uproot.open(root_file(channel)) as f:
        n_ev = int(round(f[f"averagepulse_ap_wp{wp}_APdistro"].member("fEntries") / WINDOW_SIZE))
        sel = f[f"crosscorr_signal_wp{wp}"]["pass"].array(library="np")
        amp = f["maxminusbaseline"]["amplitude"].array(library="np")[sel]
    return n_ev, float(np.median(amp))


def load_row_inputs(channel, wp):
    """(template di GENERAZIONE, nps)."""
    return template_pulse(channel, wp, GEN_TEMPLATE), load_noise(channel, wp)


def simulate_psd(S, nps, w, H_unit, f1, f2, signal_amp, dt_max, seed):
    """PSD (parametro di forma) di NSIM eventi simulati: dt_max=0 -> impulsi SINGOLI,
    dt_max>0 -> PILE-UP. Il rumore e' generato con lo spettro nps del canale.

    Si genera a BLOCCHI di CHUNK eventi: il generatore alloca sei array (n, 10000) complessi,
    quindi a NSIM intero il processo viene ucciso per memoria. Il seed di ogni blocco e'
    seed + indice del blocco, cosi' con PAIRED_NOISE singoli e pile-up restano appaiati blocco
    per blocco (stesso rumore, stessa ampiezza, stesso rapporto r)."""
    out = []
    done = 0
    for k in range(0, NSIM, CHUNK):
        n = min(CHUNK, NSIM - k)
        fpulses, *_ = sim.simulate_frequency_pulses(S, nps, DETECTOR_SIGMA, w, nsim=n,
                                                    seed=seed + k // CHUNK,
                                                    signal_scale=signal_amp, dt_max=dt_max,
                                                    fold_ratio=FOLD_RATIO)
        pulses = np.fft.ifft(fpulses, axis=1).real.astype(np.float32)
        del fpulses
        dataset = ds.NumpyDataset(pulses)
        dataset.win_length = pulses.shape[1]    # get_PSD_interpole legge win_length dal dataset
        psd, _, _ = an.get_PSD_interpole(dataset, H_unit, f1, f2)
        out.append(np.asarray(psd).ravel())
        del pulses, dataset
        done += n
    return np.concatenate(out)


def train_kernel(channel, wp, nps, row):
    """Kernel del TRAINING, ricalcolato da TRAIN_TEMPLATE secondo FILTER_TYPE:
    H = S*/NPS ("optimum"), W = S*/(|S|^2+lambda*NPS) ("wiener"), R(f)*W ("wiener_R")."""
    tpl = template_pulse(channel, wp, TRAIN_TEMPLATE)
    S, _, H = an.compute_H(tpl, nps, np.hanning, sampling_rate=SAMPLING_RATE)
    if FILTER_TYPE == "optimum":
        return H
    S_t = torch.as_tensor(S, dtype=torch.cfloat)
    n_t = torch.as_tensor(nps, dtype=torch.float32)
    W = an.compute_W_torch(S_t, n_t, torch.tensor(float(row["lambda_wiener"])))
    if FILTER_TYPE == "wiener_R":
        W = an.reliability_R(S_t, n_t, int(float(row["n_events"])), float(row["beta_R"])) * W
    return W.detach().cpu().numpy()


def run_pair(channel, wp, row):
    """BI Monte Carlo + incertezza per una coppia (canale, WP)."""
    signal_amp = float(row["signal_amp"])
    meanpulse, nps = load_row_inputs(channel, wp)
    S, w, H_unit = an.compute_H(meanpulse, nps, np.hanning, sampling_rate=SAMPLING_RATE)

    # Il KERNEL applicato ai dati e' quello del TRAINING, che puo' venire da un template diverso
    # da quello di generazione: si ricalcola da TRAIN_TEMPLATE e si verifica contro il .npy salvato.
    H_train = train_kernel(channel, wp, nps, row)
    kern = os.path.join(FILTERS_DIR, f"kernel_ch{channel}_wp{wp}.npy")
    if os.path.exists(kern):
        saved = full_spectrum(np.load(kern))
        rel = np.abs(saved - H_train).max() / max(np.abs(H_train).max(), 1e-300)
        if rel > 1e-5:
            raise RuntimeError(f"il kernel salvato non corrisponde a TRAIN_TEMPLATE='{TRAIN_TEMPLATE}',"
                               f" FILTER_TYPE='{FILTER_TYPE}', NPS_SOURCE='{NPS_SOURCE}' "
                               f"(scarto relativo {rel:.1e}): controlla RESULTS_DIR")
    H_unit = H_train                       # i filtri vanno applicati col LORO kernel

    f1 = full_spectrum(np.load(os.path.join(FILTERS_DIR, f"f1_ch{channel}_wp{wp}.npy")))
    f2 = full_spectrum(np.load(os.path.join(FILTERS_DIR, f"f2_ch{channel}_wp{wp}.npy")))

    psd_single = simulate_psd(S, nps, w, H_unit, f1, f2, signal_amp, 0.0, SEED)
    psd_pileup = simulate_psd(S, nps, w, H_unit, f1, f2, signal_amp, T_MAX,
                              SEED if PAIRED_NOISE else SEED + 1)

    cut = np.percentile(psd_single, 100 - ACCEPTANCE * 100)
    rp = float(np.mean(psd_pileup < cut))              # frazione di pile-up RIGETTATA
    bi_mc = fn.K * (1.0 - rp)
    sigma_rp, sigma_bi = an.compute_BI_uncertainty(psd_single, psd_pileup, ACCEPTANCE, rp)
    return dict(rp=rp, sigma_rp=sigma_rp, BI_mc=bi_mc, sigma_BI=sigma_bi,
                cut=cut, psd_single=psd_single, psd_pileup=psd_pileup)


def plot_pair(channel, wp, res, bi_analytic):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    lo = min(res["psd_single"].min(), res["psd_pileup"].min())
    hi = max(res["psd_single"].max(), res["psd_pileup"].max())
    bins = np.linspace(lo, hi, 120)
    ax.hist(res["psd_single"], bins=bins, histtype="step", lw=1.5, label="single pulses")
    ax.hist(res["psd_pileup"], bins=bins, histtype="step", lw=1.5, label="pile-up")
    ax.axvline(res["cut"], color="k", ls="--", lw=1.2,
               label=f"cut at {ACCEPTANCE:.0%} acceptance")
    ax.set_xlabel("pulse-shape parameter")
    ax.set_ylabel(f"events / {NSIM}")
    ax.set_title(f"m205 Ch{channel} WP{wp} — BI(MC) = {res['BI_mc']:.3e} ± {res['sigma_BI']:.1e}"
                 f"   (analytic {bi_analytic:.3e})")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"BI_mc_psd_ch{channel}_wp{wp}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"   distribuzioni -> {out}")


def make_sim_ap(rows, gen):
    """Genera, per ogni (canale, WP), un AP SIMULATO mediando lo STESSO numero di impulsi
    dell'AP vero (N dalle entries dell'APdistro). Gli impulsi sono generati dal template di
    GEN_TEMPLATE piu' rumore gaussiano con lo spettro del canale, ognuno normalizzato al proprio
    massimo come nella costruzione reale, poi si prende la MEDIANA.

    Serve a rompere l'auto-consistenza: addestrando i filtri su questo AP e valutandoli su eventi
    generati dal template originale, il rumore di template del training e' una realizzazione
    DIVERSA da quella della valutazione (cfr. il paper, sez. 4.5).

    ponytail: non si simula il jitter di allineamento (gli impulsi generati sono gia' allineati).
    Il paper nota che le imperfezioni di allineamento allargano l'AP vero: e' un effetto in piu'
    che qui non c'e'. Aggiungerlo = spostare ogni impulso di un delta casuale prima della mediana."""
    os.makedirs(SIM_AP_DIR, exist_ok=True)
    print(f"AP simulati da template '{gen}' -> {SIM_AP_DIR}\n")
    print(f"{'ch':>4s} {'wp':>3s} {'N':>4s} {'amp [V]':>9s} {'max|sim-AP|':>13s} {'RMS base':>11s}")
    for r in rows:
        ch, wp = int(r["channel"]), int(r["wp"])
        try:
            meanpulse, nps = template_pulse(ch, wp, gen), load_noise(ch, wp)
            # per generare impulsi nel TEMPO serve la FFT del template NON finestrato: compute_H
            # restituisce FFT(template * hanning), e la finestra verrebbe poi applicata una
            # seconda volta dal training.
            S_raw = np.fft.fft(meanpulse)
            w = 2 * np.pi * np.fft.fftfreq(len(meanpulse), 1.0 / SAMPLING_RATE)
            n_ev, ap_amp = ap_statistics(ch, wp)
            fpulses, *_ = sim.simulate_frequency_pulses(S_raw, nps, DETECTOR_SIGMA, w, nsim=n_ev,
                                                        seed=SEED + 1000 * ch + wp,
                                                        signal_scale=ap_amp, dt_max=0.0)
            pulses = np.fft.ifft(fpulses, axis=1).real
            pulses /= pulses.max(axis=1, keepdims=True)      # come la costruzione reale dell'AP
            ap = np.median(pulses, axis=0)
            ap /= ap.max()
            out = os.path.join(SIM_AP_DIR, SIM_AP_PATTERN.format(ch=ch, wp=wp, gen=gen))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.save(out, ap)
            print(f"{ch:>4d} {wp:>3d} {n_ev:>4d} {ap_amp:9.4f} {np.abs(ap - meanpulse).max():13.2e}"
                  f" {ap[:4000].std():11.2e}   -> {os.path.relpath(out, BASE_DIR)}")
        except Exception as e:
            tag = "[INFO] salto" if "non trovato" in str(e) else "[ERROR]"
            print(f"{ch:>4d} {wp:>3d}   {tag} {e}")


# ═════════════════════════════════════════════════════════════════════════════
# CSV condiviso fra i job
# ═════════════════════════════════════════════════════════════════════════════
def init_csv(path):
    """Crea il CSV (sovrascrivendolo) con la sola riga di header."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDNAMES).writeheader()


def append_row_to_csv(path, row):
    """Appende una riga in modo sicuro fra processi concorrenti (flock)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            if f.tell() == 0:
                w.writeheader()
            w.writerow({k: row.get(k) for k in CSV_FIELDNAMES})
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ═════════════════════════════════════════════════════════════════════════════
# Orchestratore: un job per coppia (canale, WP)
# ═════════════════════════════════════════════════════════════════════════════
def freeze_script():
    """Copia questo file in RESULTS_DIR e fa puntare i job alla COPIA, con BASE_DIR congelato.

    Il job contiene solo `python <file> --worker ...`, letto QUANDO IL JOB PARTE: senza la
    copia, cambiare la config per lanciare un'altra campagna toccherebbe anche i job ancora
    in coda. La copia sta in un'altra cartella, quindi os.path.dirname(__file__) darebbe la
    cartella dei risultati: BASE_DIR va congelato al valore vero."""
    global SCRIPT_PATH
    os.makedirs(RESULTS_DIR, exist_ok=True)
    dst = os.path.join(RESULTS_DIR, "_" + os.path.basename(SCRIPT_PATH))
    src, n = re.subn(r"^BASE_DIR(\s*)= os\.path\.dirname\(os\.path\.abspath\(__file__\)\)",
                     lambda m: f'BASE_DIR{m.group(1)}= {BASE_DIR!r}'
                               '   # congelato da freeze_script(): la copia sta altrove',
                     open(SCRIPT_PATH).read(), count=1, flags=re.M)
    if n != 1:
        raise RuntimeError("freeze_script: non trovo la riga di BASE_DIR da congelare")
    with open(dst, "w") as fh:
        fh.write(src)
    SCRIPT_PATH = dst
    print(f"[INFO] config congelata: i job useranno {os.path.relpath(dst, BASE_DIR)}")


def make_job_lines(channel, wp):
    """Corpo dello script di job: rilancia questo stesso file in modalita' worker."""
    return [f"cd {BASE_DIR}"] + ENV_SETUP_LINES + [
        f"{sys.executable} {SCRIPT_PATH} --worker --channel {channel} --wp {wp}"]


def create_sh(lines):
    os.makedirs(JOBS_DIR, exist_ok=True)
    fd, path = tempfile.mkstemp(suffix=".sh", dir=JOBS_DIR)
    with os.fdopen(fd, "w") as f:
        f.write("#!/bin/bash\n" + "\n".join(lines) + "\n")
    os.chmod(path, 0o755)
    return path


def running_job_count():
    user = os.environ.get("USER", "")
    try:
        r = subprocess.run(f"qstat -u {user} | grep '{JOB_NAME_PREFIX}' | wc -l",
                           shell=True, capture_output=True, text=True)
        return int(r.stdout.strip() or 0)
    except Exception as e:
        print("[WARN] qstat fallito:", e)
        return 0


def wait_for_slot():
    while running_job_count() >= MAX_PARALLEL_JOBS:
        print(f"Max job paralleli ({MAX_PARALLEL_JOBS}). Attendo {SLEEP_INTERVAL}s...")
        time.sleep(SLEEP_INTERVAL)


def submit_task(task_key, sh_file):
    job_name = f"{JOB_NAME_PREFIX}{task_key}"[:15]     # PBS limita la lunghezza del nome
    cmd = (f"qsub -N {job_name} {'-V ' if EXPORT_ENV else ''}-q {QUEUE} "
           f"-o localhost:{LOG_DIR}/ -e localhost:{LOG_DIR}/ "
           f"-l walltime={WALLTIME} -l mem={RAM_GB}G {sh_file}")
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    except Exception as e:
        print(f"[ERROR] eccezione su qsub per {task_key}: {e}")
        return None
    if r.returncode != 0:
        print(f"[ERROR] qsub fallito per {task_key}. stderr:\n{r.stderr}")
        return None
    jobid = (r.stdout.strip() or "").split(".")[0]
    if not jobid.isdigit():
        print(f"[ERROR] job id non valido per {task_key}: '{r.stdout.strip()}'")
        return None
    print(f"[OK] sottomesso {task_key} (job {jobid})")
    return jobid


# ═════════════════════════════════════════════════════════════════════════════
# Worker: UNA coppia (canale, WP)
# ═════════════════════════════════════════════════════════════════════════════
def run_worker(channel, wp):
    rows = [r for r in csv.DictReader(open(BI_CSV))
            if int(r["channel"]) == channel and int(r["wp"]) == wp]
    if not rows:
        print(f"[ERROR] ch {channel} wp {wp}: riga non trovata in {BI_CSV}")
        return
    r = rows[0]
    bi_an = float(r["BI"])
    try:
        res = run_pair(channel, wp, r)
    except Exception as e:
        print(f"[ERROR] ch {channel} wp {wp}: {e}")
        return
    append_row_to_csv(OUT_CSV, dict(
        channel=channel, wp=wp, vbias=r["vbias"], gen=GEN_TEMPLATE, train=TRAIN_TEMPLATE,
        nps=NPS_SOURCE, filter=FILTER_TYPE, BI_analytic=bi_an, BI_mc=res["BI_mc"],
        sigma_BI=res["sigma_BI"], rp=res["rp"], sigma_rp=res["sigma_rp"], nsim=NSIM,
        ratio=res["BI_mc"] / bi_an))
    if (channel, wp) in PLOT:
        plot_pair(channel, wp, res, bi_an)
    print(f"[OK] ch {channel} wp {wp}: BI_mc={res['BI_mc']:.4e} +- {res['sigma_BI']:.1e} "
          f"(analitico {bi_an:.4e}, rapporto {res['BI_mc']/bi_an:.3f})  ->  {OUT_CSV}")


def select_rows():
    rows = list(csv.DictReader(open(BI_CSV)))
    if ONLY_CHANNELS:
        rows = [r for r in rows if int(r["channel"]) in ONLY_CHANNELS]
    if ONLY_WPS:
        rows = [r for r in rows if int(r["wp"]) in ONLY_WPS]
    if not rows:
        raise SystemExit(f"[ERROR] nessuna riga selezionata in {BI_CSV}")
    return sorted(rows, key=lambda x: (int(x["channel"]), int(x["wp"])))


def main():
    import argparse
    ap_arg = argparse.ArgumentParser(description="BI per Monte Carlo, con incertezza (m205).")
    ap_arg.add_argument("--make-ap", nargs="*", metavar="TEMPLATE", default=None,
                        help="genera gli AP SIMULATI (stesso N dell'originale) invece del BI. "
                             "Uno o piu' template di generazione, es. --make-ap root fit "
                             "(default: GEN_TEMPLATE)")
    ap_arg.add_argument("--worker", action="store_true",
                        help="esegue UNA coppia (canale, WP): e' cosi' che partono i job")
    ap_arg.add_argument("--channel", type=int, help="canale (richiesto con --worker)")
    ap_arg.add_argument("--wp", type=int, help="working point (richiesto con --worker)")
    args = ap_arg.parse_args()

    if args.worker:
        if args.channel is None or args.wp is None:
            sys.exit("[ERROR] --worker richiede --channel e --wp")
        run_worker(args.channel, args.wp)
        return

    rows = select_rows()
    if args.make_ap is not None:
        for gen in (args.make_ap or [GEN_TEMPLATE]):
            make_sim_ap(rows, gen)
        return

    print(f"Set: {RESULTS_NAME}")
    print(f"  dedotto dal nome -> filtro '{FILTER_TYPE}', training su '{TRAIN_TEMPLATE}'"
          + (f" ({SIM_AP_FROM})" if SIM_AP_FROM else "") + f", NPS '{NPS_SOURCE}'")
    print(f"  eventi generati da '{GEN_TEMPLATE}'"
          + ("  [auto-consistente: e' anche il template del training]"
             if GEN_TEMPLATE == TRAIN_TEMPLATE else "  [incrociato]"))
    print(f"  {len(rows)} coppie (canale, WP), NSIM={NSIM}, chunk={CHUNK}, "
          f"paired_noise={PAIRED_NOISE}, fold_ratio={FOLD_RATIO}, "
          f"detector_sigma={DETECTOR_SIGMA}\n")

    tasks = [(int(r["channel"]), int(r["wp"])) for r in rows]

    if SUBMIT_MODE == "local":
        print("[INFO] SUBMIT_MODE='local': eseguo i task in sequenza (no qsub).\n")
        if RESET_CSV:
            init_csv(OUT_CSV)
        for ch, wp in tasks:
            run_worker(ch, wp)
        print(f"\nFatto. Risultati in {OUT_CSV}")
        return

    # Le cartelle devono esistere PRIMA del qsub: PBS scrive stdout/stderr in LOG_DIR e se
    # non c'e' il job va subito in stato di errore senza eseguire niente.
    for d in (RESULTS_DIR, LOG_DIR, JOBS_DIR):
        os.makedirs(d, exist_ok=True)

    # I job devono usare una COPIA di questo file, non l'originale: vedi freeze_script().
    freeze_script()
    if RESET_CSV:
        init_csv(OUT_CSV)

    submitted, failed = 0, []
    for ch, wp in tasks:
        wait_for_slot()
        task_key = f"{ch}_{wp}"
        jobid = None
        for _ in range(3):                     # retry per errori transitori di qsub
            jobid = submit_task(task_key, create_sh(make_job_lines(ch, wp)))
            if jobid is not None:
                break
            time.sleep(SLEEP_INTERVAL)
        if jobid is not None:
            submitted += 1
        else:
            failed.append(task_key)
            print(f"[WARN] impossibile sottomettere {task_key} dopo 3 tentativi.")

    print("\n" + "=" * 65)
    print(f"  {submitted}/{len(tasks)} job sottomessi.")
    if failed:
        print(f"  {len(failed)} NON sottomessi: {failed}")
    print(f"  Ogni job scrivera' la sua riga in: {OUT_CSV}")
    print(f"  Log dei job in: {LOG_DIR}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
