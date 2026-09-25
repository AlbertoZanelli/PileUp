"""
simulate_BI_error_m205.py
=========================
BI MONTE CARLO dei filtri gia' addestrati (e, se richiesto, dello stimatore di verosimiglianza M),
su impulsi SIMULATI. Non addestra niente: legge i filtri salvati dai programmi di training e li
applica a eventi generati qui.

COSA FA, IN BREVE
  Per ogni punto (canale, WP) e per ogni seed:
    1. genera NSIM impulsi SINGOLI e NSIM impulsi di PILE-UP dal template GEN_TEMPLATE (la
       "verita'") piu' rumore con lo spettro della NPS;
    2. li passa per i filtri addestrati di ogni cartella di risultati (RESULTS_NAME e COMPARE)
       e, con LIKELIHOOD = True, anche per lo stimatore M -- gli STESSI eventi per tutti;
    3. per ogni algoritmo: taglio che accetta il 90% dei singoli, frazione di pile-up che
       sopravvive, BI = K x sopravvissuti, con la sua incertezza statistica;
    4. scrive una riga per seed nel CSV del Monte Carlo di ogni cartella.
  Il BI del programma di training e' ANALITICO (J x K). Questo e' quello SIMULATO: e' il numero da
  riportare (come fa il paper) e quello su cui si giudica ogni confronto fra algoritmi.
  NB: sigma_BI e' l'errore STATISTICO del Monte Carlo (scende alzando NSIM), non l'incertezza fisica.

COME SI USA
  1. si scelgono le cartelle (RESULTS_NAME, COMPARE), il template di generazione (GEN_TEMPLATE),
     se aggiungere M (LIKELIHOOD) e quanti seed (N_SEEDS), nella CONFIGURAZIONE qui sotto;
  2.    KMP_DUPLICATE_LIB_OK=TRUE python simulate_BI_error_m205.py
     sottomette un job PBS per ogni punto e termina (SUBMIT_MODE = "local": li esegue in fila);
  3. il confronto fra cartelle si fa poi con compare_templates_m205.py, che legge i CSV scritti qui.

MAPPA DELLE FUNZIONI (chi chiama chi)
  main()                                        punto d'ingresso
   |-- modalita' ORCHESTRATORE (default)
   |     select_rows()                          quali punti simulare
   |     freeze_script()                        copia congelata di questo file per i job
   |     create_sh(make_job_lines())            lo script di un job
   |     submit_task()  <- wait_for_slot() <- running_job_count()      qsub, con limite di job
   |
   '-- modalita' WORKER (--worker --channel C --wp W), cioe' dentro un job
         run_worker(ch, wp)                     UN punto, tutti i seed
          |-- make_likelihood()                 lo stimatore M del punto (se LIKELIHOOD)
          |-- run_pair(..., seed)               UN seed: eventi + BI di tutti gli algoritmi
          |     |-- load_row_inputs()           template di generazione + NPS
          |     |     |-- template_pulse()      legge un template (fit, AP simulato, root)
          |     |     '-- load_noise()          legge la NPS
          |     |-- load_filters()              filtri addestrati di una cartella
          |     |     |-- train_kernel()        ricalcola il kernel del training (controllo)
          |     |     '-- full_spectrum()       spettro completo dalla meta' salvata
          |     |-- simulate_psd()              genera gli eventi e li passa per tutti
          |     '-- bi_from()                   taglio, BI e incertezza di un algoritmo
          |-- append_row_to_csv()               scrive una riga (sicuro fra job paralleli)
          |     |-- read_rows()                 legge il CSV controllando lo schema
          |     '-- row_key()                   identita' di una riga (punto, gen, seed)
          '-- plot_pair()                       istogrammi di un punto (solo per PLOT)
  Da dove vengono le cartelle: FOLDERS = folder_info() di ogni nome, che deduce filtro, template
  di training e NPS dal nome stesso (_parse_results_name()). likelihood_dir() da'
  la cartella di M.
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
from src.pileup_likelihood import PileupLikelihood

SCRIPT_PATH = os.path.abspath(__file__)
DATA_DIR    = os.path.join(BASE_DIR, "Processed")
MEAS_NAME   = "000205"

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE: cosa simulare
# ═════════════════════════════════════════════════════════════════════════════
# 1) RESULTS_NAME: la cartella di risultati di un programma di training (analyse_BI_m205.py per
#    il filtro ottimo, analysis_BI_m205_wiener_regolarized.py per Wiener). Dal NOME si deducono
#    tipo di filtro, template del training e sorgente della NPS (folder_info), cosi' non si
#    possono sbagliare a mano; il controllo sul kernel in load_filters verifica comunque.
#      m205_results_octopus_APsimfit10000led_npsclean         -> filtro ottimo
#      m205_results_wiener_APsimfit10000led_npsclean_swna1    -> Wiener + penalita' su s
#    RESULTS_NAME decide anche i punti da simulare, il template di M e dove vanno log e job.
RESULTS_NAME = "m205_results_octopus_APsimfit10000led_npsclean"

# 1b) COMPARE: altre cartelle da simulare INSIEME a RESULTS_NAME. Per ogni seed gli eventi si
#     generano UNA volta e passano per i filtri di TUTTE le cartelle, ognuna scrive nel proprio
#     CSV: eventi identici per costruzione (e' su questo che si regge l'errore appaiato di Delta BI
#     in compare_templates) e la generazione, ~70% del tempo, si paga una volta. [] = una sola.
COMPARE = []

# 2) GEN_TEMPLATE: il template che GENERA gli eventi, cioe' cosa si considera la verita'.
#    "fit"  -> il bestfit dello scan (liscio, senza rumore finito-N): la scelta delle campagne;
#    "root" -> il medianAP di Octopus.
#    Se coincide con il template di training il conto e' auto-consistente (solo validazione);
#    se e' diverso misura quanto costa addestrare su un template imperfetto.
GEN_TEMPLATE = "fit"

# 3) LIKELIHOOD: un SECONDO algoritmo sugli STESSI eventi, lo stimatore M
#    (src/pileup_likelihood.py; derivazione in tesi_note/stimatore_verosimiglianza.pdf).
#    Per ogni evento M dice quanto e' piu' verosimile un pile-up di due impulsi (a qualsiasi dt e r,
#    pesati con la loro distribuzione) di un impulso singolo. Non ha filtri addestrati: usa solo il
#    TEMPLATE DI TRAINING di RESULTS_NAME (lo stesso dei suoi filtri, cosi' M non sa piu' di loro
#    sulla forma vera) e la NPS. Il taglio si fissa come per il rapporto dei massimi: al 90% dei
#    singoli simulati dello stesso seed.
#    M e' un SET come le cartelle di training e scrive in una cartella sua,
#        m205_results_likelihood_<template di training di RESULTS_NAME>[_npsclean]
#    con lo stesso CSV del Monte Carlo (stesso nome, stesse colonne, una riga per seed) e un
#    BI_results_*.csv con una riga per punto (BI analitico = nan: M non ne ha uno). Quindi
#    compare_templates_m205.py lo legge come qualsiasi altro set, e il Delta BI e' appaiato.
#    Tipico: RESULTS_NAME = cartella Wwna, COMPARE = [cartella OF], LIKELIHOOD = True -> un solo
#    lancio, tre set sugli stessi impulsi. I CSV delle cartelle di training non cambiano.
#    Costo: ~1.3 ms per evento (~4 h per job con due cartelle, M e N_SEEDS = 50; walltime 24 h).
LIKELIHOOD = True

ONLY_CHANNELS = None        # lista, oppure None/[] per tutti i canali di RESULTS_NAME
ONLY_WPS      = None        # lista, oppure None/[] per tutti i WP
PLOT          = [(91, 15)]  # punti (canale, WP) di cui salvare gli istogrammi; [] = nessuno

# ── Parametri della simulazione ───────────────────────────────────────────────────────────────
NSIM    = 50_000            # eventi per popolazione; l'errore MC scala come 1/sqrt(NSIM)
CHUNK   = 500               # eventi generati per volta: tutti insieme finirebbero la memoria (il
                            # generatore alloca sei array (n, 10000) complessi). MISURATO: 500 e'
                            # anche il piu' veloce (250 -> 100 s, 500 -> 99 s, 2000 -> 145 s).
                            # CAMBIA GLI EVENTI (vedi simulate_psd): uguale in tutte le campagne
                            # che si confrontano.
SEED    = 1234
N_SEEDS = 50                 # ripetizioni INDIPENDENTI: il seed SEED + i genera una simulazione
                            # completa, una riga per seed. Cartelle simulate con gli stessi seed
                            # vedono gli STESSI eventi: il Delta BI seed per seed contiene la
                            # covarianza e la sua dispersione e' l'errore (compare_templates).
ACCEPTANCE = 0.9            # frazione di singoli che il taglio accetta
T_MAX      = 8e-4           # ritardo massimo del pile-up [s] (= T_MAX del BI analitico)
# Scelte FISSE della generazione (erano interruttori, sempre a questi valori; cambiarle genera
# eventi diversi e rende i risultati non confrontabili con le campagne fatte):
#   - nessuno spread di ampiezza oltre al rumore (DETECTOR_SIGMA = 0): e' l'ipotesi del BI analitico;
#   - singoli e pile-up con lo STESSO seed: stesso rumore, ampiezza e r, cambia solo il ritardo del
#     secondo impulso, come nel paper (misurato su ch91 WP15: non cambia media ne' precisione);
#   - r su tutto [0, 1] della distribuzione 2beta, senza ripiegarlo in [0, 0.5].

# Un seed solo -> il CSV di sempre; piu' seed -> un CSV a parte, col numero di seed nel nome.
OUT_NAME = "BI_mc_error_m205.csv" if N_SEEDS == 1 else f"BI_mc_error_m205_seeds{N_SEEDS}.csv"

# ── Esecuzione sul cluster PBS: un job per punto ────────────────────────────────────────────────
SUBMIT_MODE       = "qsub"  # "qsub" = un job per punto ; "local" = in sequenza, qui (debug)
QUEUE             = "cupid"
WALLTIME          = "24:00:00"
RAM_GB            = 2       # con CHUNK = 500 il picco e' ben sotto
MAX_PARALLEL_JOBS = 150
SLEEP_INTERVAL    = 20      # s fra due controlli degli slot liberi
JOB_NAME_PREFIX   = "MC"    # nome dei job, e filtro di qstat per contarli
EXPORT_ENV        = True    # "-V" al qsub: il job eredita l'ambiente
OVERWRITE         = True    # True: una riga con la stessa identita' (canale, WP, gen, seed) viene
                            # SOSTITUITA sul posto: rilanciare un punto lo aggiorna, non lo duplica.
ENV_SETUP_LINES   = ["source /home/zanelli/LoadOctopus.sh"]

# ── File di ingresso ─────────────────────────────────────────────────────────────────────────
FIT_DIR        = os.path.join(BASE_DIR, "residual_scan_bessel", "fits_octopus")
FIT_PATTERN    = "bestfit_ch{ch}_wp{wp}.npy"
SIM_AP_DIR     = os.path.join(BASE_DIR, "m205_AP_sim")        # build_simAP_injected_m205.py
SIM_AP_PATTERN = os.path.join("ch{ch}", "simAP_{gen}_ch{ch}_wp{wp}.npy")
NPS_DIR        = os.path.join(BASE_DIR, "m205_NPS_clean")     # build_NPS_clean_m205.py
NPS_PATTERN    = os.path.join("ch{ch}", "nps_ch{ch}_wp{wp}.npy")
WINDOW_SIZE    = 10_000
SAMPLING_RATE  = 10_000
SAMPLING_TIME  = WINDOW_SIZE / SAMPLING_RATE

# ── File di uscita ───────────────────────────────────────────────────────────────────────────
# Il CSV del Monte Carlo, uguale per tutte le cartelle (anche quella di M). Una riga = un punto,
# un template di generazione, un seed.
CSV_FIELDNAMES = ["channel", "wp", "vbias", "gen", "seed", "train", "nps", "filter",
                  "BI_analytic", "BI_mc", "sigma_BI", "rp", "sigma_rp", "nsim", "ratio"]
# Il BI_results_*.csv della cartella di M, una riga per punto: le colonne che compare_templates
# legge da ogni set. BI (analitico) = nan: M si giudica solo sul Monte Carlo.
LIK_BI_FIELDNAMES = ["channel", "wp", "vbias", "signal_amp", "sigma_analytic", "SNR", "BI", "template"]
ROW_KEY = ("channel", "wp", "gen", "seed")                   # identita' di una riga, vedi row_key


# ═════════════════════════════════════════════════════════════════════════════
# LE CARTELLE DI RISULTATI: tutto si deduce dal nome
# ═════════════════════════════════════════════════════════════════════════════
# Suffissi che il programma Wiener aggiunge in coda e che toccano SOLO il training (lambda fissa,
# fase libera, lr minimo): qui non servono, si tolgono prima di leggere template e NPS.
TRAIN_ONLY = r"(_lam[0-9.eE+-]+|_ph|_eta[0-9.eE+-]+)+$"


def _parse_results_name(name):
    """Dal nome della cartella: (tipo di filtro, template di training, AP simulato, sorgente NPS).
        m205_results_octopus_...  -> "optimum"      m205_results_wiener_..._R -> "wiener_R"
        m205_results_wiener_...   -> "wiener"
    Il suffisso della penalita' su s (_swna<w>, _sbar<s>) agisce solo nel training e non cambia
    niente qui: si toglie. Usata da folder_info."""
    if name.startswith("m205_results_octopus"):
        base, tag = "optimum", name[len("m205_results_octopus"):]
    elif name.startswith("m205_results_wiener"):
        base, tag = "wiener", name[len("m205_results_wiener"):]
    else:
        raise SystemExit(f"[ERROR] '{name}' non e' una cartella di training: deve iniziare per "
                         "'m205_results_octopus' o 'm205_results_wiener'. (Le cartelle "
                         "m205_results_likelihood_* le SCRIVE questo programma, non le legge.)")
    tag = re.sub(TRAIN_ONLY, "", tag)
    tag = re.sub(r"_(sbar|swna)[0-9.eE+-]+$", "", tag)
    nps = "clean" if tag.endswith("_npsclean") else "octopus"
    tag = tag[:-len("_npsclean")] if nps == "clean" else tag
    if base == "wiener" and tag.endswith("_R"):
        base, tag = "wiener_R", tag[:-2]
    if tag in ("", "_root"):
        train, sim_from = "root", None
    elif tag == "_fit":
        train, sim_from = "fit", None
    elif tag.startswith(("_APsim", "_APreal")):
        train, sim_from = "sim", tag[1:]
    elif tag.startswith("_sim_"):
        train, sim_from = "sim", tag[len("_sim_"):]          # nomenclatura vecchia, letta ancora
    else:
        raise SystemExit(f"[ERROR] non so dedurre il template dal nome '{name}' (resto: '{tag}').")
    return base, train, sim_from, nps


def folder_info(name):
    """Tutto quello che serve di una cartella: percorsi (CSV dei risultati del training, filtri
    addestrati, CSV del Monte Carlo da scrivere) e cosa contiene (filtro, template, NPS).
    Costruisce FOLDERS, qui sotto."""
    filt, train, sim_from, nps = _parse_results_name(name)
    d = os.path.join(BASE_DIR, name)
    csvs = glob.glob(os.path.join(d, "BI_results_*.csv"))
    if not csvs:
        raise SystemExit(f"[ERROR] cartella o CSV dei risultati mancante: {d}")
    return dict(name=name, dir=d, bi_csv=csvs[0], filters_dir=os.path.join(d, "trained_filters"),
                out_csv=os.path.join(d, OUT_NAME), filter=filt, train=train, sim=sim_from, nps=nps)


if GEN_TEMPLATE not in ("root", "fit"):
    raise SystemExit(f"[ERROR] GEN_TEMPLATE='{GEN_TEMPLATE}' non valido: 'root' o 'fit'.")
# le cartelle da simulare, senza doppioni (RESULTS_NAME ripetuto in COMPARE conta una volta)
FOLDERS = [folder_info(n) for n in dict.fromkeys([RESULTS_NAME] + list(COMPARE))]
# la NPS GENERA gli eventi: con NPS diverse le cartelle non vedrebbero gli stessi eventi
if len({f["nps"] for f in FOLDERS}) > 1:
    raise SystemExit("[ERROR] COMPARE: le cartelle usano NPS diverse, gli eventi non sarebbero gli stessi")
NPS_SOURCE  = FOLDERS[0]["nps"]
RESULTS_DIR = FOLDERS[0]["dir"]                  # log, job e copia congelata
LOG_DIR     = os.path.join(RESULTS_DIR, "logs_mc")
JOBS_DIR    = os.path.join(RESULTS_DIR, "jobs_mc")


def likelihood_dir():
    """Cartella dei risultati di M: m205_results_likelihood_<template di training di RESULTS_NAME>
    [_npsclean]. Stesse regole di nome delle cartelle di training, cosi' compare_templates ne
    deduce l'etichetta come per le altre. Usata da run_worker."""
    f = FOLDERS[0]
    tag = f["train"] if f["train"] != "sim" else (
        f["sim"] if f["sim"].startswith(("APsim", "APreal")) else "sim_" + f["sim"])
    return os.path.join(BASE_DIR, "m205_results_likelihood_" + tag
                        + ("_npsclean" if f["nps"] == "clean" else ""))


# ═════════════════════════════════════════════════════════════════════════════
# INGRESSI: template, NPS, filtri
# ═════════════════════════════════════════════════════════════════════════════
def root_file(channel):
    """Il file ROOT di Octopus del canale (serve solo per template "root" e NPS "octopus")."""
    files = glob.glob(os.path.join(DATA_DIR, f"Processed_*_{MEAS_NAME}_{channel}.root"))
    if not files:
        raise RuntimeError(f"ROOT del canale {channel} non trovato")
    return files[0]


def template_pulse(channel, wp, source, sim_from=None):
    """Un template nel TEMPO, normalizzato al picco:
        "fit"  -> il bestfit dello scan;     "sim" -> l'AP simulato `sim_from`;
        "root" -> il medianAP di Octopus dal ROOT.
    Usata per il template di GENERAZIONE (load_row_inputs), di TRAINING (train_kernel) e di M
    (make_likelihood)."""
    if source == "fit":
        path = os.path.join(FIT_DIR, FIT_PATTERN.format(ch=channel, wp=wp))
    elif source == "sim":
        path = os.path.join(SIM_AP_DIR, SIM_AP_PATTERN.format(ch=channel, wp=wp, gen=sim_from))
    else:
        with uproot.open(root_file(channel)) as f:
            v = np.asarray(f[f"averagepulse_ap_wp{wp}_medianAP"].values(), dtype=float)
        return v / v.max()
    if not os.path.exists(path):
        raise RuntimeError(f"template '{source}' non trovato: {path}")
    v = np.asarray(np.load(path), dtype=float)
    return v / v.max()


def flattop_power_factor(n):
    """N / sum(w^2) della finestra flattop: la correzione che il training applica alla NPS di
    Octopus. Usata solo da load_noise con NPS "octopus"."""
    from scipy.signal.windows import flattop
    return float(n / np.sum(flattop(n) ** 2))


def load_noise(channel, wp):
    """La NPS del punto, nella convenzione del codice (quella in cui sigma_OF =
    1/sqrt(sum |S|^2/NPS)). "clean": misurata da build_NPS_clean_m205.py, gia' pronta;
    "octopus": medianpower dal ROOT con le stesse normalizzazioni del training."""
    if NPS_SOURCE == "clean":
        path = os.path.join(NPS_DIR, NPS_PATTERN.format(ch=channel, wp=wp))
        if not os.path.exists(path):
            raise RuntimeError(f"NPS 'clean' non trovata: {path}")
        return np.asarray(np.load(path), dtype=float)
    with uproot.open(root_file(channel)) as f:
        nps = np.asarray(f[f"averagepowerspectrum_noise_wp{wp}_medianpower"].values(), dtype=float)
    nps = np.concatenate([nps, nps[-2:0:-1]])
    return nps * flattop_power_factor(WINDOW_SIZE) * WINDOW_SIZE ** 2 / SAMPLING_TIME


def load_row_inputs(channel, wp):
    """Quello che GENERA gli eventi: (template GEN_TEMPLATE, NPS). Usata da run_pair."""
    return template_pulse(channel, wp, GEN_TEMPLATE), load_noise(channel, wp)


def full_spectrum(half):
    """Spettro completo dalla meta' indipendente salvata dal training (bin da DC a Nyquist).
    Per uno spettro complesso (il kernel) la meta' speculare va CONIUGATA: e' la simmetria di un
    segnale reale nel tempo. Per f1, f2 reali il coniugio non cambia niente. Usata da load_filters."""
    half = np.asarray(half).ravel()
    mirror = half[-2:0:-1]
    return np.concatenate([half, np.conj(mirror) if np.iscomplexobj(half) else mirror])


def train_kernel(f, channel, wp, nps, row):
    """Il kernel del TRAINING della cartella `f`, ricalcolato dal suo template:
        "optimum"  -> H = S*/NPS;
        "wiener"   -> W = S*/(|S|^2 + lambda NPS), con il lambda scritto nel CSV del training;
        "wiener_R" -> R(f) W.
    Il template di training puo' essere diverso da quello di generazione: e' il kernel del
    training quello che si applica agli eventi. Usata da load_filters."""
    tpl = template_pulse(channel, wp, f["train"], f["sim"])
    S, _, H = an.compute_H(tpl, nps, np.hanning, sampling_rate=SAMPLING_RATE)
    if f["filter"] == "optimum":
        return H
    S_t = torch.as_tensor(S, dtype=torch.cfloat)
    n_t = torch.as_tensor(nps, dtype=torch.float32)
    W = an.compute_W_torch(S_t, n_t, torch.tensor(float(row["lambda_wiener"])))
    if f["filter"] == "wiener_R":
        W = an.reliability_R(S_t, n_t, int(float(row["n_events"])), float(row["beta_R"])) * W
    return W.detach().cpu().numpy()


def load_filters(f, channel, wp, nps, row):
    """(kernel, f1, f2) della cartella `f` per il punto: il kernel ricalcolato da train_kernel e
    CONTROLLATO contro quello salvato (se non combaciano entro 1e-5 si ferma: vorrebbe dire che il
    nome della cartella non descrive il training), f1 e f2 letti dai .npy. Usata da run_pair."""
    H = train_kernel(f, channel, wp, nps, row)
    kern = os.path.join(f["filters_dir"], f"kernel_ch{channel}_wp{wp}.npy")
    if os.path.exists(kern):
        saved = full_spectrum(np.load(kern))
        rel = np.abs(saved - H).max() / max(np.abs(H).max(), 1e-300)
        if rel > 1e-5:
            raise RuntimeError(f"{f['name']}: il kernel salvato non corrisponde a train="
                               f"'{f['train']}', filtro '{f['filter']}', NPS '{f['nps']}' "
                               f"(scarto relativo {rel:.1e})")
    f1, f2 = (full_spectrum(np.load(os.path.join(f["filters_dir"], f"{k}_ch{channel}_wp{wp}.npy")))
              for k in ("f1", "f2"))
    return H, f1, f2


def make_likelihood(channel, wp):
    """Lo stimatore M del punto (src/pileup_likelihood.py), costruito UNA volta per punto sul
    template di TRAINING di RESULTS_NAME e sulla NPS. Usata da run_worker se LIKELIHOOD."""
    f = FOLDERS[0]
    nps = load_noise(channel, wp)
    S, w, _ = an.compute_H(template_pulse(channel, wp, f["train"], f["sim"]), nps, np.hanning,
                           sampling_rate=SAMPLING_RATE)
    return PileupLikelihood(S, nps, w, dt_max=T_MAX)


# ═════════════════════════════════════════════════════════════════════════════
# IL CUORE: generare gli eventi e calcolare il BI
# ═════════════════════════════════════════════════════════════════════════════
def simulate_psd(S, nps, w, filters, signal_amp, dt_max, seed, stats=()):
    """Genera NSIM eventi e restituisce, per ogni algoritmo, il suo discriminante evento per evento.
        dt_max = 0 -> impulsi SINGOLI;  dt_max = T_MAX -> PILE-UP.
        filters    -> (kernel, f1, f2) di ogni cartella: discriminante = rapporto dei massimi
                      dei due segnali filtrati (get_PSD_interpole);
        stats      -> altre statistiche f(impulsi) -> un numero per evento (qui: -M).
    Restituisce una lista di array, prima quelli dei filtri e poi quelli di stats, tutti calcolati
    sugli STESSI impulsi.

    Gli eventi si generano a blocchi di CHUNK con UN solo generatore per tutta la simulazione,
    passato a ogni blocco: il flusso continua senza ripetere numeri, quindi `seed` identifica
    l'intera simulazione. Singoli e pile-up usano due generatori con lo STESSO seed, che estraggono
    in parallelo: stesso rumore, stessa ampiezza, stesso r (uniform(0, 0) consuma gli stessi numeri
    di uniform(0, T_MAX)). NB: gli eventi dipendono da CHUNK, perche' dentro un blocco le estrazioni
    vanno per tipo (tutto il rumore reale, poi l'immaginario, ampiezze, r, dt). Usata da run_pair."""
    rng = np.random.default_rng(seed)
    out = [[] for _ in range(len(filters) + len(stats))]
    for k in range(0, NSIM, CHUNK):
        n = min(CHUNK, NSIM - k)
        fpulses, *_ = sim.simulate_frequency_pulses(S, nps, 0.0, w, nsim=n, seed=rng,
                                                    signal_scale=signal_amp, dt_max=dt_max,
                                                    fold_ratio=False)
        pulses = np.fft.ifft(fpulses, axis=1).real.astype(np.float32)
        del fpulses
        dataset = ds.NumpyDataset(pulses)
        dataset.win_length = pulses.shape[1]    # get_PSD_interpole legge win_length dal dataset
        # get_PSD_interpole lavora su una copia (il DataLoader impila i blocchi): pulses resta
        # intatto per l'algoritmo successivo (test/check_compare_mode_m205.py,
        # test/check_likelihood_in_mc_m205.py)
        for o, (H, f1, f2) in zip(out, filters):
            o.append(np.asarray(an.get_PSD_interpole(dataset, H, f1, f2)[0]).ravel())
        for o, stat in zip(out[len(filters):], stats):
            o.append(np.asarray(stat(pulses)).ravel())
        del pulses, dataset
    return [np.concatenate(o) for o in out]


def bi_from(psd_single, psd_pileup):
    """BI Monte Carlo di UN algoritmo, dal suo discriminante su singoli e pile-up. Il pile-up deve
    stare nella coda BASSA (vale per il rapporto dei massimi, e per -M):
        taglio = percentile che accetta ACCEPTANCE dei singoli;
        rp     = frazione di pile-up sotto il taglio (rigettata);
        BI     = K (1 - rp),
    con l'incertezza statistica di compute_BI_uncertainty (binomiale + posizione del taglio).
    Usata da run_pair per ogni algoritmo."""
    cut = np.percentile(psd_single, 100 - ACCEPTANCE * 100)
    rp = float(np.mean(psd_pileup < cut))
    sigma_rp, sigma_bi = an.compute_BI_uncertainty(psd_single, psd_pileup, ACCEPTANCE, rp)
    return dict(rp=rp, sigma_rp=sigma_rp, BI_mc=fn.K * (1.0 - rp), sigma_BI=sigma_bi,
                cut=cut, psd_single=psd_single, psd_pileup=psd_pileup)


def run_pair(channel, wp, rows, seed=SEED, likelihood=None):
    """UN punto, UN seed: genera singoli e pile-up una volta, li passa per i filtri di ogni
    cartella di FOLDERS (`rows`: la riga del punto nel CSV di training di ciascuna, stesso ordine)
    e calcola il BI di ognuna con bi_from. Restituisce la lista dei risultati, cartella per cartella.

    `likelihood` (un PileupLikelihood, o None): se c'e', gli stessi eventi passano anche per M e si
    restituisce (risultati delle cartelle, risultato di M). M e' GRANDE per il pile-up: si passa -M,
    cosi' il pile-up sta nella coda bassa come per il rapporto dei massimi e bi_from vale per
    entrambi. Usata da run_worker (e dai test)."""
    amps = {float(r["signal_amp"]) for r in rows}
    if len(amps) > 1:
        raise RuntimeError(f"signal_amp diverso fra le cartelle {amps}: eventi non confrontabili")
    signal_amp = amps.pop()
    meanpulse, nps = load_row_inputs(channel, wp)
    S, w, _ = an.compute_H(meanpulse, nps, np.hanning, sampling_rate=SAMPLING_RATE)
    filters = [load_filters(f, channel, wp, nps, r) for f, r in zip(FOLDERS, rows)]
    stats = () if likelihood is None else (lambda p: -likelihood.statistic(p),)
    singles = simulate_psd(S, nps, w, filters, signal_amp, 0.0, seed, stats)
    pileups = simulate_psd(S, nps, w, filters, signal_amp, T_MAX, seed, stats)
    out = [bi_from(s, p) for s, p in zip(singles, pileups)]
    return out if likelihood is None else (out[:-1], out[-1])


# ═════════════════════════════════════════════════════════════════════════════
# USCITE: CSV (sicuri fra job paralleli) e istogrammi
# ═════════════════════════════════════════════════════════════════════════════
def row_key(r):
    """Identita' di una riga: punto, template di generazione e seed. Due run che differiscono per
    uno di questi sono righe diverse; la stessa identita' e' la stessa riga (OVERWRITE). Una riga
    senza seed (CSV vecchi) vale come SEED. Usata da append_row_to_csv."""
    return tuple(str(r.get(k) or (SEED if k == "seed" else "")) for k in ROW_KEY)


def read_rows(path):
    """Le righe di un CSV, fermandosi se lo schema e' incoerente: un file con righe di lunghezze
    diverse dall'header (campagna ripresa dopo aver aggiunto colonne) farebbe slittare i valori
    nella colonna sbagliata senza errori. Usata da append_row_to_csv."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for n, r in enumerate(rows, start=2):
        if r.get(None) is not None or any(v is None for v in r.values()):
            raise RuntimeError(f"{path}: la riga {n} non ha lo stesso numero di campi "
                               f"dell'header. Il file mescola due schemi: le colonne slittano. "
                               f"Riscrivilo con l'header completo, o cancellalo e rifallo.")
    return rows


def append_row_to_csv(path, row, fieldnames=CSV_FIELDNAMES):
    """Scrive `row` in `path` in modo sicuro fra job concorrenti: lock esclusivo su tutta la
    lettura-modifica-riscrittura. Con OVERWRITE una riga con la stessa identita' (row_key) viene
    sostituita sul posto invece che aggiunta. Il file si riscrive con l'header corrente, quindi un
    CSV nato con meno colonne si allinea da solo. Usata da run_worker."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = {k: row.get(k) for k in fieldnames}
    with open(path, "a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            rows = read_rows(path)
            for i, r in enumerate(rows):
                if OVERWRITE and row_key(r) == row_key(new):
                    rows[i] = r | new          # tiene le colonne vecchie non piu' scritte
                    break
            else:
                rows.append(new)
            f.seek(0)
            f.truncate()
            w = csv.DictWriter(f, fieldnames=fieldnames, restval="", extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def plot_pair(folder_dir, channel, wp, res, bi_analytic):
    """Istogrammi del discriminante di singoli e pile-up di un punto, con il taglio. Solo per i
    punti in PLOT e il primo seed. Usata da run_worker."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    lo = min(res["psd_single"].min(), res["psd_pileup"].min())
    hi = max(res["psd_single"].max(), res["psd_pileup"].max())
    bins = np.linspace(lo, hi, 120)
    ax.hist(res["psd_single"], bins=bins, histtype="step", lw=1.5, label="single pulses")
    ax.hist(res["psd_pileup"], bins=bins, histtype="step", lw=1.5, label="pile-up")
    ax.axvline(res["cut"], color="k", ls="--", lw=1.2, label=f"cut at {ACCEPTANCE:.0%} acceptance")
    ax.set_xlabel("pulse-shape parameter")
    ax.set_ylabel(f"events / {NSIM}")
    ax.set_title(f"m205 Ch{channel} WP{wp} — BI(MC) = {res['BI_mc']:.3e} ± {res['sigma_BI']:.1e}"
                 f"   (analytic {bi_analytic:.3e})")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(folder_dir, f"BI_mc_psd_ch{channel}_wp{wp}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"   distribuzioni -> {out}")


# ═════════════════════════════════════════════════════════════════════════════
# WORKER: un punto (canale, WP), tutti i seed -- e' cio' che esegue ogni job
# ═════════════════════════════════════════════════════════════════════════════
def run_worker(channel, wp):
    """Legge la riga del punto nel CSV di training di ogni cartella, costruisce M se LIKELIHOOD,
    poi per ogni seed chiama run_pair e scrive:
        - una riga nel CSV del Monte Carlo di ogni cartella;
        - con M: una riga nel CSV del Monte Carlo della cartella di M (likelihood_dir) e, al primo
          seed, la riga del punto nel suo BI_results_*.csv.
    Chiamata da main (in un job, o in fila con SUBMIT_MODE = "local")."""
    rows = []
    for f in FOLDERS:
        r = [x for x in csv.DictReader(open(f["bi_csv"]))
             if int(x["channel"]) == channel and int(x["wp"]) == wp]
        if not r:
            print(f"[ERROR] ch {channel} wp {wp}: riga non trovata in {f['bi_csv']}")
            return
        rows.append(r[0])
    lik = make_likelihood(channel, wp) if LIKELIHOOD else None    # una volta per punto
    for i in range(N_SEEDS):
        seed = SEED + i
        try:
            results = run_pair(channel, wp, rows, seed, likelihood=lik)
            if lik is not None:
                results, res_m = results
        except Exception as e:
            print(f"[ERROR] ch {channel} wp {wp} seed {seed}: {e}")
            return
        if lik is not None:
            d = likelihood_dir()
            tag = os.path.basename(d)[len("m205_results_likelihood_"):]
            if i == 0:          # la riga del punto: vbias, SNR, ... come le cartelle di training
                append_row_to_csv(os.path.join(d, f"BI_results_m205_likelihood_{tag}.csv"), dict(
                    channel=channel, wp=wp, vbias=rows[0]["vbias"],
                    signal_amp=rows[0]["signal_amp"], sigma_analytic=rows[0]["sigma_analytic"],
                    SNR=rows[0]["SNR"], BI="nan", template=tag), LIK_BI_FIELDNAMES)
            append_row_to_csv(os.path.join(d, OUT_NAME), dict(
                channel=channel, wp=wp, vbias=rows[0]["vbias"], gen=GEN_TEMPLATE, seed=seed,
                train=tag, nps=FOLDERS[0]["nps"], filter="likelihood", BI_analytic="",
                BI_mc=res_m["BI_mc"], sigma_BI=res_m["sigma_BI"], rp=res_m["rp"],
                sigma_rp=res_m["sigma_rp"], nsim=NSIM, ratio=""))
            bi_y = results[0]["BI_mc"]                    # RESULTS_NAME, sugli stessi eventi
            print(f"[OK] stimatore M ch {channel} wp {wp} seed {seed}: BI_mc={res_m['BI_mc']:.4e}"
                  f"  vs {FOLDERS[0]['name']} {bi_y:.4e}  ({100*(res_m['BI_mc']/bi_y-1):+.2f}%, "
                  f"stessi eventi)")
        for f, r, res in zip(FOLDERS, rows, results):
            bi_an = float(r["BI"])
            append_row_to_csv(f["out_csv"], dict(
                channel=channel, wp=wp, vbias=r["vbias"], gen=GEN_TEMPLATE, seed=seed,
                train=f["train"], nps=f["nps"], filter=f["filter"], BI_analytic=bi_an,
                BI_mc=res["BI_mc"], sigma_BI=res["sigma_BI"], rp=res["rp"],
                sigma_rp=res["sigma_rp"], nsim=NSIM, ratio=res["BI_mc"] / bi_an))
            if i == 0 and (channel, wp) in PLOT:
                plot_pair(f["dir"], channel, wp, res, bi_an)
            print(f"[OK] {f['name']} ch {channel} wp {wp} seed {seed}: BI_mc={res['BI_mc']:.4e} "
                  f"+- {res['sigma_BI']:.1e} (analitico {bi_an:.4e}, rapporto "
                  f"{res['BI_mc']/bi_an:.3f})")


# ═════════════════════════════════════════════════════════════════════════════
# ORCHESTRATORE: un job PBS per punto
# ═════════════════════════════════════════════════════════════════════════════
def select_rows():
    """I punti da simulare: le righe del CSV di training di RESULTS_NAME, filtrate da
    ONLY_CHANNELS e ONLY_WPS. Usata da main."""
    bi_csv = FOLDERS[0]["bi_csv"]
    rows = list(csv.DictReader(open(bi_csv)))
    if ONLY_CHANNELS:
        rows = [r for r in rows if int(r["channel"]) in ONLY_CHANNELS]
    if ONLY_WPS:
        rows = [r for r in rows if int(r["wp"]) in ONLY_WPS]
    if not rows:
        raise SystemExit(f"[ERROR] nessuna riga selezionata in {bi_csv}")
    return sorted(rows, key=lambda x: (int(x["channel"]), int(x["wp"])))


def freeze_script():
    """Copia questo file in RESULTS_DIR e fa puntare i job alla COPIA, con BASE_DIR congelato.
    Un job contiene solo `python <file> --worker ...`, e il file si legge quando il job PARTE:
    senza la copia, cambiare la configurazione per un'altra campagna cambierebbe anche i job
    ancora in coda. Usata da main prima di sottomettere."""
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
    """Le righe dello script di un job: ambiente, poi questo file (la copia congelata) in modalita'
    worker per il punto. Usata da main."""
    return [f"cd {BASE_DIR}"] + ENV_SETUP_LINES + [
        f"{sys.executable} {SCRIPT_PATH} --worker --channel {channel} --wp {wp}"]


def create_sh(lines):
    """Scrive lo script .sh di un job in JOBS_DIR e ne restituisce il percorso. Usata da main."""
    os.makedirs(JOBS_DIR, exist_ok=True)
    fd, path = tempfile.mkstemp(suffix=".sh", dir=JOBS_DIR)
    with os.fdopen(fd, "w") as f:
        f.write("#!/bin/bash\n" + "\n".join(lines) + "\n")
    os.chmod(path, 0o755)
    return path


def running_job_count():
    """Quanti job di questo programma (JOB_NAME_PREFIX) sono in coda o in esecuzione. Usata da
    wait_for_slot."""
    user = os.environ.get("USER", "")
    try:
        r = subprocess.run(f"qstat -u {user} | grep '{JOB_NAME_PREFIX}' | wc -l",
                           shell=True, capture_output=True, text=True)
        return int(r.stdout.strip() or 0)
    except Exception as e:
        print("[WARN] qstat fallito:", e)
        return 0


def wait_for_slot():
    """Aspetta finche' i job attivi sono meno di MAX_PARALLEL_JOBS. Usata da main."""
    while running_job_count() >= MAX_PARALLEL_JOBS:
        print(f"Max job paralleli ({MAX_PARALLEL_JOBS}). Attendo {SLEEP_INTERVAL}s...")
        time.sleep(SLEEP_INTERVAL)


def submit_task(task_key, sh_file):
    """Sottomette uno script con qsub e restituisce il job id, o None se qsub fallisce. Usata da
    main (con fino a tre tentativi)."""
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


def main():
    """Senza argomenti: ORCHESTRATORE (stampa la configurazione, sottomette un job per punto e
    termina; con SUBMIT_MODE = "local" esegue i punti qui, in fila). Con --worker --channel C
    --wp W: esegue UN punto, ed e' cio' che fa ogni job."""
    import argparse
    ap_arg = argparse.ArgumentParser(description="BI per Monte Carlo, con incertezza (m205).")
    ap_arg.add_argument("--worker", action="store_true",
                        help="esegue UN punto (canale, WP): e' cosi' che partono i job")
    ap_arg.add_argument("--channel", type=int, help="canale (richiesto con --worker)")
    ap_arg.add_argument("--wp", type=int, help="working point (richiesto con --worker)")
    args = ap_arg.parse_args()

    if args.worker:
        if args.channel is None or args.wp is None:
            sys.exit("[ERROR] --worker richiede --channel e --wp")
        run_worker(args.channel, args.wp)
        return

    rows = select_rows()
    for f in FOLDERS:
        print(f"Set: {f['name']}")
        print(f"  dedotto dal nome -> filtro '{f['filter']}', training su '{f['train']}'"
              + (f" ({f['sim']})" if f["sim"] else "") + f", NPS '{f['nps']}'"
              + ("  [auto-consistente]" if GEN_TEMPLATE == f["train"] else "  [incrociato]"))
    if len(FOLDERS) > 1:
        print(f"  MODALITA' CONFRONTO: {len(FOLDERS)} cartelle, eventi generati UNA volta per seed "
              f"e passati per i filtri di tutte")
    if LIKELIHOOD:
        print(f"  + stimatore M sugli stessi eventi -> {os.path.relpath(likelihood_dir(), BASE_DIR)}")
    print(f"  eventi generati da '{GEN_TEMPLATE}'")
    print(f"  righe nel CSV: {'sovrascritte' if OVERWRITE else 'accodate'} "
          f"(chiave: canale, WP, gen, seed)")
    print(f"  seed: {SEED}..{SEED + N_SEEDS - 1}, uno per simulazione da {NSIM} eventi")
    print(f"  {len(rows)} punti (canale, WP) x {N_SEEDS} seed, NSIM={NSIM}, chunk={CHUNK}\n")

    tasks = [(int(r["channel"]), int(r["wp"])) for r in rows]

    if SUBMIT_MODE == "local":
        print("[INFO] SUBMIT_MODE='local': eseguo i punti in sequenza (no qsub).\n")
        for ch, wp in tasks:
            run_worker(ch, wp)
        print(f"\nFatto. Risultati in {', '.join(f['out_csv'] for f in FOLDERS)}")
        return

    # Le cartelle devono esistere PRIMA del qsub: PBS scrive stdout/stderr in LOG_DIR e se non
    # c'e' il job va subito in errore senza eseguire niente.
    for d in (RESULTS_DIR, LOG_DIR, JOBS_DIR):
        os.makedirs(d, exist_ok=True)
    freeze_script()

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
    for f in FOLDERS:
        print(f"  Ogni job scrivera' le sue righe in: {f['out_csv']}")
    if LIKELIHOOD:
        print(f"  e quelle di M in: {os.path.join(likelihood_dir(), OUT_NAME)}")
    print(f"  Log dei job in: {LOG_DIR}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
