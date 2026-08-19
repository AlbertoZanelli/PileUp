#!/usr/bin/env python3
"""
fit_one_pulse_m205.py
=====================
Fit di un average pulse (un canale, un working point) della misura m205,
col modello pole-zero convoluto con un filtro di Bessel analogico.

Scopo:
  1) si carica l'AP (peak-normalizzato) dal file ROOT;
  2) si misurano dai dati le scale di tempo (salita e coda);
  3) da quelle si costruiscono i GUESS iniziali dei parametri;
  4) si impongono i BOUND (stabilità + banda limitata);
  5) si minimizza ai minimi quadrati (least_squares) la differenza fit - dato;
  6) si disegna AP + fit + residuo.

Sono incluse DUE funzioni-modello:
  - make_pulse_pole_zero_bessel_ct : 1 zero, poli REALI + Bessel  (versione base)
  - make_pulse_bessel_general      : N zeri, poli reali + coppia complessa (CC) + Bessel

Il parametro MODEL sceglie quale usare. theta (vettore dei parametri) =
  [ t0 , zeri... , poli_reali... , (sigma_cc, omega_cc) se CC ]
Il modello ha il picco a t=0 ed è normalizzato a 1: t0 è solo la posizione del picco.

Esecuzione (non serve ROOT, basta uproot):
  KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/bin/python3.13 test/fit_one_pulse_m205.py
"""

import os
import glob
import csv
import numpy as np
from scipy.signal import besselap
from scipy.optimize import least_squares, minimize_scalar
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE  (tutto qui in testa)
# ══════════════════════════════════════════════════════════════════════════════
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # cartella PileUp
PROCESSED_DIR = os.path.join(BASE, "Processed")
MEAS_NAME = "000205"

# SOURCE = "root"  -> average pulse dal file ROOT del canale/WP (comportamento iniziale).
# SOURCE = "cfile" -> impulso da una macro ROOT .C nella cartella Processed/ (TGraph con gli
#   array Graph_fx*=tempo[s], Graph_fy*=ampiezza; 10000 campioni a 10 kHz, stessa struttura
#   degli AP). Viene peak-normalizzato a 1 e poi si procede identicamente (l'APdistro non esiste
#   per i .C non ci sono i singoli impulsi -> errore = sigma di baseline costante).
# SOURCE = "npy"   -> AP salvato come vettore .npy (es. i median AP allineati al MASSIMO di
#   build_medianAP_maxalign_m205.py). Stessa griglia degli AP ROOT (10000 campioni a 10 kHz),
#   quindi da qui in poi il flusso e' identico; l'errore per time-bin viene dai singoli impulsi
#   riallineati al massimo. Il percorso si costruisce da NPY_PATTERN con CHANNEL e WP.
SOURCE  = "npy"         # "root" | "cfile" | "npy"
CFILE   = "LED1.C"     # nome del file .C in Processed/  (solo se SOURCE="cfile")
NPY_PATTERN = "m205_AP_pulses/medianAP_maxalign_ch{ch}_wp{wp}.npy"   # solo se SOURCE="npy"

CHANNEL = 91           # canale da fittare        (SOURCE="root" o "npy")
WP      = 15           # working point (dispari)  (SOURCE="root" o "npy")
SAMPLING_RATE = 10_000  # Hz, per ricostruire l'asse dei tempi dai .npy

MODEL   = "bessel_1zero"     # "bessel_1zero"  oppure  "bessel_general"
BESSEL_ORDER = 6             # ordine del filtro di Bessel (FISSO)
FCUT         = 2500          # Hz, taglio del Bessel (FISSO)

N_REAL  = 3       # numero di poli reali
N_ZERO  = 1      # numero di zeri  (per "bessel_1zero" viene forzato a 1)
USE_CC  = False         # coppia complessa coniugata (solo per "bessel_general")

PARAM_BOUND = 31416.0  # |poli|,|zeri| <= questo [rad/s]: banda limitata -> niente
                       # oscillazioni ad alta frequenza nel fit. 31416 = 2*pi*5000 = NYQUIST:
                       # e' il limite fisico. Col vecchio 8000 (=1273 Hz) i fit a >=6 poli
                       # finivano CON DUE POLI INCOLLATI AL BOUND (verificato) -> vincolo
                       # artificiale sul fronte di salita.
T0_WINDOW   = 0.01   # t0 vincolato a ± questo [s] attorno al picco (±5 ms)

# --- Errore per time-bin dai SINGOLI IMPULSI che formano l'AP -----------------------
# USE_PULSE_ERRORS = False -> sigma di baseline COSTANTE, banda ±3σ, fit non pesato.
# USE_PULSE_ERRORS = True  -> errore per time-bin dalla dispersione dei singoli impulsi
#   (extract_AP_pulses_m205.py), NON binnata: e' l'errore vero della mediana, molto piu'
#   grande sul fronte di salita che sulla coda. WEIGHT_FIT decide se pesare il fit con 1/err.
#   Sostituisce il vecchio errore preso dall'istogramma 2D APdistro (binnato in ampiezza).
USE_PULSE_ERRORS = True
WEIGHT_FIT       = True   # peso 1/err nel fit (solo se USE_PULSE_ERRORS)
PULSE_PATTERN    = "m205_AP_pulses/pulses_ch{ch}_wp{wp}.npy"   # impulsi che formano l'AP
# ERR_METHOD = "bootstrap" -> errore della mediana per RICAMPIONAMENTO degli impulsi: nessuna
#   ipotesi sulla forma della distribuzione (solo impulsi indipendenti). E' il default.
# ERR_METHOD = "gauss"     -> 1.2533*std/sqrt(N): il fattore sqrt(pi/2) vale SOLO per una
#   distribuzione gaussiana. Verificato su ch91 WP15: coincide col bootstrap entro il 4% in
#   baseline e coda, ma SOVRASTIMA del ~25% sul fronte di salita (li' la dispersione tra
#   impulsi e' dominata dal jitter e non e' gaussiana).
ERR_METHOD  = "bootstrap"
BOOT_N      = 800    # ricampionamenti (seed fisso -> risultato riproducibile)

# --- Cost function del fit ----------------------------------------------------------
# COST = "time_ls"     -> minimi quadrati nel TEMPO (iniziale): Σ_t (modello − dato)².
# COST = "white_resid" -> forza lo SPETTRO del residuo a non superare il RUMORE BIANCO.
#   Fa la FFT del residuo e penalizza |FFT(residuo)(f)| solo dove ECCEDE il livello del rumore
#   bianco A_white = σ_baseline·√N (σ = RMS della baseline nel pretrigger dell'AP). Cosi' le
#   oscillazioni a bassa frequenza (eccesso sopra il bianco) vengono minimizzate, mentre il
#   rumore bianco alle alte freq (già al livello) non e' penalizzato.
#   NB (verificato): "white_resid" NON rimuove l'onda a bassa freq (limite del modello, non della
#   cost) e peggiora l'ottimizzazione.
# COST = "white_match" -> cost a DUE termini: (a) residuo nel TEMPO (ancora il fit al dato) +
#   (b) residuo nello SPETTRO che spinge |FFT(residuo)(f)| a COMBACIARE col rumore bianco
#   A_white = σ_baseline·√N (RMS del pretrigger dell'AP), non solo a starci sotto. Il termine (b)
#   e' (|FFT(residuo)| − A_white) su TUTTE le frequenze (bilaterale): dove il residuo eccede il
#   bianco (onda a bassa freq) viene spinto giu', dove e' gia' bianco (alte freq) il termine ~0.
#   LAMBDA_FREQ pesa (b) rispetto ad (a); entrambi i termini sono in scala σ (freq / √N).
#   COST = "time_ls" per tornare al comportamento iniziale.
# COST = "nps" -> minimi quadrati nel dominio delle FREQUENZE pesati col RUMORE:
#   Sigma_f |FFT(modello - dato)(f)|^2 / NPS(f)  (DC escluso, parte reale e immaginaria come
#   residui separati). E' la stessa metrica che determina sigma_OF, quindi il fit spende i
#   parametri dove il filtro ottimo/BI li usa davvero e non insegue il fronte ad alta
#   frequenza, dove l'AP e' sotto il pavimento di rumore del template. NPS letta dal ROOT
#   del canale (averagepowerspectrum_noise_wp<wp>_medianpower): serve SOURCE="root" o "npy".
# Punti di partenza del multi-start: fattori (rise, decay, osc) applicati ai guess fisici di
# initial_guess. Piu' start = piu' probabile trovare il minimo GLOBALE (ma il fit dura di piu').
#   VERIFICATO su ch91 con 9p z4: con 8 start il fit finiva in un minimo LOCALE su 6 WP su 15
#   (es. wp5 chi=7.3 invece di 2.1, wp27 2.9 invece di 0.67). Con la griglia sotto (28 start)
#   il problema sparisce; il fit passa da ~20 s a ~60 s per WP.
STARTS = [(rs, ds, 1) for rs in (0.2, 0.3, 0.5, 1, 2, 3, 5) for ds in (0.5, 1, 2, 3)]

COST        = "time_ls"
LAMBDA_FREQ = 2         # peso del termine spettrale in "white_match" (0 -> solo tempo)

# NB: il nome del PNG e la riga del CSV includono il nome del modello (costruiti in main()).
OUTDIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_CSV = os.path.join(OUTDIR, "fit_one_pulse_params.csv")


# ══════════════════════════════════════════════════════════════════════════════
# MODELLI
# ══════════════════════════════════════════════════════════════════════════════
def make_pulse_pole_zero_bessel_ct(bessel_order, fcut, zero, *poles):
    """Impulso pole-zero a 1 zero e poli REALI, convoluto con un Bessel analogico
    (ordine bessel_order, taglio fcut Hz). Ritorna f(t), normalizzata a picco 1 a t=0.

    Idea: il sistema H(s) = (s - zero) / prod(s - poli) ha risposta all'impulso data
    dalla somma di esponenziali (fratti semplici). Convolvendo col Bessel (altri poli,
    fissi) si sommano altri esponenziali. I coefficienti B_i, C_j vengono dai residui."""
    poles = np.asarray(poles, dtype=float)          # <-- poli REALI
    wc = 2 * np.pi * fcut

    # residui del pole-zero:  k_i = (p_i - zero) / prod_{j!=i}(p_i - p_j)
    diff = poles[:, None] - poles[None, :]
    denom = np.prod(np.where(np.eye(len(poles), dtype=bool), 1.0, diff), axis=1)
    k = (poles - zero) / denom

    # poli del Bessel (normalizzati), scalati al taglio wc
    _, p_norm, g_norm = besselap(bessel_order)
    p_filt = wc * p_norm
    g_filt = g_norm * wc ** bessel_order
    diff_f = p_filt[:, None] - p_filt[None, :]
    denom_f = np.prod(np.where(np.eye(len(p_filt), dtype=bool), 1.0, diff_f), axis=1)
    A = g_filt / denom_f

    # combinazione: coefficienti dei due gruppi di esponenziali
    coef = k[:, None] * A[None, :] / (poles[:, None] - p_filt[None, :])
    B = np.sum(coef, axis=1)     # pesa exp(p_i t)      (poli del pulse)
    C = -np.sum(coef, axis=0)    # pesa exp(lambda_j t) (poli del Bessel)

    def f_raw(t):
        t = np.asarray(t, float)
        out = np.zeros_like(t, dtype=complex)
        m = t >= 0
        tt = t[m]
        out[m] = (np.sum(B[:, None] * np.exp(poles[:, None] * tt), axis=0)
                  + np.sum(C[:, None] * np.exp(p_filt[:, None] * tt), axis=0))
        return out.real

    # normalizza: trova il picco e riscala così che f(0) = 1, picco a t=0
    t_peak = minimize_scalar(lambda x: -f_raw(x), bounds=(0, 0.1), method="bounded").x
    pk = f_raw(t_peak) or 1.0
    return lambda tt: f_raw(np.asarray(tt, float) + t_peak) / pk


def make_pulse_bessel_general(bessel_order, fcut, zeros, poles):
    """Versione GENERALE di make_pulse_pole_zero_bessel_ct. Stessa matematica (fratti
    semplici del pole-zero convoluto col Bessel), con TRE differenze — segnate sotto con
    '### DIFFERENZA':
      1) i poli possono essere COMPLESSI  -> ammette una coppia complessa coniugata (CC),
         cioè un'oscillazione smorzata, che i poli reali non possono dare;
      2) N zeri invece di 1 -> numeratore = PRODOTTO su tutti gli zeri;
      3) normalizzazione del picco su GRIGLIA invece che con minimize_scalar (più robusta
         quando la risposta oscilla per via dei poli complessi).
    Coincide esattamente con la funzione base nel caso 1 zero / poli reali."""
    poles = np.asarray(poles, dtype=complex)        # ### DIFFERENZA 1: complex (non float)
    zeros = np.asarray(zeros, dtype=complex)         #     -> permette la coppia CC
    wc = 2 * np.pi * fcut

    diff = poles[:, None] - poles[None, :]
    denom = np.prod(np.where(np.eye(len(poles), dtype=bool), 1.0, diff), axis=1)
    # ### DIFFERENZA 2: numeratore = prod_k (p_i - z_k) su N zeri
    #     (nella funzione base era il singolo fattore (p_i - zero))
    num = np.ones(len(poles), dtype=complex)
    for z in zeros:
        num *= (poles - z)
    k = num / denom

    # --- da qui in poi IDENTICO alla funzione base: Bessel + combinazione dei residui ---
    _, p_norm, g_norm = besselap(bessel_order)
    p_filt = wc * p_norm
    g_filt = g_norm * wc ** bessel_order
    diff_f = p_filt[:, None] - p_filt[None, :]
    denom_f = np.prod(np.where(np.eye(len(p_filt), dtype=bool), 1.0, diff_f), axis=1)
    A = g_filt / denom_f
    coef = k[:, None] * A[None, :] / (poles[:, None] - p_filt[None, :])
    B = np.sum(coef, axis=1)
    C = -np.sum(coef, axis=0)

    def f_raw(t):
        t = np.asarray(t, float)
        out = np.zeros_like(t, dtype=complex)
        m = t >= 0
        tt = t[m]
        out[m] = (np.sum(B[:, None] * np.exp(poles[:, None] * tt), axis=0)
                  + np.sum(C[:, None] * np.exp(p_filt[:, None] * tt), axis=0))
        return out.real                              # somma reale (le CC si cancellano l'immaginario)

    # ### DIFFERENZA 3: come si trova il picco per normalizzare.
    #   - se NON ci sono poli complessi la risposta è unimodale -> minimize_scalar
    #     (preciso), ESATTAMENTE come nella funzione base;
    #   - se c'è una coppia CC la risposta può oscillare (multimodale) -> massimo GLOBALE
    #     su una griglia fitta (robusto a gobbe secondarie che ingannerebbero minimize_scalar).
    if np.any(np.abs(poles.imag) > 1e-12):
        tg = np.linspace(0.0, 0.15, 3000)
        tpk = tg[int(np.argmax(f_raw(tg)))]
    else:
        tpk = minimize_scalar(lambda x: -f_raw(x), bounds=(0, 0.1), method="bounded").x
    pk = f_raw(tpk) or 1.0
    return lambda tt: f_raw(np.asarray(tt, float) + tpk) / pk


# ══════════════════════════════════════════════════════════════════════════════
# 1) CARICAMENTO dell'average pulse (peak-normalizzato)
# ══════════════════════════════════════════════════════════════════════════════
def load_ap(channel, wp):
    files = glob.glob(os.path.join(PROCESSED_DIR, f"Processed_*_{MEAS_NAME}_{channel}.root"))
    if not files:
        raise SystemExit(f"ROOT del canale {channel} non trovato in {PROCESSED_DIR}")
    with uproot.open(files[0]) as f:
        h = f[f"averagepulse_ap_wp{wp}_medianAP"]
        t = np.asarray(h.axis().centers(), float)
        v = np.asarray(h.values(), float)
    return t, v / v.max()                            # picco = 1


def load_c_pulse(cfile):
    """Carica un impulso da una macro ROOT .C (TGraph) in Processed/: estrae gli array
    Graph_fx*=tempo[s] e Graph_fy*=ampiezza (10000 campioni, 10 kHz, stessa struttura degli AP)
    e ritorna (t, v/v.max()) peak-normalizzato a 1 — come load_ap, ma senza sottrarre baseline."""
    import re
    path = os.path.join(PROCESSED_DIR, cfile)
    if not os.path.exists(path):
        raise SystemExit(f"file .C non trovato: {path}")
    txt = open(path).read()
    def grab(kind):                                  # kind = "x" (tempo) oppure "y" (ampiezza)
        m = re.search(r"Graph_f%s\d*\[\d+\]\s*=\s*\{(.*?)\}" % kind, txt, re.S)
        if m is None:
            raise SystemExit(f"array Graph_f{kind}* non trovato in {path}")
        return np.fromstring(m.group(1), sep=",")
    t, v = grab("x"), grab("y")
    return t, v / v.max()                            # picco = 1


def load_npy_ap(channel, wp):
    """Carica un AP salvato come vettore .npy (NPY_PATTERN) e ricostruisce l'asse dei tempi
    come i centri dei bin degli AP ROOT: t_i = (i + 0.5)/SAMPLING_RATE. Ritorna (t, v/v.max())."""
    path = os.path.join(BASE, NPY_PATTERN.format(ch=channel, wp=wp))
    if not os.path.exists(path):
        raise SystemExit(f"file .npy non trovato: {path}")
    v = np.asarray(np.load(path), float)
    t = (np.arange(len(v)) + 0.5) / SAMPLING_RATE
    return t, v / v.max()                            # picco = 1


# ══════════════════════════════════════════════════════════════════════════════
# 2) MISURA delle scale di tempo dai dati (servono per i guess)
# ══════════════════════════════════════════════════════════════════════════════
def measure_timescales(t, v):
    """t_rise = tempo di salita 10%->90%;  t_dec = tempo di discesa a 1/e;
    t_peak = istante del picco. Sono le scale fisiche da cui si ricavano i poli."""
    dt = t[1] - t[0]
    imax = int(np.argmax(v)); peak = v[imax]
    up = v[:imax]
    i10 = np.where(up > 0.10 * peak)[0]
    i90 = np.where(up > 0.90 * peak)[0]
    t_rise = max((t[i90[0]] - t[i10[0]]) if (len(i10) and len(i90)) else 5 * dt, dt)
    be = np.where(v[imax:] < peak / np.e)[0]
    t_dec = (t[imax + be[0]] - t[imax]) if len(be) else 10 * dt
    return t_rise, t_dec, t[imax]


def baseline_sigma(v, frac=0.40):
    """sigma = RMS del RUMORE, stimato sulla BASELINE prima dell'impulso: deviazione
    standard dei campioni nella prima frazione `frac` della finestra (pre-trigger, ben
    prima del picco che sta al ~50%). Serve a esprimere il residuo in 'numero di sigma'
    e a disegnare la banda di ±3 sigma."""
    return float(v[:int(frac * len(v))].std()) or 1.0


def pulses_sigma(channel, wp, align_max):
    """Errore per TIME-BIN dai SINGOLI impulsi che formano l'AP (PULSE_PATTERN, salvati da
    extract_AP_pulses_m205.py), senza passare da nessun istogramma.

    Gli impulsi salvati sono allineati come li allinea Octopus (mezza salita): per l'AP
    allineato al MASSIMO vanno riallineati allo stesso modo (align_on_max), altrimenti la
    dispersione non e' quella dell'AP che si sta fittando.

    ERR_METHOD="bootstrap" (default): errore della mediana per ricampionamento degli impulsi,
    senza ipotesi sulla forma della distribuzione. ERR_METHOD="gauss": 1.2533*std/sqrt(N), dove
    1.2533 = sqrt(pi/2) porta dall'errore della MEDIA a quello della MEDIANA ma vale solo per
    una gaussiana (sul fronte di salita sovrastima del ~25%).

    FLOOR al livello della baseline: al picco tutti gli impulsi valgono 1 per costruzione
    (normalizzati al proprio massimo) -> std = 0; e negli ultimi campioni lo zero-padding
    dello shift di allineamento da' anch'esso std = 0. Senza floor quei bin avrebbero peso
    infinito. Ritorna None se il file degli impulsi non c'e' -> il chiamante ricade sul
    sigma di baseline costante."""
    path = os.path.join(BASE, PULSE_PATTERN.format(ch=channel, wp=wp))
    if not os.path.exists(path):
        return None
    pulses = np.load(path)
    if align_max:
        import sys
        sys.path.insert(0, BASE)
        from build_medianAP_maxalign_m205 import align_on_max
        pulses = align_on_max(pulses)
    if ERR_METHOD == "bootstrap":
        rng = np.random.default_rng(0)                     # seed fisso: errore riproducibile
        n = len(pulses)
        meds = np.empty((BOOT_N, pulses.shape[1]))
        for b in range(BOOT_N):
            meds[b] = np.median(pulses[rng.integers(0, n, n)], axis=0)
        err = meds.std(axis=0, ddof=1)
    else:
        err = 1.2533 * pulses.std(axis=0, ddof=1) / np.sqrt(len(pulses))
    n_base = int(0.40 * pulses.shape[1])                   # regione di pretrigger
    return np.maximum(err, np.median(err[:n_base]))        # floor = errore sulla baseline


def noise_nps(channel, wp):
    """NPS del canale/WP dal ROOT (`averagepowerspectrum_noise_wp<wp>_medianpower`), meta'
    indipendente dello spettro (DC..Nyquist), stessa griglia di np.fft.rfft sulla finestra.
    Serve solo come PESO per COST="nps": la scala costante e' irrilevante."""
    files = glob.glob(os.path.join(PROCESSED_DIR, f"Processed_*_{MEAS_NAME}_{channel}.root"))
    if not files:
        return None
    with uproot.open(files[0]) as f:
        key = f"averagepowerspectrum_noise_wp{wp}_medianpower"
        if key not in {k.split(';')[0] for k in f.keys()}:
            return None
        return np.asarray(f[key].values(), dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
# 3) GUESS iniziali dei parametri, dai tempi misurati
# ══════════════════════════════════════════════════════════════════════════════
def initial_guess(t_rise, t_dec, t_peak, rs=1.0, ds=1.0, osc=1.0):
    """Guess FISICI dai tempi misurati (un polo reale = un esponenziale: p = -1/tau).
      - POLI REALI: costanti di tempo geometricamente spaziate tra SALITA e CODA.
        A 2 poli sono ESATTAMENTE  p0 = -1/t_rise  e  p1 = -1/t_dec; con piu' di 2 poli
        gli estremi restano questi e gli intermedi si distribuiscono in mezzo.
      - ZERI: consigliati TRA poli consecutivi (media geometrica) -> z_k tra p_k e p_{k+1}.
        E' solo un guess: i bound NON li vincolano a stare tra i poli.
      - COPPIA CC (se usata): smorzamento ~1/t_dec, pulsazione ~2pi/t_dec.
    rs/ds/osc sono fattori per provare start leggermente diversi (multi-start)."""
    taus = np.geomspace(t_rise * rs, t_dec * ds, N_REAL)   # tau dei poli: da t_rise a t_dec
    reals = -1.0 / taus                                     # p0=-1/t_rise ... p_last=-1/t_dec
    # zeri fra poli consecutivi: tau_z = media geometrica delle due tau adiacenti
    gaps = np.sqrt(taus[:-1] * taus[1:]) if N_REAL >= 2 else np.array([t_rise * rs])
    zeros = -1.0 / np.resize(gaps, N_ZERO) if N_ZERO else np.empty(0)
    theta = [t_peak, *zeros, *reals]
    if USE_CC and MODEL == "bessel_general":
        theta += [-1.0 / t_dec, (2 * np.pi / t_dec) * osc]                  # sigma<0, omega>0
    return np.array(theta, float)


# ══════════════════════════════════════════════════════════════════════════════
# 4) BOUND sui parametri
# ══════════════════════════════════════════════════════════════════════════════
def make_bounds(t_peak):
    """- t0: stretto attorno al picco (± T0_WINDOW): il picco è ben noto dai dati.
       - zeri e poli reali: [-PARAM_BOUND, 0]  -> parte reale <= 0 (STABILI: esponenziali
         decrescenti) e modulo <= PARAM_BOUND (BANDA LIMITATA: il fit non può inseguire
         wiggle ad alta frequenza).
       - coppia CC: sigma in [-PARAM_BOUND, 0] (smorzata), omega in [0, PARAM_BOUND]
         (oscillazione a bassa frequenza, fisica)."""
    B = PARAM_BOUND
    lo = [t_peak - T0_WINDOW] + [-B] * N_ZERO + [-B] * N_REAL
    hi = [t_peak + T0_WINDOW] + [0.0] * N_ZERO + [0.0] * N_REAL
    if USE_CC and MODEL == "bessel_general":
        lo += [-B, 0.0]; hi += [0.0, B]
    return np.array(lo, float), np.array(hi, float)


# ══════════════════════════════════════════════════════════════════════════════
# Costruzione del modello da theta e residuo
# ══════════════════════════════════════════════════════════════════════════════
def unpack_theta(theta):
    """Scioglie theta in (t0, zeros, poles). Ordine: [t0, zeri, poli_reali, (cc_sigma, cc_omega) se CC]."""
    t0 = theta[0]
    zeros = theta[1:1 + N_ZERO]
    reals = theta[1 + N_ZERO:1 + N_ZERO + N_REAL]
    if USE_CC and MODEL == "bessel_general":
        cs, co = theta[-2], theta[-1]
        poles = list(reals) + [complex(cs, co), complex(cs, -co)]          # coppia CC
    else:
        poles = list(reals)
    return t0, zeros, poles


def pulse_from_theta(theta, t):
    """Costruisci la funzione-modello e valutala sull'asse t (shift t0)."""
    t0, zeros, poles = unpack_theta(theta)
    if MODEL == "bessel_1zero":
        f = make_pulse_pole_zero_bessel_ct(BESSEL_ORDER, FCUT, zeros[0], *poles)
    else:
        f = make_pulse_bessel_general(BESSEL_ORDER, FCUT, zeros, poles)
    return f(t - t0)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN: carica, fitta (minimi quadrati), disegna
# ══════════════════════════════════════════════════════════════════════════════
def main():
    if MODEL == "bessel_1zero":
        globals()["N_ZERO"] = 1                      # il modello base ha esattamente 1 zero
        globals()["USE_CC"] = False                  # e solo poli reali

    # sorgente dell'impulso: ROOT (canale/WP) oppure macro .C in Processed/. In entrambi i casi
    # il risultato e' (t, v) con v peak-normalizzato a 1. `src_label` e' l'etichetta per titoli/
    # stampe; `ch_id`,`wp_id` sono le chiavi per CSV e nome del PNG (per i .C: nome file e wp=0).
    if SOURCE == "cfile":
        t, v = load_c_pulse(CFILE)
        src_label = os.path.splitext(os.path.basename(CFILE))[0]     # es. "LED1"
        ch_id, wp_id = src_label, 0
    elif SOURCE == "npy":
        t, v = load_npy_ap(CHANNEL, WP)
        src_label = f"Ch {CHANNEL} · WP {WP} (max-aligned)"
        # chiave CSV distinta da quella dell'AP di Octopus dello stesso (canale, WP),
        # altrimenti l'upsert sovrascriverebbe la riga dell'AP allineato a mezza salita
        ch_id, wp_id = f"{CHANNEL}max", WP
    else:
        t, v = load_ap(CHANNEL, WP)
        src_label = f"Ch {CHANNEL} · WP {WP}"
        ch_id, wp_id = CHANNEL, WP

    t_rise, t_dec, t_peak = measure_timescales(t, v)
    sigma = baseline_sigma(v)                        # RMS del rumore sulla baseline (fallback)
    print(f"{src_label} | modello={MODEL}  N_real={N_REAL} N_zero={N_ZERO} CC={USE_CC}")
    print(f"tempi misurati:  t_rise={t_rise*1e3:.3f} ms  t_dec={t_dec*1e3:.3f} ms  t_peak={t_peak:.4f} s")

    # ERRORE per time-bin: dalla dispersione dei singoli impulsi che formano l'AP (SOURCE
    # "root" o "npy", se abilitato e presenti) oppure sigma di baseline costante. `sig` e' un
    # array lungo come t; `weighted` dice se il fit lo pesa.
    use_pulse_err = USE_PULSE_ERRORS and SOURCE in ("root", "npy")
    err = pulses_sigma(CHANNEL, WP, align_max=(SOURCE == "npy")) if use_pulse_err else None
    if use_pulse_err and err is None:
        print("  [warn] impulsi non trovati: ricado sul sigma di baseline costante")
    sig = err if err is not None else np.full(len(t), sigma)
    weighted = WEIGHT_FIT and err is not None
    print(f"errore: {'per-bin dai singoli impulsi' if err is not None else f'baseline costante ({sigma:.2e})'}"
          f"   | fit {'PESATO 1/err' if weighted else 'non pesato'}")

    lo, hi = make_bounds(t_peak)

    # COST function: "time_ls" = minimi quadrati nel tempo; "white_resid" = penalizza lo spettro
    # del residuo dove eccede il livello di rumore bianco A_white = sigma_baseline·√N (sigma dalla
    # baseline di pretrigger). Le alte freq (rumore bianco) restano libere, la struttura a bassa
    # freq (eccesso sopra il bianco) e' penalizzata.
    cost_mode = COST
    A_white = sigma * np.sqrt(len(t))                # livello |FFT| atteso del rumore bianco
    # white_resid / white_match: al termine nel TEMPO (che ancora il fit all'impulso) si affianca
    # un termine sullo SPETTRO del residuo (rfft, len(t)//2 bin AC scartando la DC). white_resid
    # penalizza solo l'ECCESSO sopra il bianco; white_match penalizza lo scostamento BILATERALE
    # dal bianco (spinge lo spettro del residuo a combaciare col rumore bianco).
    n_resid = (len(t) + len(t) // 2) if cost_mode in ("white_resid", "white_match") else len(t)
    w_nps = None
    if cost_mode == "nps":
        nps_half = noise_nps(CHANNEL, WP) if SOURCE in ("root", "npy") else None
        if nps_half is None or len(nps_half) != len(t) // 2 + 1:
            raise SystemExit("COST='nps': NPS non disponibile per questo canale/WP")
        w = 1.0 / np.sqrt(nps_half[1:])                # DC escluso
        w_nps = w / np.median(w)                       # scala arbitraria: solo leggibilita'
        n_resid = 2 * len(w_nps)
    print(f"cost = {cost_mode}" + (f"  (lambda_freq={LAMBDA_FREQ})" if cost_mode == "white_match" else ""))

    # 5) FIT: minimizza la cost scelta. Si provano pochi start fisici e si tiene il MIGLIORE.
    def residual(theta):
        try:
            y = pulse_from_theta(theta, t)
        except Exception:
            return np.full(n_resid, 1e3)
        if cost_mode == "white_resid":
            d = y - v
            excess = np.maximum(np.abs(np.fft.rfft(d)[1:]) - A_white, 0.0)   # eccesso sopra il bianco
            r = np.concatenate([d, excess / np.sqrt(len(t))])               # ancora nel tempo + spettro
        elif cost_mode == "white_match":
            d = y - v                                                        # residuo FISICO (per la FFT)
            time_term = d / sig if weighted else d                          # termine nel tempo (pesato 1/err come time_ls)
            spec = np.abs(np.fft.rfft(d)[1:]) - A_white                      # scostamento (bilaterale) dal bianco
            r = np.concatenate([time_term, LAMBDA_FREQ * spec / np.sqrt(len(t))])   # tempo(pesato) + spettro->bianco
        elif cost_mode == "nps":
            D = np.fft.rfft(y - v)[1:]                                   # DC escluso
            r = np.concatenate([D.real, D.imag]) * np.concatenate([w_nps, w_nps])
            r = r / np.sqrt(len(t))                                      # scala leggibile
        else:
            r = (y - v) / sig if weighted else (y - v)
        return np.nan_to_num(r, nan=1e6, posinf=1e6, neginf=-1e6)

    starts = list(STARTS)
    if USE_CC and MODEL == "bessel_general":
        starts += [(1, 1, 2), (1, 1, 0.5)]
    best = None
    for rs, ds, osc in starts:
        th0 = np.clip(initial_guess(t_rise, t_dec, t_peak, rs, ds, osc), lo + 1e-12, hi - 1e-12)
        r = least_squares(residual, th0, bounds=(lo, hi), method="trf", max_nfev=3000)
        fit = pulse_from_theta(r.x, t)
        if not np.all(np.isfinite(fit)):
            continue
        cost = float(np.mean(residual(r.x) ** 2))    # obiettivo effettivo (time_ls o freq_whitened)
        if best is None or cost < best[0]:
            best = (cost, r.x, fit, r.jac)           # jac: serve per la covarianza dei parametri
    if best is None:
        raise SystemExit("nessuno start è convertito")
    _, theta, fit, jac = best

    # metriche: RMS assoluta; residuo in unita' di errore (per-bin o baseline) e sua RMS (chi)
    rms = float(np.sqrt(np.mean((fit - v) ** 2)))
    rms_sigma = rms / sigma                          # per il CSV (rispetto al rumore di baseline)
    resid_sigma = (fit - v) / sig                    # residuo in NUMERO DI ERRORI
    chi = float(np.sqrt(np.mean(resid_sigma ** 2)))  # RMS del residuo in unita' di errore

    # estrae la coppia CC (ultimi due parametri, se presente)
    cc_sig_v = cc_om_v = None
    if USE_CC and MODEL == "bessel_general":
        cc_sig_v, cc_om_v = float(theta[-2]), float(theta[-1])

    # stampa i parametri finali (in modo leggibile)
    print(f"\nRMS fit = {rms:.3e}   (chi = RMS/err = {chi:.2f})")
    print(f"  t0          = {theta[0]:.5f} s")
    print(f"  zeri  [rad/s]= {np.array2string(theta[1:1+N_ZERO], precision=1)}")
    print(f"  poli reali  = {np.array2string(theta[1+N_ZERO:1+N_ZERO+N_REAL], precision=1)}")
    if cc_sig_v is not None:
        print(f"  coppia CC   : sigma={cc_sig_v:.1f} rad/s  omega={cc_om_v:.1f} rad/s "
              f"(= {cc_om_v/(2*np.pi):.0f} Hz)")

    # ── ERRORI dei parametri: covarianza dal JACOBIANO nel minimo ─────────────────
    #   cov = (J^T J)^-1 * s^2,  s^2 = SSR/(n-p). Col fit PESATO 1/err i residui hanno
    #   varianza 1 e s^2 ~ 1, quindi cov e' gia' in unita' fisiche.
    #   La pseudo-inversa e' via SVD: se due poli sono quasi coincidenti J^T J e' quasi
    #   SINGOLARE e l'inversa esplode -> e' proprio questo che l'errore deve mostrare.
    #   Il NUMERO DI CONDIZIONE dice quanto: >1e8 = parametri non identificabili singolarmente.
    def param_errors(jac, resid):
        n_p = jac.shape[1]
        s2 = float(np.sum(resid ** 2) / max(len(resid) - n_p, 1))
        u, sv, vt = np.linalg.svd(jac, full_matrices=False)
        cond = float(sv.max() / sv.min()) if sv.min() > 0 else np.inf
        sv_inv = np.where(sv > sv.max() * 1e-14, 1.0 / np.maximum(sv, 1e-300), 0.0)
        cov = (vt.T * sv_inv ** 2) @ vt * s2
        return np.sqrt(np.abs(np.diag(cov))), cov, cond

    err_par, cov, cond = param_errors(jac, residual(theta))
    sd = np.sqrt(np.abs(np.diag(cov)))
    corr = cov / np.outer(np.where(sd > 0, sd, 1), np.where(sd > 0, sd, 1))
    print(f"\nerrori dei parametri (numero di condizione = {cond:.2e}"
          + ("  -> DEGENERE: parametri non identificabili singolarmente)" if cond > 1e8 else ")"))
    for nm, x, e in zip((["t0"] + [f"zero{i+1}" for i in range(N_ZERO)]
                         + [f"pole{i+1}" for i in range(N_REAL)]
                         + (["cc_sigma", "cc_omega"] if cc_sig_v is not None else [])),
                        theta, err_par):
        rel = abs(e / x) if x else np.inf
        print(f"  {nm:9s} = {x:12.4f} ± {e:<12.4g} ({100*rel:8.1f}%)")
    iu = np.triu_indices(len(theta), 1)
    k = int(np.argmax(np.abs(corr[iu])))
    print(f"  correlazione massima: {corr[iu][k]:+.5f} tra i parametri {iu[0][k]} e {iu[1][k]}")

    # ── RAILING: parametri finiti SUL PROPRIO BOUND ───────────────────────────────
    #   Un parametro incollato al bound non e' un minimo (la derivata non e' zero): il
    #   vincolo lo sta bloccando e gli altri parametri si deformano per compensare.
    #   Si segnala anche chi e' "vicino" al bound (entro l'1% dell'intervallo): candidato
    #   al railing se si cambia modello o dato.
    #   Criterio: per zeri e poli NON si usa una tolleranza lineare sull'intervallo (i poli
    #   coprono 4 ordini di grandezza, un polo lento risulterebbe sempre "vicino allo 0").
    #     - bound INFERIORE -PARAM_BOUND: railing se |x| >= 0.999*PARAM_BOUND;
    #     - bound SUPERIORE 0: railing se tau = 1/|x| >= durata della finestra (esponenziale
    #       indistinguibile da una costante -> il dato non lo vincola), "vicino" se tau >= meta'.
    #   t0 e' lineare: tolleranza sull'intervallo.
    T_win = float(t[-1] - t[0])
    par_names = (["t0"] + [f"zero{i+1}" for i in range(N_ZERO)]
                 + [f"pole{i+1}" for i in range(N_REAL)]
                 + (["cc_sigma", "cc_omega"] if cc_sig_v is not None else []))
    rail, near = [], []
    for i, nm in enumerate(par_names):
        x = theta[i]
        if nm == "t0":
            d = min(abs(x - lo[i]), abs(x - hi[i])) / (hi[i] - lo[i])
            lvl = 0 if d <= 1e-3 else 1 if d <= 1e-2 else 2
        else:
            slow = 1.0 / abs(x) if x else np.inf                  # tau [s]
            lvl = (0 if (abs(x) >= 0.999 * PARAM_BOUND or slow >= T_win)
                   else 1 if slow >= T_win / 2 else 2)
        (rail if lvl == 0 else near if lvl == 1 else []).append(f"{nm}={x:.4g}")
    if rail:
        print(f"  [RAILING] sul bound: {', '.join(rail)}  -> allarga il bound e rifitta")
    elif near:
        print(f"  [ok] nessun railing (vicini al bound: {', '.join(near)})")
    else:
        print("  [ok] nessun railing: tutti i parametri sono interni ai bound")

    # nome del MODELLO provato (identifica PNG e riga del CSV)
    cc_flag = 1 if (USE_CC and MODEL == "bessel_general") else 0
    model_name = f"{N_REAL}p{'+cc' if cc_flag else ''} z{N_ZERO}"

    # esporta i parametri (piena precisione) nel CSV che visual_pulse carica dal menu'
    # "start from". UPSERT: se esiste gia' una riga per lo stesso (channel, wp, model)
    # la sovrascrive, altrimenti la aggiunge -> un modello per riga, confrontabili.
    HEADER = ["channel", "wp", "rank", "model", "n_real", "nzer", "cc", "bessel_order",
              "fcut", "t0", "zeros", "real_poles", "cc_sigma", "cc_omega", "rms_over_sigma"]
    row = [ch_id, wp_id, 0, model_name, N_REAL, N_ZERO, cc_flag, BESSEL_ORDER, FCUT,
           f"{theta[0]:.8f}",
           " ".join(f"{z:.6f}" for z in theta[1:1 + N_ZERO]),
           " ".join(f"{p:.6f}" for p in theta[1 + N_ZERO:1 + N_ZERO + N_REAL]),
           f"{cc_sig_v:.6f}" if cc_sig_v is not None else "",
           f"{cc_om_v:.6f}" if cc_om_v is not None else "",
           f"{rms_sigma:.4f}"]
    rows = []
    if os.path.exists(PARAMS_CSV):
        with open(PARAMS_CSV) as f:
            existing = list(csv.reader(f))
        if existing and existing[0] == HEADER:
            ix = {h: i for i, h in enumerate(HEADER)}
            key = (str(ch_id), str(wp_id), model_name)     # chiave di upsert
            rows = [r for r in existing[1:]
                    if r and (r[ix["channel"]], r[ix["wp"]], r[ix["model"]]) != key]
    rows.append([str(x) for x in row])
    with open(PARAMS_CSV, "w", newline="") as f:
        w = csv.writer(f); w.writerow(HEADER); w.writerows(rows)
    print(f"✓ parametri del modello '{model_name}' salvati in {PARAMS_CSV}")

    name_tag = (f"ch{CHANNEL}_wp{WP}" if SOURCE == "root" else
                f"ch{CHANNEL}_wp{WP}_maxalign" if SOURCE == "npy" else src_label)
    cost_tag = "" if cost_mode == "time_ls" else f"_{cost_mode}"
    out_png = os.path.join(OUTDIR, f"fit_one_pulse_{name_tag}_{model_name.replace(' ', '_')}{cost_tag}.png")

    # 6) PLOT: AP + fit + banda ±3sigma (sopra); residuo in unità di sigma (sotto)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    data_lbl = "pulse data" if SOURCE == "cfile" else "AP data"   # ("npy" = AP anch'esso)
    ax1.plot(t, v, "k.", ms=3, label=data_lbl)
    ax1.plot(t, fit, "r-", lw=1.5, label=f"fit ({MODEL})")
    band_lbl = r"fit $\pm\,3\cdot$err (single pulses)" if err is not None else r"fit $\pm\,3\sigma$ (baseline)"
    ax1.fill_between(t, fit - 3 * sig, fit + 3 * sig, color="red", alpha=0.15, label=band_lbl)
    ax1.set_ylabel("pulse (peak-normalized)")
    ax1.set_title(f"{src_label} — pole-zero + Bessel({BESSEL_ORDER}@{FCUT}Hz)"
                  f" — RMS={rms:.2e}  (χ={chi:.1f})")
    ax1.legend(loc="upper right"); ax1.grid(True, ls="--", alpha=0.4)

    # box con i PARAMETRI risultanti dal fit
    plines = [f"$t_0$ = {theta[0]:.5f} s", "poles [rad/s]: " +
              ", ".join(f"{p:.0f}" for p in theta[1 + N_ZERO:1 + N_ZERO + N_REAL])]
    if N_ZERO:
        plines.insert(1, "zeros [rad/s]: " + ", ".join(f"{z:.0f}" for z in theta[1:1 + N_ZERO]))
    if cc_sig_v is not None:
        plines.append(f"CC: $\\sigma$={cc_sig_v:.0f}, $\\omega$={cc_om_v:.0f} rad/s "
                      f"({cc_om_v/(2*np.pi):.0f} Hz)")
    ax1.text(0.98, 0.72, "\n".join(plines), transform=ax1.transAxes, ha="right", va="top",
             fontsize=8.5, family="monospace",
             bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    # residuo in NUMERO DI ERRORI (per-bin dai singoli impulsi o baseline costante), rif. a 0 e ±3
    ax2.plot(t, resid_sigma, "b-", lw=1.0)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.axhline(+3, color="0.5", ls="--", lw=0.8); ax2.axhline(-3, color="0.5", ls="--", lw=0.8)
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel(r"(fit $-$ data) / err")
    ax2.grid(True, ls="--", alpha=0.4)
    # riporto anche la RMS assoluta (oltre alla RMS/σ nel titolo)
    ax2.text(0.99, 0.05, f"RMS = {rms:.2e}", transform=ax2.transAxes,
             ha="right", va="bottom", fontsize=9)
    # zoom sull'impulso (dal poco prima del picco a un po' dopo)
    ax1.set_xlim(t_peak - 5 * t_rise, t_peak + 8 * t_dec)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print(f"✓ figura salvata in {out_png}")

    # con COST="nps" il residuo MINIMIZZATO e' quello in frequenza pesato col rumore: senza
    # guardarlo non si vede cosa ha fatto il fit (nel tempo sembra sempre peggio di time_ls).
    if cost_mode == "nps":
        freq = np.fft.rfftfreq(len(t), t[1] - t[0])[1:]
        D = np.fft.rfft(fit - v)[1:]
        # normalizzato alla propria mediana: la NPS del ROOT e' in unita' grezze mentre l'AP e'
        # peak-normalizzato, quindi il livello ASSOLUTO non vuol dire niente; quello che conta e'
        # se il rapporto e' PIATTO (residuo bianco rispetto al rumore = il fit ha preso tutto
        # quello che c'era sopra il rumore) o se ha strutture (li' il modello sbaglia).
        ratio = np.abs(D) / np.sqrt(nps_half[1:])
        ratio = ratio / np.median(ratio)
        fig2, ax = plt.subplots(figsize=(9, 4.2))
        ax.loglog(freq, ratio, lw=0.5, color="tab:blue", alpha=0.5, label="per-bin")
        # mediana in bin logaritmici: senza questa la curva per-bin e' illeggibile
        edges = np.geomspace(freq[0], freq[-1], 60)
        idx = np.digitize(freq, edges)
        fb = np.array([freq[idx == i].mean() for i in range(1, len(edges)) if (idx == i).any()])
        rb = np.array([np.median(ratio[idx == i]) for i in range(1, len(edges)) if (idx == i).any()])
        ax.loglog(fb, rb, lw=1.8, color="tab:orange", label="median in log bins")
        ax.axhline(1.0, color="tab:red", ls="--", lw=1.2, label="flat = residual at the noise level")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(r"|FFT(fit $-$ data)| / $\sqrt{NPS}$  (norm. to median)")
        ax.set_title(f"{src_label} — {model_name}, COST=nps — residuo in frequenza pesato col rumore")
        ax.grid(True, which="both", ls="--", alpha=0.4); ax.legend()
        fig2.tight_layout()
        out2 = out_png.replace(".png", "_spectrum.png")
        fig2.savefig(out2, dpi=140)
        print(f"✓ residuo in frequenza in {out2}")


if __name__ == "__main__":
    main()
