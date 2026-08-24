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
import csv
import glob

import numpy as np
import uproot
import torch

import src.analysis as an
import src.dataset as ds
import src.simulation as sim
import utility.functions as fn

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE
# ═════════════════════════════════════════════════════════════════════════════
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "Processed")
MEAS_NAME   = "000205"

# Set di risultati da simulare: cartella col CSV e i filtri addestrati.
RESULTS_DIR = os.path.join(BASE_DIR, "m205_results_octopus")     # analyse_BI_m205.py (filtro ottimo)
# analyse_BI_m205.py scrive "BI_results_m205<tag>.csv", col tag della modalita' (_fit,
# _sim_fitinj, ...): si prende quello che c'e' nella cartella invece di indovinare il nome.
BI_CSV      = (glob.glob(os.path.join(RESULTS_DIR, "BI_results_m205*.csv")) or
               [os.path.join(RESULTS_DIR, "BI_results_m205.csv")])[0]
FILTERS_DIR = os.path.join(RESULTS_DIR, "trained_filters")
OUT_CSV     = os.path.join(RESULTS_DIR, "BI_mc_error_m205.csv")

# ── Quale template GENERA gli eventi e quale ha ADDESTRATO i filtri ──────────────────
# Sono due cose distinte, ed e' proprio la loro differenza che si vuole misurare (il paper:
# "to avoid the bias of using an identical template for injection and training").
#   GEN_TEMPLATE   -> da cosa si generano gli eventi simulati (la "verita'").
#   TRAIN_TEMPLATE -> su cosa sono stati addestrati i filtri di RESULTS_DIR. Serve al controllo
#                     sul kernel: se non combacia, il programma si ferma invece di mescolare.
# Valori: "root" (medianAP dal ROOT) | "fit" (bestfit dello scan) | "sim" (AP simulato, --make-ap).
#   GEN="root", TRAIN="root" -> consistenza interna (analitico vs MC), il caso auto-consistente;
#   GEN="root", TRAIN="sim"  -> stima ONESTA: filtri addestrati su un'altra realizzazione del
#                               rumore di template, valutati sulla verita'. La differenza tra i
#                               due e' il guadagno fittizio dell'auto-consistenza.
GEN_TEMPLATE   = "root"
TRAIN_TEMPLATE = "root"

FIT_DIR     = os.path.join(BASE_DIR, "residual_scan_bessel", "fits_octopus")
FIT_PATTERN = "bestfit_ch{ch}_wp{wp}.npy"
# AP simulati prodotti da --make-ap: una sottocartella per canale.
SIM_AP_DIR     = os.path.join(BASE_DIR, "m205_AP_sim")
SIM_AP_PATTERN = os.path.join("ch{ch}", "simAP_{gen}_ch{ch}_wp{wp}.npy")
SIM_AP_FROM    = "fitinj"   # quale AP simulato leggere quando un TEMPLATE vale "sim":
                            # "fitinj" | "rootinj" | "fitgen" | "rootgen" (vedi
                            # build_simAP_injected_m205.py). I vecchi "fit"/"root",
                            # generati dalla NPS di Octopus, sono superati.
_SIM_OK = ("fitinj", "rootinj", "fitgen", "rootgen")
if "sim" in (GEN_TEMPLATE, TRAIN_TEMPLATE) and SIM_AP_FROM not in _SIM_OK:
    raise SystemExit(f"[ERROR] SIM_AP_FROM='{SIM_AP_FROM}' non valido: usare uno di {_SIM_OK}.")

# ── Sorgente della NPS: deve essere LA STESSA con cui sono stati addestrati i filtri ────
# "octopus": medianpower dal ROOT (mediana usata come media, array one-sided specchiato:
#   1.84 volte la potenza vera, misurato su ch91).
# "clean": NPS misurata dalle finestre vere (build_NPS_clean_m205.py), gia' nella convenzione
#   del simulatore. Se non combacia con quella del training, il controllo sul kernel se ne
#   accorge e il programma si ferma.
# ── Tipo di filtro con cui sono stati addestrati i risultati di RESULTS_DIR ────────────
# "optimum": kernel H = S*/NPS di analyse_BI_m205.py.
# "wiener":  kernel W = S*/(|S|^2 + lambda*NPS), lambda addestrato letto dal CSV
#   (colonna `lambda_wiener`), SENZA regolarizzazione.
# "wiener_R": lo stesso, ma con W <- R(f)*W (reliability_R), usando `beta_R` e `n_events`
#   dal CSV. E' una modalita' a se' e non si deduce dal CSV: se la sbagli, il controllo sul
#   kernel se ne accorge e il programma si ferma.
# Il resto della catena non cambia: il filtro totale applicato ai dati e' g_i = f_i * kernel.
FILTER_TYPE = "optimum"     # "optimum" | "wiener" | "wiener_R"

NPS_SOURCE  = "octopus"     # "octopus" | "clean"
NPS_DIR     = os.path.join(BASE_DIR, "m205_NPS_clean")
NPS_PATTERN = os.path.join("ch{ch}", "nps_ch{ch}_wp{wp}.npy")

ONLY_CHANNELS = None        # lista, oppure None/[] per tutti i canali del CSV
ONLY_WPS      = None        # lista, oppure None/[] per tutti i WP

# Parametri della simulazione (gli stessi del calcolo analitico in analyse_BI_m205.py)
NSIM        = 20_000        # eventi per popolazione; l'errore MC scala come 1/sqrt(NSIM)
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
    dt_max>0 -> PILE-UP. Il rumore e' generato con lo spettro nps del canale."""
    fpulses, *_ = sim.simulate_frequency_pulses(S, nps, DETECTOR_SIGMA, w, nsim=NSIM, seed=seed,
                                                signal_scale=signal_amp, dt_max=dt_max,
                                                fold_ratio=FOLD_RATIO)
    pulses = np.fft.ifft(fpulses, axis=1).real.astype(np.float32)
    dataset = ds.NumpyDataset(pulses)
    dataset.win_length = pulses.shape[1]        # get_PSD_interpole legge win_length dal dataset
    psd, _, _ = an.get_PSD_interpole(dataset, H_unit, f1, f2)
    return np.asarray(psd).ravel()


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
    args = ap_arg.parse_args()

    rows = select_rows()
    if args.make_ap is not None:
        for gen in (args.make_ap or [GEN_TEMPLATE]):
            make_sim_ap(rows, gen)
        return

    print(f"{len(rows)} coppie (canale, WP): eventi da '{GEN_TEMPLATE}', filtri '{FILTER_TYPE}' "
          f"addestrati su '{TRAIN_TEMPLATE}' ({os.path.basename(RESULTS_DIR)}), NPS '{NPS_SOURCE}', "
          f"NSIM={NSIM}, fold_ratio={FOLD_RATIO}, detector_sigma={DETECTOR_SIGMA}\n")

    out = []
    print(f"{'ch':>4s} {'wp':>3s} {'BI analitico':>13s} {'BI Monte Carlo':>16s} "
          f"{'sigma_BI':>10s} {'MC/analitico':>13s}")
    for r in rows:
        ch, wp = int(r["channel"]), int(r["wp"])
        bi_an = float(r["BI"])
        try:
            res = run_pair(ch, wp, r)
        except Exception as e:
            print(f"{ch:>4d} {wp:>3d}   [ERROR] {e}")
            continue
        print(f"{ch:>4d} {wp:>3d} {bi_an:13.4e} {res['BI_mc']:11.4e} ± {res['sigma_BI']:.1e} "
              f"{res['sigma_BI']:10.1e} {res['BI_mc']/bi_an:12.3f}")
        out.append(dict(channel=ch, wp=wp, vbias=r["vbias"], gen=GEN_TEMPLATE,
                        train=TRAIN_TEMPLATE, BI_analytic=bi_an,
                        BI_mc=res["BI_mc"], sigma_BI=res["sigma_BI"], rp=res["rp"],
                        sigma_rp=res["sigma_rp"], nsim=NSIM, ratio=res["BI_mc"] / bi_an))
        if (ch, wp) in PLOT:
            plot_pair(ch, wp, res, bi_an)

    fields = ["channel", "wp", "vbias", "gen", "train", "BI_analytic", "BI_mc", "sigma_BI", "rp", "sigma_rp",
              "nsim", "ratio"]
    with open(OUT_CSV, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=fields)
        wcsv.writeheader()
        wcsv.writerows(out)
    print(f"\n-> {OUT_CSV}")


if __name__ == "__main__":
    main()
