"""
visual_pulse_m205.py
====================
Tool INTERATTIVO per scegliere a occhio la STRUTTURA e i guess del fit pole-zero(+CC)+Bessel
su un average pulse della run 205, da riusare in fit_pulses_m205.py. Adattamento di
test/visual_pulse_test.py ai dati m205 (AP da ROOT + overlay del dato).

Interfaccia:
  - grafico AP (nero) + modello (rosso) e, sotto, il RESIDUO/sigma che si aggiorna live;
  - RadioButtons per la STRUTTURA del modello: n. poli reali, n. zeri, coppia complessa
    coniugata CC (No/Yes), ordine del Bessel (0 = nessuno);
  - per OGNI parametro: uno SLIDER e una TEXTBOX (inserimento da tastiera), sincronizzati;
  - in alto e' scritto ESPLICITAMENTE il modello in uso.
I valori correnti sono i guess: copiali in fit_pulses_m205.py. Il modello coincide con
make_pulse_pole_zero_bessel_ct nel caso 1 zero + poli reali (guess compatibili).

NB: serve un backend GRAFICO interattivo (esegui in locale, non headless).
Run:  KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/bin/python3.13 test/visual_pulse_m205.py
"""

import sys
import csv
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, TextBox
from scipy.signal import besselap
from scipy.optimize import minimize_scalar
import uproot

BASE = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE))

# ═════════════════════════════════════════════════════════════════════════════
# Config
# ═════════════════════════════════════════════════════════════════════════════
# SOURCE = "root"  -> average pulse dal file ROOT del canale/WP (comportamento iniziale).
# SOURCE = "cfile" -> impulso da una macro ROOT .C in Processed/ (TGraph Graph_fx*=tempo[s],
#   Graph_fy*=ampiezza; peak-normalizzato a 1, stessa struttura degli AP). I preset di
#   fit_one_pulse per i .C sono salvati con channel=<nome file> (es. "LED1"), wp=0.
SOURCE    = "root"     # "root"  oppure  "cfile"
CFILE     = "LED1.C"   # nome del .C in Processed/  (solo se SOURCE="cfile")

CHANNEL   = 91
WP        = 1
MEAS      = "000205"
DATA_DIR  = BASE / "Processed"
HIST_TMPL = "averagepulse_ap_wp{wp}_medianAP"
VBIAS_LIST = np.array([0.6, 1.0, 1.4, 1.8, 2, 3, 4, 5, 6, 8, 10, 20, 26, 30, 40])
MAX_POLES = 6
MAX_ZEROS = 5
INIT = dict(n_real=4, n_zeros=1, cc=False, order=6, fcut=2500)   # struttura iniziale
ORDER_CHOICES = [0, 2, 4, 6]


def load_ap(channel, wp):
    """AP peak-normalizzato e asse tempi (s) CENTRATO sul picco, dal file ROOT di m205."""
    fp = glob.glob(str(DATA_DIR / f"Processed_*_{MEAS}_{channel}.root"))
    if not fp:
        sys.exit(f"[ERROR] ROOT del canale {channel} non trovato in {DATA_DIR}")
    with uproot.open(fp[0]) as f:
        h = f[HIST_TMPL.format(wp=wp)]
        v = np.asarray(h.values(), dtype=float)
        t = np.asarray(h.axis().centers(), dtype=float)
    v = v / v.max()
    t_peak = t[int(np.argmax(v))]
    return t - t_peak, v, t_peak            # asse centrato sul picco + istante del picco (per convertire t0)


def load_c_pulse(cfile):
    """Impulso da una macro ROOT .C (TGraph) in Processed/: estrae Graph_fx*=tempo[s] e
    Graph_fy*=ampiezza, peak-normalizza a 1 e ritorna (t-t_peak, v, t_peak) come load_ap
    (asse centrato sul picco + istante assoluto del picco, per convertire t0)."""
    import re
    path = DATA_DIR / cfile
    if not path.exists():
        sys.exit(f"[ERROR] file .C non trovato: {path}")
    txt = path.read_text()
    def grab(kind):                                       # kind = "x" (tempo) / "y" (ampiezza)
        m = re.search(r"Graph_f%s\d*\[\d+\]\s*=\s*\{(.*?)\}" % kind, txt, re.S)
        if m is None:
            sys.exit(f"[ERROR] array Graph_f{kind}* non trovato in {path}")
        return np.fromstring(m.group(1), sep=",")
    t, v = grab("x"), grab("y")
    v = v / v.max()
    t_peak = t[int(np.argmax(v))]
    return t - t_peak, v, t_peak


def load_top_results(channel, wp, n=5):
    """Legge i top-N modelli (dallo scan Bessel) per (channel, wp) dal CSV
    residual_scan_bessel/top{n}_fit_params_ch<ch>.csv. Ritorna lista di preset ordinati per rank."""
    path = BASE / "residual_scan_bessel" / f"top{n}_fit_params_ch{channel}.csv"
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if int(r["channel"]) != channel or int(r["wp"]) != wp:
                continue
            zeros = [float(x) for x in r["zeros"].split()] if r["zeros"].strip() else []
            real = [float(x) for x in r["real_poles"].split()]
            cc = int(r["cc"]) == 1
            out.append(dict(rank=int(r["rank"]), model=r["model"], n_real=int(r["n_real"]),
                            n_zeros=int(r["nzer"]), cc=cc, order=int(r["bessel_order"]),
                            fcut=float(r["fcut"]), t0=float(r["t0"]), zeros=zeros, real=real,
                            cc_sigma=float(r["cc_sigma"]) if r["cc_sigma"] else None,
                            cc_omega=float(r["cc_omega"]) if r["cc_omega"] else None,
                            rms_sigma=float(r["rms_over_sigma"])))
    out.sort(key=lambda d: d["rank"])
    return out


def load_fit_one_pulse(channel, wp):
    """Carica TUTTI i modelli fittati da fit_one_pulse (test/fit_one_pulse_params.csv, un
    modello per riga) per (channel, wp). Stesso formato dei preset dello scan -> ognuno
    selezionabile dal menu' 'start from' per riprodurre ESATTAMENTE quell'impulso."""
    path = Path(__file__).resolve().parent / "fit_one_pulse_params.csv"
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["channel"] != str(channel) or r["wp"] != str(wp):   # chiave stringa: vale sia per int (root) sia per label .C
                continue
            zeros = [float(x) for x in r["zeros"].split()] if r["zeros"].strip() else []
            real = [float(x) for x in r["real_poles"].split()]
            out.append(dict(rank=0, model=r["model"], n_real=int(r["n_real"]),
                            n_zeros=int(r["nzer"]), cc=int(r["cc"]) == 1, order=int(r["bessel_order"]),
                            fcut=float(r["fcut"]), t0=float(r["t0"]), zeros=zeros, real=real,
                            cc_sigma=float(r["cc_sigma"]) if r["cc_sigma"] else None,
                            cc_omega=float(r["cc_omega"]) if r["cc_omega"] else None,
                            rms_sigma=float(r["rms_over_sigma"])))
    out.sort(key=lambda d: d["model"])
    return out


def make_pulse_bessel_general(bessel_order, fcut, zeros, poles):
    """Pole-zero (poli reali e/o complessi = coppia CC, N zeri) convoluto con Bessel analogico
    (ordine bessel_order, taglio fcut Hz). Normalizzato a picco 1 a t=0. Generalizza
    src.simulation.make_pulse_pole_zero_bessel_ct a N zeri e poli complessi."""
    poles = np.asarray(poles, dtype=complex)
    zeros = np.asarray(zeros, dtype=complex)
    wc = 2 * np.pi * fcut
    diff = poles[:, None] - poles[None, :]
    denom = np.prod(np.where(np.eye(len(poles), dtype=bool), 1.0, diff), axis=1)
    num = np.ones(len(poles), dtype=complex)
    for z in zeros:
        num *= (poles - z)
    k = num / denom
    if bessel_order == 0:
        B, exp_p = k, poles

        def f_raw(t):
            t = np.asarray(t, float); out = np.zeros_like(t, dtype=complex)
            mask = t >= 0; tt = t[mask]
            out[mask] = np.sum(B[:, None] * np.exp(exp_p[:, None] * tt), axis=0)
            return out.real
    else:
        _, p_norm, g_norm = besselap(bessel_order)
        p_filt = wc * p_norm; g_filt = g_norm * wc ** bessel_order
        diff_f = p_filt[:, None] - p_filt[None, :]
        denom_f = np.prod(np.where(np.eye(len(p_filt), dtype=bool), 1.0, diff_f), axis=1)
        A = g_filt / denom_f
        coef = k[:, None] * A[None, :] / (poles[:, None] - p_filt[None, :])
        B = np.sum(coef, axis=1); C = -np.sum(coef, axis=0)

        def f_raw(t):
            t = np.asarray(t, float); out = np.zeros_like(t, dtype=complex)
            mask = t >= 0; tt = t[mask]
            out[mask] = (np.sum(B[:, None] * np.exp(poles[:, None] * tt), axis=0)
                         + np.sum(C[:, None] * np.exp(p_filt[:, None] * tt), axis=0))
            return out.real

    # Ricerca del picco per normalizzare a 1 (identica a fit_one_pulse):
    #  - poli tutti REALI -> risposta unimodale -> minimize_scalar (continuo, preciso);
    #  - coppia CC (poli complessi) -> puo' oscillare -> massimo globale su griglia (robusto).
    if np.any(np.abs(poles.imag) > 1e-12):
        tg = np.linspace(0.0, 0.15, 3000)
        tpk = tg[int(np.argmax(f_raw(tg)))]
    else:
        tpk = minimize_scalar(lambda x: -f_raw(x), bounds=(0, 0.1), method="bounded").x
    pk = f_raw(tpk)
    pk = pk if pk > 0 else 1.0
    return lambda tt: f_raw(np.asarray(tt, float) + tpk) / pk


def physical_defaults(t, v):
    """Default (1/s) dai tempi misurati (formula dell'engine): fast=t_rise/3, slow=max(5*t_dec,
    10*t_rise). Riempie MAX_POLES poli, MAX_ZEROS zeri, e la coppia CC (sigma, omega)."""
    peak = float(v.max()); imax = int(np.argmax(v)); dt = t[1] - t[0]
    up = v[:imax]
    i10 = np.where(up > 0.1 * peak)[0]; i90 = np.where(up > 0.9 * peak)[0]
    t_rise = max((t[i90[0]] - t[i10[0]]) if (len(i10) and len(i90)) else 5 * dt, dt)
    be = np.where(v[imax:] < peak / np.e)[0]
    t_dec = (t[imax + be[0]] - t[imax]) if len(be) else 10 * dt
    fast, slow = t_rise / 3.0, max(t_dec * 5, t_rise * 10)
    poles = np.sort(-1.0 / np.geomspace(fast, slow, MAX_POLES))          # da veloce a lento
    zeros = np.sort(-1.0 / np.geomspace(t_rise / 3, t_rise * 3, MAX_ZEROS))
    cc_sigma = -1.0 / t_dec
    cc_omega = 2 * np.pi / t_dec
    return poles, zeros, cc_sigma, cc_omega


def model_str(n_real, n_zeros, cc, order, fcut):
    s = f"{n_real} real pole{'s' if n_real != 1 else ''} + {n_zeros} zero{'s' if n_zeros != 1 else ''}"
    if cc:
        s += " + CC pair"
    s += f", Bessel order {order} @ {fcut:.0f} Hz" if order > 0 else ", no Bessel"
    return s


def main():
    # sorgente del dato: AP da ROOT (canale/WP) oppure impulso da .C. In entrambi i casi
    # (t centrato sul picco, v peak-norm, t_peak assoluto). ch_key/wp_key sono le chiavi con
    # cui cercare i preset di fit_one_pulse (per i .C: nome file e wp=0); src_label per titoli.
    if SOURCE == "cfile":
        t, v, t_peak_raw = load_c_pulse(CFILE)
        src_label = Path(CFILE).stem                   # es. "LED1"
        ch_key, wp_key, head_txt = src_label, 0, src_label
    else:
        t, v, t_peak_raw = load_ap(CHANNEL, WP)
        src_label = f"Ch {CHANNEL} · WP {WP}"
        ch_key, wp_key = CHANNEL, WP
        head_txt = f"Ch {CHANNEL} · WP {WP} ({VBIAS_LIST[WP // 2]:g} V)"

    sigma = float(v[:int(0.40 * len(v))].std()) or 1.0
    p_def, z_def, cc_sig, cc_om = physical_defaults(t, v)
    st = dict(INIT)                                    # struttura corrente (mutabile)
    guard = {"busy": False}                            # evita ricorsione slider<->textbox

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0.05, 0.57, 0.52, 0.37])
    axr = fig.add_axes([0.05, 0.42, 0.52, 0.12], sharex=ax)
    ax.plot(t * 1e3, v, ".", ms=3, color="k", label="pulse data" if SOURCE == "cfile" else "AP data")
    mline, = ax.plot(t * 1e3, np.zeros_like(t), "-", lw=1.8, color="#c1121f", label="model")
    ax.set_xlim(-5, 60); ax.set_ylim(-0.05, 1.08)
    ax.set_ylabel("AP amp (peak-norm.)"); ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right"); ax.tick_params(labelbottom=False)
    title = ax.set_title("", fontsize=12, fontweight="bold")
    axr.axhspan(-3, 3, color="#4a90d9", alpha=0.2, lw=0)
    axr.axhline(0, color="gray", ls=":", lw=0.7)
    rline, = axr.plot(t * 1e3, np.zeros_like(t), "-", lw=0.9, color="#c1121f")
    axr.set_ylabel("residual / σ"); axr.set_xlabel("t - t_peak [ms]"); axr.grid(True, alpha=0.3)
    rms_txt = fig.text(0.31, 0.955, "", fontsize=11, family="monospace", ha="center")

    # ── righe parametro: (slider + textbox) sincronizzati ─────────────────────
    rows = ["t0", "fcut"] + [f"p{i}" for i in range(1, MAX_POLES + 1)] \
           + [f"z{i}" for i in range(1, MAX_ZEROS + 1)] + ["cc_sigma", "cc_omega"]
    # t0 in SECONDI ASSOLUTI (come fit_one_pulse), range ±6 ms attorno al picco dei dati
    ranges = {"t0": (t_peak_raw - 0.006, t_peak_raw + 0.006), "fcut": (200, 5000),
              "cc_sigma": (-1e5, -1), "cc_omega": (1, 40000)}
    for i in range(1, MAX_POLES + 1):
        ranges[f"p{i}"] = (-1e5, -1)
    for i in range(1, MAX_ZEROS + 1):
        ranges[f"z{i}"] = (-1e5, -1)
    init_val = {"t0": t_peak_raw, "fcut": st["fcut"], "cc_sigma": cc_sig, "cc_omega": cc_om}
    for i in range(MAX_POLES):
        init_val[f"p{i+1}"] = p_def[i]
    for i in range(MAX_ZEROS):
        init_val[f"z{i+1}"] = z_def[i]

    def fmt(name, val):
        return f"{val:.5f}" if name == "t0" else f"{val:.0f}"

    S, T = {}, {}
    y0 = 0.93
    for r in rows:
        lo, hi = ranges[r]
        ax_s = fig.add_axes([0.63, y0, 0.20, 0.026], facecolor="lightgoldenrodyellow")
        ax_t = fig.add_axes([0.86, y0, 0.055, 0.026])
        S[r] = Slider(ax_s, r, lo, hi, valinit=init_val[r], valfmt="%.5f" if r == "t0" else "%.0f")
        T[r] = TextBox(ax_t, "", initial=fmt(r, init_val[r]))
        y0 -= 0.056

    # ── RadioButtons di struttura ─────────────────────────────────────────────
    def radio(rect, title_, labels, active):
        a = fig.add_axes(rect, facecolor="lightgoldenrodyellow"); a.set_title(title_, fontsize=9)
        return RadioButtons(a, labels, active=active)

    rb_nreal = radio([0.05, 0.05, 0.075, 0.32], "n real poles",
                     [str(i) for i in range(1, MAX_POLES + 1)], st["n_real"] - 1)
    rb_nzero = radio([0.14, 0.05, 0.075, 0.32], "n zeros",
                     [str(i) for i in range(0, MAX_ZEROS + 1)], st["n_zeros"])
    rb_cc = radio([0.23, 0.22, 0.09, 0.11], "CC pair", ["No", "Yes"], 1 if st["cc"] else 0)
    rb_ord = radio([0.23, 0.05, 0.09, 0.14], "Bessel order",
                   [("0 (none)" if o == 0 else str(o)) for o in ORDER_CHOICES],
                   ORDER_CHOICES.index(st["order"]))

    def visible_rows():
        vis = {"t0": True, "fcut": st["order"] > 0}
        for i in range(1, MAX_POLES + 1):
            vis[f"p{i}"] = i <= st["n_real"]
        for i in range(1, MAX_ZEROS + 1):
            vis[f"z{i}"] = i <= st["n_zeros"]
        vis["cc_sigma"] = vis["cc_omega"] = st["cc"]
        return vis

    def apply_visibility():
        for r, on in visible_rows().items():
            S[r].ax.set_visible(on); T[r].ax.set_visible(on)

    def current_poles_zeros():
        real = [S[f"p{i}"].val for i in range(1, st["n_real"] + 1)]
        zeros = [S[f"z{i}"].val for i in range(1, st["n_zeros"] + 1)]
        if st["cc"]:
            sig, om = S["cc_sigma"].val, S["cc_omega"].val
            poles = real + [complex(sig, om), complex(sig, -om)]
        else:
            poles = real
        return poles, zeros

    def update(_=None):
        poles, zeros = current_poles_zeros()
        try:
            # t0 ASSOLUTO in secondi (come fit_one_pulse): t e' centrato sul picco,
            # quindi il tempo assoluto e' (t + t_peak_raw); il modello e' f(t_abs - t0).
            y = make_pulse_bessel_general(st["order"], S["fcut"].val, zeros, poles)(t + t_peak_raw - S["t0"].val)
            ok = np.all(np.isfinite(y))
        except Exception:
            ok = False
        if ok:
            mline.set_ydata(y)
            r = (v - y) / sigma
            rline.set_ydata(r)
            axr.relim(); axr.autoscale_view(scalex=False)
            rms = float(np.sqrt(np.mean(r ** 2)))
            rms_txt.set_text(f"RMS/σ = {rms:.1f}")
        title.set_text(f"MODEL: {model_str(st['n_real'], st['n_zeros'], st['cc'], st['order'], S['fcut'].val)}"
                       f"    —    {head_txt}")
        fig.canvas.draw_idle()

    # ── sincronizzazione slider <-> textbox (con guardia anti-ricorsione) ─────
    def on_slider(name):
        def cb(val):
            if guard["busy"]:
                return
            guard["busy"] = True
            T[name].set_val(fmt(name, val))
            guard["busy"] = False
            update()
        return cb

    def on_text(name):
        def cb(text):
            if guard["busy"]:
                return
            try:
                val = float(text)
            except ValueError:
                return
            lo, hi = ranges[name]
            val = min(max(val, lo), hi)
            guard["busy"] = True
            S[name].set_val(val)
            guard["busy"] = False
            update()
        return cb

    for r in rows:
        S[r].on_changed(on_slider(r))
        T[r].on_submit(on_text(r))

    def on_struct(_=None):
        if guard["busy"]:                              # ignora i set_active fatti da apply_preset
            return
        st["n_real"] = int(rb_nreal.value_selected)
        st["n_zeros"] = int(rb_nzero.value_selected)
        st["cc"] = (rb_cc.value_selected == "Yes")
        st["order"] = ORDER_CHOICES[[("0 (none)" if o == 0 else str(o))
                                     for o in ORDER_CHOICES].index(rb_ord.value_selected)]
        apply_visibility()
        update()

    for rb in (rb_nreal, rb_nzero, rb_cc, rb_ord):
        rb.on_clicked(on_struct)

    # ── "start from": tendina (RadioButtons) coi 5 migliori fit dello scan Bessel ─
    presets = load_top_results(CHANNEL, WP) if SOURCE == "root" else []   # scan Bessel: solo ROOT
    fops = load_fit_one_pulse(ch_key, wp_key)         # fit di fit_one_pulse (uno per modello; per i .C keyati sul nome file)
    phys = dict(n_real=INIT["n_real"], n_zeros=INIT["n_zeros"], cc=INIT["cc"], order=INIT["order"],
                fcut=INIT["fcut"], t0=t_peak_raw, zeros=list(z_def[:INIT["n_zeros"]]),
                real=list(p_def[:INIT["n_real"]]), cc_sigma=cc_sig, cc_omega=cc_om)
    all_presets = [phys] + fops + presets
    labels = (["physical init"]
              + [f"fit_one_pulse {p['model']} (σ={p['rms_sigma']:.0f})" for p in fops]
              + [f"#{p['rank']} {p['model']} (σ={p['rms_sigma']:.0f})" for p in presets])
    rb_preset = radio([0.35, 0.05, 0.21, 0.32],
                      f"start from  ({src_label} best fits)", labels, 0)

    def apply_preset(p):
        guard["busy"] = True                           # blocca i callback durante il set massivo
        st.update(n_real=p["n_real"], n_zeros=p["n_zeros"], cc=p["cc"], order=p["order"])
        rb_nreal.set_active(p["n_real"] - 1)
        rb_nzero.set_active(p["n_zeros"])
        rb_cc.set_active(1 if p["cc"] else 0)
        rb_ord.set_active(ORDER_CHOICES.index(p["order"]))

        def setv(name, val):
            lo, hi = ranges[name]
            v_ = min(max(val, lo), hi)
            S[name].set_val(v_); T[name].set_val(fmt(name, v_))

        setv("t0", p["t0"])                            # t0 ASSOLUTO in secondi (come fit_one_pulse)
        setv("fcut", p["fcut"])
        for i, val in enumerate(p["real"], 1):
            setv(f"p{i}", val)
        for i, val in enumerate(p["zeros"], 1):
            setv(f"z{i}", val)
        if p.get("cc_sigma") is not None:
            setv("cc_sigma", p["cc_sigma"]); setv("cc_omega", p["cc_omega"])
        guard["busy"] = False
        apply_visibility()
        update()

    def on_preset(label):
        if guard["busy"]:
            return
        apply_preset(all_presets[labels.index(label)])

    rb_preset.on_clicked(on_preset)

    apply_visibility()
    update()
    plt.show()


if __name__ == "__main__":
    main()
