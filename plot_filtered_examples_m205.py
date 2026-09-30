"""
plot_filtered_examples_m205.py
==============================
Un impulso simulato (singolo e pile-up) PRIMA e DOPO i filtri addestrati, per i tre algoritmi:
    filtro ottimo (OF), Wiener senza penalita' (W), Wiener con penalita' (Wwna).
Ogni pannello: l'impulso originale e le due tracce filtrate con i due filtri di banda,
f1 x kernel e f2 x kernel, sulla stessa scala di tempo. I filtri sono normalizzati a guadagno 1,
quindi le tracce filtrate si leggono nelle stesse unita' dell'impulso.

Evento generato come nel Monte Carlo (template fit x ampiezza della ROI + rumore dalla NPS), con lo
STESSO rumore per il singolo e per il pile-up. Tracce filtrate nella convenzione di
get_PSD_interpole (baseline tolta, picco riportato al centro della finestra); A1, A2 e Y = A1/A2
sono quelli che calcola il Monte Carlo.

Punto di default: ch34 WP3, dove senza penalita' lambda e' crollata (~1e-4) e il BI e' il 36%
peggiore di quello dell'OF.

    python plot_filtered_examples_m205.py                      # il punto di default
    python plot_filtered_examples_m205.py 34 15 0.4 0.6        # canale, WP, r, dt [ms]
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import simulate_BI_error_m205 as mc
import src.analysis as an
import src.dataset as ds

# ── CONFIGURAZIONE ────────────────────────────────────────────────────────────────────────────
CHANNEL, WP = 34, 3
R, DT = 0.5, 0.5e-3           # il pile-up d'esempio: frazione d'energia del secondo impulso, ritardo [s]
SEED = 3                      # rumore dell'evento
T_RANGE = (-10.0, 40.0)       # finestra di tempo disegnata [ms]
ALGOS = [                     # (etichetta, cartella dei filtri)
    ("optimum filter", "m205_results_octopus_APsimfit10000led_npsclean"),
    ("Wiener, no penalty", "m205_results_wiener_APsimfit10000led_npsclean"),
    ("Wiener + penalty", "m205_results_wiener_APsimfit10000led_npsclean_swna1"),
]
C_RAW, C_F1, C_F2 = "0.55", "#2a78d6", "#eb6834"
if len(sys.argv) > 2:                                   # da riga di comando: canale WP [r dt_ms]
    CHANNEL, WP = int(sys.argv[1]), int(sys.argv[2])
if len(sys.argv) > 4:
    R, DT = float(sys.argv[3]), float(sys.argv[4]) * 1e-3


def main():
    nps = mc.load_noise(CHANNEL, WP)
    S, w, _ = an.compute_H(mc.template_pulse(CHANNEL, WP, "fit"), nps, np.hanning, sampling_rate=mc.SAMPLING_RATE)
    n = len(S)
    filters, amp = [], None
    for label, folder in ALGOS:
        f = mc.folder_info(folder)
        row = [r for r in csv.DictReader(open(f["bi_csv"])) if int(r["channel"]) == CHANNEL and int(r["wp"]) == WP][0]
        amp = float(row["signal_amp"])
        lam = row.get("lambda_wiener")
        filters.append((label + (f"  ($\\lambda$ = {float(lam):.2g})" if lam else ""), mc.load_filters(f, CHANNEL, WP, nps, row)))

    # ── l'evento: singolo e pile-up con lo stesso rumore ─────────────────────────────────────
    rng = np.random.default_rng(SEED)
    noise = (rng.normal(size=n) + 1j * rng.normal(size=n)) * np.sqrt(nps)
    events = {"single pulse": np.fft.ifft(amp * S + noise).real,
              f"pile-up  ($\\Delta t$ = {DT*1e3:g} ms, r = {R:g})":
                  np.fft.ifft(amp * S * ((1 - R) + R * np.exp(-1j * w * DT)) + noise).real}

    tgt = n // 2
    t = (np.arange(n) - tgt) / mc.SAMPLING_RATE * 1e3
    m = (t >= T_RANGE[0]) & (t <= T_RANGE[1])
    fig, axes = plt.subplots(len(filters), len(events), figsize=(13, 3.6 * len(filters)), sharex=True,
                             gridspec_kw=dict(hspace=0.22, wspace=0.16))
    for i, (label, (K, f1, f2)) in enumerate(filters):
        dset = ds.NumpyDataset(np.asarray(list(events.values()), np.float32)); dset.win_length = n
        Y, A1, A2 = an.get_PSD_interpole(dset, K, f1, f2)            # quello che usa il MC
        for j, (name, x) in enumerate(events.items()):
            a = axes[i, j]
            x0 = x - np.mean(x[:tgt - 100])                            # baseline, come get_PSD_interpole
            fx = np.fft.fft(x0)
            a.plot(t[m], x0[m], color=C_RAW, lw=1, label="simulated pulse")
            for f, c, lab in ((f1, C_F1, "after band filter 1"), (f2, C_F2, "after band filter 2")):
                a.plot(t[m], np.roll(np.fft.ifft(f * K * fx).real, tgt)[m], color=c, lw=1.6, label=lab)
            a.axhline(0, color="0.8", lw=0.8, zorder=0)
            a.text(0.98, 0.95, f"$A_1$ = {A1[j]/amp:.2f} A\n$A_2$ = {A2[j]/amp:.2f} A\nY = {Y[j]:.3f}",
                   transform=a.transAxes, ha="right", va="top", fontsize=9, color="0.25",
                   bbox=dict(fc="white", ec="none", alpha=0.8))
            if i == 0:
                a.set_title(name, fontsize=12)
            if j == 0:
                a.set_ylabel(f"{label}\namplitude", fontsize=10)
            if i == len(filters) - 1:
                a.set_xlabel("time from the pulse [ms]")
        print(f"{label:40s} singolo Y = {Y[0]:.3f}   pile-up Y = {Y[1]:.3f}")
    axes[0, 0].legend(fontsize=9, loc="lower right", framealpha=0.9)
    fig.suptitle(f"m205 ch{CHANNEL} WP{WP}: a simulated event before and after the trained filters "
                 f"(A = {amp:.3g}, same noise in both columns)", fontsize=12, y=0.93)
    out = os.path.join(mc.BASE_DIR, f"filtered_examples_ch{CHANNEL}_wp{WP}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
