#!/usr/bin/env python3
"""
study_grid_m205.py
==================
Studio delle griglie di M (src/pileup_likelihood.py): quanto devono essere fitte in dt e in r perche' il rigetto
non cambi. Stessi eventi della campagna (simulate_BI_error_m205.py: generatore v2, stesso seed, stessa NPS e
template), generati UNA volta per punto: ogni blocco passa per tutte le varianti di M, quindi le differenze di
R fra varianti sono appaiate e hanno un errore molto piu' piccolo di R.

Varianti (il resto di M come nella campagna):
  campaign  passo 10 us in dt (e nei tempi d'arrivo), r = 0.05, 0.10, ..., 0.95 (19 valori)
  N<n>      scan in dt: passo = dt_max / n, n in DT_N; r come la campagna. Il passo dei tempi d'arrivo e'
            lo stesso del passo in dt, come in M. Riferimento: il passo piu' fine.
  r<n>      scan in r: passo 10 us; r da 0.05 a 0.95 col passo della campagna dimezzato una, due, tre
            volte (37, 73, 145 valori). Riferimento: il piu' fitto. campaign = r19.

Un punto (sul server, un job per punto, vedi sotto):
    python study_grid_m205.py CH WP [--nsim 20000]       -> grid_study_m205/grid_ch{CH}_wp{WP}.npz
Analisi (in locale, sui .npz sincronizzati):
    python study_grid_m205.py                            -> grid_study_m205/grid_study_m205.csv, grid_{dt,r}_m205.png

Controllo: se c'e' il file eventi della campagna (SAVE_EVENTS), M della variante campaign deve coincidere evento
per evento con quello salvato lì (stessi eventi, stesso M): il run si ferma se non e' cosi'.

Sul server, dalla cartella del progetto, i 15 punti (WP1, WP15, WP29 di ogni canale). Memoria: picco misurato sul
Mac 0.91 GB (ch31 WP29: le matrici di N320) e 0.79 GB (ch34 WP1: r145), di cui 0.29 GB solo per gli import;
1500 MB lasciano margine per Linux. Ogni job stampa il suo picco: si puo' scendere quando li abbiamo visti.
    for ch in 31 34 71 83 91; do for wp in 1 15 29; do echo "cd \\$PBS_O_WORKDIR && source /home/zanelli/LoadOctopus.sh && KMP_DUPLICATE_LIB_OK=TRUE python study_grid_m205.py $ch $wp" | qsub -q cupid -N grid_${ch}_${wp} -j oe -l walltime=02:00:00 -l mem=1500mb; done; done
"""
import os
import sys
import csv
import glob
import time
import argparse
import resource

import numpy as np
import simulate_BI_error_m205 as s

OUT_DIR = os.path.join(s.BASE_DIR, "grid_study_m205")
CHANNELS, WPS = (31, 34, 71, 83, 91), (1, 15, 29)
NSIM = 20_000
DT_N = (10, 20, 40, 80, 160, 320)          # dt_max / passo
R_STEPS = (0.025, 0.0125, 0.00625)          # passo in r (la campagna: 0.05)
CAMP_STEP, CAMP_R = 1e-5, np.linspace(0.05, 0.95, 19)
B = 200                                     # ricampionamenti bootstrap per l'errore di Delta R
SUB = 100                                   # eventi per chiamata di M: meno memoria, stessi eventi (il blocco
                                            # generato resta CHUNK della campagna, che fissa la sequenza casuale)
COLORS = ("#2a78d6", "#eb6834", "#1b9e77")  # WP basso, medio, alto
INK, MUTED = "#222222", "#8a8a8a"


def variants(dt_max):
    """{nome: opzioni di PileupLikelihood}."""
    v = {"campaign": dict(step=CAMP_STEP, r_values=CAMP_R)}
    v.update({f"N{n}": dict(step=dt_max / n, r_values=CAMP_R) for n in DT_N})
    for d in R_STEPS:
        r = np.round(np.arange(0.05, 0.95 + d / 2, d), 6)
        v[f"r{len(r)}"] = dict(step=CAMP_STEP, r_values=r)
    return v


def run_point(ch, wp, nsim, out_dir):
    row = [x for x in csv.DictReader(open(s.FOLDERS[0]["bi_csv"])) if int(x["channel"]) == ch and int(x["wp"]) == wp][0]
    vbias, signal_amp = float(row["vbias"]), float(row["signal_amp"])
    dt_max, K, jitter = s.point_dt(ch, vbias)
    meanpulse, nps = s.load_row_inputs(ch, wp)                       # chi GENERA gli eventi, come run_pair
    S, w, _ = s.an.compute_H(meanpulse, nps, np.hanning, sampling_rate=s.SAMPLING_RATE)
    opts = variants(dt_max)
    liks = {v: s.make_likelihood(ch, wp, dt_max, **o) for v, o in opts.items()}
    kw = dict(n_photons=s.fn.n_photons(ch, cupid=s.CUPID), e_roi=s.fn.E_ROI_MEV,
              shift_max=1.0 / s.SAMPLING_RATE, jitter_max=jitter, nsim=nsim)
    out, secs, best = {}, dict.fromkeys(liks, 0.0), {}

    def stat(v):
        def f(pulses):
            t0 = time.perf_counter()
            M, d, r = (np.concatenate(x) for x in zip(*(liks[v].statistic(pulses[i:i + SUB], best=True)
                                                         for i in range(0, len(pulses), SUB))))
            secs[v] += time.perf_counter() - t0
            best.setdefault(v, []).append((d, r))
            return M
        return f

    for pop, dm in (("single", 0.0), ("pileup", dt_max)):
        best.clear()
        info = []
        Ms = s.simulate_psd_v2(S, nps, w, [], signal_amp, dm, s.SEED, tuple(stat(v) for v in liks), info_out=info, **kw)
        for v, M in zip(liks, Ms):
            out[f"M_{pop}_{v}"] = M.astype(np.float32)
            if pop == "pileup":
                out[f"dt_hat_{v}"] = np.concatenate([d for d, _ in best[v]]).astype(np.float32)
                out[f"r_hat_{v}"] = np.concatenate([r for _, r in best[v]]).astype(np.float32)
        if pop == "pileup":
            out["dt"] = np.concatenate([i["dt"] for i in info]).astype(np.float32)
            out["r"] = np.concatenate([i["r"] for i in info]).astype(np.float32)

    ev = os.path.join(s.likelihood_dir(), "events", f"events_ch{ch}_wp{wp}_seed{s.SEED}{s._V2}.npz")
    if os.path.exists(ev):                    # stessi eventi e stesso M della campagna
        z = np.load(ev)
        for pop in ("single", "pileup"):
            n = min(nsim, len(z[f"M_{pop}"]))
            assert np.allclose(out[f"M_{pop}_campaign"][:n], z[f"M_{pop}"][:n], rtol=1e-5, atol=1e-5), \
                f"M della variante campaign diverso dalla campagna ({pop}): eventi o M non sono gli stessi"
        print(f"OK  campaign = file eventi della campagna, primi {n} eventi")
    names = list(liks)
    path = os.path.join(out_dir, f"grid_ch{ch}_wp{wp}.npz")
    np.savez_compressed(path, channel=ch, wp=wp, vbias=vbias, dt_max=dt_max, K=K, nsim=nsim, variants=names,
                        step=[opts[v]["step"] for v in names], n_r=[len(opts[v]["r_values"]) for v in names],
                        seconds=[secs[v] for v in names], **out)
    for v in names:
        print(f"{v:9s} passo {opts[v]['step'] * 1e6:6.2f} us  n_r {len(opts[v]['r_values']):3d}  "
              f"{1e3 * secs[v] / (2 * nsim):7.2f} ms/evento", flush=True)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e6 if sys.platform == "darwin" else 1e3)
    print(f"-> {os.path.relpath(path, s.BASE_DIR)}   memoria di picco {peak:.0f} MB")


# ── analisi ───────────────────────────────────────────────────────────────────────────────────────
def rejection(Ms, Mp):
    """R al 90% di accettanza dei singoli: M alto = pile-up."""
    return np.mean(Mp > np.percentile(Ms, 100 * s.ACCEPTANCE))


def summarize(z):
    """Righe del CSV di un punto: R, Delta R (bootstrap appaiato) rispetto al riferimento del suo scan, stime."""
    names = [str(v) for v in z["variants"]]
    n = len(z["M_single_campaign"])
    rng = np.random.default_rng(0)
    i_s, i_p = rng.integers(0, n, (B, n)), rng.integers(0, len(z["dt"]), (B, len(z["dt"])))
    R = {v: rejection(z[f"M_single_{v}"], z[f"M_pileup_{v}"]) for v in names}
    boot = {v: np.array([rejection(z[f"M_single_{v}"][a], z[f"M_pileup_{v}"][b]) for a, b in zip(i_s, i_p)]) for v in names}
    ref = {"dt": f"N{max(DT_N)}", "r": max((v for v in names if v.startswith("r")), key=lambda v: int(v[1:]))}
    rows = []
    for i, v in enumerate(names):
        tag = z[f"M_pileup_{v}"] > np.percentile(z[f"M_single_{v}"], 100 * s.ACCEPTANCE)
        for scan in ("dt", "r"):
            if not (v == "campaign" or v[0] == ("N" if scan == "dt" else "r")):
                continue
            rows.append(dict(channel=int(z["channel"]), wp=int(z["wp"]), vbias=float(z["vbias"]),
                             dt_max_ms=1e3 * float(z["dt_max"]), scan=scan, variant=v,
                             N=float(z["dt_max"]) / float(z["step"][i]), step_us=1e6 * float(z["step"][i]),
                             n_r=int(z["n_r"][i]), R=R[v], dR=R[v] - R[ref[scan]],
                             sigma_dR=float(np.std(boot[v] - boot[ref[scan]])), sigma_R=float(np.std(boot[v])),
                             BI=float(z["K"]) * (1 - R[v]),
                             err_dt_us=1e6 * float(np.median(np.abs(z[f"dt_hat_{v}"] - z["dt"])[tag])),
                             err_r=float(np.median(np.abs(z[f"r_hat_{v}"] - z["r"])[tag])),
                             ms_per_event=1e3 * float(z["seconds"][i]) / (2 * n)))
    return rows


def plot(rows, scan, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    x_of = (lambda r: r["N"]) if scan == "dt" else (lambda r: r["n_r"])
    quantities = (("dR", "ΔR vs finest grid [points of %]", 100),
                  ("err_dt_us" if scan == "dt" else "err_r",
                   "median |Δt̂ − Δt|, tagged [µs]" if scan == "dt" else "median |r̂ − r|, tagged", 1),
                  ("ms_per_event", "time of M [ms / event]", 1))
    chans = sorted({r["channel"] for r in rows})
    fig, axes = plt.subplots(3, len(chans), figsize=(3.8 * len(chans), 9.5), sharex=True, squeeze=False,
                             layout="constrained")
    for j, ch in enumerate(chans):
        wps = sorted({r["wp"] for r in rows if r["channel"] == ch})
        for k, wp in enumerate(wps):
            pts = sorted((r for r in rows if r["channel"] == ch and r["wp"] == wp and r["scan"] == scan
                          and (scan == "r" or r["variant"] != "campaign")), key=x_of)
            camp = [r for r in rows if r["channel"] == ch and r["wp"] == wp and r["scan"] == scan and r["variant"] == "campaign"]
            lab = f"WP {wp} ({pts[0]['vbias']:g} V, dt_max {pts[0]['dt_max_ms']:.2f} ms)"
            for ax, (q, _, f) in zip(axes[:, j], quantities):
                x, y = [x_of(r) for r in pts], [f * r[q] for r in pts]
                yerr = [f * r["sigma_dR"] for r in pts] if q == "dR" else None
                ax.errorbar(x, y, yerr=yerr, color=COLORS[k % 3], marker="o", ms=4, lw=1.5, capsize=2, label=lab)
                if scan == "dt" and camp:      # la griglia della campagna (10 us): un punto a parte
                    c = camp[0]
                    ax.errorbar(x_of(c), f * c[q], yerr=f * c["sigma_dR"] if q == "dR" else None, color=COLORS[k % 3],
                                marker="*", ms=11, ls="none", capsize=2)
        axes[0, j].set_title(f"Ch {ch}")
        axes[0, j].axhline(0, color=MUTED, lw=1)
        axes[0, j].legend(fontsize=8, frameon=False)
        axes[-1, j].set_xlabel("dt_max / step (grid points in Δt)" if scan == "dt" else "number of r values")
        ticks = DT_N if scan == "dt" else sorted({r["n_r"] for r in rows if r["scan"] == "r"})
        for ax in axes[:, j]:
            ax.set_xscale("log")
            ax.set_xticks(ticks, [str(x) for x in ticks])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_xlim(min(ticks) / 1.3, max(ticks) * 1.3)
            ax.spines[["top", "right"]].set_visible(False)
    for a in axes[-1]:
        a.set_yscale("log")
        a.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
        a.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for ax, (_, lab, _) in zip(axes[:, 0], quantities):
        ax.set_ylabel(lab)
    what = ("Δt grid of M (step = dt_max / N, also for the arrival time; r as in the campaign). "
            "Stars: campaign grid, 10 µs" if scan == "dt" else
            "r grid of M (0.05 ... 0.95, step 0.05 / 2^k; Δt step 10 µs as in the campaign). 19 values = campaign")
    fig.suptitle(f"{what}\nΔR vs the finest grid, same events (bootstrap error); 90% singles acceptance",
                 x=0.01, ha="left", fontsize=11)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"-> {os.path.relpath(out, s.BASE_DIR)}")


def analyse(out_dir):
    rows = [r for f in sorted(glob.glob(os.path.join(out_dir, "grid_ch*_wp*.npz"))) for r in summarize(dict(np.load(f)))]
    if not rows:
        raise SystemExit(f"nessun grid_ch*_wp*.npz in {out_dir}")
    path = os.path.join(out_dir, "grid_study_m205.csv")
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"-> {os.path.relpath(path, s.BASE_DIR)}")
    for scan in ("dt", "r"):
        plot(rows, scan, os.path.join(out_dir, f"grid_{scan}_m205.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("point", type=int, nargs="*", help="CH WP: simula un punto; niente: analisi")
    ap.add_argument("--nsim", type=int, default=NSIM, help="eventi per popolazione")
    ap.add_argument("--out-dir", default=OUT_DIR)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.point:
        run_point(*a.point, a.nsim, a.out_dir)
    else:
        analyse(a.out_dir)


if __name__ == "__main__":
    main()
