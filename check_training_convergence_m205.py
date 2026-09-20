#!/usr/bin/env python3
"""
check_training_convergence_m205.py
==================================
L'addestramento dei filtri e' arrivato a regime? Legge le storie salvate da
analysis_BI_m205_wiener_regolarized.py (`<RESULTS>/training_history/hist_ch<ch>_wp<wp>.npz`,
chiavi J e lam) e, per ogni punto, misura:

  calo      [%] quanto la funzione di costo e' ancora scesa nell'ULTIMO 10% dei passi. E' la
                loss (J + penalita') se la campagna l'ha salvata, altrimenti J, in relativo. J e' quella
                FISICA (senza penalita') ed e' anche il BI: BI = J[-1] * K. Se qui c'e' ancora
                un calo, il BI dipende da quando ci si e' fermati.
  residuo_J [%] quanto scenderebbe ANCORA J continuando all'infinito, stimato dai decrementi
                finali: se calano in modo geometrico di ragione rho, il residuo e'
                dJ_ultimo * rho/(1-rho). rho >= 1 (nessun segno di arresto) -> "inf".
  deriva_l  [%] di quanto e' cambiato lambda nell'ultimo 10% dei passi.

Config: RESULTS_NAME e TAIL. Uso:
    KMP_DUPLICATE_LIB_OK=TRUE python3 check_training_convergence_m205.py
"""
import os, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_NAME = "m205_results_wiener_APsimfit10000led_npsclean_swna1"
TAIL = 0.10              # frazione finale dei passi su cui si misurano calo e deriva
GRID = (5, 3)


def tail_metrics(y):
    """(calo nell'ultimo TAIL [%], residuo stimato [%]) di una curva decrescente."""
    k = max(2, int(len(y) * TAIL))
    drop = 100.0 * (y[-k] - y[-1]) / abs(y[-1])
    d = -np.diff(y[-k:])                              # decrementi finali (>0 se scende)
    pos = d[d > 0]
    if len(pos) < 3 or d[-1] <= 0:
        return drop, 0.0
    rho = float(np.median(pos[1:] / pos[:-1]))        # ragione geometrica dei decrementi
    if not np.isfinite(rho) or rho >= 1:
        return drop, np.inf
    return drop, 100.0 * d[-1] * rho / (1 - rho) / abs(y[-1])


def main():
    d = os.path.join(BASE_DIR, RESULTS_NAME, "training_history")
    files = sorted(glob.glob(os.path.join(d, "hist_ch*_wp*.npz")))
    if not files:
        raise SystemExit(f"[ERROR] nessuna storia in {d}")
    rows = []
    for p in files:
        ch, wp = map(int, re.search(r"hist_ch(\d+)_wp(\d+)", p).groups())
        h = np.load(p)
        J, lam = h["J"], h["lam"]
        # loss = J + penalita': e' la funzione davvero minimizzata, salvata solo dalle
        # campagne nuove. Se c'e', il verdetto si legge su di lei, non su J.
        loss = h["loss"] if "loss" in h.files and len(h["loss"]) else None
        drop, resid = tail_metrics(loss if loss is not None else J)
        k = max(2, int(len(lam) * TAIL))
        rows.append(dict(ch=ch, wp=wp, n=len(J), J=J, lam=lam, loss=loss, drop=drop, resid=resid,
                         lam_drift=100.0 * (lam[-1] - lam[-k]) / abs(lam[-1])))
    print(f"{RESULTS_NAME}\n{len(rows)} punti, {rows[0]['n']} passi, coda = ultimo {TAIL:.0%}\n")
    print(f"{'ch':>4s} {'calo [%]':>22s} {'residuo [%]':>22s} {'deriva_lambda [%]':>22s}")
    print(f"{'':4s} {'mediana      max':>22s} {'mediana      max':>22s} {'mediana      max':>22s}")
    fmt = lambda v: "   inf" if not np.isfinite(v) else f"{v:6.3f}"
    for ch in sorted({r["ch"] for r in rows}):
        g = [r for r in rows if r["ch"] == ch]
        cols = []
        for key in ("drop", "resid", "lam_drift"):
            v = np.array([r[key] for r in g])
            fin = v[np.isfinite(v)]
            cols.append(f"{fmt(np.median(fin) if len(fin) else np.inf):>10s} "
                        f"{fmt(v.max()):>11s}")
        print(f"{ch:>4d} " + " ".join(cols))
        # una figura per canale: J (sinistra) e lambda (destra), un pannello per WP
        fig, axes = plt.subplots(*GRID, figsize=(4.4 * GRID[1], 3.0 * GRID[0]), squeeze=False)
        axes = axes.ravel()
        for a, r in zip(axes, sorted(g, key=lambda x: x["wp"])):
            a.plot(r["J"], color="C0", lw=1.4, label="J")
            if r["loss"] is not None:
                a.plot(r["loss"], color="C3", lw=1.2, label="loss = J + penalty")
                a.legend(fontsize=7)
            a.set_ylabel("J", color="C0", fontsize=9)
            a.tick_params(axis="y", labelcolor="C0", labelsize=8)
            a.tick_params(axis="x", labelsize=8)
            a2 = a.twinx()
            a2.plot(r["lam"], color="C1", lw=1.2, ls="--")
            a2.set_ylabel("λ", color="C1", fontsize=9)
            a2.tick_params(axis="y", labelcolor="C1", labelsize=8)
            a.set_title(f"WP{r['wp']}   last-10% drop {r['drop']:.2f}%,  λ drift "
                        f"{r['lam_drift']:+.1f}%", fontsize=9)
            a.set_xlabel("step", fontsize=8)
        for a in axes[len(g):]:
            a.axis("off")
        fig.suptitle(f"{RESULTS_NAME}\nCh{ch} — training: J (physical, = BI/K) and λ", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = os.path.join(BASE_DIR, RESULTS_NAME, f"training_convergence_ch{ch}.png")
        fig.savefig(out, dpi=110)
        plt.close(fig)
    print(f"\nfigure -> {RESULTS_NAME}/training_convergence_ch<ch>.png")
    bad = [r for r in rows if not np.isfinite(r["resid"]) or r["resid"] > 0.1]
    print(f"punti con residuo stimato > 0.1% (o senza segno di arresto): {len(bad)}/{len(rows)}"
          + ("" if not bad else "  " + ", ".join(f"ch{r['ch']}wp{r['wp']}" for r in bad[:12])))


def selftest():
    """Su una curva con limite NOTO, J = J_inf + A*exp(-n/tau), il residuo stimato deve
    ricostruire la distanza vera dal limite (e 'inf' se la curva non si ferma)."""
    n = np.arange(500)
    J_inf, A, tau = 0.30, 0.5, 80.0
    J = J_inf + A * np.exp(-n / tau)
    true = 100.0 * (J[-1] - J_inf) / J[-1]
    drop, resid = tail_metrics(J)
    assert abs(resid / true - 1) < 0.15, (resid, true)
    assert not np.isfinite(tail_metrics(1.0 - 1e-4 * n)[1])        # discesa lineare: non si ferma
    assert tail_metrics(np.full(500, 0.3))[1] == 0.0               # gia' piatta
    print(f"SELFTEST_OK  residuo stimato {resid:.4f}% vs vero {true:.4f}%")


if __name__ == "__main__":
    import sys
    selftest() if "--selftest" in sys.argv else main()
