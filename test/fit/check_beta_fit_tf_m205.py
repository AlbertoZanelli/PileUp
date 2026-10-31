#!/usr/bin/env python3
"""
check_beta_fit_tf_m205.py
=========================
Risoluzione temporale beta (banda RMS del template pesata sulla NPS) dei fit "time+freq"
(residual_scan_bessel/fits_octopus_tf) contro l'AP di Octopus, per tutti i punti, con il vecchio fit
come riferimento. L'errore su beta(AP) viene dal BOOTSTRAP dei singoli impulsi che formano l'AP.
Il confronto con l'AP si fa sul fit sfumato di un campione, come nel fit (l'AP e' la mediana di
impulsi che cadono a caso fra due campioni); si riporta anche il fit non sfumato, che e' il template.

Il rumore dell'AP aggiunge potenza ad alta frequenza e GONFIA beta(AP), mentre il fit e' liscio: il
confronto diretto e' distorto. Per questo si costruiscono AP SINTETICI con verita' nota: fit sfumato +
residui veri degli impulsi (p_i - scala_i * AP) con segno casuale, poi mediana. La loro beta media e' cio'
che ci si aspetta di misurare se il fit e' giusto; z_syn = (beta(AP) - media) / dispersione.

Scrive comparisons/beta_fit_tf_vs_AP_m205.png e la tabella beta_fit_tf_vs_AP_m205.csv.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=. python3 test/fit/check_beta_fit_tf_m205.py
"""
import os, sys, csv
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import src.analysis as an
import scan_residuals_bessel_m205 as sb
import simulate_BI_error_m205 as mc

CHANNELS = [31, 34, 71, 83, 91]
WPS = list(range(1, 30, 2))
VBIAS = [0.6, 1.0, 1.4, 1.8, 2, 3, 4, 5, 6, 8, 10, 20, 26, 30, 40]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]
N_BOOT = 200
FS = 10_000
NEW_DIR = os.path.join(BASE, "residual_scan_bessel", "fits_octopus_tf")
OLD_DIR = os.path.join(BASE, "residual_scan_bessel", "fits_octopus")
f = np.fft.rfftfreq(10_000, 1 / FS)
w = 2 * np.pi * np.fft.fftfreq(10_000, 1 / FS)


def smear(x):
    F = np.fft.rfft(x) * np.sinc(f / FS)
    y = np.fft.irfft(F, n=len(x))
    return y / y.max()


def beta(x, nps):
    S = an.compute_H(x / x.max(), nps, np.hanning, sampling_rate=FS)[0]
    return float(an.compute_sigma_OF(S, nps)) / float(an.compute_sigma_OF(w * S, nps)) / (2 * np.pi)


def main():
    rng = np.random.default_rng(0)
    rows = []
    for ch in CHANNELS:
        for wp in WPS:
            nps = mc.load_noise(ch, wp)
            P = sb.load_pulses(ch, wp)
            ap = np.median(P, axis=0); ap /= ap.max()
            boot = [beta(np.median(P[rng.integers(0, len(P), len(P))], axis=0), nps) for _ in range(N_BOOT)]
            b_ap, s_ap = beta(ap, nps), float(np.std(boot, ddof=1))
            new = np.load(os.path.join(NEW_DIR, f"bestfit_ch{ch}_wp{wp}.npy")).astype(float)
            old = np.load(os.path.join(OLD_DIR, f"bestfit_ch{ch}_wp{wp}.npy")).astype(float)
            b_s, b_n, b_o = beta(smear(new), nps), beta(new, nps), beta(old, nps)
            ys = smear(new)
            res = P / P.max(axis=1, keepdims=True); res = res - ((res @ ap) / (ap @ ap))[:, None] * ap
            syn = [beta(np.median(ys + rng.choice((-1.0, 1.0), len(P))[:, None] * res, axis=0), nps) for _ in range(N_BOOT)]
            b_syn, s_syn = float(np.mean(syn)), float(np.std(syn, ddof=1))
            rows.append(dict(channel=ch, wp=wp, vbias=VBIAS[wp // 2], beta_AP=b_ap, sigma_AP=s_ap, beta_boot_mean=float(np.mean(boot)),
                             beta_syn=b_syn, sigma_syn=s_syn, z_syn=(b_ap - b_syn) / s_syn, ratio_syn=b_ap / b_syn,
                             beta_new_smeared=b_s, beta_new=b_n, beta_old=b_o, ratio_new=b_s / b_ap, ratio_old=b_o / b_ap,
                             z_new=(b_s - b_ap) / s_ap, z_old=(b_o - b_ap) / s_ap))
    out = os.path.join(BASE, "comparisons", "beta_fit_tf_vs_AP_m205.csv")
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for ch, c, m in zip(CHANNELS, COLORS, MARKERS):
        r = [x for x in rows if x["channel"] == ch]
        vb = [x["vbias"] for x in r]
        ax[0].plot(vb, [x["ratio_old"] for x in r], color=c, marker=m, ms=6, lw=2, label=f"Ch {ch}")
        ax[1].plot(vb, [x["ratio_new"] for x in r], color=c, marker=m, ms=6, lw=2, label=f"Ch {ch}")
        ax[2].plot(vb, [x["z_syn"] for x in r], color=c, marker=m, ms=6, lw=2, label=f"Ch {ch}")
    for a in ax[:2]:
        a.axhspan(0.9, 1.1, color="0.92", zorder=0); a.axhline(1, color="0.5", lw=1)
        a.set(ylim=(0.75, 1.7), ylabel="beta(fit) / beta(Octopus AP)")
    ax[0].set_title("old fits (time chi2)"); ax[1].set_title("new fits (time + frequency chi2)")
    ax[2].axhspan(-2, 2, color="0.92", zorder=0); ax[2].axhline(0, color="0.5", lw=1)
    ax[2].set(ylabel="(beta AP - beta synthetic AP) / sigma", title="new fits + AP noise vs measured AP")
    for a in ax:
        a.set(xscale="log", xlabel="V_bias [V]")
        a.grid(True, color="0.9", lw=0.6); a.spines[["top", "right"]].set_visible(False)
    ax[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, "comparisons", "beta_fit_tf_vs_AP_m205.png"), dpi=140)

    for lab, key, zk in (("vecchi", "ratio_old", "z_old"), ("nuovi", "ratio_new", "z_new")):
        v, z = np.array([x[key] for x in rows]), np.array([x[zk] for x in rows])
        print(f"{lab}: rapporto beta fit/AP mediana {np.median(v):.3f}, min {v.min():.3f}, max {v.max():.3f}; "
              f"entro 5%: {np.sum(abs(v - 1) < .05)}, entro 10%: {np.sum(abs(v - 1) < .1)} su {len(v)}; "
              f"|z|<2: {np.sum(abs(z) < 2)}, |z|<3: {np.sum(abs(z) < 3)}; z mediana {np.median(z):+.1f}")
    z, r = np.array([x["z_syn"] for x in rows]), np.array([x["ratio_syn"] for x in rows])
    print(f"AP sintetici (fit + rumore dell'AP): beta(AP)/beta(sint) mediana {np.median(r):.4f}, min {r.min():.3f}, max {r.max():.3f}; "
          f"z_syn mediana {np.median(z):+.2f}, |z|<2: {np.sum(abs(z) < 2)}, |z|<3: {np.sum(abs(z) < 3)} su {len(z)}")
    print("\n ch wp  V_bias  beta_AP +- sigma   nuovo sfumato (non sfumato)  vecchio |  r_nuovo  r_vecchio   z_nuovo | sintetico +- s   z_syn")
    for x in rows:
        print(f"{x['channel']:3d} {x['wp']:2d} {x['vbias']:6.1f}  {x['beta_AP']:6.1f} +- {x['sigma_AP']:4.1f}   {x['beta_new_smeared']:6.1f} ({x['beta_new']:6.1f})      "
              f"{x['beta_old']:6.1f}  |  {x['ratio_new']:.3f}    {x['ratio_old']:.3f}    {x['z_new']:+5.1f}  | {x['beta_syn']:6.1f} +- {x['sigma_syn']:4.1f}  {x['z_syn']:+5.1f}")


if __name__ == "__main__":
    main()
