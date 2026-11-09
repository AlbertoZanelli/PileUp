#!/usr/bin/env python3
"""
plot_BI_vs_parameters_likelihood_m205.py
========================================
Stimatore M (likelihood ratio), campagna v2tf: BI e rigetto dei pile-up contro i parametri del punto
di lavoro, come le figure BI_vs_parameters di plot_BI_results.py (di cui riusa lettura e griglie).

  - BI = media sui 50 seed del Monte Carlo (la dispersione fra seed e' ~0.7%, barre invisibili);
  - R  = frazione di pile-up rigettata (rp del MC): BI = K (1 - R), con K proporzionale al rise
         time (dt_max = rise time). R isola la capacita' di discriminare dall'effetto di K;
  - beta e SNR*beta dal CSV del filtro ottimo: stesso template e stesso rumore (SNR identici).

    KMP_DUPLICATE_LIB_OK=TRUE python3 plot_BI_vs_parameters_likelihood_m205.py
"""
import os
import pandas as pd
import plot_BI_results as pb

TAG = "APsimfit10000ledv2tf_npsclean_dtrise"
LIK = os.path.join(pb.BASE_DIR, f"m205_results_likelihood_{TAG}")
OF_CSV = os.path.join(pb.BASE_DIR, f"m205_results_octopus_{TAG}", f"BI_results_m205_{TAG}.csv")


def load_rows():
    mc = pd.read_csv(os.path.join(LIK, "BI_mc_error_m205_v2_seeds50.csv"))
    mc = mc.groupby(["channel", "wp"]).agg(BI=("BI_mc", "mean"), R=("rp", "mean"))
    of = pd.read_csv(OF_CSV).set_index(["channel", "wp"])
    df = of[["vbias", "signal_amp", "sigma_analytic", "SNR", "beta_Hz", "rho_t"]].join(mc, how="inner").reset_index()
    df["R"] *= 100
    rows = df.to_dict("records")
    pb.merge_timing(rows, pb.read_amplitudes(pb.AMP_CSV))
    rbol = pb.read_rbol(pb.RBOL_TXT)
    for r in rows:
        r["R_bol"] = rbol.get((r["channel"], round(r["vbias"], 3)))
        r["decay_over_rise"] = r["decaytime_ms"] / r["risetime_ms"] if r["risetime_ms"] else None
        # dt_max / sigma_t con sigma_t ~ 1/(SNR beta): quante risoluzioni temporali stanno nella finestra dei ritardi
        r["window_over_sigma_t"] = r["rho_t"] * r["risetime_ms"] * 1e-3 if r["risetime_ms"] else None
    return rows


def specs(y, ylabel, logy):
    p = [("vbias", r"$V_{bias}$ (V)", True), ("risetime_ms", "Rise time 10-90% (ms)", False),
         ("decaytime_ms", "Decay time (ms)", False), ("decay_over_rise", "Decay / rise time", False),
         ("SNR", "SNR", False), ("rho_t", r"SNR$\cdot\beta$ (Hz)", False),
         ("beta_Hz", r"$\beta$ (Hz)", False), ("window_over_sigma_t", r"SNR$\cdot\beta\cdot$rise time  (= dt_max / $\sigma_t$)", False),
         ("signal_amp", "Amplitude (V)", True), ("sigma_analytic", r"$\sigma$ noise (V)", True),
         ("R_bol", r"$R_{bol}$ (M$\Omega$)", True), ("hf_power", "AP HF power (>500 Hz)", True)]
    return [dict(xkey=k, ykey=y, xlabel=xl, ylabel=ylabel, title=f"{ylabel.split(' ')[0]} vs {xl.split(' (')[0]}",
                 logx=lx, logy=logy) for k, xl, lx in p]


def main():
    rows = load_rows()
    colors = pb.channel_colors([r["channel"] for r in rows])
    print(f"{len(rows)} punti -> {os.path.relpath(LIK, pb.BASE_DIR)}/")
    for plot, tag in ((pb.plot_grid, ""), (pb.plot_grid_vbias, "_vbiascolor")):   # tinta = canale, ombra = V_bias
        plot(rows, specs("BI", "BI (counts/keV/kg/yr)", True), 4,
             os.path.join(LIK, f"BI_vs_parameters{tag}_likelihood_m205.png"), colors,
             "Likelihood-ratio estimator M: BI vs parameters (MC, 50 seeds)")
        plot(rows, specs("R", "R pile-up rejection (%)", False), 4,
             os.path.join(LIK, f"rejection_vs_parameters{tag}_likelihood_m205.png"), colors,
             "Likelihood-ratio estimator M: pile-up rejection R vs parameters (MC, 50 seeds)")

    df = pd.DataFrame(rows)
    keys = [s["xkey"] for s in specs("BI", "", True)]
    print("\nCorrelazione di Spearman sui 75 punti (tutti i canali insieme):")
    print("  parametro            BI      R")
    for k in keys:
        print(f"  {k:18s} {df[k].corr(df['BI'], method='spearman'):+.2f}  {df[k].corr(df['R'], method='spearman'):+.2f}")


if __name__ == "__main__":
    main()
