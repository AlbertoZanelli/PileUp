"""
root_pulse_fit.py
=================
Fit dell'average pulse col MOTORE DI ROOT (TF1 + Minuit), il più fedele possibile
al programma originale FitPulse.C:

  - Le funzioni del modello sono le funzioni C LETTERALI di FitPulse.C, compilate
    via ROOT.gInterpreter.Declare:
      * fitfuncNpMz   -> tutti i poli reali            (cc=False)
      * FitFakePulse  -> npol-2 poli reali + 1 coppia complessa coniugata (cc=True)
    Convenzione parametri del C (par[0]=Npol, par[1]=Nzer, poi t0, amp, zeri, poli,
    baseline, tilt, preamp); per cc gli ultimi due "poli" sono sigma (reale, <0) e
    omega (>0) della coppia complessa.
  - Macchina di fit come nel C: TGraphErrors con errore per punto costante = RMS del
    baseline (computeRMS di FitPulse.C), TF1 con Npol/Nzer/PreAmp FISSI e SetParLimits
    sugli altri, fit con opzione "RSQ" su un range ristretto attorno al pulse
    [pretrigger-PRE_S, pretrigger+POST_S] (nel C: pretrigger-0.02 .. +0.09).
  - UNICA differenza rispetto al C: l'INIT dei parametri (t0, ampiezza, poli, zeri) e'
    adattato per-pulse ai dati m205 (onset, rise, 1/e decay). L'init letterale del C
    (amp=MaxMin*3500, poli a -62/-260, pretrigger 0.3) NON converge sull'AP m205, che
    e' peak-normalizzato e ha il picco a ~0.5 s.

Interfaccia identica a plot_AP_spectra_m205.fit_average_pulse:
    fit_average_pulse(t, v, nzer, npol, cc) -> dict con
      fit  : modello valutato su tutto l'asse t (np.ndarray)
      win  : maschera del range di fit
      theta: vettore completo dei parametri del TF1 (convenzione C)
      sigma: RMS del baseline (errore per punto, per la banda dei residui)
      rms  : RMS del residuo (dato - fit) sul range di fit
      npol, cc, nzer, t_range
"""

import numpy as np
import ROOT

ROOT.gROOT.SetBatch(True)

# Range di fit attorno al pretrigger (come FitPulse.C: -0.02 .. +0.09 s).
PRE_S, POST_S = 0.02, 0.09

# ── Funzioni C LETTERALI di FitPulse.C, compilate una volta sola ──────────────
_DECLARED = False


def _declare():
    global _DECLARED
    if _DECLARED:
        return
    ROOT.gInterpreter.Declare(r'''
#include "TComplex.h"
#include "TMath.h"

double FitFakePulse(double *x, double *par) {
    double * new_par = par;
    int npol = static_cast<int>(new_par[0]);
    int nzer = static_cast<int>(new_par[1]);
    par = nullptr;
    double provvisorio[npol+nzer+5];
    for( int i = 2; i<npol+nzer+5+2; i++){ provvisorio[i-2] = new_par[i]; }
    par = provvisorio;
    double value = par[4+npol+nzer]*(exp(x[0]-par[0]))+par[2+npol+nzer]+par[3+npol+nzer]*(x[0]-par[0]);
    double Res[npol];
    if((x[0]>par[0])) {
        for (int i=0;i<npol-2;i++) {
            Res[i]=1.;
            for (int j=0;j<nzer;j++) Res[i]*=(par[2+nzer+i]-par[2+j]);
            for (int j=0;j<npol-2;j++) if (i!=j) Res[i]/=(par[2+nzer+i]-par[2+nzer+j]);
            Res[i]/=(pow((par[2+nzer+i]-par[0+nzer+npol]),2)+par[1+nzer+npol]*par[1+nzer+npol]);
        }
        Res[npol-2]=1.;
        for (int j=0;j<nzer;j++) Res[npol-2]*=(pow((par[0+nzer+npol]-par[2+j]),2)+par[1+nzer+npol]*par[1+nzer+npol]);
        for (int j=0;j<npol-2;j++) Res[npol-2]/=(pow((par[0+nzer+npol]-par[2+nzer+j]),2)+par[1+nzer+npol]*par[1+nzer+npol]);
        Res[npol-2]/=(par[1+nzer+npol]*par[1+nzer+npol]);
        Res[npol-2]=TMath::Sqrt(Res[npol-2]);
        TComplex phi= - TComplex::I();
        for (int j=0;j<nzer;j++) phi*=(par[0+nzer+npol] + TComplex::I()*par[1+nzer+npol] - par[2+j]);
        for (int j=0;j<npol-2;j++) phi/=(par[0+nzer+npol] + TComplex::I()*par[1+nzer+npol] - par[2+nzer+j]);
        phi/=(2.0*par[1+nzer+npol]);
        double value1=0.;
        for (int i=0;i<npol-2;i++) value1+=Res[i]*exp((x[0]-par[0])*par[2+nzer+i]);
        value1+=Res[npol-2]*exp((x[0]-par[0])*par[0+nzer+npol])*cos(par[1+nzer+npol]*(x[0]-par[0]) + atan2(phi.Im(),phi.Re()));
        value +=  200*par[1]* value1;
    }
    return value;
}

double fitfuncNpMz(double *x, double *par) {
    double * new_par = par;
    int npol = static_cast<int>(new_par[0]);
    int nzer = static_cast<int>(new_par[1]);
    par = nullptr;
    double provvisorio[npol+nzer+5];
    for( int i = 2; i<npol+nzer+5+2; i++){ provvisorio[i-2] = new_par[i]; }
    par = provvisorio;
    double value = par[4+npol+nzer]*(exp(x[0]*par[2+nzer]))+par[2+npol+nzer]+par[3+npol+nzer]*(x[0]-par[0]);
    double Res[npol];
    if(x[0] > par[0]) {
        for (int i=0;i<npol;i++) {
            Res[i]=1;
            for (int j=0;j<nzer;j++) Res[i]*=(par[2+nzer+i]-par[2+j]);
            for (int j=0;j<npol;j++) if (i!=j) Res[i]/=(par[2+nzer+i]-par[2+nzer+j]);
        }
        double value1=0.;
        for (int i=0;i<npol;i++) value1+=Res[i]*TMath::Exp((x[0]-par[0])*par[2+nzer+i]);
        value1*=par[1];
        value+=value1;
    }
    return value;
}
''')
    _DECLARED = True


def _computeRMS(y, index_fin):
    """RMS di y[:index_fin] (analogo di computeRMS in FitPulse.C)."""
    if index_fin <= 0:
        return 0.0
    seg = y[:index_fin]
    return float(np.sqrt(np.mean((seg - seg.mean()) ** 2)))


def fit_average_pulse(t, v, nzer=1, npol=3, cc=False):
    """Fit di un average pulse peak-normalizzato col TF1 di ROOT (funzioni + settaggi
    di FitPulse.C, init adattato a m205). Vedi il docstring del modulo."""
    _declare()
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    n = len(t)
    dt = t[1] - t[0]
    imax = int(np.argmax(v))
    peak = float(v.max())

    # Onset (pretrigger m205) = ultimo campione sotto il 5% del picco prima del picco.
    below = np.where(v[:imax] < 0.05 * peak)[0]
    i0 = below[-1] if len(below) else max(imax - 10, 0)
    pretrigger = t[i0]
    t_0 = pretrigger - PRE_S
    t_fin = pretrigger + POST_S
    idx_fin = int(np.argmin(np.abs(t - t_fin)))

    # Errore per punto costante = RMS del baseline (SetPointError di FitPulse.C).
    sigma = _computeRMS(v, idx_fin)
    if not sigma > 0:
        sigma = 1.0

    # Timescale per l'init adattato (rise 10-90%, 1/e decay).
    up = v[:imax]
    i10 = np.where(up > 0.1 * peak)[0]
    i90 = np.where(up > 0.9 * peak)[0]
    t_rise = max((t[i90[0]] - t[i10[0]]) if (len(i10) and len(i90)) else 5 * dt, dt)
    after = v[imax:]
    be = np.where(after < peak / np.e)[0]
    t_dec = (t[imax + be[0]] - t[imax]) if len(be) else 10 * dt
    fast, slow = t_rise / 3.0, max(t_dec * 5, t_rise * 10)
    base0 = float(np.mean(v[:max(i0 - 5, 1)]))

    nreal = npol - 2 if cc else npol
    poles0 = -1.0 / np.geomspace(fast, slow, nreal) if nreal else np.array([])
    zeros0 = -1.0 / np.geomspace(t_rise / 2.0, t_rise * 2.0, nzer) if nzer else np.array([])

    # TGraphErrors con errore costante = sigma (come il C).
    x = t.astype("float64")
    graph = ROOT.TGraphErrors(n, x, v.astype("float64"),
                              np.zeros(n), np.full(n, sigma))

    func = ROOT.FitFakePulse if cc else ROOT.fitfuncNpMz
    npar = npol + nzer + 7
    f1 = ROOT.TF1("bolo", func, 0.0, float(t[-1]), npar)
    f1.SetNpx(1000)

    f1.FixParameter(0, npol)                                  # Npol
    f1.FixParameter(1, nzer)                                  # Nzer
    f1.SetParameter(2, pretrigger); f1.SetParLimits(2, pretrigger - 0.02, pretrigger + 0.02)  # t0
    f1.SetParameter(3, 1.0);        f1.SetParLimits(3, 1e-9, 1e6)   # amp (auto-scale sotto)
    for j in range(nzer):
        f1.SetParameter(4 + j, float(zeros0[j])); f1.SetParLimits(4 + j, -1e5, -1e-3)
    for j in range(nreal):
        f1.SetParameter(4 + nzer + j, float(poles0[j])); f1.SetParLimits(4 + nzer + j, -1e5, -1e-3)
    if cc:                                                     # coppia CC: sigma (<0) + omega (>0)
        f1.SetParameter(4 + nzer + npol - 2, -1.0 / slow)
        f1.SetParLimits(4 + nzer + npol - 2, -1e5, -1e-3)
        f1.SetParameter(4 + nzer + npol - 1, 600.0)
        f1.SetParLimits(4 + nzer + npol - 1, 1.0, 1e4)
    f1.SetParameter(4 + npol + nzer, base0); f1.SetParLimits(4 + npol + nzer, -1.0, 1.0)   # baseline
    f1.SetParameter(5 + npol + nzer, 0.0);  f1.SetParLimits(5 + npol + nzer, -1e4, 1e4)    # tilt
    f1.FixParameter(6 + npol + nzer, 0.0)                     # PreAmp

    # Amplitude auto-scale: picco del modello (amp=1) portato al picco dei dati.
    tt = np.linspace(pretrigger, pretrigger + max(t_dec * 3, 0.02), 300)
    mpk = max((f1.Eval(float(x_)) - base0) for x_ in tt)
    f1.SetParameter(3, peak / mpk if mpk > 0 else 1.0)

    graph.Fit(f1, "RSQ", "", t_0, t_fin)

    theta = np.array([f1.GetParameter(k) for k in range(npar)], dtype=float)
    fit = np.array([f1.Eval(float(tt_)) for tt_ in t], dtype=float)
    win = (t >= t_0) & (t <= t_fin)
    rms = float(np.sqrt(np.mean((fit[win] - v[win]) ** 2)))
    return {"fit": fit, "win": win, "theta": theta, "sigma": sigma, "rms": rms,
            "npol": npol, "cc": cc, "nzer": nzer, "t_range": (float(t_0), float(t_fin))}
