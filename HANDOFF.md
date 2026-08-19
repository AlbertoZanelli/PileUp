# PileUp — Handoff

Thesis work on the CUPID LMO light-detector average pulses (AP) of run **m205**.
**Active thread now: add a template-reliability regularization R(f) to the Wiener-filter
training**, to stop the learned filters from overtraining on the finite-N high-frequency
template noise, and to make the BI independent of how many pulses go into the average.
Earlier threads (single-pulse fit, AP/noise & reliability diagnostics, argonauts-vs-octopus,
normalization study) are summarized at the bottom.

**NEXT AGENT: use ponytail mode** (the user asked). Laziest thing that works, config-at-top,
no scaffolding.

---

## Environment (IMPORTANT)
- **Project root**: `~/Desktop/Tesi_Erasmus/PileUp` (git repo; `.root` data gitignored but
  PRESENT locally in `Processed/`).
- **Run with**: `KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/bin/python3.13 <script.py>`
  - python3.13 has numpy/scipy/matplotlib/uproot (needs `KMP_DUPLICATE_LIB_OK=TRUE`). **No ROOT, no torch.**
  - **pyrootAlbi** conda env (`/opt/anaconda3/envs/pyrootAlbi/bin/python`) has ROOT + numpy/
    scipy/matplotlib/pandas + uproot + python-pptx + **pypdf** + **torch**. Use it for training
    (torch) and to read Octopus `.root` via uproot.
  - The **user commits** work themselves — do NOT commit. Discussion in Italian, plot text in
    English. Programs **essential, config-at-the-top**, few/no CLI flags.

## Data conventions (m205)
- **10 kHz**, window **10000** samples (1 s). ROOT files `Processed/Processed_*_000205_<ch>.root`.
  - AP hist `averagepulse_ap_wp<wp>_medianAP` (peak-norm ~1, **peak at ~0.5 s**, pretrigger 50%).
  - Noise PSD `averagepowerspectrum_noise_wp<wp>_medianpower` (one-sided, V²/Hz, flat-top window).
  - **N = 38 pulses** in each average (median of ~36–38 real pulses per time-column).
  - WP→V_bias via `VBIAS_LIST[wp//2]`, **odd wp only**, 15 WPs.
- **9 channels**: 31,34,37,40,41,71,83,91,94. **5 "good"**: 31,34,71,83,91. **4 "bad"**: 37,40,41,94.
- **No injected-pulse datasets for m205.** BI is the analytical `J_final·K` (from S, nps,
  ratio_distribution) — there is NO labelled held-out set. This matters for β selection (below).

---

## ACTIVE THREAD — template reliability R(f) regularization

### The idea (relatore + `~/Downloads/template_reliability_optimal_filter.pdf`, estimator A)
Template = median of N pulses → `X̄[k] = S[k] + Ē[k]`, `E{|Ē|²}=P_η/N`. The learned band
filters can exploit this finite-N HF noise (overtraining). Fix: a per-bin reliability
$$R(f)=\frac{S_{above}}{S_{above}+\beta\,\text{ANPS}/N+\epsilon},\quad S_{above}=\max(\text{APPS}-\beta\,\text{ANPS}/N,\,0)$$
`R∈[0,1]` ≈1 where signal clearly beats the **template** noise floor ANPS/N, ≈0 where the bin
is just residual noise. `R=0.5` at template SNR (power) `ρ=2β`. Then multiply the built filter
by R **once**. Inputs: APPS=`|S|²`, ANPS=per-event noise, **N=38**, β (currently 2), ε tiny.

### Where R goes in the code
Core fn: **`optimize_filters_wiener_lambda(S, w, t, r, nps, signal_amp, ratio_distribution, …)`**
in `src/analysis.py:729`. The applied filter there is `f_j · W_unit`, where
`W_unit = compute_W_torch(S, nps, lam)` (`src/analysis.py:695`):
`W = conj(S)/(AvgPS_n + lam·nps_n)`, `AvgPS_n=|S|²/a`, `nps_n=nps·a/b²`, `a=Σ√|S|²`, `b=Σ√nps`.
**Key: W (and R) are INVARIANT to a constant scale of nps** (a/b² cancels it) → compute R from
the SAME normalized spectra (`AvgPS_n`, `nps_n`) so |S|² and nps are comparable (PDF sec 2 unit
warning). `Sabove = max(AvgPS_n − β·nps_n/N, 0)`, `R = Sabove/(Sabove+β·nps_n/N+ε)`.
Fold once: `W_robust = R·W_unit`, use it everywhere in the loop (S_H, the `mean(|f·W·S|)`
normalization, J) and **return it** in place of W_unit. "Do not use R twice."

### PLAN for the next agent (what to build)
1. **`src/analysis.py`** — add a small **separate function** (like `compute_W_torch`), e.g.
   `reliability_R(S, nps, N_events, beta, eps)` returning a real torch tensor R (full spectrum,
   Hermitian-symmetric so filters stay real). Then give `optimize_filters_wiener_lambda` new args
   `use_R=False, N_events=None, beta_R=2.0, eps_R=<small>`; when `use_R`, set `W_unit = R*W_unit`
   inside the loop. **Default off → existing behavior byte-identical.** Estimator A only (only S+nps
   available; split/LOO cross-power need per-event FFTs we don't have).
2. **`analysis_BI_m205_wiener_regolarized.py`** — copy of `analyse_BI_m205_wiener.py`, pass
   `use_R=True, N_events=38, beta_R=…` into the optimizer. Same data loading (meanpulse=medianAP,
   nps=medianpower→full ×5.708×window²×(1/sampling_time), S=fft(meanpulse·hanning)).
3. **β selection for m205** (no injected data, BI=analytical J → can't tune β on the same-template
   J, self-defeating). Use **cross-fit on the 38 real pulses**: split 38→19+19, build template_A/_B,
   train with R(β) on A, **evaluate J on template_B** (and vice-versa); scan β∈[1e-2,10]; pick the β
   with min **cross-J** (or where J_self≈J_cross). Repeat 5–10 random splits and average (19 is noisy).
   This also enforces goal-2 (N-independence). Cheap prior to center the grid: β from the noise-power
   fluctuation quantile, or β=2. **Needs the per-time-column pulses** — the `APdistro` TH2D
   (`averagepulse_ap_wp<wp>_APdistro`, 10000×1000) holds them; median over a random half = template_A/B.

### Diagnostics already built (use them, don't rebuild)
- **`compare_AP_noise_ch34_ch91_m205.py`** — 6-panel (AP time, APPS, ANPS, + per-channel
  reliability panel with APPS/ANPS-N/β·ANPS-N + R(f) + soft cutoff **f\***, + overlay of the
  trained filter's **signal output |F·S|²** and **noise output |F|²·NPS**; last panel R(f) both
  channels). Config `N_EVENTS=38, BETA_R=2, SMOOTH_R_HZ=100, FILTER_DIR=m205_results_wiener/
  trained_filters, WIENER_CSV=…`. Filter overlay uses saved `f1/f2_ch<ch>_wp<wp>.npy` + scalar
  `lambda_wiener` from the CSV → reconstructs `W_unit` (invariant to nps scale) → `F_j=f_j·W_unit`.
  **Finding (WP15):** f*≈511 Hz (ch34), ≈202 Hz (ch91, noisier→narrower). The trained filter's
  **signal output sits below f\***, its **noise output above f\*** → R would cut the noise, keep the
  signal → **R helps**; ch34 fine at β=2, ch91 wants lower β.
- **`scan_R_vs_beta_m205.py`** — 3×4 grid, one β per panel (Ch91 WP15), each with APPS+ANPS (left,
  log) and R(f) (right, 0–1). Config `CHANNEL, WP, N`. Shows the soft cutoff tightening as β grows.

---

## WHAT WORKED
- R is a soft, per-bin SNR gate; f* (R=0.5) sits at template SNR ρ=2β; it adapts per channel via
  the channel's own ANPS/N. Diagnostics show the current filters overtrain (noise output above f*).
- Reconstructing `W_unit` from saved f1/f2 + scalar `lambda_wiener` (CSV) — invariant to nps scale.

## WHAT DIDN'T WORK / DON'T DO
- **Don't make β a trainable torch parameter** and don't tune it on the training J — a regularizer
  minimized on its own training objective collapses (β→0). Select β by cross-fit / held-out only.
- **No injected datasets exist for m205** → the honest selector is the 38-pulse cross-fit, not BI.
- Estimator B/C (split/LOO cross-power) and cross-fitting the whole BI pipeline: need per-event
  FFTs / a data-pipeline change — out of scope for the first R implementation.

---

## Earlier threads (DONE — background)
- **Single-pulse fit** `test/fit_one_pulse_m205.py` — essential pole-zero+Bessel fitter for one
  channel/WP; two model fns (`make_pulse_pole_zero_bessel_ct` 1-zero real, `make_pulse_bessel_general`
  N-zero+CC), physical guesses (2 poles = −1/t_rise,−1/t_dec; zeros between poles), bounds
  `[-PARAM_BOUND,0]` (8000), σ=baseline RMS, ±3σ band, residual in σ. Writes
  `fit_one_pulse_params.csv` (upsert per (ch,wp,model); PNG named per model) loaded by
  **`test/visual_pulse_m205.py`** from its "start from" menu. Both now use **t0 in ABSOLUTE
  seconds** and the same peak-finding → identical pulse (was a confusion: visual used ms-offset).
- **`scan_residuals_bessel_m205.py`** — parallel Bessel residual scan; bounds re-aligned to
  fit_pulses (`PARAM_BOUND=8000`, real+CC), stronger multi-start.
- **`compare_argonauts_vs_octopus_m205.py`** — % differences (RT/DT/HF) Argonauts vs Octopus.
- **`compare_BI_vs_hfpower_m205.py`** — ΔBI% vs ΔHF-power% scatter (needs argonauts edmean for all
  8 BI channels; copied the missing 4 from `Cose_vecchie/.../AnalisiArgonauts_m205/`).
- **`risetime_and_amplitude_.py`** — added `AVERAGE_PULSES_BIN_WP` mode (m205 argonauts, parses
  ch+wp from `m205_ch<ch>_combined_WP<wp>.bin_edmean.bin`, 10 kHz); ROOT import made optional
  (`HAVE_ROOT`) so it runs on the homebrew python when plots are off.
- **Octopus normalization study**: confirmed the noise `medianpower` = median over events of
  `2|X_norm|²/df` (one-sided ×2 on AC bins, ×1 DC/Nyquist; FFT 1/N-normalized); the `normalized`
  graph = `sqrt(power_arithmetic_mean)/Gain(=910)`; flat-top factor 5.708 = fEnergyCorrection²=N/Σw²
  (Octopus processes noise with FlatTop + CoherentGain=0, so the ×5.708 recovers true V²/Hz).
- **BI/Wiener** (`analyse_BI_m205.py`=optimum, `_wiener.py`=scalar λ, `_wiener_freq.py`=λ(f));
  saved filters in `m205_results_wiener*/trained_filters/` (`f1/f2_ch<ch>_wp<wp>.npy`, `_freq` also
  `lambda_*`); BI/λ in `BI_results_m205_wiener.csv` (`lambda_wiener` column). Load curves, overview
  canvases, m204 cross-check, slide decks — all done.
