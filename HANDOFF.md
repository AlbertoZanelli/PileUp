# PileUp — Handoff

Thesis work (Milano-Bicocca, CUPID/CROSS) on **pile-up rejection** for the CUPID LMO light
detectors, run **m205**. The chain is now closed end to end:
real pulses → **fitted template** → **simulated AP** (`APsimfit10000led`) → training
(optimum filter vs Wiener with a penalty on s) → **Monte-Carlo BI** with errors → comparison
plots and thesis slides. The final result is in **FINAL RESULT** below. Older threads are
summarized at the bottom.

**NEXT AGENT: use ponytail mode** (the user asked). Laziest thing that works, config-at-top,
no scaffolding. Discussion in Italian, plot text in English. **The user commits, not you.**
**Never run training locally** ("prepara tutto e poi lo runno sul server") — EXCEPT short
diagnostic runs on one point, which the user has explicitly asked for and which are fine.
When told "non devi fare niente", only read.

---

## ⚠ READ THIS FIRST — 2026-09-20/21 overturned the FINAL RESULT below

**1. OF and Wiener are the SAME filter family. Measured, not argued.** The applied filter is
`g_i = f_i · kernel` and `f_i` is a free NON-NEGATIVE REAL function of frequency. The ratio
`H_optimum / W_wiener` is real and positive at every frequency (imaginary part 8e-8 of the
modulus, min sign +7e-6, ch34 wp15), so the band filters absorb the kernel difference exactly:
`{f·H} = {f·W} = {positive function × S*}`. **No choice of λ can leave a family you are already
in.** Any OF-vs-Wiener difference is optimization dynamics, not physics.

**2. λ is not identifiable (gauge).** Rescaling a trained point by the exact kernel ratio
W(λ₀)/W(λ₁) and moving λ by ×0.1, ×3, ×100 leaves J unchanged to **1e-6** and s1, s2 identical
to 6 digits. The `lambda_wiener` column is gauge, never compare it across points/campaigns.

**3. The old −3% gain was an artefact of unequal training.** OF used `N_TRIALS = 300`, Wiener
500. With both at 500 (the `_hist` campaigns) the analytic gain drops to ≈ −0.8%.

**4. And that residual analytic gain evaporates in the MC.** ch34, 15 points, all at 500 steps:

| variant | ΔBI analytic | ΔBI **Monte Carlo** | better than OF | MC/an | s₁ |
|---|---|---|---|---|---|
| Wwna (λ trainable) | −0.96% | **+0.01%** | 7/15 | 1.019 | 0.030 |
| λ=1, no penalty | −1.53% | **+0.61%** | 1/15 | 1.027 | 0.062 |
| λ=1, with penalty | −1.28% | **+0.46%** | 1/15 | 1.026 | 0.049 |
| OF | — | — | — | **1.009** | — |

The analytic gain is exactly cancelled by the analytic model being MORE optimistic for Wiener
(MC/an 1.019–1.027 vs 1.009), and the ordering follows s₁: the more the filter amplifies noise,
the more the first-order σ_Y expansion lies. **Never conclude from the analytic BI.**

**NEXT STEP THE USER CHOSE: free the PHASE of the band filters** (idea 2 below) — today `f` is
real ≥ 0, so the total filter's phase is nailed to S*'s; a complex `f` genuinely enlarges the
family, and phase is exactly what carries the pile-up DELAY. The next agent runs the test and
reports whether the gain is tangible **on the MC**.

---

## Environment (IMPORTANT)
- **Project root**: `~/Desktop/Tesi_Erasmus/PileUp` (git repo; `.root` data gitignored but
  PRESENT locally in `Processed/`). Server copy: `/data/users/azanelli/PileUp`.
  **Sync is via git only** — `ssh zanelli@cupid-login.lngs.infn.it` does not work from here
  (no key auth). Tell the user which files to commit/pull; several times the two copies diverged.
- **Run with**: `KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/bin/python3.13 <script.py>`
  - python3.13: numpy 2 (use `np.trapezoid`, not `np.trapz`)/scipy/matplotlib/uproot/python-pptx.
    **No ROOT, no torch.**
  - **pyrootAlbi** (`/opt/anaconda3/envs/pyrootAlbi/bin/python`): ROOT + torch + uproot + pypdf.
- LibreOffice is **not installed** and PowerPoint AppleScript export produces nothing: pptx
  can't be rendered locally — check layouts geometrically.
- Cluster is **PBS**: `qsub`/`qstat`/`qdel`. Kill a campaign with
  `qstat -u $USER | grep <PREFIX> | awk '{print $1}' | xargs qdel`.
  Job prefixes: `BI`/`BIF`/`BIS` (optimum), `BIW`/`BIWF`/`BIWR` (Wiener), `BSCAN` (fits).
- **Measured**: BI worker peak RAM **2.04 GB** (`RAM_GB = 4` is fine), fastest with
  **2 threads** (`-l nodes=1:ppn=2` + `OMP_NUM_THREADS=2` is the real speed lever).

## Data conventions (m205)
- **10 kHz**, window **10000** samples (1 s). `Processed/Processed_*_000205_<ch>.root`.
  - AP: `averagepulse_ap_wp<wp>_medianAP` (peak-norm 1, pretrigger 50%). Octopus aligns the
    pulses at **mid-rise = sample 5000**; the peak is 3–32 samples later (jitter std 0.32 samples).
  - Noise: `averagepowerspectrum_noise_wp<wp>_medianpower` is NOT E|FFT|² (see NPS section).
  - **N per AP** = `APdistro` entries / 10000 → **36–39, varies per WP**.
  - WP→V_bias via `VBIAS_LIST[wp//2]`, **odd wp only**, 15 WPs.
- **9 channels**: 31,34,37,40,41,71,83,91,94. The BI campaigns use **31, 34, 71, 83, 91**
  (the ones with a fitted template).
- `amplitudes_m205.csv` = **ROI-equivalent** amplitude (~mV) = LED amplitude / **430.9**,
  constant to 0.0% over all WPs. **ROI amplitude → MC signal (`signal_amp`)**; **LED amplitude
  (~0.5 V, `maxminusbaseline.amplitude`) → only to build the simulated AP**, as the real AP is
  built from LED pulses.

---

## FINAL RESULT (campaign `APsimfit10000led`, NPS clean, MC gen = fit, NSIM = 50 000)

> **STALE — do not quote.** These OF numbers come from a 300-step training against Wiener's 500
> (see READ THIS FIRST). The 50-seed MC on these same folders gives ΔBI medians −2.29% (ch31),
> −2.52 (34), −2.36 (71), −2.78 (83), −0.91 (91), global −1.59%, Wwna better in 71/75, z from
> −20 to −52 — but at EQUAL training steps the gain is ≈ −0.8% analytic and ≈ 0 on the MC.

Folders: `m205_results_octopus_APsimfit10000led_npsclean` (OF),
`m205_results_wiener_APsimfit10000led_npsclean` (W, no penalty),
`m205_results_wiener_APsimfit10000led_npsclean_swna1` (Wwna, penalty `("wna", 1.0)`).
75 points each (5 ch × 15 WP). BI_X = MC BI.

| ch | MC/an OF | MC/an W | MC/an Wwna | W/OF | Wwna/OF | Wwna wins |
|---|---|---|---|---|---|---|
| 31 | 1.021 | 1.032 | 1.032 | 0.989 | 0.980 | 15/15 |
| 34 | 1.006 | 1.019 | 1.013 | 0.971 | 0.968 | 11/15 |
| 71 | 1.009 | 1.029 | 1.019 | 0.982 | 0.976 | 15/15 |
| 83 | 1.004 | 1.022 | 1.018 | 0.976 | 0.972 | 13/15 |
| 91 | 1.034 | **1.200** | 1.036 | **1.121** | 0.990 | 13/15 |

- **All: W/OF median 0.992 (49/75), Wwna/OF median 0.983 (67/75).** Small but systematic gain,
  larger at high bias: **≥10 V −3.5% (25/25)**, ≤2 V −0.6%.
- The penalty is what makes Wiener safe: without it ch91 overtrains (λ collapses, MC/an 1.20,
  W 12% worse than OF). With it MC/an stays ≤ 1.036 everywhere (paper: < 8%). Wwna λ ranges
  0.8–30 (ch91 38–780), max s₁ 0.10.
- CUPID pile-up target **5×10⁻⁵ counts/(keV kg yr)** (half of the 1e-4 budget): reached only by
  **ch71 at 20–40 V** (both OF and Wwna). Min BI per channel (Wwna): ch31 5.1e-5, ch34 6.4e-5,
  ch71 4.5e-5, ch83 5.7e-5, ch91 1.2e-4.
- Comparison figures: `comparisons/OF-APsimfit10000led_vs_Wwna-APsimfit10000led`,
  `..._vs_W-...`, the three-way one, and `Wwna-APsimfit10000led_vs_W-APsimfit10000led`.
- Numbers for the slides are the MC ones (conservative, like the paper). The conclusion the
  user presents: **small improvement with Wwna on APsimfit10000led**.

---

## 1. Single pulses from the raw binary — SOLVED, exact

`extract_AP_pulses_m205.py`. **The median of the extracted pulses equals `medianAP` bit-exactly
(max|diff| = 0.0) for all 15 WPs of ch91 and ch34.**

Binary format (`rawType = "Cupid"`), reverse-engineered and verified:
- 12-byte header: `uint32` (unknown, 27680) | `float32` sampling rate (10000) | `float32`
  full scale (10.069444 V);
- one sample per 4 bytes, `uint32` LE, **24-bit ADC in the top 3 bytes** (low byte always 0),
  offset binary: `V = (u32/256/2**23 - 1) * fullscale`;
- 1 GiB segments, `<run>_<prefix>_<ch:03d>_<seg:03d>.bin`. **All m205 AP events are in `_000`**.

Recipe: events with `crosscorr_signal_wp<wp>.pass` → window of 10000 starting at
`triggersample - 5000 + (midsample - 5000)` (read already shifted: `np.roll` wraps) → subtract
`baseline.baseline` (Octopus computes it on the **uncorrected** window, first 4900 samples).

Pulses in `m205_AP_pulses/ch<ch>/pulses_ch<ch>_wp<wp>.npy`, **8 channels × 15 WP (all but
ch37)**, peak-normalized. On the server `BIN_DIR = "/data2/LSC/DATA/RUN14/000205"`.

## 2. Per-time-bin errors — bootstrap, no Gaussian assumption

Bootstrap over the pulses (already contains 1/√N — **do not divide again**). The Gaussian
`1.2533·std/√N` overestimates by ~25% on the rise. A **floor** at the baseline error is
required (the peak bin has std = 0 by construction).

## 3. Template fits — `scan_residuals_bessel_m205.py`

Config: `CHANNELS`, `AP_SOURCE` (`"octopus"` default = pulses as saved, `"root"`, `"maxalign"`),
`MODELS`, `BESSEL_ORDER = 6`, `FCUT = 2500` (real electronics values, keep fixed).
Output `residual_scan_bessel/fits_<AP_SOURCE>/`.
- χ plateaus at **7 poles**; `PARAM_BOUND` = Nyquist (31416 rad/s); **28 multi-starts** needed;
  railing check after every fit; condition 1e9–1e13 → the poles are not measurements, the
  template is. `COST = "nps"` exists, default stays `time_ls`.
- `bestfit_*.npy` for **ch31, 34, 71, 83, 91**. Missing 40, 41, 94 (scan with **`RESET = False`**).

## 4. BI programs — template modes

| program | knobs | output |
|---|---|---|
| `analyse_BI_m205.py` (optimum filter) | `TEMPLATE_SOURCE = "root"｜"fit"｜"sim"`, `SIM_SOURCE`, `NPS_SOURCE` | `m205_results_octopus[_<tag>][_npsclean]` |
| `analysis_BI_m205_wiener_regolarized.py` | same + `USE_R`, `BETA_R`, `S_PENALTY`, `N_TRIALS`, `RUN_TAG` | `m205_results_wiener_<tag>[_npsclean][_swna1]` |
| `simulate_BI_error_m205.py` | `RESULTS_NAME`, `COMPARE`, `GEN_TEMPLATE`, `NSIM`, `CHUNK`, `N_SEEDS`, `OVERWRITE` | `BI_mc_error_m205.csv` + PNG in the results folder |

- `SIM_SOURCE = "APsimfit10000led"`; `sim_folder_tag(tag)` keeps tags starting with
  `APsim`/`APreal` as they are (old tags get `sim_` prefix). Unknown tags only warn.
- `ONLY_CHANNELS` in each (check it before a campaign: the Wiener one was last left at `[83]`).
- **Regularization `S_PENALTY = ("wna", 1.0)`**: adds `w·(s₁² + s₂²)` to the **training loss
  only** (analytic BI), so s cannot grow to exploit the template noise. Folder suffix `_swna1`.
  In plot labels: `W` = no penalty, `Wwna` = penalty.
- **Training convergence — checked 2026-09-20 with `check_training_convergence_m205.py`** (reads
  `<results>/training_history/hist_ch*_wp*.npz`, writes `training_convergence_ch<ch>.png`):
  **J plateaus by step ~150, λ does NOT settle.** With penalty log λ still rises at step 500 in
  70/75 points (slope +8.4e-4/step over the last 100 vs +1.4e-3 the 100 before → decelerating,
  ratio 0.58), λ median 7.6; without penalty λ still drifts (67/75), median 1.29, one point at 0
  (the known collapse). Residual J descent estimated from the tail: ch31 median 0.7% (no-penalty
  run 2.2%), ch34/71/83/91 < 0.06%. J is nearly flat in λ (−0.67%/e-fold with penalty, +0.61%
  without): a flat valley, so λ moves at almost no cost in J. Since BI = J[-1]·K, an under-trained
  point gives a slightly HIGH BI → the Wwna-vs-OF gain is conservative (OF has no training).
  **The loss actually minimised (J + penalty) was never saved**: `optimize_filters_wiener_lambda`
  now takes `history=` (dict → loss, s1, s2 per step, backward compatible) and the npz stores them.
  New knob `RUN_TAG` suffixes the results folder, so a test run cannot overwrite a campaign.
  Test still TO RUN on the server: `RUN_TAG="_conv3000"`, `N_TRIALS=3000`, `ONLY_CHANNELS=[31]`.
- **λ is NOT identifiable — exact degeneracy, measured 2026-09-20.** J, s1 and s2 depend ONLY on the
  product g_i = f_i·W(λ). Taking a trained point (ch31 wp29) and rescaling the band filters by the
  exact kernel ratio W(λ₀)/W(λ₁) while moving λ by ×0.1, ×3, ×100 leaves J unchanged to **1e-6** and
  s1, s2 identical to 6 digits. So λ drifting while the loss is flat = the optimizer sliding along a
  flat (gauge) direction; the applied filter does not move. Consequences: the `lambda_wiener` column
  is gauge, NOT physics — never compare λ across points or campaigns; the "λ collapse" without
  penalty is only a symptom (what matters is s, the noise amplification of the product); judge
  convergence on the loss and on s1/s2, never on λ. Natural fix if it ever matters: fix λ = 1 and
  train f1, f2 only (same reachable optimum, one parameter less, no drift).
- Both training programs now write `n_trials` and `train_s` (seconds of the optimizer alone) in the
  results CSV and print them; `analyse_BI_m205.py` uses `N_TRIALS = 500` like the Wiener one (was
  300), so the two curves and times are comparable. ~2.4 s/step locally → ~20 min per point at 500.
- `reliability_R(...)` / `USE_R` (R(f) template regularization) still exists; superseded by the
  penalty, only meaningful with the `root` template.
- **Paper bug** (their `build_mean_pulse_filteralignement`): the phase `exp(−2πi·shift/N)` lacks
  the frequency index → the sub-sample alignment is a no-op.

## 5. Monte-Carlo BI — `simulate_BI_error_m205.py`

Applies the **already trained** filters to simulated singles and pile-up (`dt_max=8e-4`),
90% cut on singles, counts survivors; `an.compute_BI_uncertainty` gives σ_BI (MC statistics
only, 1/√NSIM). Built-in check recomputes `H_unit` from the template and refuses to run on a
mismatch (caught two real bugs).

Event generation: template (`GEN_TEMPLATE="fit"`) × ROI amplitude + noise from the NPS:
`N = √nps·(a+ib)` with a,b ~ N(0,1) → E|N|² = 2·nps, `.real` of the ifft halves it →
E|FFT(x)|² = nps. **Do not add √2** (old bug, fixed in `src/simulation.py`).

New this session:
- `OVERWRITE` (replaces RESET_CSV): a row with the same `ROW_KEY = (channel, wp, gen, seed)` is
  replaced (read-modify-write under flock); otherwise appended. `read_rows` refuses a CSV with
  a mixed/old schema instead of silently misreading it.
- `N_SEEDS` (default 50; the earlier version was lost, re-added 2026-09-10): seed SEED + i
  (1234…1283) = ONE full MC run of NSIM events per population. Inside the run ONE Generator
  (`default_rng(seed)`) is passed to every chunk, so the stream continues without repeats.
  Singles and pile-up get two Generators with the same seed (paired); unpaired → `[seed, 1]`.
  Events still depend on CHUNK (draw order inside a block: all real noise, all imag, amp, r, Δt)
  → **same CHUNK in all folders**. Now `CHUNK = 500`: fastest (one 50 000-event run 99 s
  locally, vs 145 s at 2000 and 165 s at 4000, 3.6 GB peak). **Compare mode**: `COMPARE = [...]`
  folders are simulated together with `RESULTS_NAME` — events generated ONCE per seed, every
  folder's filters applied, each folder writes its own CSV; identical events by construction,
  1.8× faster than one folder at a time (2000 events: 6.8 s vs 12.5 s). Logs/jobs/frozen copy in
  RESULTS_NAME. Check: `test/check_compare_mode_m205.py` (each folder together == alone, bit for bit). One row per seed (`seed` column, in
  `ROW_KEY`; a row without seed = SEED). The old single-seed CSV used per-block seeds
  (seed + k): the current code does not reproduce it.
  **Separate files**: `N_SEEDS == 1` → `BI_mc_error_m205.csv`, `N_SEEDS > 1` →
  `BI_mc_error_m205_seeds<N>.csv`. Compare picks one with `MC_CSV` (now the `_seeds50` one).
  Earlier measurement: **ρ ≈ +0.998** between two models on the same events (3 seeds).
- `compare_templates_m205.py` reads all seeds, then `align_seeds` keeps per point only the seeds
  common to ALL loaded sets (same events for every filter): BI_mc = **median** over seeds,
  σ = √(π/2)·σ_p/√n, σ_p = (P84−P16)/2 of the seeds (`med_err`/`robust_sigma`: unbiased
  with tails, skew or an outlier, where std overestimates by 20–50%, measured with n = 50; 1 seed → σ of compute_BI_uncertainty).
  Seeds used: 1234…1283 (printed at MC start). `signal_amp` identical in the 3 final
  folders → same seed = same events verified. For two different
  sets both on MC with ≥2 common seeds, `paired()` gives ΔBI seed by seed → ΔBI = median,
  error = √(π/2)·σ_p/√n (covariance included), also used by z; summary prints the shrink factor vs
  quadrature and the effective ρ. With 1 seed it falls back to quadrature (ρ = 0).
- `READOUT` removed (always the original max readout). `_parse_results_name` understands
  `_APsim*` / `_APreal*`.

## 6. Simulated AP — `build_simAP_injected_m205.py` (rewritten, per the relatore)

Goal: a template that has the **same construction as the real AP** but many more pulses, so
its noise imprint is negligible. Generate N pulses from the **fit**, add NPS noise, take the
mean (no per-pulse normalization), peak-normalize at the end.
- `MODE = "mc"` (generated like the MC) | `"realnoise"` (fit + real noise windows).
- `N_PULSES = 10000` (or `"match"` = same N as the real AP), `GEN_CHUNK = 500`.
- `AP_AMPLITUDE = "led"` (default) | `"roi"`. The AP is peak-normalized, so the amplitude
  **only sets its noise σ/A**. With the ROI amplitude the pulses are noise-dominated (raw SNR
  2.6–17) and the AP was *noisier* than the 38-pulse real one; with the LED amplitude it has the
  real σ/A, as in the paper (which builds the template from the injected data).
- Tag `APsim<template><N>[led]` (e.g. `APsimfit10000led`), output
  `m205_AP_sim/ch<ch>/simAP_<tag>_ch<ch>_wp<wp>.npy`. The old `fitinj`/`rootinj`/`fitgen`/
  `rootgen` files are still on disk but **buried** (confusing, do not use).
- Selftest built in (tolerance 0.85–1.15 on the noise).
- Can be regenerated on the server from the script alone (no need to sync the .npy).

## 7. Why Wiener overtrains — template noise (the key physics point)

The template's residual noise is a **deterministic imprint** in both S and H; a trainable λ
learns it and the analytic BI "hallucinates" a gain that the MC (generated from the smooth fit)
does not confirm → MC/an > 1 and W worse than OF. Measured:
- `fit` template: W/OF 0.993 (45/70), MC/an 0.955–1.085.
- `APsimfit10000` (ROI amplitude): W/OF 1.010 (22/75), MC/an 1.03–1.21 → worse.
- `APsimfit10000led` + penalty: see FINAL RESULT.
Ruled out as causes: `jitter_max`, trigger jitter, min(OF, W), penalty strength.

**Two SNRs — never mix them:**
- σ_OF = 1/√Σ|S|²/nps → OF SNR **10–150**; medians ch31 22.3, ch34 100.5, ch71 36.7,
  ch83 67.3, ch91 24.9.
- raw peak SNR = M/√Σnps → medians 9.3 / 11.8 / 17.0 / 11.8 / 2.6. Ratio 5–15.

## THE NPS FACTOR — resolved by the clean NPS

Octopus `medianpower` is not E|FFT|²: ×2 (one-sided kept in a mirrored array) and ×ln2 (median
of an exponential) → **1.386× the truth** on the independent-window test (the code comment in
`analyse_BI_m205.py` quotes 1.84 on ch91 with the full selection). **Do not use `_power`**
(destroyed by contaminated events).
Now **`NPS_SOURCE = "clean"`** is the default everywhere: `build_NPS_clean_m205.py` measures
the NPS from real noise windows of the binary (Octopus selection + RMS cut, mean periodogram),
reproduces the RMS of independent windows to 0.4%, in `m205_NPS_clean/ch<ch>/nps_ch<ch>_wp<wp>.npy`.
σ drops 18–30% and the BI drops. Folders carry `_npsclean`; `-npsoct` in labels = Octopus NPS.

## 8. Comparison plots — `compare_templates_m205.py`

Config at top: `SETS` (result folders), `PAIRS` (`("OF-fit:mc", "OF-root:mc")`; `:mc`/`:an`
allows **MC vs analytic of the same model**), `MC_GEN`, `BI_SOURCE = "mc"|"analytic"|"both"`,
`SLIDE`, `ONLY_CHANNELS`, `BI_TARGET = 5e-5`.
- Names deduced from folder names by `describe()` (OF/W/Wwna, template, `-npsoct` only when
  not clean). Missing data are **not plotted** (never invented errors).
- ΔBI = (BI_A − BI_B)/BI_B·100, written on the y axis with the legend names; error
  `100·R·√((σA/A)² + (σB/B)²)` (ρ = 0). For MC-vs-analytic, z uses only σ_MC.
- `SLIDE = True`: one big BI panel + ΔBI panel, per-channel figures (`plot_channels`), log x,
  scientific notation, big markers (`SLIDE_MS = 9`), dark-blue delta (`#0d2c6b`), red dashed
  CUPID target line in every BI panel.
- `read_csv_checked` guards against mixed-schema CSVs (happened with the Wiener CSV when two
  columns were added: `~/Downloads/BI_results_m205_wiener_root_npsclean.csv` repaired, backup `.bak`).
- **Sync `compare_templates_m205.py` to the server** (BI_TARGET line was added last; the user
  reported a plot without it — it was the old version on the server).

## 9. Thesis slides

- `make_slide_figures_m205.py` → `slides_figures/fig01–fig13` (spectrum, pile-up pulses,
  ingredients, model, discriminator, BI vs V with target, overtraining, regularization, MC event,
  filtered event, single/pile-up pulses and their filtered versions). `FILT_SET` =
  OF APsimfit10000led; colors `C_SINGLE = "#04254F"`; English text. Not yet committed.
- Deck: `~/Desktop/Slide standard - Tesi Alberto Zanelli.pptx` (user's latest). Template:
  decorations are GROUP shapes, figures are picture fills on FREEFORM, titles Bree Serif
  `#04264F`, body Canva Sans. A `build_slides_pileup_m205.py` (python-pptx) that rebuilt the
  pile-up slides per the relatore's outline lived in the old scratchpad — **gone**, rewrite if
  needed.
- Artifacts: scaletta https://claude.ai/code/artifact/8d66128b-1b69-44d0-81b9-b33fd5e56121 ,
  5-min Italian speech https://claude.ai/code/artifact/0a2ef884-eaaa-4132-b078-855bb8d4f5f5
  (mentions, as future work, choosing the WP also on the expected BI).
- Pending, not started: translate two slides of `~/Desktop/Presentazione standard1.pptx` to
  English (the user interrupted — ask before doing it).

---

## WHAT WORKED
- Reading the window **already shifted** from the stream; taking Octopus's `baseline`.
- Bootstrap errors + weighted fit.
- Folder names carrying every mode (`_<tag>`, `_npsclean`, `_swna1`), and plot labels deduced
  from them.
- Self-checks that fail loudly (kernel vs template, CSV schema, railing).
- Simulated AP from the fit at the **LED** amplitude with 10000 pulses + penalty on s.

## WHAT DIDN'T WORK / DON'T DO
- **Don't make β or λ-free knobs trainable on the training J** — they collapse (β→0, λ→0).
- Half-spectrum reconstruction of a COMPLEX array needs `concat([h, conj(h[-2:0:-1])])`.
- `TensorDataset` in `get_PSD_interpole` has no `win_length`: use `ds.NumpyDataset`.
- Sub-sample alignment, free Bessel cutoff/order, offsets/slopes in the fit: no gain.
- Generating time-domain pulses from `compute_H`'s S (already Hann-windowed).
- **Building the simulated AP at the ROI amplitude**: noise-dominated, AP noisier than the real one.
- Generating a simulated AP from the **real AP** (its noise is copied in every pulse).
- `timing_filter` in `src/analysis.py`: tried and reverted.
- **Chasing a gain on the ANALYTIC BI**: it rewards filters that amplify noise, because the
  first-order σ_Y model is more optimistic for them (MC/an 1.009 OF → 1.027 λ=1). Every
  conclusion must be taken on the MC.
- **Fixing λ = 1** (ch34, with and without the s-penalty, 500 steps): analytic −1.3/−1.5%, but
  MC **+0.5/+0.6%**, i.e. worse than OF in 14/15 points. Same for λ trainable: MC +0.01%.
- Freezing λ from step 0 (+1.15% cost, never catches up) and lr of the filters at 3e-1
  (+1.06%, unstable).
- Choosing λ without training: the "equivalent λ" of the trained filters has median ≈ 2 but the
  Wiener family alone misses the trained filter by 0.25 dex, and three analytic criteria
  (crossover at the OF-weight peak / median / RMS band β) correlate only 0.24–0.34 with it.
  λ = 1 is the defensible default, and it is what Wiener theory says anyway.

## BUGS FIXED IN SHARED CODE (`src/`)
- `src/simulation.py`: `simulate_frequency_pulses` injected √2 too much noise (MC/an +21% → +2.8%).
- `extract_AP_pulses_m205.py` SyntaxError from a half-merged edit.

## THE PAPER (`~/Desktop/Tesi_Erasmus/MasterThesis___Alberto_Zanelli/Pileup_Paper_EPJC-2.pdf`)
- **It uses ONLY the classical optimum filter**, `H(ω) = ŝ*(ω)/P_N(ω)` normalised so
  `∫H·ŝ = 1`. The words Wiener / regularization / Tikhonov / deconvolution appear **zero** times
  (checked with pypdf). λ, the s-penalty and R(f) are all our additions, not theirs.
- Analytic vs simulated BI **< 8%**, simulated is higher; they **report the simulated BI**.
- Template is the largest systematic (up to 13%); single high-energy pulses give artificially
  low BI through the "residual high-frequency noise imprint" — exactly our overtraining.
- They recompute the average pulse from the injected data (our `APsim...led` is the analogue).
- DAQ: 10 kHz, 6th-order Bessel @ 2.5 kHz.

---

## NEXT STEPS

**1. START HERE — free the PHASE of the band filters (the user picked this).**
Today `f = activation(f_param)` is real ≥ 0 and mirrored hermitian, so the total filter is
`positive × S*`: the phase is frozen and OF/Wiener span the same set (see READ THIS FIRST).
A COMPLEX `f` (two real parameter vectors, modulus and phase, or real+imag) enlarges the family
for real, and phase is what encodes the DELAY between the two piled-up pulses — the thing being
discriminated. Where to touch: `optimize_filters_wiener_lambda` / `optimize_filters_wiener` in
`src/analysis.py` (the `torch.cat([f, f[1:-1].flip(0)])` mirroring must become
`torch.cat([f, conj(f[1:-1].flip(0))])` for a complex f, like `full_spectrum` does for the
kernel), plus `compute_vars_wiener` (check whether it assumes `f` real: it uses `|f|²` and
`f·Re(conj(W)S)` — the second term needs `Re(conj(f·W)·S)` when f is complex).
Test protocol: one channel (34 is the cleanest), 500 steps, same template and NPS, then
**simulate_BI_error (MC) and compare on the MC BI, not the analytic one**; quote ΔBI with the
paired error (`compare_templates_m205.py` already does it if you run ≥2 seeds).
Guard rail: a complex f can fake a gain by exploiting template noise — check MC/an stays ≈ 1.01
and s stays ≤ 0.05, exactly as for the λ tests.

**Other ideas for a REAL gain** (same family problem, ordered by effort/benefit):
2. Train on the right objective: the analytic J is biased by 1–3% and biased DIFFERENTLY per
   filter, which is why analytic gains evaporate. A differentiable surrogate of the MC survival
   fraction (soft cut on simulated events) optimises what is actually measured.
3. More than two bands + a multivariate discriminant (linear/quadratic on 3–4 filtered
   amplitudes) instead of the ratio of two — OF stays a special case, so it cannot do worse.
4. Neyman–Pearson likelihood ratio between "single" and "pile-up", marginalised over amplitude,
   delay and energy ratio: the only route with a theoretical optimality guarantee.
5. Reduce the amplitude-estimator bias (discrete argmax → the periodic cost spikes, one every
   ~170 steps, up to +6.6%): a smoother interpolation lowers s and shrinks MC/an.

**Training mechanics, measured (ch31 wp29, reduced 40×40 grid, 1500 steps unless stated):**
- 500 steps is NOT converged: the loss flattens only around **1500** (conv3000 test at full
  grid: M drops another −2.4% between 500 and 3000, BI −3.0%).
- `lr` of the band filters **3e-2** halves the steps (472 vs 972 to stay within 0.5% of the
  asymptote) for +0.10% cost; 1e-1 → 295 steps for +0.33%; 3e-1 is unstable (+1.06%).
- λ converges for free if its lr decays to 0 (cosine, or off at half): drift +2.4% → 0.0–0.2%,
  cost unchanged. **Freezing λ from step 0 costs +1.15%** and never catches up.
- `eta_min` is a FLOOR, not a factor: with `eta_min = 1e-2` the filters' lr never decays and
  λ's stops at 1e-2; passing `lr_lambda = 0` made λ's lr RISE 0 → 1e-2 (now guarded).

**Housekeeping still open:**
6. Future work (thesis): choose the WP using the expected BI, not only the SNR.
7. Never done: β selection by 19+19 cross-fit; bootstrap over the 38 pulses for the physical
   BI uncertainty; fits for ch40/41/94.
8. Decide with the relatore how to present the NPS change (clean vs Octopus) vs collaboration
   numbers.
9. The L-curve for the penalty weight `w` (J vs s₁²+s₂² as w scans 0…10, pick the corner) would
   replace "we chose w = 1" with a criterion that has a name (Hansen) and a reference. λ is NOT
   the parameter to apply it to — the penalty weight is.

## New tools and knobs (2026-09-19/21) — all pushed
- `simulate_BI_error_m205.py`: `N_SEEDS` (50 runs, one row per seed, `seed` column),
  `CHUNK = 500` (fastest; **CHUNK changes the events**, keep it equal across campaigns),
  **compare mode** `COMPARE = [...]` (events generated ONCE per seed and passed through every
  folder's filters, each writes its own CSV — identical events by construction, 1.8× faster).
  Check: `test/check_compare_mode_m205.py`.
- `compare_templates_m205.py`: reads all seeds, `align_seeds` keeps only the seeds common to all
  sets, BI = median, σ = √(π/2)·σ_p/√n with `robust_sigma` = (P84−P16)/2, ΔBI **paired** seed by
  seed (covariance included: the paired error is 0.35× the quadrature, effective ρ ≈ 0.88).
- `plot_BI_distributions_m205.py`: `dist_BI_ch<ch>.png`, `dist_dBI_ch<ch>.png`.
- `check_training_convergence_m205.py`: tail metrics + per-channel grids + `DETAIL = [(ch, wp)]`
  four-panel figure (loss, zoom, λ, s₁/s₂). `--selftest` included.
- Both training programs: `RUN_TAG` (results-folder suffix for test runs; the name parsers of
  compare/simulate strip it), `n_trials` and `train_s` columns, `history=` in the optimizers
  (loss, s1, s2 per step, saved in the npz), `N_TRIALS = 500` in the OF one too (was 300).
- Wiener program: `TRAIN_LAMBDA` / `LAMBDA_VALUE`. With `TRAIN_LAMBDA = False` it calls the
  OTHER function, `optimize_filters_wiener` (λ is not a parameter at all), and the folder gets
  `_lam<value>`. Verified: with the same initial filters and λ = 1 the two functions give J
  identical **bit for bit**, so a λ-fixed vs λ-free comparison differs in exactly one thing.
- 50-seed MC results (`BI_mc_error_m205_seeds50.csv`, 4 folders): σ_p(BI_s)/σ_formula = 0.88,
  i.e. `compute_BI_uncertainty` **overestimates the per-run MC error by ~14%**; the new
  event scheme agrees with the old single-seed runs to 0.9954 ± 0.52%.

## Earlier threads (DONE — background)
- `test/fit_one_pulse_m205.py` (single-pulse fit, CSV has no `cost` column).
- `build_medianAP_maxalign_m205.py`, `compare_argonauts_vs_octopus_m205.py`,
  `compare_BI_vs_hfpower_m205.py`, `risetime_and_amplitude_.py`, `scan_residuals_m205.py`,
  `plot_AP_spectra_m205.py`, `compare_wiener_vs_optimum_m205.py`.
- Octopus normalization: flat-top factor 5.708 = N/Σw² is correct.
- BI/Wiener variants: `_wiener.py` (scalar λ), `_wiener_freq.py` (λ(f)); filters in
  `*/trained_filters/` as the independent half-spectrum.
