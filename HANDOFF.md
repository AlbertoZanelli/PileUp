# PileUp — Handoff

Thesis work on the CUPID LMO light-detector average pulses (AP) of run **m205**.
Three threads are now live and interlocked: (1) the **R(f) template regularization** in the
Wiener training, (2) a **fitted (denoised) template** used in place of the AP, and (3) a
**Monte-Carlo BI with uncertainty** that validates the analytical BI. Along the way the
single pulses that build the AP were recovered from the raw binaries, which unlocked
everything else. Older threads are summarized at the bottom.

**NEXT AGENT: use ponytail mode** (the user asked). Laziest thing that works, config-at-top,
no scaffolding. Discussion in Italian, plot text in English. **The user commits, not you.**

---

## Environment (IMPORTANT)
- **Project root**: `~/Desktop/Tesi_Erasmus/PileUp` (git repo; `.root` data gitignored but
  PRESENT locally in `Processed/`). Server copy: `/data/users/azanelli/PileUp` (**local edits
  must be synced by hand — several times the two diverged and a merge left a SyntaxError**).
- **Run with**: `KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/bin/python3.13 <script.py>`
  - python3.13: numpy/scipy/matplotlib/uproot. **No ROOT, no torch.**
  - **pyrootAlbi** (`/opt/anaconda3/envs/pyrootAlbi/bin/python`): ROOT + torch + uproot + pypdf.
    Needed for anything that trains (torch) or reads PDFs.
- Cluster is **PBS**: `qsub`/`qstat`/`qdel`. Kill a campaign with
  `qstat -u $USER | grep <PREFIX> | awk '{print $1}' | xargs qdel`.
  Job prefixes: `BI`/`BIF`/`BIS` (optimum), `BIW`/`BIWF`/`BIWR` (Wiener), `BSCAN` (fits).
- **Measured**: BI worker peak RAM **2.04 GB** (`RAM_GB = 4` is fine), and it is fastest with
  **2 threads** (78 s → 46 s per 20 steps; 4 and 8 threads are worse). The qsub line requests
  no cores — adding `-l nodes=1:ppn=2` + `OMP_NUM_THREADS=2` is the real speed lever, not RAM.

## Data conventions (m205)
- **10 kHz**, window **10000** samples (1 s). `Processed/Processed_*_000205_<ch>.root`.
  - AP: `averagepulse_ap_wp<wp>_medianAP` (peak-norm 1, peak at ~0.5 s, pretrigger 50%).
  - Noise: `averagepowerspectrum_noise_wp<wp>_medianpower` (see the **NPS section** — it is
    NOT E|FFT|²). Also available: `_power` (arithmetic mean), `_trimmedpower`,
    `_geometricpower`, `_harmpower`.
  - **N per AP** = `averagepulse_ap_wp<wp>_APdistro` entries / 10000 → **36–39, varies per WP**.
  - WP→V_bias via `VBIAS_LIST[wp//2]`, **odd wp only**, 15 WPs.
- **9 channels**: 31,34,37,40,41,71,83,91,94.
- `amplitudes_m205.csv` holds the **ROI-equivalent** amplitude: verified to be the LED
  amplitude divided by **430.9, constant to 0.0% across all 15 WPs**. Use `signal_amp` for BI
  events; use `maxminusbaseline.amplitude` (LED) when simulating the AP construction.

---

## 1. Single pulses from the raw binary — SOLVED, exact

`extract_AP_pulses_m205.py`. **The median of the extracted pulses equals `medianAP` bit-exactly
(max|diff| = 0.0) for all 15 WPs of ch91 and ch34.**

Binary format (`rawType = "Cupid"`), reverse-engineered and verified:
- 12-byte header: `uint32` (unknown, 27680) | `float32` sampling rate (10000) | `float32`
  full scale (10.069444 V);
- one sample per 4 bytes, `uint32` LE, **24-bit ADC in the top 3 bytes** (low byte always 0),
  offset binary: `V = (u32/256/2**23 - 1) * fullscale`;
- 1 GiB segments, `<run>_<prefix>_<ch:03d>_<seg:03d>.bin`. **All m205 AP events are in `_000`**
  (last one at 96.2% of the segment).

Recipe: events with `crosscorr_signal_wp<wp>.pass` → window of 10000 starting at
`triggersample - 5000 + (midsample - 5000)` (read already shifted: `np.roll` wraps and costs
1.2e-3 on the AP) → subtract `baseline.baseline`, which Octopus computes on the **uncorrected**
window (mean of the first 4900 samples). Recomputing the baseline on the shifted window leaves
a 3e-6 residual.

Pulses live in `m205_AP_pulses/ch<ch>/pulses_ch<ch>_wp<wp>.npy`, **8 channels × 15 WP
(all but ch37)**, peak-normalized, ~45 MB per channel. On the server set
`BIN_DIR = "/data2/LSC/DATA/RUN14/000205"` and `CHANNELS = [...]`.

## 2. Per-time-bin errors — bootstrap, no Gaussian assumption

The AP is a median of N pulses; its per-bin error comes from **bootstrap** over the pulses
(the spread of the resampled medians already contains 1/√N — **do not divide by √N again**).
The Gaussian formula `1.2533·std/√N` (√(π/2) = median-vs-mean factor) **overestimates by ~25%
on the rising edge**, where the distribution is not Gaussian (skew +0.28, 28% of bins fail
Shapiro). Baseline and tail agree within 4%.
A **floor** at the baseline error is required: the peak bin has std = 0 by construction
(every pulse is normalized to its own max) and so does the zero-padded tail after alignment.

## 3. Template fits — `scan_residuals_bessel_m205.py`

Config: `CHANNELS` (list), `AP_SOURCE`, `MODELS`, `BESSEL_ORDER = 6`, `FCUT = 2500`.
Output goes to `residual_scan_bessel/fits_<AP_SOURCE>/` (the folder name carries AP_SOURCE
**because a RESET once wiped fits made with another template**).

`AP_SOURCE`:
- **`"octopus"` (default, preferred)** — AP = median of the saved pulses *as they are*, i.e.
  aligned at mid-rise exactly like Octopus. Identical to the ROOT AP (0.0), but with real
  bootstrap errors. Needs the pulses.
- `"root"` — medianAP from ROOT, errors from the `APdistro` TH2D with a floor at the AP's
  baseline RMS. **The floor is mandatory**: the histogram's amplitude bins are 0.006 wide,
  7× the noise, so in quiet regions all pulses fall in one bin and the measured spread is
  exactly zero. Validated against bootstrap: −8% in baseline, −9% on the rise.
- `"maxalign"` — pulses re-aligned on the maximum (`build_medianAP_maxalign_m205.align_on_max`),
  giving an AP that differs from the ROOT one by 0.019 on the rise.

What the fits established:
- **χ plateaus at 7 poles** (3p 4.90 → 4p 4.87 → 6p 1.97 → 7p 1.05 → 9p 1.00); beyond that you
  are fitting the 38-pulse noise.
- `PARAM_BOUND` must be **Nyquist (31416 rad/s)**: with the old 8000 two poles railed.
- **28 multi-starts are necessary**: with 8, the fit landed in a local minimum on 6 WP out of 15
  (wp5 χ=7.3 instead of 2.1). `STARTS` is now a config-level grid.
- Automatic **railing check** after every fit (physical criterion: |x| ≥ 0.999·bound, or
  τ = 1/|x| longer than the window). Poles rail at Nyquist on several high-bias WPs — the pulse
  is faster than 10 kHz can resolve.
- **Parameter errors** from the Jacobian + condition number. Condition is 1e9–1e13 everywhere:
  the individual poles are NOT measurements (errors up to 380%, correlations 0.9997), the
  template is. Two near-coincident poles are NOT removable: dropping one takes χ from 1.05 to
  1.98 (a double pole gives t·e^{pt}, and a near-cancelling pole/zero pair is a real slow
  correction of 8%).
- A `COST = "nps"` mode exists (frequency-domain least squares weighted by 1/NPS). On 7p z3 it
  is much better in the band that matters (Δbeta +6.0% → −0.85%); on 9p z4 it does not converge.
  Default stays `time_ls`.

Current state: `bestfit_*.npy` for **ch31, 34, 71, 83, 91**; `_fits/*.npz` only for **ch31**
(10 models × 15 WP) because RESET wiped the rest. Missing: 40, 41, 94 (and 37, no pulses).
With 10 candidate models `best_fit` picks 8p z5/z6, **not** the largest — but the winner varies
per WP, so for a homogeneous template set fix a single model.

## 4. BI programs — template modes

All three now share the same two knobs:

| program | modes | output |
|---|---|---|
| `analyse_BI_m205.py` (optimum filter) | `TEMPLATE_SOURCE = "root"｜"fit"｜"sim"` | `m205_results_octopus[_fit｜_sim]` |
| `analysis_BI_m205_wiener_regolarized.py` | `TEMPLATE_SOURCE`, `USE_R`, `BETA_R = 2.0` | `m205_results_wiener_<tag>` |
| `simulate_BI_error_m205.py` | `GEN_TEMPLATE` / `TRAIN_TEMPLATE` | CSV + PNG in `RESULTS_DIR` |

Both BI programs have `ONLY_CHANNELS` (list or None) and skip pairs whose external template is
missing, with an `[INFO]`, instead of submitting doomed jobs. Folders, CSV names and job
prefixes all carry the mode tag, so runs never overwrite each other.

`reliability_R(S, nps, N_events, beta, eps_frac)` lives in `src/analysis.py` next to
`compute_W_torch`, and `optimize_filters_wiener_lambda(..., use_R=False, N_events, beta_R,
eps_R)` folds `W ← R·W` **once** (R does not depend on λ, so it is computed outside the loop).
Default off ⇒ byte-identical to the old behaviour. `eps_R = 1e-12`: at the original 1e-6 it
acted as a **second, unwanted gate** (it exceeded the smallest β·NPS/N and shifted R by up to
0.4 in the HF tail).

## 5. Monte-Carlo BI with uncertainty — `simulate_BI_error_m205.py`

Applies the **already trained** filters (no retraining) to simulated singles (`dt_max=0`) and
pile-up (`dt_max=8e-4`), takes the 90% cut on the singles and counts survivors;
`an.compute_BI_uncertainty` gives σ_BI (binomial + cut-position via KDE).
`--make-ap` instead generates **simulated APs** with the same N as the real one, into
`m205_AP_sim/ch<ch>/simAP_ch<ch>_wp<wp>.npy`.

A built-in check recomputes `H_unit` from `TRAIN_TEMPLATE` and compares it with the saved
kernel; it refuses to run on a mismatch. **It has already caught two real bugs.**

**σ_BI is the MC statistical error** (shrinks as 1/√NSIM), NOT the physical uncertainty on the
BI. Its value is the cross-check: BI_MC must equal BI_analytic.

Result (ch91 WP15, after the fixes below): **BI_MC / BI_analytic = 1.028**, well inside the
paper's <8%. σ(single) MC/analytic = 0.985, i.e. **`compute_sigma_ratio` (linear propagation)
is correct** — an earlier accusation against it was wrong, it was the simulator.

---

## THE NPS FACTOR — measured, explained, decision pending

This closes a long-standing puzzle (the old Argonauts-vs-Octopus comparison with per-channel
factors F = 1.35–1.68).

**`nps_project = 2·ln2 × E|FFT|²_true = 1.386 ×`**, measured **1.384** on 100 independent
noise windows (10000 samples, ≥2 s from any LED/signal trigger, read from the ch91 binary).
Decomposed and verified separately:
- **×2** — one-sided convention kept inside an array that is then mirrored and used as a
  two-sided PSD. Confirmed twice: `power`/true = **2.07**, `medianpower`/median = **2.01**.
- **×ln2** — Octopus stores the **median over events**, and the median of an exponential is
  ln2 of the mean. Measured **0.703–0.714 on 8 channels**, expected 0.693.

Consequences: `sigma_analytic` is **+21% too large**, SNR ~21% too low, **BI conservative**.
`nps /= 2*np.log(2)` brings σ_OF within **3%** of the truth (residual from the <10 Hz bins,
where the factor is 1.66–1.72 because the statistics there are not pure exponential).

**Do NOT switch to `_power`**: the arithmetic mean is destroyed by contaminated events (ratio
of sums 21.5 against the same truth). The median is the only sane estimator on that sample —
Octopus is right to use it, it just needs the /ln2 if you want the mean.

**Decision, to be taken with the relatore, not silently**: correcting changes every number
produced so far (σ, SNR, BI, and the trained filters through λ) and makes them incomparable
with the collaboration's. It changes **no relative comparison** (template vs template, R on/off,
WP vs WP) because it is a common factor. Recommendation: **do not correct now**, declare it as
a measured systematic.

---

## WHAT WORKED
- Reading the window **already shifted** from the stream instead of `np.roll` (no wrap-around).
- Taking Octopus's stored `baseline` instead of recomputing it → exact reproduction.
- Bootstrap errors + weighted fit: χ becomes interpretable against 1.
- Letting the output folder name carry the mode (`fits_<AP_SOURCE>`, `_fit`, `_sim`, `_root_R`).
- Self-checks that fail loudly (kernel vs template, median vs medianAP, railing) — they caught
  a conjugation bug, a template/filter mismatch and an active parameter bound.
- Reusing `compare_wiener_vs_optimum_m205.py`: it was already generic, two dict entries were
  enough. Colours are now **per suffix** (`SET_COLORS`), so a set keeps its colour across figures.

## WHAT DIDN'T WORK / DON'T DO
- **Don't make β trainable** and don't tune it on the training J — it collapses to β→0.
  β stays 2.0; the honest selector is the 19+19 cross-fit on the real pulses (never done).
- **Half-spectrum reconstruction of a COMPLEX array**: the kernel needs
  `concat([h, conj(h[-2:0:-1])])`. The docstrings in all BI programs said otherwise (correct
  only for the real f1/f2) — fixed in 4 files. This produced a phase-flipped kernel.
- `TensorDataset` in `get_PSD_interpole`: it has no `win_length`. Use `ds.NumpyDataset` and set
  `.win_length` (the example in `src/simulation.py.__main__` is stale).
- Sub-sample alignment of the pulses: does **not** improve the fit (χ 1.00 → 1.03) even though
  it changes the AP by 2.4e-2 on the rise. The rise residual is the model, not the alignment.
- Free Bessel cutoff / order: flat above 1500 Hz, fully degenerate with the free poles. And
  6th order @ 2.5 kHz are the **real** electronics values (paper, sec. 4.1) — keep them fixed.
- Adding an offset, a slope, or box-car ADC averaging to the fit: ≤3% on χ, not worth it.
- Generating pulses from `compute_H`'s S when you need a **time-domain** template: that S is
  `FFT(AP × hanning)` and the window would be applied twice.
- Simulating the AP at the ROI amplitude: the pulses would be noise-dominated and the
  peak-normalized median meaningless. Use the LED amplitude.

## BUGS FIXED IN SHARED CODE (`src/`)
- `src/simulation.py`: `simulate_frequency_pulses` (and `_fixed_dt_r`) injected **√2 too much
  noise** (a factor 2 in power). The noise is non-Hermitian and `.real` of the ifft halves the
  variance, so the `× np.sqrt(2)` was wrong. Verified with the exact OF estimator:
  σ_sim/σ_OF = 1.408 before, **0.988** after. This alone moved MC-vs-analytic from **+21% to
  +2.8%**. `src/plots.py` used the same function.
  A new optional `fold_ratio=False` folds r → min(r, 1−r) onto the analytical convention
  (tested: it changes nothing, 1.213 vs 1.214).
- `extract_AP_pulses_m205.py` had a **SyntaxError** from a half-merged edit between the local
  and server versions — repaired keeping the per-channel subfolder layout.

## OPEN — the simulated-AP noise (partly solved)

A simulated AP generated from the **real AP** cannot be an independent noise realization: the
template's own residual noise (1.74e-4) is **identical in all N generated pulses**, so the
median does not reduce it. Measured decomposition (ch91 WP15):

| | real | simulated | ratio |
|---|---|---|---|
| per-pulse noise (normalized) | 9.27e-4 | 1.125e-3 | 1.213 (= the NPS factor) |
| AP noise (median of 38) | 1.743e-4 | 2.716e-4 | 1.558 |

`σ_simAP = sqrt(σ_template² + (1.2533·σ_new/√N)²)` predicts 2.875e-4 — and generating from the
**fit** (smooth by construction, σ_template = 0) gives **2.127e-4**, matching the
new-noise-only expectation. **So: set `GEN_TEMPLATE = "fit"` when making simulated APs.**
What remains is only the NPS factor (2.13e-4 vs 1.74e-4 = 1.22 ≈ √1.386).

## THE PAPER (`~/Desktop/Pileup_Paper_EPJC-2.pdf`)
- Sec. 4.5: analytic vs simulated BI **< 8%** (Table 2: <10%), simulated is the **higher** one;
  they attribute it to correlated/non-stationary noise (+1.9% average on data) and **report the
  simulated BI** as the conservative choice.
- **Template is the largest systematic: up to 13%.** Single high-energy pulses give the lowest
  BI but it is an artefact — "residual high-frequency noise imprint ... may be exploited by the
  optimized weighting functions" (exactly the overtraining R(f) targets). Average pulse
  recommended; the phenomenological model is **1.5–7.4% worse** when used for training.
- To avoid the injection/training bias **they recompute the average pulse from the injected
  data**. They never quantify the same-template case — so measuring it would be original.
- Grid: the trained filters are stable (<0.6%) but the **analytical** BI needs Δt < 8 µs and
  r < 0.005 (N = 100) for 1% stability. Our grid already matches.
- DAQ: 10 kHz, **6th-order Bessel @ 2.5 kHz**, rise times 0.5–0.8 ms (ours: 0.7 ms).

---

## NEXT STEPS

**Overnight, independent, no conflicts:**
1. `analysis_BI_m205_wiener_regolarized.py` with `TEMPLATE_SOURCE="root"`, `USE_R=True`,
   `ONLY_CHANNELS=None` → 135 jobs, the R(f) deliverable.
2. `simulate_BI_error_m205.py` with `ONLY_CHANNELS=None`, `NSIM=20000` → **serial**, 3–4 h in
   one job. Closes analytic-vs-MC on all 9 channels (we have one point, +2.8%).
3. `analyse_BI_m205.py` with `TEMPLATE_SOURCE="fit"`, `ONLY_CHANNELS=None` → ~75 jobs
   (channels without a template are skipped). **Do not run together with a scan**, which
   rewrites `bestfit_*.npy`.

**Then:**
4. Scan the missing channels: `CHANNELS = [40, 41, 94]` and **`RESET = False`** (True wipes the
   other channels' fits).
5. The four-way template study, which is what the paper's Table 1 (i)/(ii) does:
   `GEN_TEMPLATE` × `TRAIN_TEMPLATE` ∈ {root, fit} — how much a noisy template flatters itself.
6. Decide with the relatore whether to apply `nps /= 2ln2`.
7. Still never done: **β selection by 19+19 cross-fit** on the real pulses (now trivial, the
   pulses are on disk); and a **bootstrap over the 38 pulses** for the physical BI uncertainty
   (the only one that is not MC statistics).

## Earlier threads (DONE — background)
- **Single-pulse fit** `test/fit_one_pulse_m205.py`: now reads `SOURCE = "root"|"cfile"|"npy"`,
  bootstrap errors from the pulses (`USE_PULSE_ERRORS`, `ERR_METHOD`), `PARAM_BOUND` at Nyquist,
  28 starts, railing check, parameter errors + condition number, optional `COST="nps"`.
  CSV `test/fit_one_pulse_params.csv` keyed by (channel, wp, model) — `91max` marks the
  max-aligned AP. **The CSV has no `cost` column**, so rows from different costs are
  indistinguishable; changing the header makes `main()` drop every existing row.
- `build_medianAP_maxalign_m205.py` — max-aligned median APs + `align_on_max`, reused elsewhere.
- `compare_argonauts_vs_octopus_m205.py`, `compare_BI_vs_hfpower_m205.py`,
  `risetime_and_amplitude_.py`, `scan_residuals_m205.py` (its qsub machinery is reused by the
  Bessel scan), `plot_AP_spectra_m205.py` (`compute_psd` reused by `--psd`).
- **Octopus normalization study**: `medianpower` = median over events of `2|X_norm|²/df`;
  the flat-top factor 5.708 = N/Σw² is correct. See the NPS section for what this implies.
- **BI/Wiener**: `analyse_BI_m205.py` (optimum), `_wiener.py` (scalar λ), `_wiener_freq.py`
  (λ(f)); filters in `*/trained_filters/` as the independent half-spectrum.
