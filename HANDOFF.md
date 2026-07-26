# PileUp — Handoff

Thesis work on the **pile-up Background Index (BI)** for CUPID LMO light detectors.
Active thread now: **improving the BI-estimation filter** on run **m205** (optimum
filter → Wiener with trainable λ → Wiener with λ(f)), comparing the variants, and
building load-curve / filter plots. Earlier threads (m204 pipeline cross-check,
timing figure of merit ρ_t = SNR·β, slide deck) are DONE and summarized at the end.

---

## Environment (IMPORTANT — changed)
- **Project root moved** to `~/Desktop/Tesi_Erasmus/PileUp` (git repo; `.root`/`.bin`
  data gitignored but PRESENT locally in `Processed/`). Old paths `~/Downloads/PileUp`
  and `~/Desktop/PileUp` no longer exist.
- **Run scripts with:** `KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/bin/python3.13 <script.py>`
  - **python3.13** has numpy/pandas/scipy/matplotlib/torch/uproot (crashes on import
    without the `KMP_DUPLICATE_LIB_OK=TRUE` OpenMP workaround). Use it to RUN everything.
  - **python3.14** exists but has ONLY numpy — use it just for `-m py_compile` syntax checks.
  - Both interpreters are **externally-managed (PEP 668)**: `pip install` fails without
    `--break-system-packages`. User does **not** want venvs. **openpyxl is NOT installed** →
    read `.xlsx` with the standard library (zipfile + xml), see load-curve script.
- The user commits work themselves (recent commits: kernel saving, flattop factor).

## Data conventions (m205, verified)
- **10 kHz**, window **10000** samples (1 s). ROOT files `Processed/Processed_*_000205_<ch>.root`.
  Per-WP hists: AP `averagepulse_ap_wp<wp>_medianAP`, NPS
  `averagepowerspectrum_noise_wp<wp>_medianpower` (5001 one-sided → concat to 10000
  two-sided). WP→V_bias via `VBIAS_LIST[wp//2]`, **odd wp only**, 15 WPs: 0.6…40 V.
- **9 channels** total: 31, 34, 37, 40, 41, 71, 83, 91, 94. The **5 "good"** analyzed:
  **31, 34, 71, 83, 91**; the **4 "bad"** (oscillations): 37, 40, 41, 94.
- **ch91 is anomalous** (BI flat/high ~1.6e-4, unresponsive to bias) — flag/isolate it.

---

## The BI estimator & the filter (core of the current work)

Pile-up discriminator = ratio of two band-filtered amplitudes `Y = A1/A2 = μ1/μ2`.
Two band filters **f1, f2** are trained (Adam) to minimize **J = pile-up
misidentification rate** (analytic, Gaussian μ/σ from the AP+NPS). **BI = J·K**.
The filter applied to data is the **product** `g_i(f) = f_i(f) · kernel(f)`, where the
kernel is the optimal filter `H = S*/NPS` (optimum) or the Wiener kernel
`W = S*/(|S|² + λ·NPS)` (Wiener variants). All the math lives in `src/analysis.py`
(`compute_H`, `compute_W_torch`, `optimize_filters`, `optimize_filters_wiener_lambda`,
`optimize_filters_wiener_lambda_freq`, `compute_J`, `compute_mu_sigma`, `compute_vars`).

### Three parallel BI-estimation programs (cluster: 1 qsub job per (channel, WP))
All share the orchestrator/worker structure, concurrency-safe CSV append (flock),
and now all **save f1, f2 and the applied kernel as `.npy`** (independent half,
`kernel_ch{ch}_wp{wp}.npy`; complex) in `<results>/trained_filters/`. Total filter =
`f_i · kernel`. `RESET_CSV=True` clears the CSV **and** the filters dir.

1. **`analyse_BI_m205.py`** — OPTIMUM filter. Kernel = `H_unit`. Output dir
   **`m205_results_octopus/`** (was wrongly `m205_results` — FIXED this session).
   NOTE: its `f1/f2/kernel` .npy don't exist yet (never re-run since filter-saving added).
2. **`analyse_BI_m205_wiener.py`** — Wiener, **scalar trainable λ**. `W = S*/(|S|²+λ·NPS)`,
   λ trained with f1/f2. Output **`m205_results_wiener/`**. `lambda_wiener` in CSV; kernel=W.
   Job prefix `BIW`. **Result: Wiener lowers BI ~7.4% vs optimum on every channel/WP**
   (biggest on ch91 +12.6%).
3. **`analyse_BI_m205_wiener_freq.py`** — Wiener, **λ(f) per-frequency** (function
   `optimize_filters_wiener_lambda_freq`, added to `src/analysis.py`; λ(f) is the
   independent half, mirrored to be Hermitian; optional `lambda_smooth` reg). Output
   **`m205_results_wiener_freq/`**. CSV has λ stats (`lambda_mean/median/min/max`);
   saves `lambda_ch*.npy` (the λ(f) curve) + kernel=W. Job prefix `BIWF`.

### Flattop windowing factor (this session)
The hardcoded NPS factor `5.708` was replaced in **all three** programs by
`flattop_power_factor(N) = N/Σ(flattop(N)²) = 1/mean(w²)` (≈ **5.7077** for N=10000,
i.e. identical to 5.708, diff 0.005%). Cached, scipy imported lazily (orchestrator stays light).

---

## Plotting: `plot_BI_results.py` (refactored this session)
- **Config-driven, NO CLI (argparse removed).** One variable **`SUFFIX`** at the top
  ("" = optimum → `m205_results_octopus`; "_wiener"; "_wiener_freq") derives
  RESULTS_DIR, BI_CSV, FILTERS_DIR, TIMING_CSV (fallback), OUTDIR, and the output name
  suffix. Base "" maps to the `_octopus` folder via `SUFFIX or "_octopus"`.
- **`BAD_CHANNELS` flag**: False → exclude [37,40,41,94] (analyze the 5 good); True →
  exclude [31,34,71,83,91] (analyze the 4 bad) and append `_bad` to all output names.
- **Removed** the BI-min-per-channel bar chart + `BI_summary.csv`, and the 3D scatter.
- **Kept**: `BI_vs_parameters` (3×3), `BI_vs_parameters_vbiascolor` (hue=channel,
  shade=V_bias), `params_vs_Vbias`, `BI_vs_Vbias`, `lambda_vs_Vbias` (scalar Wiener only).
- **Filter plots go in `<results>/filter_plots/`**: `trained_filters_ch*` (f1,f2 vs freq),
  `trained_lambda_ch*` (λ(f), freq variant), and **`total_filters_ch*`** (|g_i|=|f_i·kernel|).
  The total filters read the saved `kernel_ch*.npy`; if absent (current Wiener runs), they
  **reconstruct** W from ROOT (AP+NPS, same normalization/flattop as the worker) + λ. Helpers
  `_load_ap_nps`, `_wiener_kernel_half` live here and are reused by the comparison program.

## Comparison: `compare_wiener_vs_optimum_m205.py` (project root)
- Generic **A (baseline) vs B (evaluated)**. `improvement_% = 100·(BI_A − BI_B)/BI_A`.
- Args: `--csv-a/--csv-b` (aliases `--optimum-csv/--wiener-csv`), `--label-a/--label-b`,
  `--tag`, `--outdir`, `--exclude`. Robust ROOT-detection finds the m205 data folders.
- **Output goes in a dedicated folder `comparisons/<tag>/`** (e.g. `comparisons/OF_vs_WF/`,
  `comparisons/WF_vs_WFfreq/`): improvement plots, per-channel bar, BI-vs-Vbias & BI-vs-SNRβ
  grids (A vs B), improvement CSV.
- **Total-filter comparison** in `comparisons/<tag>/total_filters/`: one image per channel
  for **f1** and one for **f2** — grid per WP, each cell = `|g|` of both models overlaid
  (navy=A, amber=B) + a **ratio B/A sub-panel** below. Reuses the kernel reconstruction from
  `plot_BI_results` (imports it). Skips a model that lacks f1/f2 .npy.
- **Verified working: WF-scalar vs WF-freq** (both have f1/f2). **OF vs WF total-filter
  comparison is skipped** until `analyse_BI_m205.py` is re-run (optimum has no f1/f2 yet).
- (This file was found corrupted/duplicated inside `m204_comparison/`; rewritten clean and
  `git mv`-d back to the project root.)

## Load curves
- **`plot_load_curve_full_m205.py`** — reference-style 4-axis load curve (V_bol blue, AP
  Amplitude red, OF RMS purple, OF SNR green) vs **I_bol = V_bias/R_load (R = 2.069 GΩ)**,
  in a **3×3 grid** (9 channels), manual `add_axes` layout so panels keep ~1.26:1 aspect and
  the offset y-axes fit. Merges **V_bol/I_bol from the Excel** file with AP/RMS/SNR from
  `m205_results_octopus/BI_results_m205.csv`. Output `m205_load_curves/`. Reads `.xlsx` with
  the **standard library** (no openpyxl) — the reader is **inlined** (does NOT import the old
  `plot_Vbol_Ibol_RUN14.py`, which was consolidated in and deleted). Produces ONE 3×3 image
  `load_curve_full_m205.png` (only I_bol = V_bias/R_load; the older two-version/stacked layout
  is gone). **⚠ git caveat:** git HEAD still holds an OLD/BROKEN copy (imports the deleted
  `plot_Vbol_Ibol_RUN14`, "RUN14" naming, two versions). The correct final file is now on disk
  (recovered from Trash: `...23-55-55-041.py`) — **when committing, make sure this working-tree
  version is the one committed**, and do NOT `git checkout` it (that would bring back the broken one).
- Excel: **`20260406_RUN14_load_curves_25mK.xlsx`** — filename says "RUN14" but it is run 205
  (user: don't use "RUN14" anywhere in code/labels/folders). Columns: `Name` = "<ch>-<desc>"
  (channel = int before the dash), `Bias_V`, `V_Bol` (V), `I_Bol` (A), `R_Bol` (Ω),
  `R_Load` (= 2.069e9). 16 bias points (0.6…50 V); BI CSV only to 40 V so merges to 15.

## Slide deck (done earlier this session)
- `PileUp_update _06072926.pptx` on the Desktop (23 slides: added a Wiener section and an
  OF-vs-Wiener comparison section). Built by a `build_deck.py` in a prior session scratchpad
  (regenerates from scratch; formulas are rendered mathtext images, palette navy/amber).
  No LibreOffice → QA via python-pptx text extraction only.

---

## What worked
- Wiener trainable-λ filter → **−7.4% BI** on every channel vs the optimum filter.
- λ(f) generalization (per-frequency λ) implemented and running.
- Saving f1/f2/kernel as `.npy` (independent half) → total filters `g=f·kernel` plottable;
  kernel also reconstructible from ROOT+λ when not saved.
- Config-driven `plot_BI_results` (SUFFIX) + dedicated `comparisons/<tag>/` folders.
- Flattop factor derived from the window (`N/Σw²`) instead of the magic 5.708.

## What didn't work / gotchas
- **OF total filters need the optimum re-run** (its f1/f2 aren't saved yet).
- `m205_results` (bare) never existed — the optimum data lives in `m205_results_octopus`
  (OUTPUT_DIR was fixed accordingly).
- The comparison script had been silently **corrupted (every line duplicated)** by a bad
  merge — if a file looks doubled, rewrite it clean.
- pip is blocked (PEP 668) and the user rejects venvs → **don't try to install packages**;
  work with what python3.13 already has, and parse `.xlsx` with the stdlib.

## Next steps (open)
1. **Re-run `analyse_BI_m205.py`** (optimum) so it emits `f1/f2/kernel` .npy → enables the
   **OF-vs-Wiener total-filter comparison** and OF filter plots. (Beware `RESET_CSV=True`
   wipes `m205_results_octopus/BI_results_m205.csv` + its `trained_filters/`.)
2. **Run the λ(f) analysis end-to-end** on the cluster to populate `m205_results_wiener_freq/`
   fully, then compare **BI**: is λ(f) better than scalar λ? (compare `--tag WF_vs_WFfreq`).
3. `git checkout plot_load_curve_full_m205.py` if the load-curve figure is still wanted.
4. **Algorithm improvement brainstorm** (user wants to improve the estimator "a lot").
   Highest-leverage ideas discussed: (a) replace the analytic Gaussian J with a
   **likelihood-ratio / ROC-based** objective and weight delays by the real pile-up time
   distribution & the Qββ energy; (b) **validate on injected pile-up in real noise** instead
   of trusting the Gaussian approximation; (c) **full deconvolution + peak finding** (push λ
   further) or **λ(f)**; (d) **denoise the AP template** (physical fit) to kill the residual
   HF-noise bias found on m204; (e) add the **χ² goodness-of-fit** pile-up tag; regularize/
   reparametrize f1,f2 (smoothness / small basis) against overfitting.

## User preferences (recurring)
- Compare quantities as **percentage differences**, linear scale (not ratios/log).
- **Plot text in English**; discussion in Italian.
- **β without 2π** (Hz); the 2π only appears in σ_t = 1/(2π·SNR·β).
- Keep programs **essential, readable, config-at-the-top** (few/no CLI flags); wants
  honest caveats and to challenge the supervisor's SNR-centric model with defensible statements.

---
## Earlier threads (DONE — background)
- **m204 pipeline cross-check** (`analyze_BI_octo_vs_argo.py` → `m204_comparison/`): BI
  discrepancy driven by **residual noise in the Octopus AP**, quantified by AP HF-power
  (>500 Hz), Pearson r=+0.88 vs ΔBI. Compare with % differences.
- **Timing FoM ρ_t = SNR·β** (`analyze_timing_SNR_m205.py`, `timing_SNR_m205.csv`): β =
  noise-weighted RMS bandwidth (no window, no 2π). ρ_t straightens the BI–SNR
  non-monotonicity (mean |Spearman| 0.57→0.96). Along the load curve BI is monotonic vs σ
  and risetime but NOT vs SNR (BI-optimal WP at 40 V, past the SNR peak) — "maximize SNR" ≠
  "minimize BI". A multivariate "σ not SNR" claim did NOT hold (collinearity, 5 channels) —
  the defensible statement is the monotonicity one.
