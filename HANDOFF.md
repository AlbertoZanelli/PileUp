# PileUp — Handoff

Thesis work on the **pile-up Background Index (BI)** for CUPID LMO light detectors.
Two threads: (1) cross-checking BI between two processing pipelines on run **m204**,
(2) studying BI dependencies on the load-curve run **m205**, and building a
progress-update slide deck.

## Environment (IMPORTANT)
- Project root: `~/Desktop/PileUp` (git repo; data `.root`/`.bin` are gitignored).
  (It used to be in `~/Downloads/PileUp`, which no longer exists.)
- **Run everything with:** `KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/bin/python3.13 <script.py>`
  - The system `/usr/bin/python3` (3.9) lacks pandas/scipy/matplotlib/torch.
  - The homebrew 3.13 has them all but crashes on import without the `KMP_DUPLICATE_LIB_OK=TRUE` OpenMP workaround.
- pptx deck is built with a venv: `<scratchpad>/pptxenv/bin/python` (has python-pptx + Pillow via system-site-packages). The scratchpad path is session-specific; recreate with `python3.13 -m venv --system-site-packages pptxenv && pptxenv/bin/pip install python-pptx` if gone.
- No LibreOffice here → **cannot pixel-render the .pptx** for visual QA. Content QA done via python-pptx text extraction only.

## Data conventions (verified)
- **m204**: 2000 Hz, window 800 samples. Octopus AP = ROOT hist `averagepulse_ap_medianAP`; Argonauts AP = `*.bin_edmean.bin`. Octopus NPS normalization differs from Argonauts (see `analyze_BI_singlerun.py` `load_ap_nps`).
- **m205**: **10 kHz**, window 10000 samples (1 s). Load-curve ROOT files `Processed/Processed_*_000205_<ch>.root`, per-WP hists `averagepulse_ap_wp<wp>_medianAP` and NPS `averagepowerspectrum_noise_wp<wp>_medianpower` (5001 one-sided → concat to 10000 two-sided). WP→V_bias via `VBIAS_LIST[wp//2]`, odd wp only. 15 WPs: 0.6…40 V. 5 channels analyzed: **31, 34, 71, 83, 91** (37/40/41/94 excluded).

---

## Thread 1 — m204: BI Octopus vs Argonauts  (DONE)
**Goal:** understand why the two pipelines give different BI on some channels.
**Result:** the discrepancy is driven by **residual noise in the Octopus average pulse**,
NOT by signal amplitude. Quantified as **AP high-frequency power (>500 Hz)**: its
symmetric %-difference vs ΔBI gives Pearson **r = +0.88** (best of all drivers;
amplitude/σ/risetime give |r| ≤ 0.6). Always compare with **percentage differences**.
- Script: `analyze_BI_octo_vs_argo.py` → outputs in `m204_comparison/`
  (`BI_diff_drivers_m204.png`, `AP_PSD_overlay_all_m204.png`, `AP_overlay_all_m204.png`,
  `AP_noise_overlay_m204.png`, `BI_diff_analysis_m204.csv`). ch41 excluded from that analysis.

## Thread 2 — m205: BI dependencies + timing figure of merit  (DONE, active)
**Per-WP observables** produced by `risetime_and_amplitude_.py` (mode
`ANALYSIS_MODE="LOAD_CURVE"`): risetime, **decaytime** (added), amplitude, **HF-power**
(replaced the old freq-sigma). CSV → `m205_results_octopus/amplitudes_m205.csv`.
(Modes AVERAGE_PULSES_BIN / AVERAGE_PULSES_ROOT also exist; channel regex is configurable via `BIN_CH_REGEX`.)

**BI dependency plots**: `plot_BI_results.py` → `m205_results_octopus/`
  - `BI_vs_parameters_m205.png` (channel-colored) and `BI_vs_parameters_vbiascolor_m205.png`
    (hue = channel, shade = V_bias). 3×3 grid, **fixed order**: row1 V_bias/risetime/decaytime,
    row2 SNR/amplitude/σ, row3 SNR÷risetime / **SNR·β** / HF-power.
  - `params_vs_Vbias_m205.png`, `BI_vs_Vbias_m205.png`, `BI_min_per_channel_m205.png`, `BI_summary_m205.csv`.
  - Lines are **ordered by V_bias** (not by x) so folded variables don't draw spurious spikes.

**Key physics finding (defensible line for the supervisor):**
- Along the load curve BI is **monotonic vs σ and risetime** (smaller → lower BI) but
  **NOT vs SNR** (SNR peaks at intermediate bias; the BI-optimal WP is at 40 V, past the
  SNR peak, for all channels). So "modeling BI on SNR + risetime alone is not adequate";
  "maximize SNR" ≠ "minimize BI". (Supervisor deck `~/Downloads/LDRequirementsAndSensitivity.pdf`
  argues +33% SNR helps pile-up — this is the point being nuanced.)

**Timing figure of merit (the good result):** amplitude SNR = A/σ_OF ignores WHERE noise
sits. Pile-up needs the fast edges → weight by frequency (Cramér-Rao arrival-time bound).
- `analyze_timing_SNR_m205.py`: `sigma_and_bandwidth()` uses the pipeline's
  `src.analysis.compute_sigma_OF`, **NO window** on the average pulse.
  σ_OF = compute_sigma_OF(S,nps); σ_mod = compute_sigma_OF(f·S,nps);
  **β = σ_OF/σ_mod = √(Σf²|S|²/N ÷ Σ|S|²/N) in Hz (NO 2π** — user's explicit choice;
  the 2π cancels and is kept only in σ_t = 1/(2π·SNR·β)). β takes SNR/BI from the CSV.
  → `timing_SNR_m205.csv` (cols incl. `beta_Hz`, `rho_t=SNR·β`, `sigma_t_ms`, `sigma_OF_nowin`, `sigma_mod_nowin`) and `BI_vs_timing_SNR_m205.png`.
- **ρ_t = SNR·β straightens the BI–SNR non-monotonicity**: mean |Spearman(BI,·)|
  **0.57 (SNR) → 0.96 (ρ_t)**; folded channels 34/83/91 fixed.
- Weight visualization: `plot_timing_weight_m205.py` (5 panels, one WP each) and
  `plot_timing_weight_allWP_m205.py` (one grid per channel, all WPs). Blue = amplitude
  weight |S|²/N (low-f); red = timing weight (2πf)²|S|²/N (peaks ~1/risetime). **No window.**
- `plot_AP_spectra_m205.py`: AP power spectra per channel, all WPs, colored by V_bias.
- `diagnose_timing_channels_m205.py`: shows the residual ρ_t non-monotonicity enters at
  the high-bias tail (SNR falls faster than β rises) for ch34/83; **ch91 is anomalous**
  (BI flat/high ~1.6e-4, unresponsive to bias — treat separately).

## Slide deck  (11 slides, active)
- Built by `<scratchpad>/build_deck.py` → `~/Desktop/PileUp/PileUp_m204_update.pptx`
  (name is legacy; now covers m204 **and** m205).
- S1 title, S2 BI pipeline, S3 AP/NPS comparison, S4–6 m204 result/diagnosis/HF-power,
  **S7** m205 intro, **S8** BI vs V_bias, **S9** BI-vs-parameters table (the vbiascolor grid),
  **S10** how SNR is computed, **S11** SNR·β figure of merit.
- Formulas on S10/S11 are **rendered images** (matplotlib mathtext, `_formula_cache/`),
  not Unicode. Palette navy/amber. `build_deck.py` regenerates the whole deck from scratch.
- Google Slides: import the .pptx (no native Google Slides API here).

## What worked
- Percentage-difference driver ranking (m204); AP HF-power metric.
- V_bias-ordered lines + per-channel hue / V_bias shade for folded load-curve plots.
- Cramér-Rao timing FoM ρ_t = SNR·β to explain the SNR non-monotonicity.
- Rendered formula images instead of Unicode in the deck.

## What didn't work / dead ends
- Claiming "BI depends on σ not SNR": a multivariate/partial-correlation test did NOT
  support it (collinearity; 5 channels). That deep-dive was **deleted** at the user's
  request. The defensible statement is the monotonicity one above.
- Using a window (Hann) on the average pulse for σ_OF/β — user wants **no window** (the
  template is ~zero at edges). Weight plots also de-windowed.
- Putting the 2π inside β (rad/s) — user reverted; β stays in Hz (2π only in σ_t).

## Next steps (open)
1. **Visual QA of the deck**: no LibreOffice here → eyeball S7/S10/S11 fit in PowerPoint/Google
   Slides, or install LibreOffice for a rendered pass. Fix any text overflow.
2. Optional: rename deck `PileUp_m204_update.pptx` → `PileUp_update.pptx`.
3. Optional: native editable PowerPoint equations instead of formula images (harder; can't verify here).
4. Decide how to present **ch91** (anomalous detector) — likely flag/isolate it.
5. `analyse_BI_m205.py` reads amplitudes from `Processed/amplitudes_m205.csv`; the new
   enriched CSV is in `m205_results_octopus/`. Update that path if re-running the m205 BI estimation.

## User preferences (recurring)
- Compare quantities as **percentage differences**, linear scale, not ratios/log.
- Plot text in **English**.
- Formulas: rendered/proper, not raw Unicode; **β without 2π** (Hz).
- Wants honest caveats and to challenge the supervisor's SNR-centric model with defensible statements.
