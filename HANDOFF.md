# PileUp — Handoff

Thesis work (Milano-Bicocca, CUPID/CROSS) on **pile-up rejection** for the CUPID LMO light
detectors, run **m205**. Chain: real pulses → **fitted template** → **simulated AP**
(`APsimfit10000led`) → training (optimum filter vs Wiener with a penalty on s) → **Monte-Carlo BI**
with errors → comparison plots → thesis.

**NEXT AGENT: use ponytail mode** (the user asked). Laziest thing that works, config-at-top,
no scaffolding. Discussion in Italian, plot text in English. **The user commits unless they
explicitly ask you to** (they did on 2026-09-22 and 09-23: "committa e pusha").
**Never run training locally** ("prepara tutto e poi lo runno sul server") — EXCEPT short
diagnostic runs on one point, which the user has explicitly asked for and which are fine.
When told "non devi fare niente", only read. The user wants things explained SIMPLY BUT
PRECISELY, with sources, and every claim checked on the Monte Carlo, never on the analytic BI.

---

## ★ STATE ON 2026-09-23 — START HERE (last commit `a06d0bd`, pushed)

**Goal now.** Put a defensible pile-up-rejection result in the thesis. Two threads converged:
(1) is the trainable Wiener filter really better than the optimum filter (OF)? (2) is there a
better discriminant than the paper's ratio of maxima of two band filters (called **Y**)?

**Where it stands.**
- **Wiener vs OF — answered: same filter family, the campaign gain was the learning-rate
  schedule** (READ THIS FIRST §6). Trained Wiener filters rewritten as OF filters give the same J
  on 45/45 points; ablation on ch31 wp7: of the −1.79% campaign gain, −1.38% is the OF's lr
  decaying to 1e-5 vs the Wiener's constant lr; at EQUAL schedule the Wwna is not better
  (+0.43 ± 0.21% decaying, −0.06 ± 0.18% constant, two points). λ growing = the optimiser
  walking back to the OF kernel, driven by the s-penalty; harmless (validation curves).
- **Headroom — measured** (§7): Y is already within **2–5% of the Neyman–Pearson limit of ANY
  estimator** on the waveform. 78% of the BI comes from Δt < 0.2 ms, unresolvable at this SNR.
  More SNR is worth 3–10× any estimator (×√2 SNR → −18…−23% on the limit).
- **A better discriminant exists: the model-averaged likelihood ratio M** (§8,
  `src/pileup_likelihood.py`, derivation with sources in `tesi_note/stimatore_verosimiglianza.pdf`).
  No training. On the CAMPAIGN MC events vs Wwna: **ch31 wp7 −1.96 ± 0.32%, ch91 wp15
  −1.51 ± 0.23%, ch34 wp15 −0.31 ± 0.23% (n.s.)**; ~40% of the theoretical headroom. It gains
  where Y saturates (Δt > 0.4 ms: ch31 0.6–0.8 ms survival 1.9% → 0.3%).
- **`simulate_BI_error_m205.py` runs M as a SET** (`LIKELIHOOD = True`): same events as the
  folders' filters, own folder `m205_results_likelihood_APsimfit10000led_npsclean`, read by
  `compare_templates_m205.py` like any other set (paired ΔBI with common seeds). The program
  was cleaned and commented (function map at the top); outputs verified identical to all digits.

**What worked (this round).** Measuring the headroom BEFORE designing an estimator; one-factor-
at-a-time ablations with the same init and paired MC events; two independent MC samples (choose
on one, measure on the other); reproducing the campaign CSV to all digits before trusting a new
pipeline; making a new algorithm a "set" so the existing comparison tooling just works.

**What didn't work (don't redo).** Free phase in the band filters (§5: MC worse at every stage);
Y + width estimator T (correlated 0.95–0.99: same information); Y OR T; 2D histogram likelihood
of (Y, T) (too noisy); the GLRT (max over Δt) alone (+9% worse than Y on ch31: wrong criterion,
the BI is an AVERAGE — use the mixture, i.e. M).

**Next steps, in order.**
1. **The thesis numbers for M**: on the server, `simulate_BI_error_m205.py` with
   `RESULTS_NAME = m205_results_wiener_APsimfit10000led_npsclean_swna1_hist`,
   `COMPARE = [m205_results_octopus_APsimfit10000led_npsclean_hist]`, `LIKELIHOOD = True`,
   `N_SEEDS = 50` (~4 h per job, 75 jobs). Then `compare_templates_m205.py` with
   SETS = [Wwna, OF, `m205_results_likelihood_APsimfit10000led_npsclean`],
   `MC_CSV = "BI_mc_error_m205_seeds50.csv"`. Needs `git pull` on the server (src/pileup_likelihood.py).
2. **Robustness of M to the template**: so far M used the simulated AP, built FROM the fit that
   generates the events (nearly the truth). Test M with a genuinely different template (Octopus's
   38-pulse medianAP, `root`) before claiming it for real data.
3. **Fair OF-vs-Wwna at campaign scale** (§6): Wwna with `ETA_MIN = 1e-5` (knob exists, folder
   suffix `_eta1e-05`) vs the existing OF `_hist`; or OF with constant lr (needs an `eta_min`
   knob in `optimize_filters`).
4. Literature check on M before calling it new (the LaTeX note says so explicitly); the
   citations in `tesi_note/` were written from memory — verify them.
5. Thesis structure proposed to the user: method → equivalence of the families → where the
   Wiener gain comes from (ablation) → headroom → M → validity of the analytic model →
   negative results. Figures: `ablation_ch31_wp7.png`, `ablation_2x2_schedule_vs_model.png`,
   `tesi_note/fig_M_survival_ch31_wp7.pdf`, `width_estimator_explained.png`.

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

**5. FREE PHASE — DONE 2026-09-21. It enlarges the family for real, and it still LOSES on
the MC.** `phase=` in `optimize_filters_wiener`, `PHASE` in the Wiener training script, both
default False. Measured on ch34 wp15, 40×40 grid, 20 000 events per population, all from the
same random start (`test/check_phase_gain_m205.py`, ~8 min):

| stage | BI analytic | BI **Monte Carlo** | MC/an | ΔBI_mc | s1, s2 | σ_Y an/mc |
|---|---|---|---|---|---|---|
| real, 500 steps | 9.35e-5 | 9.49e-5 | 1.01 | — | 0.053, 0.042 | 1.02 |
| +phase 10 | 1.04e-4 | 1.08e-4 | 1.04 | **+14.3%** | 0.057, 0.031 | 1.02 |
| +phase 25 | 9.74e-5 | 1.05e-4 | 1.07 | **+10.2%** | 0.051, 0.035 | 1.00 |
| +phase 50 | 9.15e-5 | 1.04e-4 | 1.14 | **+9.6%** | 0.049, 0.037 | 0.93 |
| +phase 100 | 6.05e-5 | 1.04e-4 | 1.71 | **+9.1%** | 0.051, 0.045 | 0.42 |
| phase, 500 steps | 3.78e-5 | 2.36e-4 | 6.24 | **+149%** | 0.059, 0.057 | **0.11** |

The analytic BI improves (−60% at 500 steps) and the true BI is worse at **every** stage, from
the first ten steps. So the phase is a genuine enlargement of the family (unlike λ, which is
gauge) but the analytic metric pays for it in fiction.

**The guard that catches it is σ_Y of the SINGLES, analytic vs MC** — new, and better than s.
The s-penalty sees nothing (s stays 0.03–0.06 everywhere), because what breaks is not the
resolution of each band but the σ_Y of the RATIO: the phase buys a small σ_Y as the RESIDUE of a
cancellation between numerator and denominator, exactly the λ-collapse mechanism. The first-order
σ_Y then underestimates the true one by **8.7×** (0.0019 vs 0.0166 measured), and the ratio
σ_Y(an)/σ_Y(mc) on single pulses tracks it perfectly: 1.02 with real f, 0.11 with the phase.
Cheap, one point, no full campaign: **use it as the acceptance test for any new filter family.**

**BUG FOUND AND FIXED on the way — `compute_A` put the cut in the wrong place.** It read
`1 - N_sigma*sigmaY[0,0]`, i.e. it assumed the SINGLES sit at muY = 1. That is exact for real
f ≥ 0 (the normalization `mean(|f·W·S|) = 1` makes `f·W·S` real non-negative, so the peak is
exactly 1) and WRONG for a complex f, where the singles measured 0.833 (MC: 0.836) and the cut
therefore sat ~7σ above them. Now it reads `muY[0,0] - N_sigma*sigmaY[0,0]`. **Every real-f
number is unchanged** (ch34 wp15 re-run after the fix: BI_an 9.348e-5 and BI_mc 9.488e-5,
identical to before), so no campaign moves. The first version of this section quoted a phase
collapse of four orders of magnitude and a +127% MC: that was the misplaced cut, not the filter.

**Not a parametrization artefact.** Modulus and phase are trained JOINTLY (one Adam, one loss,
four parameter vectors): in a 50-step run |f| moves 19% AND the phase 0.18 rad. The polar form
`|f|·exp(ip)` is the same complex family as the cartesian `Re + i·Im` (2 d.o.f. per bin); a
cartesian run was tried and moves FURTHER into the phase (mean |p| 1.57 rad vs 0.18), never
less. The polar one is in `src/analysis.py` because it is one line shorter and `|f| > 0`
everywhere here, so its singularity is never reached.

**CONSEQUENCE: the objective, not the family, is the bottleneck.** Enlarging the filter family
is pointless while the training metric can be gamed — go to idea 2 below (differentiable
surrogate of the MC survival fraction). The phase code stays in, off by default, as the measured
proof; re-test it the day the objective is the MC one, because that day the family may finally
pay.

---

**6. WHERE THE WIENER-vs-OF GAIN COMES FROM — causal analysis, 2026-09-23.**
The effect is real but local: at 500 steps each, same events (`_hist` campaigns, seed 1234,
50 000 events), ΔBI_mc Wwna vs OF = **ch31 −0.60% (15/15)**, ch91 −0.43% (10/15), ch34/71/83 ≈ 0.
Three measurements then settle WHY:
- **Reachability — the family is identical, measured on 45/45 points** (ch31, 34, 91 × 15 WP).
  Each trained Wiener filter, rewritten as `f_OF = f_W · W/H`, is a valid OF band filter (factor
  real to Im/Re ≤ 1.2e-7, f_OF ≥ 0) and gives **the same J to ≤ 2e-4** with the OF code. So OF
  CAN reach every Wiener optimum; the OF actually trained lands +0.7–1.0% (analytic) above it in
  44/45 points. "Wiener reaches a better filter" is FALSE; "the OF does not get there" is true.
- **The OF's J is flat at 500 steps because its optimizer is switched off**: `optimize_filters`
  has cosine annealing to `eta_min = 1e-5`, the Wiener-λ keeps lr 1e-2 constant. Same steps,
  unequal training — the same kind of artefact as the old 300-vs-500. Hidden confounder.
- **Ablation, one factor at a time** (`test/ablation_of_vs_wiener_m205.py`, ch31 wp7, 50×50 grid,
  same init, 30 000 paired MC events, paired bootstrap errors; figure `ablation_ch31_wp7.png`):
  A OF lr→1e-5 (= campaign) → B constant lr **−1.38 ± 0.35%** → C Wiener kernel λ=1 **+2.24 ± 0.37%**
  → D λ trainable **−1.63 ± 0.28%** → E + penalty wna **−0.98 ± 0.24%** (= campaign Wwna). Net A→E
  −1.79 ± 0.38%, of which **−1.38% is the learning-rate schedule**; the Wiener-specific rest
  (E vs B) is −0.41%, within errors. The endpoints reproduce the campaign gap (−1.9% vs −2.0%
  analytic). The Wiener kernel at λ=1 is a WORSE parametrization than the OF (s up, MC/an up);
  trainable λ climbs (6.1, and 29.9 with the penalty) i.e. back towards the OF weighting
  (W → S*/(λ·NPS) ∝ H): **the growth of λ is the optimizer rediscovering the OF kernel.**
- **The λ drift, re-measured on the loss actually minimised (J + penalty)**: only 0–8% of λ's
  log-growth happens after the loss is flat (0.1%). λ does not drift at constant cost; it grows
  while the PENALTY term is still being reduced at almost constant J (penalty = 0.4–2% of the
  loss; highest on ch31). The older statement below ("J plateaus by ~150, λ does NOT settle") was
  measured on J alone and is the signature of the penalty, not a gauge drift.
- **Saving bug FIXED**: `optimize_filters_wiener_lambda` returned the kernel of `exp(log_lambda)`
  AFTER the last optimizer step, i.e. one λ step ahead of the returned f1, f2
  (mean|f·W·S| = 0.996–0.998 instead of 1, J off by 0.1–0.2%). Now it returns the λ and kernel
  of the last iteration; verified mean|f·W·S| = 1.0000001. Existing folders keep the old kernel
  (consistent with their CSV λ, as `simulate_BI_error_m205.py` checks).
**2×2 {OF, Wwna} × {lr decays to 1e-5, constant lr}** (arm F = Wwna with the OF's schedule, via
the new `eta_min=` of `optimize_filters_wiener_lambda`; figure `ablation_2x2_schedule_vs_model.png`;
A, B, E reproduced BIT FOR BIT from the previous run, so the procedure is deterministic):

| contrast | ch31 wp7 | ch34 wp15 (control) |
|---|---|---|
| Wwna − OF, decaying lr (F vs A) | +0.37 ± 0.27% | +0.52 ± 0.33% |
| Wwna − OF, constant lr (E vs B) | −0.41 ± 0.28% | +0.17 ± 0.23% |
| schedule, OF (B vs A) | −1.38 ± 0.35% | −0.13 ± 0.30% |
| schedule, Wwna (E vs F) | −2.15 ± 0.39% | −0.48 ± 0.27% |
| interaction | −0.78 ± 0.36% | −0.35 ± 0.37% |

**At equal schedule the Wwna is not better than the OF**: weighted over the two points +0.43 ±
0.21% (decaying lr, Wwna slightly WORSE) and −0.06 ± 0.18% (constant lr, nothing). The schedule
matters where training has not finished (ch31) and hardly at all where it has (ch34). With the
decaying lr λ stops at 6.2 instead of 29.9 — the decay also stops the λ drift. Also on ch34 the
OF's s is 0.009 against 0.027–0.032 for the Wwna even WITH the penalty: the OF parametrization is
intrinsically lower in noise amplification. Caveat: two points, 50×50 grid, one init seed.
**Consequence for the thesis**: at campaign scale, compare at equal schedule. Two options, both
now possible: Wwna with `ETA_MIN = 1e-5` (new knob in the Wiener program, folder suffix `_eta1e-05`)
against the existing OF `_hist`; or OF with constant lr — not yet run at campaign scale. `optimize_filters` still has `eta_min` hardcoded to 1e-5: for the OF-constant option it needs a knob
(or run the OF through `optimize_filters_wiener` with kernel H, which is provably the same J).

**7. ANOTHER ESTIMATOR? MEASURE THE HEADROOM FIRST — 2026-09-23.** The user asked for a
discriminant beyond the max of the two filtered signals (reasoning: "the max depends on power, a
squared modulus, not on phases"). That premise is wrong — the peak `max_t Re[ifft(f·K·X)]` is a
linear functional WITH phases (a random phase lowers it, measured); only the energy is
phase-blind. But the conclusion is right for another reason: expanding the pile-up around its
time centroid, `p = s + ½ r(1−r)Δt² s'' + ⅙ r(1−r)(2r−1)Δt³ s''' + …` — the first order is a pure
time shift (indistinguishable from a single pulse), the first discriminating term is a SYMMETRIC
broadening, and asymmetry (the only thing a phase could exploit) enters at third order and
vanishes at r = ½.
- **Headroom** (`test/estimator_headroom_m205.py`): for each (Δt, r) of the J grid, the distance of
  the pile-up from the single-pulse manifold (min over amplitude and time) in noise σ = the
  Neyman–Pearson limit for ANY test on the waveform at 90% acceptance. Against the MC BI of the
  Wwna `_hist` filters: **ch34 +1.7%, ch83 +2.6%, ch71 +1.8%, ch91 +4.4%, ch31 ≈ +5%** above the
  limit. The trained band ratio is already within 2–5% of the best possible estimator.
- **Width estimator** — `T = Re<X e^{iωt_OF}, g>/‖g‖`, `g = S(M2/M0 − ω²)`: the direction of s''
  orthogonalized to s (and to s' by symmetry), read at the OF time. Linear, unit variance, no
  training, no ratio. Analytic BI vs the limit: +1.1% (ch34), +1.6% (83), +1.5% (71), +6.1% (91),
  ≈ +13% (31). **MC check** (`test/width_estimator_mc_m205.py`, 30 000 paired events): singles
  T = 0.00 ± 1.00 as designed; ch34 wp15 BI 9.381e-5 vs Wwna 9.368e-5 (**+0.14 ± 0.33%**, MC/an of
  the width estimator 0.994); ch31 wp7 7.798e-5 vs 6.609e-5 (**+18.0 ± 0.8%**). At high SNR it
  EQUALS the trained filters with zero training and an exact analytic model; at low SNR only the
  larger Δt are resolvable, the second-order expansion is not enough, and it loses.
- **Information beats estimators** (`--snr`): the same limit with SNR ×√2 / ×2 vs today's Wwna:
  ch31 wp7 −23% / −37%, ch34 wp15 −18% / −32%, ch91 wp15 −22% / −36%, ch71 wp21 −18% / −31%.
  The best conceivable estimator gains 1–6%; √2 more SNR gains ~20%. Whatever raises the SNR
  (WP choice, noise, combining independent channels) is worth 3–10× any new discriminant.
- Remaining estimator idea with real (small) headroom: the two-pulse likelihood-ratio test
  (GLRT) for the LOW-SNR channels (ch31, ch91: ~4–6% to the limit). Computable per event from the
  OF correlation y(t) alone plus the noise-metric autocorrelation R(Δt) of the template:
  max over (t, Δt) of the two-pulse fit minus the one-pulse fit. Upper bound on its gain: the
  headroom above; in practice less (composite alternative over Δt).

**8. A SECOND DISCRIMINANT THAT WORKS: the model-averaged likelihood ratio M — 2026-09-23.**
Protocol for everything below: two independent MC samples (seed 1111 chooses the combination,
seed 2222 measures it), 30 000 events per population, cut at 90% of the measured singles, paired
bootstrap errors, Y = ratio of maxima with the Wwna `_hist` filters. Points ch31 wp7, ch91 wp15,
ch34 wp15.
- **Y + width T: nothing** (best linear −0.8% on ch31, 0 elsewhere). Y and T are correlated
  0.95–0.99 on the singles' noise: the trained filters ARE the width filter.
- **Where the headroom is** (ch31): 78% of the BI comes from Δt < 0.2 ms, where Y already sits ON
  the limit (87.8% vs 87.6% survival, 68.3% vs 68.5%) — unresolvable at this SNR by any estimator.
  All the headroom is at Δt > 0.2 ms, where Y and T SATURATE: once the pulses separate, the max of
  the high band locks onto the larger pulse alone and Y stops growing with Δt (≈ max(r, 1−r)).
- **Y OR two-pulse GLRT Λ** (free amplitudes, Δ ∈ [0.15, 1] ms): ch31 −1.93 ± 0.19%, ch91 −1.13
  (OR) / −1.64 (linear), ch34 −0.15 / −0.68. Correlation Y–Λ 0.71–0.84: new information.
- **M = log Σ_{Δt,r} π(Δt,r) exp((L(Δt,r) − L1)/2)**, with L(Δt,r) the matched-filter fit of the
  composite template S[(1−r) + r e^{−iωΔt}] (amplitude and time profiled, computed from the event's
  OF correlation y(t) as (1−r)y(t) + r·y(t+Δt)), π = uniform Δt × `pdf_ratio2b(r)`. Neyman–Pearson
  for an alternative AVERAGED over (Δt, r), i.e. exactly the BI's figure of merit. NO training,
  NO tuning: only the template and the NPS. **M alone vs Y: ch31 −2.32 ± 0.42%, ch91 −1.75 ± 0.27%,
  ch34 −0.41 ± 0.31%** (with the simulated-AP template, the one Y is trained on; with the true fit
  template −2.24 / −1.72 / −0.48 — same, so the gain is not from knowing the truth). Y OR M adds
  ~0.2% on ch31 only. Theoretical limit vs Y on the same sample: −5.6 / −5.1 / −2.4% — M captures
  ~40% of it; the limit is per-(Δt, r) clairvoyant and not reachable by one test, so the reachable
  optimum is somewhere in between.
- **Confirmed on the CAMPAIGN MC events themselves**: regenerating the seed-1234, 50 000-event MC
  of `_swna1_hist` reproduces Y's BI_mc of the CSV to all digits (same events), and on those events
  M vs Y = **ch31 −1.96 ± 0.32%, ch91 −1.51 ± 0.23%, ch34 −0.31 ± 0.23%** — consistent with the two
  independent samples above. The fit templates are already peak-normalized (max = 1), as the
  campaign's `template_pulse` assumes.
- Reproduce: `test/likelihood_estimator_m205.py <ch> <wp> [n]` (~15 min at 30 000 events).
- **Where M lives now (2026-09-23)**: `src/pileup_likelihood.py`, class `PileupLikelihood(S, nps, w,
  dt_max)` built once per (channel, WP), `.statistic(pulses)` → M per event. Each block is tagged
  with the equation number of `tesi_note/stimatore_verosimiglianza.pdf` (the LaTeX derivation, with
  sources). Self-check: `python src/pileup_likelihood.py` (y(t) = OF correlation within 1e-6 without
  noise and 3e-3 σ with noise — the only difference is the 1e-6 of weight in the dropped
  frequencies; noiseless single → M = ln(1/81) ≤ 0; noiseless pile-up → M > 0). Same M as the
  validated inline version to 6e-13; ~1.3 ms per event (not 6: that estimate included Y).
- **In the campaign MC, as a SET of its own**: `simulate_BI_error_m205.py`, knob
  **`LIKELIHOOD = False`**. When True, for every seed the SAME events that go through the filters
  of RESULTS_NAME and of every COMPARE folder also go through M (built by `make_likelihood` on
  RESULTS_NAME's TRAINING template), with the cut at 90% of the same seed's singles. M writes into
  its OWN results folder, `likelihood_dir()` = `m205_results_likelihood_<training template of
  RESULTS_NAME>[_npsclean]` (e.g. `m205_results_likelihood_APsimfit10000led_npsclean`): the same MC
  CSV as every folder (same name OUT_NAME, same CSV_FIELDNAMES, filter = "likelihood", one row per
  seed) plus `BI_results_m205_likelihood_<tag>.csv`, one row per point with vbias/SNR/sigma from
  RESULTS_NAME and BI = nan (M has no analytic BI). So `compare_templates_m205.py` reads it as any
  other set (it only had to learn the `m205_results_likelihood` prefix → label "likelihood ratio
  M"), and with common seeds its ΔBI vs OF/Wwna is PAIRED. The training folders' CSVs are untouched.
  Mechanics: `simulate_psd(..., stats=())` runs extra per-event statistics on the same pulses after
  the filters; `run_pair(..., likelihood=None)` passes −M (pile-up in the LOW tail, like Y) so
  `bi_from` computes cut, BI and σ for both; with `likelihood` it returns (folder results, M
  result), without it the old list. `append_row_to_csv` takes `fieldnames`.
  **Typical run**: RESULTS_NAME = Wwna `_swna1_hist`, COMPARE = [OF `_hist`], LIKELIHOOD = True,
  N_SEEDS = 50 → one launch, three sets on the same pulses. Then in compare_templates:
  SETS = [Wwna, OF, `m205_results_likelihood_APsimfit10000led_npsclean`], MC_CSV = the `_seeds50`
  file. Verified end to end on ch31 wp7 (seed 1234, 50 000 events, compare mode): Wwna 6.760484e-5
  and OF 6.850779e-5 identical to the campaign CSVs, M 6.627696e-5 identical to the independent
  calculation; compare_templates plots M as a third set, ratio to Wwna 0.980. 281 s per seed per
  point with two folders + M → ~4 h per job at 50 seeds (walltime 24 h).
  Checks: `python src/pileup_likelihood.py`; `test/check_likelihood_in_mc_m205.py` (~2 min: folder
  PSDs identical bit for bit with and without M, compare mode unchanged, M accepts exactly 90% of
  singles). `src/pileup_likelihood.py` must be on the server (git) for the jobs to import it.
- **Templates used so far**: events always generated from the FIT; Y's filters and M both use the
  simulated AP `APsimfit10000led` (built from the fit, so very close to it). The first M test used
  the fit itself (same result); the theoretical limit and the width estimator T used the fit.
  NOT yet tested: M with a template really different from the truth (e.g. Octopus's 38-pulse
  medianAP, `root`).
- **To make it a thesis number**: one seed pair and three points are not a campaign. Next step:
  add M as a second discriminant in `simulate_BI_error_m205.py` (it needs only S, NPS and the event
  spectrum; ~6 ms per event single-thread → run on the cluster) and compare Y vs M with the 50-seed
  paired MC on all channels and WPs.

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

**1. Free PHASE of the band filters — DONE and NEGATIVE, see READ THIS FIRST §5.** Do not
redo it; the code is in (`PHASE = False` by default) and the MC verdict is that it makes the BI
worse at every training stage. What it DID settle: the phase is a genuine enlargement of the
family (unlike λ, which is gauge), so the blocker is the training metric.

**Other ideas for a REAL gain** (ordered by effort/benefit):
2. **START HERE — train on the right objective.** The analytic J is biased by 1–3% and biased
   DIFFERENTLY per filter, which is why analytic gains evaporate — and with a free phase the
   bias reaches four orders of magnitude, so the metric is what is broken, not the family. A
   differentiable surrogate of the MC survival fraction (soft cut on simulated events, the
   noisy argmax kept in the graph) optimises what is actually measured. Every larger family
   (phase, >2 bands, likelihood ratio) is blocked behind this.
3. More than two bands + a multivariate discriminant (linear/quadratic on 3–4 filtered
   amplitudes) instead of the ratio of two — OF stays a special case, so it cannot do worse.
4. ~~Neyman–Pearson likelihood ratio~~ **DONE 2026-09-23: the estimator M** (READ THIS FIRST §8,
   `src/pileup_likelihood.py`), amplitude and time profiled, averaged over (Δt, r).
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

## Why Wiener beats OF on ch31 — optimisation, not family (2026-09-23)
> **Superseded by READ THIS FIRST §6 (ablation):** the dominant cause is the OF's learning-rate
> schedule (cosine to 1e-5, i.e. switched off at the end — hence its "plateau"), not λ's
> trainability. The hypothesis and the "not yet tested" list below are kept for history.
Equal-step comparison (OF `_hist` vs Wwna `swna1_hist`, both 500 steps, MC seed 1234 × 50 000,
same events, i.e. PAIRED):

| ch | ΔBI analytic | ΔBI MC | W better on MC | MC/an OF | MC/an Wwna |
|---|---|---|---|---|---|
| 31 | −1.02% | **−0.60%** | **15/15** | 1.031 | 1.035 |
| 34 | −0.96% | +0.01% | 7/15 | 1.009 | 1.019 |
| 71 | −0.71% | −0.05% | 8/15 | 1.017 | 1.023 |
| 83 | −0.75% | −0.03% | 9/15 | 1.015 | 1.023 |
| 91 | −0.70% | −0.43% | 10/15 | 1.032 | 1.037 |

- **Exact identity** that organises the whole comparison, with ρ = BI_MC/BI_an:
  `BI_MC^W / BI_MC^OF = (J^W / J^OF) · (ρ^W / ρ^OF)`. First factor = what the OPTIMISER found
  (≈ 0.99 on every channel); second = how much MORE optimistic the analytic model is for Wiener
  (≥ 1, channel dependent). ch31: 0.990 × 1.004 → −0.6% survives. ch34: 0.990 × 1.010 → 0.
- **Reachability, measured** (`test/check_family_reach_m205.py`): mapping the trained Wiener
  filters into OF coordinates (f_OF = f_W·W/H, real positive) and evaluating with the OF code
  gives the same J to **≤ 6e-6 on 15/15 WPs of ch31**. The Wiener solution IS a point of the OF
  family; the OF-trained filters sit 0.4–1.9% higher in J.
- **The OF is not slow, it is stuck**: residual descent over the last 100 steps is 0.01–0.05%
  on every channel for OF (a plateau), while Wwna is still descending on ch31 (0.54%) and ch91
  (0.11%). So the gap is not "OF needs more steps" — at that rate it would not close.
- Consistent with the ch34 fixed-λ runs (READ THIS FIRST §4): λ = 1 fixed is WORSE than OF on the
  MC (+0.5/+0.6%), λ trainable ≈ OF. The working hypothesis: a trainable λ is a ONE-parameter
  global reshaping of the kernel that moves the whole spectrum at once (Adam, lr_λ = 1e-1 = 10×
  the filters'), letting the optimiser leave a basin the bin-by-bin f cannot leave. It is a
  better-conditioned parametrisation of the SAME family, not a larger family.
- **Not yet tested, needed to close the causal chain for the thesis** (one channel, ch31, server):
  (a) Wwna with λ FIXED (`TRAIN_LAMBDA = False`, `LAMBDA_VALUE = 1`, penalty on): if the −0.6%
  disappears, the gain is λ's trainability; (b) OF initialised AT the Wiener λ=1 shape
  (f_init ∝ W/H): separates the initial point from the parametrisation of the updates.
- Limitation: one training per point (random init). swna1 vs swna1_hist (same config, two
  inits) differ by 0.00% in median but up to 2.7% on single points — comparable to the per-WP
  gaps; the 15/15 sign on ch31 is what makes it robust, ch91's 10/15 is not.

## Validation curve during training — λ growing is NOT costing anything (2026-09-21)
The worry: with a trainable λ (ch34 and friends) λ climbs by a lot while J stays nearly flat —
the band filters reabsorb the kernel — so the loss cannot say whether the early, small-λ points
(before the filter tends to the OF) are actually worth more. Answer: **apply the current filters
to simulated events every N steps.** New `validate=` / `val_every=` hook in
`optimize_filters_wiener_lambda` (calls `f(step, f1, f2, W_unit, J, lam)` with everything
detached, return value ignored — the caller records) plus `validate_lambda_curve_m205.py`
(config at the top: channel, WP, `N_TRIALS`, `VAL_EVERY`, `NSIM_VAL`, `GRID`, `S_PENALTY_W`).
The event bank is generated ONCE from the `fit` template while the training uses the simulated
AP, so there is no self-consistency, exactly as in the campaigns; the same events at every
validation make the SHAPE of the curve far more precise than a single point.

**Measured, ch34 wp15, 500 steps, grid 100×100, penalty ("wna", 1.0), 5000 events/population:**

| step | λ | BI analytic | BI MC | MC/an | s1, s2 |
|---|---|---|---|---|---|
| 0 | 1.00 | 2.35e-4 | 2.38e-4 | 1.011 | 0.049, 0.049 |
| 75 | 1.53 | 9.98e-5 | 9.86e-5 | 0.989 | 0.056, 0.029 |
| 150 | 2.49 | 9.53e-5 | 9.50e-5 | 0.997 | 0.043, 0.026 |
| 300 | 4.75 | 9.36e-5 | 9.28e-5 | 0.991 | 0.032, 0.022 |
| 499 | 6.14 | 9.35e-5 | 9.22e-5 | 0.986 | 0.028, 0.021 |

- **The best MC point is the LAST one**: stopping early is worth +0.0%. The hypothesis that the
  small-λ points are better is NOT confirmed here.
- **No overtraining at all**: MC/an stays in 0.986–1.011 for the whole run, with no upward
  drift. The analytic BI is even slightly CONSERVATIVE from step 50 on.
- Everything happens in the first ~75 steps (2.38e-4 → 9.86e-5). From step 150 to 499 the MC
  gains only **−3%** while λ goes 2.5 → 6.1 — the flat gauge valley, seen from the MC side.
- **s1 falls 0.049 → 0.028 as λ rises**: growing λ makes the filter amplify noise LESS, and the
  MC rewards it slightly. It is the opposite of the λ-collapse pathology (which happens at
  λ → 0, without the penalty).
- Cost: bank ~10–20 s once, **~9.4 s per validation point** (5000+5000 events), 21 points ≈
  3.3 min on top of ~20 min of training, i.e. **+17%**. Both scale linearly with `NSIM_VAL`
  and 1/`VAL_EVERY`. A single point carries ~2.3% statistical error (survival fraction 0.27),
  mostly COMMON to the curve.
**Paired run WITHOUT the penalty, same point, same events** (`S_PENALTY = None`):

| | λ at step 499 | s1, s2 | BI_mc final | MC/an range |
|---|---|---|---|---|
| penalty ("wna", 1.0) | 6.14, **still rising** | 0.028, 0.021 | **9.215e-5** | 0.986–1.011 |
| no penalty | **2.01** (peaks 2.16 at ~375, then comes back) | 0.044, 0.033 | 9.285e-5 | 0.989–1.021 |

- **The λ climb IS the penalty.** Without it λ converges — it even turns around and decreases.
  The handoff's older "λ does NOT settle in 70/75 points" was measured WITH the penalty and is
  its signature, not a pathology: the penalty pushes s down, and less noise amplification means
  a more OF-like filter, i.e. a larger λ.
- **The penalty does not cost, it gains ~0.75%** on the MC (9.215e-5 vs 9.285e-5, same events),
  and it is ahead in 5 of the 6 late validation points (steps 250–499) by 0.3–1.0%. CAVEAT: the
  step-to-step scatter within one run is ~0.4%, so with one seed this is ~2σ — suggestive, not
  yet quotable. Confirm on several WPs or seeds before putting it in the thesis.
- **This weakens the "switch to the barrier" idea.** s never exceeds 0.07 in either run, so a
  barrier at s_max = 0.15 would be identically zero and would reproduce the no-penalty curve
  exactly — giving up that 0.75%. The wna's lack of a threshold, which looked like a defect
  (it touches the healthy points too), is mildly BENEFICIAL here.
- No overtraining in either run: MC/an never drifts up.
- Figures: `validation_curve_ch34_wp15_{swna1,nopen}.png` and
  `validation_curve_ch34_wp15_penalty_vs_none.png`, points in the matching `.csv`.
- **Still open — ch91**, the channel where λ collapsed without the penalty (MC/an 1.20, W 12%
  worse than OF). That is the only place where "the penalty is needed" and "the barrier is
  enough" give different answers; ch34 cannot settle it.

## Validation inside the campaign + per-channel grid (2026-09-22)
The diagnostic loop is now part of the campaign, so a normal training run produces its own
validation curves and `compare_templates_m205.py` draws them.

- `src/simulation.py`: **`build_event_bank(S, nps, w, signal_amp, nsim, dt_max, chunk, seed)`**
  — the event bank of `simulate_BI_error_m205.py` extracted into one place (numpy only, no
  dependency on analysis/dataset: the caller wraps it in `NumpyDataset`). `S` and `w` are
  passed, not rebuilt, so compute_H's convention is not duplicated. Same chunked generation
  with ONE Generator, so `seed` identifies the whole bank — and `chunk` still changes the
  events, keep it equal across campaigns.
- `src/analysis.py`: the `validate=` / `val_every=` hook is now in **both**
  `optimize_filters_wiener` (λ fixed — `lam` comes back as `nan`, it is not a parameter there)
  and `optimize_filters_wiener_lambda`. It calls `f(step, f1, f2, W_unit, J, lam)` with
  everything detached and ignores the return: the caller records.
- `analysis_BI_m205_wiener_regolarized.py`: config **`VALIDATE`, `VAL_EVERY = 25`,
  `VAL_NSIM = 2000`** and `make_validator(...)`. The bank is generated ONCE per (channel, WP)
  from the **fit** template (the truth) while training uses `TEMPLATE_SOURCE`; with
  `TEMPLATE_SOURCE = "fit"` the two coincide and the program warns that the curve no longer
  measures overtraining. The BI is `an.compute_BI(..., window_fct=np.ones)` — the same cut as
  the campaign MC, and `np.ones` matters: `compute_BI`'s default is `np.hanning`, which
  `simulate_BI_error_m205.py` does NOT use. Curves go into
  `training_history/hist_ch<ch>_wp<wp>.npz` as `val_step, val_lam, val_BI_an, val_BI_mc,
  val_s1, val_s2`; the file keeps its old keys, so `check_training_convergence_m205.py` is
  unaffected. **Cost +7%** (2000 events, every 25 steps, ~80 s on ~20 min) and +160 MB of RAM
  → with `VALIDATE = True` set **`RAM_GB = 4`**.
- **`VAL_SEED = 9001`, deliberately OUTSIDE the final MC's seeds** (`SEED = 1234`, i.e.
  1234…1283 in `simulate_BI_error_m205.py`). Generator, CHUNK and stream are the same, so with
  seed 1234 the validation bank would be EXACTLY the first `VAL_NSIM` events of the seed-1234
  run of the final MC — verified: the first 1000 events of a 2000-event bank are bit for bit a
  1000-event bank. Choosing anything on the validation curve (where to stop, which penalty) and
  then quoting a BI measured on those same events is leakage: small (the final number is the
  median over 50 seeds) but free to avoid. Apart from the seed and the number of events, the
  validation MC is IDENTICAL to the campaign one: same generator, `DETECTOR_SIGMA = 0`,
  `FOLD_RATIO = False`, CHUNK 500, one Generator per population, paired singles/pile-up
  (`PAIRED_NOISE = True`), dt_max = 8e-4, fit template, no window on the events, cut at the
  10th percentile of the singles, `BI = K(1-rp)`. Statistics: ~3.7% per point at 2000 events
  vs ~0.7% at 50 000.
  **NB: the ch34 and ch91 curves of 2026-09-21/22 were made with seed 1234** (the old default):
  they stay paired among themselves, but they do not compare event by event with newer ones.
- `compare_templates_m205.py`: **`PLOT_VALIDATION`** and `plot_validation()` → one figure per
  channel, one panel per WP, `validation_ch<ch>.png` in the comparison folder: dashed = the
  analytic BI being minimised, solid = the Monte-Carlo BI of the same filters, same units.
  Folders trained without `VALIDATE` simply have no curve. Rows adapt to the number of WPs.
  Read it: a MINIMUM of the MC curve where the analytic keeps falling = training past that step
  is optimising the wrong thing.
- Verified end to end on a short run (ch34, wp15 and wp17, 7 steps, 500 events): npz keys,
  figure and grid all produced; the smoke folder was deleted afterwards.
- Pre-existing, untouched: `compare_templates_m205.py` with a SINGLE set crashes in
  `plot_channels(what="delta")` because `PAIRS` is empty (needs ≥2 sets, as always).

## Phase filters — what the code now does (2026-09-21)
- `src/analysis.py`:
  - `_peak_response(g, jitter_max)`: peak of the filtered template, searched in the same
    ±jitter_max band the estimator uses. New helper.
  - `compute_vars_wiener(..., jitter_max=20)`: with f REAL it is **bit for bit what it was**
    (`sum(f·Re(WS))`); with f COMPLEX the response is `_peak_response`. This is not cosmetic:
    a phase can only LOWER the peak (`|Σ| ≤ Σ|·|`), and the old sum would have paired the
    zero-phase response with the free-phase noise, i.e. a fake gain by construction.
  - `optimize_filters_wiener(..., phase=False)`: `f = |f|·exp(i·p)`, `p` trainable, 0 at DC and
    Nyquist, mirror `cat([f, f[1:-1].flip(0).conj()])` (`.conj()` is a no-op on real tensors, so
    there is ONE code path). `p` starts at 0, so phase training starts exactly at the real
    optimum. Not implemented for `optimize_filters_wiener_lambda` (raises).
- `analysis_BI_m205_wiener_regolarized.py`: `PHASE` config flag, folder suffix `_ph` (the name
  parsers of simulate/compare already treat everything after `_lam<v>` as RUN_TAG, so nothing
  else was needed). Complex f1/f2 save and reload unchanged: `_independent_half` slices and
  `full_spectrum` in `simulate_BI_error_m205.py` already conjugates when the dtype is complex,
  and `get_PSD_interpole` already takes the `.real` of the ifft over ±20 samples.
- `compute_A` in `src/analysis.py`: the acceptance cut is now `muY[0,0] - N_sigma*sigmaY[0,0]`
  (was a hardcoded `1`). Exact identity for real f, correctness requirement for complex f. See
  READ THIS FIRST §5.
- Checks: `test/check_phase_filters_m205.py` (real branch unchanged, hermitian mirror, a pure
  delay is neutral, a random phase is penalised, phase=True starts at the real J) and
  `test/check_phase_gain_m205.py` (the MC verdict above). Both run locally with
  **`OMP_NUM_THREADS=1`** — with more threads torch+scipy segfault on this Mac.
- **NOT COMMITTED** (the user commits): `src/analysis.py`, `src/simulation.py`,
  `compare_templates_m205.py`, `validate_lambda_curve_m205.py`,
  `analysis_BI_m205_wiener_regolarized.py`, `test/check_phase_filters_m205.py`,
  `test/check_phase_gain_m205.py`, `HANDOFF.md`.

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
