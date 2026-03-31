# Paper-Faithful Testing Protocol for A1B2 Experiments

This document consolidates the metric-alignment findings and turns them into a concrete test protocol for future runs and figure checks.

It is intended as a strict reference for:
- running analyses consistently,
- validating that reported metrics are paper-faithful,
- catching missing data or metric drift early.

---

## Executive Conclusions

From comparison between:
- paper metric definitions,
- original `transfer-interference` implementation,
- current `a1b2_modular` implementation,

the key conclusion is:

1. **Core metrics are largely aligned** across repos:
   - Accuracy normalization
   - Transfer window (final A1 winter vs initial B winter)
   - Interference from von Mises (`1 - A_weight_A2`)
   - Generalization criterion (`test_trial == 1`, A1, late blocks)
   - Splitter/lumper classification (`B_LL_A2 > A_LL_A2`)

2. **Main risks are operational, not conceptual**:
   - inconsistent filtering in notebooks,
   - mixing proxy metrics with primary paper metrics,
   - missing condition/run coverage hidden by aggregation.

3. **One specific caution**:
   - `summer_accuracy` should be explicitly restricted to Task A in analysis code/notebooks if claiming strict paper equivalence.

---

## Required Testing Grid

Use the following `init_scale` values in all future comparisons:

- `0.001`, `0.01`, `0.1`, `1.0`

Minimum core grid for paper-faithful comparisons:

- `arch`: `single_module_rnn`, `two_module_rnn`
- `routing` (for two-module): `shared`, `task_routed`
- `sparsity`: at least `no_comms` (plus optional `0.5`, `1.0` for robustness)
- `dim_hidden`: `6`, `12`, `25`, `50`
- `conditions`: `same`, `near`, `far`
- `phases`: `post_A`, `post_B`, `post_A2`

If any grid cell is missing, report it explicitly in coverage tables before plotting.

---

## Metric Definitions to Enforce (Paper-Faithful)

## 1) Accuracy
- `accuracy = 1 - resp_error / pi` (radians equivalent of /180 degrees).
- Winter-focused interpretation unless otherwise specified.

## 2) Transfer
- `transfer_error_diff = mean(first 6 B winter) - mean(last 6 A1 winter)`.

## 3) Interference (primary)
- `interference = 1 - A_weight_A2` from von Mises mixture fits.

## 4) Generalization
- Winter test-stimulus accuracy in A1 late learning (`block >= 5`).

## 5) Summer accuracy
- Average summer accuracy in Task A (enforce A1 filter explicitly).

## 6) Stimulus onset accuracy
- Debrief AFC score (`correct_afc`, percentage).

## 7) Splitter/Lumper classification
- `is_lumper = (B_LL_A2 > A_LL_A2)`, else splitter.

---

## Primary vs Secondary Metrics (Do Not Mix)

Use this convention in all notebooks:

- **Primary paper metrics**
  - `transfer_error_diff`
  - `interference` (von Mises based)
  - `generalisation_acc`
  - `summer_accuracy`
  - `correct_afc`

- **Secondary/proxy metrics**
  - `retest_error_diff` (accuracy shift proxy, not a substitute for interference weight)

If `retest_error_diff` is shown, label it as proxy/secondary.

---

## Step-by-Step Validation Workflow

1. **Run coverage audit first**
   - build table: `condition/run_id/path_exists/n_npz`.
   - fail or warn on missing expected cells.

2. **Confirm von Mises fit availability**
   - required for any interference/splitter-lumper claim.
   - check fit CSV presence for every included run.

3. **Assemble one canonical long dataframe**
   - include `participant`, `condition`, `phase`, `feature_idx`, `model_group`,
     `init_scale`, `dim_hidden`, `sparsity`, `metric columns`.

4. **Compute metrics from strict formulas**
   - do not rely on ad hoc notebook shortcuts.

5. **Output participant-count report**
   - per metric x phase x condition x model group x init scale.

6. **Output missing-cell report**
   - expected vs loaded cells.

7. **Plot phase-aware accuracy**
   - show `post_A`, `post_B`, `post_A2` explicitly.

8. **Plot derived deltas**
   - `post_B - post_A` (transfer delta),
   - `post_A2 - post_A` (interference delta proxy).

9. **Run stats after coverage checks**
   - ANOVA + post hoc for condition effects,
   - targeted group contrasts where relevant.

10. **Archive outputs**
    - save tables (coverage, participant counts, stats) and figure artifacts together.

---

## Acceptance Checklist Before Claiming Paper-Faithful Results

Mark each as pass/fail:

- [ ] Grid includes `init_scale = 0.001` plus `0.01`, `0.1`, `1.0`.
- [ ] Coverage table shows all expected cells and flags missing ones.
- [ ] von Mises files exist for all runs included in interference analyses.
- [ ] `interference` uses `1 - A_weight_A2` (not retest proxy).
- [ ] `transfer_error_diff` uses last-6 A1 winter vs first-6 B winter.
- [ ] `summer_accuracy` is explicitly Task A scoped.
- [ ] Phase-separated plots include `post_A`, `post_B`, `post_A2`.
- [ ] Missingness/participant counts shown before inferential plots.
- [ ] Stats run on the same filtered dataset used in figures.

---

## Recommended Next Action

Before running new figures, update experiment condition lists to include `init_scale = 0.001` for all target architectures and compare tables first (coverage + participant counts) before plotting.

