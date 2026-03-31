# Paper-Faithful Metric Checklist (Holton 2025)

This document summarizes how to keep analyses in this repository faithful to the original Holton transfer-interference metric definitions, and compares:

- original metric idea from the paper text,
- implementation in the original `transfer-interference` repo,
- implementation in `a1b2_modular`.

It also flags practical mismatches to avoid when building figures.

---

## 1) Accuracy

## Paper definition

- Response error is angular distance between true location and response.
- Accuracy is normalized as:
  - `accuracy = 1 - (error / 180 deg)`  
  - equivalent to `1 - (error / pi)` in radians.
- Winter responses are primary unless noted.

## Original repo implementation (`transfer-interference`)

```python
# transfer-interference/src/analysis/preprocessing.py
df['accuracy'] = 1-(df['resp_error']/np.pi)
```

## a1b2 implementation

```python
# a1b2_modular/a1b2/data/preprocessing.py
df['accuracy'] = 1 - (df['resp_error'] / np.pi)
```

```python
# a1b2_modular/a1b2/models/ffn.py
wrapped_difference = wrap_to_pi(predictions - ground_truth)
normalized_error = np.abs(wrapped_difference) / np.pi
return 1 - normalized_error
```

## Verdict

- **Faithful**. Formula and angular normalization are equivalent to paper definition.

---

## 2) Transfer (Task B switch cost)

## Paper definition

- Change in winter accuracy from:
  - final block of Task A (block 10)
  - to first block of Task B (block 11).
- Interpreted as cost when introducing Task B stimuli/rule.

## Original repo implementation (`transfer-interference`)

```python
# transfer-interference/src/analysis/participant.py
final_A1 = p_data[(p_data['task_section']=='A1') & (p_data['feature_idx']==1)].iloc[-6:]['accuracy'].mean()
initial_B = p_data[(p_data['task_section']=='B') & (p_data['feature_idx']==1)].iloc[:6]['accuracy'].mean()
group_df.loc[group_df['participant']==p, 'transfer_error_diff'] = initial_B - final_A1
```

## a1b2 implementation

```python
# a1b2_modular/a1b2/analysis/participant.py
final_A1 = p_data[(p_data['task_section'] == 'A1') & (p_data['feature_idx'] == 1)].iloc[-6:]['accuracy'].mean()
initial_B = p_data[(p_data['task_section'] == 'B') & (p_data['feature_idx'] == 1)].iloc[:6]['accuracy'].mean()
group_df.loc[group_df['participant'] == p, 'transfer_error_diff'] = initial_B - final_A1
```

```python
# a1b2_modular/a1b2/analysis/transfer_interference.py
A1_accuracy = schedule_data[subj]['accuracy'][0, 1::2].copy()
B_accuracy = schedule_data[subj]['accuracy'][1, 1::2].copy()
final_A1_acc = np.mean(A1_accuracy[-6:])
initial_B_acc = np.mean(B_accuracy[0:6])
error_diff = initial_B_acc - final_A1_acc
```

## Verdict

- **Faithful**. Uses winter-only, last 6 A1 vs first 6 B, same difference direction.

---

## 3) Interference (rule-B use at A2)

## Paper definition

- Interference quantified as probability of using Rule B during A2 retest.
- Derived using a 2-component von Mises mixture with fixed means (`theta_A`, `theta_B`) and free:
  - mixture weight (`pi`)
  - shared concentration (`kappa`).

## Original repo implementation (`transfer-interference`)

```python
# transfer-interference/src/models/vonmises.py
def fit_mixture_model(sample, mu_A, mu_B):
    rule_use, loglik, init_params = iter_fit_mixture_vonmises(sample, mu=[mu_B, mu_A])
    return {
        'A_weight': rule_use[0,1],
        'kappa': rule_use[2,0]
    }
```

```python
# transfer-interference/src/analysis/participant.py
group_df.loc[group_df['participant']==p, 'interference'] = 1 - group_df.loc[
    group_df['participant']==p, 'A_weight_A2'
].values[0].astype(np.float32)
```

## a1b2 implementation

```python
# a1b2_modular/a1b2/models/vonmises.py
def fit_mixture_model(sample, mu_A, mu_B):
    rule_use, loglik, init_params = iter_fit_mixture_vonmises(sample, mu=[mu_B, mu_A])
    return {
        'A_weight': rule_use[0,1],
        'kappa': rule_use[2,0]
    }
```

```python
# a1b2_modular/a1b2/analysis/participant.py
group_df.loc[group_df['participant'] == p, 'interference'] = 1 - group_df.loc[
    group_df['participant'] == p, 'A_weight_A2'
].values[0].astype(np.float32)
```

## Verdict

- **Faithful**. Same operationalization: interference = Rule B weight at A2 (`1 - A_weight_A2`).

---

## 4) Generalization

## Paper definition

- Winter accuracy for test stimulus, averaged over second half of Task A training.

## Original repo implementation (`transfer-interference`)

```python
# transfer-interference/src/analysis/participant.py
group_df.loc[group_df['participant']==p, 'generalisation_acc'] = p_data[
    (p_data['test_trial']==1) &
    (p_data['task_section']=='A1') &
    (p_data['block']>=5)
]['accuracy'].mean()
```

## a1b2 implementation

```python
# a1b2_modular/a1b2/analysis/participant.py
group_df.loc[group_df['participant'] == p, 'generalisation_acc'] = p_data[
    (p_data['test_trial'] == 1) & (p_data['task_section'] == 'A1') & (p_data['block'] >= 5)
]['accuracy'].mean()
```

## Verdict

- **Faithful** to intended paper logic.

---

## 5) Summer accuracy

## Paper definition

- Average summer response accuracy in Task A.

## Original repo implementation (`transfer-interference`)

```python
# transfer-interference/src/analysis/participant.py
group_df.loc[group_df['participant']==p, 'summer_accuracy'] = p_data[p_data['feature_idx']==0]['accuracy'].mean()
```

## a1b2 implementation

```python
# a1b2_modular/a1b2/analysis/participant.py
group_df.loc[group_df['participant'] == p, 'summer_accuracy'] = p_data[p_data['feature_idx'] == 0]['accuracy'].mean()
```

## Verdict

- **Matching between repos**, but both implementations may include summer trials beyond A1 unless `trial_df` is pre-filtered upstream.
- For strict paper-faithful reporting, explicitly filter to Task A when computing this metric.

---

## 6) Stimulus onset accuracy (debrief)

## Paper definition

- Post-task memory for whether stimulus onset occurred in first vs second half.

## Original repo implementation (`transfer-interference`)

```python
# transfer-interference/src/analysis/participant.py
afc_dat = trial_df[trial_df['task_section']=='debrief'].groupby('participant')['correct_afc'].mean().reset_index()
afc_dat['correct_afc'] = 100 * afc_dat['correct_afc']
```

## a1b2 implementation

```python
# a1b2_modular/a1b2/analysis/participant.py
afc_dat = trial_df[trial_df['task_section'] == 'debrief'].groupby('participant')['correct_afc'].mean().reset_index()
afc_dat['correct_afc'] = 100 * afc_dat['correct_afc']
```

## Verdict

- **Faithful** and consistent.

---

## 7) Splitter/lumper classification

## Paper definition

- Compare A-centered vs B-centered fits at retest A2 (same complexity), classify by better log likelihood.

## Original repo implementation (`transfer-interference`)

```python
# transfer-interference/src/models/vonmises.py
def compare_models(sample, mu_A, mu_B):
    A_LL, B_LL = compare_mus(mu_A, mu_B, sample)
    return {'A_LL': A_LL, 'B_LL': B_LL}
```

```python
# transfer-interference/src/analysis/participant.py
group_df['is_lumper'] = (group_df['B_LL_A2'] > group_df['A_LL_A2'])
group_df['group'] = np.where(group_df['is_lumper']==1, 'lumpers', 'splitters')
```

## a1b2 implementation

```python
# a1b2_modular/a1b2/analysis/participant.py
group_df['is_lumper'] = (group_df['B_LL_A2'] > group_df['A_LL_A2'])
group_df['group'] = np.where(group_df['is_lumper'] == 1, 'lumpers', 'splitters')
```

## Verdict

- **Faithful** and consistent.

---

## 8) ANN-specific response construction for von Mises

This is an implementation detail (not in paper prose) but important for reproducibility.

## Original repo implementation (`transfer-interference`)

```python
# transfer-interference/scripts/03_fit_vonmises.py
response_angle = winter_radians - summer_radians
response_angle = basic.wrap_to_pi(response_angle)
responses[section] = np.concatenate([response_angle[i:i+n_stim]
                                     for i in range(0, len(response_angle), n_stim*100)])
```

## a1b2 implementation

```python
# a1b2_modular/scripts/03_fit_vonmises.py
response_angle = winter_radians - summer_radians
response_angle = wrap_to_pi(response_angle)
responses[section] = np.concatenate([response_angle[i:i + n_stim]
                                     for i in range(0, len(response_angle), n_stim * 100)])
```

## Verdict

- Equivalent implementation and sampling logic.

---

## Similarities and differences: `transfer-interference` vs `a1b2`

## Similarities (strong)

- Core metric formulas for accuracy, transfer, interference, generalization are the same.
- Same von Mises fit structure and LL-based splitter/lumper logic.
- Same practical variables used in Figure 4 style analyses:
  - `interference`, `transfer_error_diff`, `generalisation_acc`, `summer_accuracy`, `correct_afc`, `retest_error_diff`.

## Differences (practical, not conceptual)

- `a1b2` adds additional analysis helpers (RNN geometry/PCA/principal angles), but these do not redefine the classic behavioral metrics.
- `a1b2` has a slightly extended data-loading path (RNN extras), which can change what is available, not what transfer/interference means.
- The most likely source of drift is notebook-level filtering/aggregation choices, not core metric functions.

---

## Paper-faithful checklist (use this for notebook QA)

For each figure/table, verify:

1. **Accuracy normalization**
   - Uses angular error normalized by `pi` (equivalent to 180 degrees).

2. **Winter probe focus**
   - Core transfer/generalization analyses use winter responses (`feature_idx == 1`).

3. **Transfer window**
   - Uses `last 6 A1 winter` vs `first 6 B winter`.

4. **Interference metric**
   - If claiming paper interference, use `1 - A_weight_A2` from von Mises fits.
   - Do not substitute `retest_error_diff` without clear relabeling.

5. **Generalization metric**
   - Restrict to `test_trial == 1`, `task_section == 'A1'`, `block >= 5`.

6. **Summer accuracy scope**
   - Restrict to Task A if claiming direct paper metric.

7. **Splitter/lumper assignment**
   - Based on LL comparison at A2 (`B_LL_A2 > A_LL_A2` for lumpers).

8. **Condition handling**
   - Keep condition-wise reporting (`same`, `near`, `far`) and avoid collapsing unless explicitly justified.

9. **Phase reporting**
   - When relevant, distinguish `post_A`, `post_B`, `post_A2`.

10. **Metric naming clarity**
   - Distinguish:
     - `transfer_error_diff` (switch cost),
     - `interference` (von Mises weight),
     - `retest_error_diff` (accuracy-based retest shift).

---

## Recommended figure defaults for paper-faithful comparisons

- Primary interference claims:
  - use von-Mises-based `interference`.
- Primary transfer claims:
  - use `transfer_error_diff`.
- Use `retest_error_diff` only as supplementary/robustness metric.
- Always include participant-count table per condition and metric to expose missingness.

