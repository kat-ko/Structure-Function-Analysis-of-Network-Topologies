# Notebook Data Flow Analysis

## Overview
After running the preprocessing scripts (`01_preprocess_data.py`, `02_run_simulations.py`, `03_fit_vonmises.py`), the notebooks perform statistical analyses and create publication figures. This document explains the complete data pipeline, transformations, and analyses performed in each notebook.

---

## Notebook 1: `figure2_transfer_interference.ipynb`

### Purpose
Compare transfer and interference effects between humans and ANNs across conditions (same, near, far). This notebook produces the main comparison figures showing how both humans and ANNs exhibit similar patterns of transfer and interference.

### Data Inputs
1. **Human trial data**: `data/participants/trial_df.csv`
   - Columns: participant, condition, task_section, accuracy, feature_idx, block, test_trial, etc.
   - Trial-level data with one row per trial
   
2. **ANN simulation data**: `data/simulations/rich_50/`
   - Contains `.npz` files: `sim_{participant}.npz`
   - Each file contains: predictions, labels, accuracy, losses, test_stim, hiddens_post_phase_0, hiddens_post_phase_1
   
3. **Von Mises fit parameters**:
   - `data/participants/human_vonmises_fits.csv`
   - `data/simulations/rich_50_vonmises_fits.csv`
   - Columns: participant, condition, A_weight_A1, A_weight_B, A_weight_A2, kappa_A1, kappa_B, kappa_A2, A_LL_B, B_LL_B, A_LL_A2, B_LL_A2

### Data Flow

#### Step 1: Load Raw Data
```python
participant_data = pd.read_csv(project_root / 'data/participants/trial_df.csv')
ann_data = ann.load_ann_data(project_root / 'data/simulations/rich_50')
```

**Data structures created:**
- **participant_data**: DataFrame with shape (n_trials, n_columns)
  - Key columns: participant, condition, task_section, accuracy, feature_idx, block
  - Each row = one trial for one participant
  
- **ann_data**: Dictionary with structure:
  ```python
  {
      'same': [participant_dict_1, participant_dict_2, ...],
      'near': [participant_dict_1, participant_dict_2, ...],
      'far': [participant_dict_1, participant_dict_2, ...]
  }
  ```
  Each participant_dict contains:
  - `predictions`: shape (3, n_trials, 4) - [cos_feat1, sin_feat1, cos_feat2, sin_feat2]
  - `accuracy`: shape (3, n_trials) - accuracy per phase per trial
  - `losses`: shape (3, n_trials) - MSE loss per phase per trial
  - `hiddens_post_phase_0`: shape (12, dim_hidden) - hidden states after A1
  - `hiddens_post_phase_1`: shape (12, dim_hidden) - hidden states after B

#### Step 2: Transfer Analysis

**For Humans:**
```python
human_transfer = participant.compute_transfer_humans(participant_data)
```

**Detailed transformation:**
1. **Extract A1 final performance:**
   - Filters: `task_section == 'A1'` AND `feature_idx == 1` (winter feature)
   - Groups by: participant, condition, study
   - Takes final 6 trials: `x.values[-6:]`
   - Computes: `accuracy_A1 = mean(final_6_trials)`

2. **Extract B initial performance:**
   - Filters: `task_section == 'B'` AND `feature_idx == 1` (winter feature)
   - Groups by: participant, condition, study
   - Takes first 6 trials: `x.values[0:6]`
   - Computes: `accuracy_B = mean(first_6_trials)`

3. **Calculate transfer metric:**
   - `error_diff = accuracy_B - accuracy_A1`
   - Positive values = transfer cost (worse performance on B)
   - Negative values = transfer benefit (better performance on B)

4. **Output DataFrame:**
   - Columns: participant, condition, study, accuracy_A1, accuracy_B, error_diff
   - One row per participant per condition

**For ANNs:**
```python
ann_transfer = ann.compute_transfer_anns(ann_data)
```

**Detailed transformation:**
1. **For each condition** (same/near/far) and **each participant:**
   - Extracts accuracy array: `accuracy[phase, trial]`
     - Phase 0 = A1, Phase 1 = B, Phase 2 = A2
   - Takes winter responses only: `accuracy[0, 1::2]` (odd indices)
     - Even indices (0, 2, 4...) = summer feature
     - Odd indices (1, 3, 5...) = winter feature

2. **Compute metrics:**
   - `final_A1_acc = mean(A1_accuracy[-6:])` - final 6 trials of A1
   - `initial_B_acc = mean(B_accuracy[0:6])` - first 6 trials of B
   - `error_diff = initial_B_acc - final_A1_acc`

3. **Output DataFrame:**
   - Columns: participant, condition, error_diff
   - One row per participant per condition

**Statistical Analysis:**
```python
human_stats = stats.compare_transfer(human_transfer, metric_col='error_diff')
ann_stats = stats.compare_transfer(ann_transfer, metric_col='error_diff')
```

**What happens:**
1. **Split by condition:**
   - `same = df[df['condition'] == 'same'][metric_col]`
   - `near = df[df['condition'] == 'near'][metric_col]`
   - `far = df[df['condition'] == 'far'][metric_col]`

2. **Calculate descriptives:**
   - Mean and standard error for each condition
   - Uses `scipy.stats.sem()` for standard error

3. **One-way ANOVA:**
   - `f_stat, p_value = stats.f_oneway(same, near, far)`
   - Tests if there are any differences across conditions
   - Computes eta-squared effect size

4. **Post-hoc t-tests:**
   - Same vs Near: `stats.ttest_ind(same, near, alternative='greater')`
   - Same vs Far: `stats.ttest_ind(same, far, alternative='greater')`
   - Near vs Far: `stats.ttest_ind(near, far, alternative='greater')`
   - All use one-tailed tests (expecting same > near > far)

5. **Effect sizes:**
   - Cohen's d for each comparison:
     ```python
     pooled_sd = sqrt(((n1-1)*var(x1) + (n2-1)*var(x2)) / (n1+n2-2))
     cohens_d = (mean(x1) - mean(x2)) / pooled_sd
     ```

6. **Output dictionary:**
   ```python
   {
       'descriptives': {
           'same': {'mean': ..., 'sem': ...},
           'near': {'mean': ..., 'sem': ...},
           'far': {'mean': ..., 'sem': ...}
       },
       'anova': {
           'f_stat': ..., 'p_value': ..., 'eta_squared': ...
       },
       'posthoc': {
           'same_vs_near': {'t_stat': ..., 'p_value': ..., 'cohens_d': ...},
           'same_vs_far': {'t_stat': ..., 'p_value': ..., 'cohens_d': ...},
           'near_vs_far': {'t_stat': ..., 'p_value': ..., 'cohens_d': ...}
       }
   }
   ```

**Visualization:**
```python
fig, ax = figure_utils.plot_transfer(
    human_transfer, 'error_diff', condition_order, 
    ylabel='Task B switch cost\n(Δ accuracy)', 
    xlim=(-0.5, 2.5), ylim=(-1.1, 0.5), 
    schedule_colours=schedule_colours, 
    p_values=np.nan, addtests=0
)
```

**Plot components:**
1. **Stripplot** (bottom layer): Individual data points with jitter
2. **Pointplot with error bars** (middle layer): Mean ± SE
3. **Pointplot markers** (top layer): Mean points with white fill
4. **Significance bars** (if addtests=1): Lines and stars for p < 0.05

**Output files:**
- `fig2A_human_transfer_all.png` - All human participants
- `supp_human_transfer_study1.png` - Study 1 only
- `supp_human_transfer_study2.png` - Study 2 only
- `fig2C_ann_transfer.png` - ANN results

#### Step 3: Interference Analysis

**Load Von Mises Parameters:**
```python
human_vonmises_params = pd.read_csv(project_root / 'data/participants/human_vonmises_fits.csv')
ann_vonmises_params = pd.read_csv(project_root / 'data/simulations/rich_50_vonmises_fits.csv')
```

**Data structure:**
- **Columns**: participant, condition, study (humans only)
- **Mixture model parameters**:
  - `A_weight_A1`: Weight of Rule A component in A1 phase (0-1)
  - `A_weight_B`: Weight of Rule A component in B phase
  - `A_weight_A2`: Weight of Rule A component in A2 phase
  - `kappa_A1`, `kappa_B`, `kappa_A2`: Concentration parameters
- **Model comparison**:
  - `A_LL_B`, `B_LL_B`: Log-likelihoods for pure A vs pure B models in B phase
  - `A_LL_A2`, `B_LL_A2`: Log-likelihoods for pure A vs pure B models in A2 phase

**Filtering:**
```python
human_vonmises_filtered = human_vonmises_params.loc[
    human_vonmises_params['B_LL_B'] > human_vonmises_params['A_LL_B']
]
```
- **Purpose**: Only include participants who learned Rule B
- **Logic**: If B_LL_B > A_LL_B, participant used Rule B more than Rule A in B phase
- **Effect**: Removes participants who didn't learn the task

**Statistical Analysis:**
```python
ann_results = stats.compare_interference(ann_vonmises_params)
human_results = stats.compare_interference(study_params)
```

**What happens:**
1. **Extract condition data:**
   ```python
   near_A1 = study_params.loc[study_params['condition']=='near', 'A_weight_A1'].values
   far_A1 = study_params.loc[study_params['condition']=='far', 'A_weight_A1'].values
   near_A2 = study_params.loc[study_params['condition']=='near', 'A_weight_A2'].values
   far_A2 = study_params.loc[study_params['condition']=='far', 'A_weight_A2'].values
   ```

2. **Compute interference differences:**
   ```python
   near_diff = near_A1 - near_A2  # How much A_weight decreased
   far_diff = far_A1 - far_A2
   ```
   - Large difference = high interference (forgot Rule A)
   - Small difference = low interference (remembered Rule A)

3. **Three statistical tests:**

   **a. A1 weights comparison:**
   - Tests: `near_A1 vs far_A1`
   - Hypothesis: Do conditions differ in initial learning?
   - Test: `stats.ttest_ind(near_A1, far_A1, alternative='less')`
   - Expectation: near_A1 < far_A1 (near condition learns less initially)

   **b. A2 weights comparison:**
   - Tests: `near_A2 vs far_A2`
   - Hypothesis: Do conditions differ in interference?
   - Test: `stats.ttest_ind(near_A2, far_A2, alternative='less')`
   - Expectation: near_A2 < far_A2 (near condition shows more interference)

   **c. A1-A2 difference comparison:**
   - Tests: `near_diff vs far_diff`
   - Hypothesis: Do conditions differ in interference magnitude?
   - Test: `stats.ttest_ind(near_diff, far_diff, alternative='greater')`
   - Expectation: near_diff > far_diff (near condition shows larger interference)

4. **Effect sizes:**
   - Cohen's d computed for each comparison using pooled standard deviation

5. **Output dictionary:**
   ```python
   {
       'A1_weights': {
           't_stat': ..., 'df': ..., 'cohens_d': ..., 'p_value': ...,
           'means': {'near': ..., 'far': ...},
           'standard_errors': {'near': ..., 'far': ...}
       },
       'A2_weights': {...},
       'A1_A2_difference': {...}
   }
   ```

**Visualization:**
```python
fig, ax = figure_utils.plot_interference(
    human_vonmises_filtered, 
    "A_weight_A2", 
    schedule_colours, 
    ylabel='Retest interference\n$p$(Rule B)'
)
```

**What happens:**
1. **Transform data:**
   - Computes: `A2_interference = 1 - A_weight_A2`
   - This is the probability of using Rule B at A2 (interference measure)
   - Filters to near and far only (excludes 'same')

2. **Create plot:**
   - Stripplot: Individual data points
   - Pointplot: Mean ± SE with error bars
   - Markers: Mean points
   - Y-axis: 0 to 1 (probability scale)

3. **Output files:**
   - `fig2B_human_interference.png` - All human participants
   - `supp_interference_study1.png` - Study 1 only
   - `supp_interference_study2.png` - Study 2 only
   - `fig2D_ann_interference.png` - ANN results

---

## Notebook 2: `figure3_anns.ipynb`

### Purpose
Analyze representational properties of ANNs: loss curves during training, PCA dimensionality of hidden representations, geometric structure of representations, and principal angles between task subspaces.

### Data Inputs
1. **ANN simulation data**: `data/simulations/rich_50/`
   - Same structure as Notebook 1
   - Contains: losses, predictions, hiddens_post_phase_0, hiddens_post_phase_1

2. **Participant data** (for reference): `data/participants/trial_df.csv`
   - Used to understand data structure, not directly analyzed

### Data Flow

#### Step 1: Load Data
```python
participant_data = pd.read_csv(project_root / 'data/participants/trial_df.csv')
ann_data = ann.load_ann_data(project_root / 'data/simulations/rich_50')
```

#### Step 2: Loss Curve Analysis

**What happens:**
```python
fig, ax = figure_utils.plot_loss_curves(ann_data, schedule, schedule_colours)
```

**Detailed transformation:**
1. **For each schedule** (same/near/far):
   - Extracts losses: `losses[phase, trial]` shape (3, n_trials)
     - Phase 0 = A1, Phase 1 = B, Phase 2 = A2
   - Flattens across phases: `np.concatenate(losses, axis=0)`
     - Creates continuous time series: A1 → B → A2
   - Takes summer responses only: `losses[1::2]` (even indices)
     - Even indices = summer feature responses
   - Repeats for all participants in condition

2. **Aggregate across participants:**
   - Creates matrix: `sched_losses[participant, trial]`
   - Computes: `mean_losses = nanmean(sched_losses, axis=0)`
   - Computes: `std_losses = nanstd(sched_losses, axis=0)`

3. **Visualization:**
   - Plots mean loss over time
   - Adds vertical lines at phase transitions
   - X-axis labels: 'A', 'B', 'A'
   - Y-axis: Loss (MSE) from 0 to 0.5

4. **Output files:**
   - `fig3B_loss_curves_same.png`
   - `fig3C_loss_curves_near.png`
   - `fig3D_loss_curves_far.png`

**Interpretation:**
- Shows how loss decreases during training
- Phase transitions show interference (loss increases when switching tasks)
- Near condition should show larger interference than far

#### Step 3: PCA Dimensionality Analysis

**What happens:**
```python
agg_df_long = ann.compute_pca_components(ann_data, variance_threshold=0.99)
```

**Detailed transformation:**
1. **For each participant and condition:**
   - Extracts hidden representations:
     - `A_hids = hiddens_post_phase_0` - shape (12, dim_hidden)
       - 12 stimuli (6 A + 6 B), dim_hidden features
     - `B_hids = hiddens_post_phase_1` - shape (12, dim_hidden)
       - Same structure after B training

2. **Fit PCA:**
   ```python
   pca_A = PCA().fit(A_hids)  # Fit to all components
   pca_B = PCA().fit(B_hids)
   ```

3. **Find dimensionality:**
   ```python
   cumsum_A = np.cumsum(pca_A.explained_variance_ratio_)
   n_components_A = np.argmax(cumsum_A >= 0.99) + 1
   ```
   - Finds minimum number of components to explain 99% variance
   - Lower n_components = more compressed representation

4. **Store results:**
   - Creates DataFrame with columns: participant, condition, task ('post A' or 'post B'), n_pca

5. **Visualization:**
   ```python
   fig, ax = figure_utils.plot_pca_components(agg_df_long, task_colours)
   ```
   - Bar plot comparing n_pca across conditions and tasks
   - Shows if representations become more/less compressed after B training

6. **Output file:**
   - `fig3E_pca_rich_50.png`

**Interpretation:**
- Measures representational dimensionality
- Lower n_pca = more efficient representation
- Changes from post-A to post-B show how B training affects representation

#### Step 4: Geometric Structure Analysis

**What happens:**
1. **Extract hidden representations:**
   - Gets `hiddens_post_phase_1` for all 12 stimuli
   - Already ordered: first 6 = A stimuli, last 6 = B stimuli

2. **Fit 2D PCA:**
   ```python
   pca = PCA(n_components=2)
   hiddens_pca = pca.fit_transform(hiddens)
   ```

3. **Project stimuli:**
   - A_stim_hiddens = hiddens_pca[:6] - A stimuli in 2D
   - B_stim_hiddens = hiddens_pca[6:] - B stimuli in 2D

4. **Visualization:**
   ```python
   plot_2d_pca(ax, A_stim_hiddens, color_A, 'Task A Stimuli')
   plot_2d_pca(ax, B_stim_hiddens, color_B, 'Task B Stimuli')
   ```
   - Plots points connected in order (forms a shape)
   - Connects last point to first (closes the shape)
   - Shows geometric organization of representations

5. **Output files:**
   - `fig3_F_geom_rich_50.png` - Same condition
   - `fig3_G_geom_rich_50.png` - Near condition
   - `fig3_H_geom_rich_50.png` - Far condition

**Interpretation:**
- Shows how stimuli are organized in representational space
- Circular/ordered structure = systematic representation
- Overlap between A and B = interference

#### Step 5: Principal Angles Analysis

**What happens:**
```python
pa_df = ann.get_principal_angles(ann_data)
```

**Detailed transformation:**
1. **For each participant and condition:**
   - Extracts hidden representations from `hiddens_post_phase_1`
   - Splits: 
     - `A_hids = hiddens[0:6, :]` - A stimuli representations
     - `B_hids = hiddens[6:, :]` - B stimuli representations

2. **Compute principal angles:**
   ```python
   # Step 1: Fit 2D PCA to each
   pca_A = PCA(n_components=2)
   pca_B = PCA(n_components=2)
   V_A = pca_A.fit_transform(A_hids)  # Principal components
   V_B = pca_B.fit_transform(B_hids)
   
   # Step 2: Compute inner product matrix
   inner_product = np.dot(pca_A.components_, pca_B.components_.T)
   
   # Step 3: SVD
   _, singular_values, _ = np.linalg.svd(inner_product)
   
   # Step 4: Principal angles
   principal_angles = np.arccos(np.clip(singular_values, -1.0, 1.0))
   principal_angles_degrees = np.degrees(principal_angles)
   ```

3. **Interpretation:**
   - Principal angles measure alignment between A and B subspaces
   - Small angle (close to 0°) = subspaces are aligned (interference)
   - Large angle (close to 90°) = subspaces are orthogonal (no interference)

4. **Store results:**
   - DataFrame: participant, condition, principal_angle_between

5. **Visualization:**
   - Compares principal angles across conditions
   - Expectation: near < far (near condition has more aligned subspaces)

6. **Output file:**
   - `fig3I_principal_angle_rich_50.png`

---

## Notebook 3: `figure4_individual_differences.ipynb`

### Purpose
Analyze individual differences in learning strategies. Compares human "splitters" (use Rule A at A2) vs "lumpers" (use Rule B at A2), and ANN "rich" vs "lazy" networks to see if similar individual differences emerge.

### Data Inputs
1. **Human data**: 
   - `data/participants/trial_df.csv` - Trial-level data
   - `data/participants/human_vonmises_fits.csv` - Von Mises parameters

2. **ANN data**:
   - `data/simulations/rich_50/` - Rich regime simulations
   - `data/simulations/lazy_50/` - Lazy regime simulations
   - `data/simulations/rich_50_vonmises_fits.csv` - Rich von Mises fits
   - `data/simulations/lazy_50_vonmises_fits.csv` - Lazy von Mises fits

### Data Flow

#### Step 1: Load and Filter Data

**For Humans:**
```python
participant_trial_df = pd.read_csv(project_root / 'data/participants/trial_df.csv')
participant_group_df = pd.read_csv(project_root / 'data/participants/human_vonmises_fits.csv')
```

**Filtering:**
```python
# Filter to near condition only
near_participants_trial = participant_trial_df[
    participant_trial_df['condition'] == 'near'
].copy()

# Filter by log-likelihood (only participants who learned Rule B)
near_participants_group = participant_group_df.loc[
    (participant_group_df['B_LL_B'] > participant_group_df['A_LL_B']) &
    (participant_group_df['condition'] == 'near')
].copy()
```

**For ANNs:**
```python
rich_trial_data = ann.load_ann_data(project_root / 'data/simulations/rich_50')
lazy_trial_data = ann.load_ann_data(project_root / 'data/simulations/lazy_50')
rich_group_params = pd.read_csv(project_root / 'data/simulations/rich_50_vonmises_fits.csv')
lazy_group_params = pd.read_csv(project_root / 'data/simulations/lazy_50_vonmises_fits.csv')
```

**Filtering:**
- Only uses 'near' condition data
- Matches participant IDs between trial data and von Mises fits

#### Step 2: Create Histogram

**What happens:**
```python
fig, ax = figure_utils.plot_near_hist(near_participants_group, schedule_colours)
```

**Transformation:**
1. Computes interference: `1 - A_weight_A2`
   - This is the probability of using Rule B at A2
   - Range: 0 (no interference) to 1 (complete interference)

2. Creates histogram:
   - Bins interference values
   - Shows distribution of individual differences
   - Y-axis: Count of participants
   - X-axis: Retest interference p(Rule B)

3. **Output file:**
   - `fig4A_near_hist.png`

**Interpretation:**
- Bimodal distribution = two groups (splitters vs lumpers)
- Unimodal distribution = continuous variation

#### Step 3: Compute Behavioral Metrics

**For Humans:**
```python
near_participants_group = participant.add_behav_metrics(
    near_participants_group, 
    near_participants_trial
)
```

**Detailed transformation:**

1. **Classify participants:**
   ```python
   is_lumper = (group_df['B_LL_A2'] > group_df['A_LL_A2'])
   group = np.where(is_lumper==1, 'lumpers', 'splitters')
   ```
   - **Lumpers**: B_LL_A2 > A_LL_A2 (use Rule B at A2)
   - **Splitters**: A_LL_A2 > B_LL_A2 (use Rule A at A2)

2. **For each participant, compute metrics:**

   **a. Interference:**
   ```python
   interference = 1 - A_weight_A2
   ```
   - From von Mises mixture model
   - Probability of using Rule B at A2

   **b. Summer accuracy:**
   ```python
   summer_accuracy = p_data[p_data['feature_idx']==0]['accuracy'].mean()
   ```
   - Mean accuracy on summer feature (feature_idx==0)
   - Measures baseline performance

   **c. Transfer error difference:**
   ```python
   final_A1 = p_data[(p_data['task_section']=='A1') & 
                     (p_data['feature_idx']==1)].iloc[-6:]['accuracy'].mean()
   initial_B = p_data[(p_data['task_section']=='B') & 
                      (p_data['feature_idx']==1)].iloc[:6]['accuracy'].mean()
   transfer_error_diff = initial_B - final_A1
   ```
   - Switch cost when transitioning from A1 to B
   - Positive = cost, negative = benefit

   **d. Retest error difference:**
   ```python
   A2_accuracy = p_data[(p_data['task_section']=='A2') & 
                        (p_data['feature_idx']==1)]['accuracy'].mean()
   retest_error_diff = A2_accuracy - final_A1
   ```
   - Performance change from A1 to A2
   - Negative = interference (worse at A2)

   **e. Generalization accuracy:**
   ```python
   generalisation_acc = p_data[
       (p_data['test_trial']==1) & 
       (p_data['task_section']=='A1') & 
       (p_data['block']>=5)
   ]['accuracy'].mean()
   ```
   - Accuracy on test trials (novel stimuli) in second half of A1
   - Measures ability to generalize

   **f. Correct AFC (debrief task):**
   ```python
   afc_dat = trial_df[trial_df['task_section']=='debrief'].groupby('participant')['correct_afc'].mean()
   correct_afc = 100 * afc_dat['correct_afc']
   ```
   - Accuracy on forced-choice debrief task
   - Measures explicit knowledge

3. **Output DataFrame:**
   - Columns: participant, condition, group, interference, summer_accuracy, transfer_error_diff, retest_error_diff, generalisation_acc, correct_afc, plus all von Mises parameters

**For ANNs:**
```python
ann_behav_df = ann.add_ann_metrics(
    rich_trial_data['near'], 
    lazy_trial_data['near'],
    rich_group_params.loc[rich_group_params['condition']=='near'],
    lazy_group_params.loc[lazy_group_params['condition']=='near']
)
```

**Detailed transformation:**

1. **For each network type** (rich/lazy):
   - Loops through all participants in 'near' condition

2. **Extract accuracy arrays:**
   ```python
   A1_accuracy = schedule_data[subj]['accuracy'][0, 1::2]  # Winter only
   B_accuracy = schedule_data[subj]['accuracy'][1, 1::2]
   A2_accuracy = schedule_data[subj]['accuracy'][2, 1::2]
   ```

3. **Compute same metrics as humans:**
   ```python
   final_A1_acc = np.mean(A1_accuracy[-6:])  # Final 6 trials
   initial_B_acc = np.mean(B_accuracy[0:6])  # First 6 trials
   A2_accuracy_mean = np.mean(A2_accuracy)  # All A2 trials
   
   transfer_error_diff = initial_B_acc - final_A1_acc
   retest_error_diff = A2_accuracy_mean - final_A1_acc
   summer_accuracy = np.mean(schedule_data[subj]['accuracy'][0, 0::2])  # Summer feature
   
   # Generalization: test trials only
   test_stim = schedule_data[subj]['test_stim'][0, 1::2].astype(int)
   all_A1_accuracy = schedule_data[subj]['accuracy'][0, 1::2].copy()
   all_A1_accuracy[test_stim==0] = np.nan  # Mask non-test trials
   generalisation_accuracy = np.nanmean(all_A1_accuracy)
   
   # Interference from von Mises fits
   interference = 1 - group_params.loc[
       group_params['participant']==participant_id, 'A_weight_A2'
   ].values[0]
   ```

4. **Add group label:**
   - 'rich' for rich regime networks
   - 'lazy' for lazy regime networks

5. **Output DataFrame:**
   - Same structure as human data
   - Columns: group, participant, initialB, transfer_error_diff, retest_error_diff, summer_accuracy, generalisation_acc, interference

#### Step 4: Combine Data

```python
grouped_df_all = pd.concat([near_participants_group, ann_behav_df])
grouped_df_all['ann'] = grouped_df_all['participant'].apply(
    lambda x: 1 if 'sim_' in str(x) else 0
)
```

**What happens:**
1. Concatenates human and ANN DataFrames
2. Adds 'ann' flag: 1 for ANNs, 0 for humans
3. All participants now in single DataFrame with consistent columns

#### Step 5: Statistical Comparisons

**For each metric:**
```python
stats.behav_group_comparisons(grouped_df_all, var)
```

**What happens:**

1. **For humans:**
   ```python
   splitters = grouped_df_all.loc[grouped_df_all['group']=='splitters', var].values
   lumpers = grouped_df_all.loc[grouped_df_all['group']=='lumpers', var].values
   
   t_stat_human, p_val_human = stats.ttest_ind(splitters, lumpers)
   df_human = len(splitters) + len(lumpers) - 2
   d_human = (np.mean(splitters) - np.mean(lumpers)) / np.sqrt((np.var(splitters) + np.var(lumpers)) / 2)
   ```
   - Two-sample t-test: splitters vs lumpers
   - Computes Cohen's d
   - Prints: t-statistic, p-value, df, Cohen's d, means, SEs

2. **For ANNs:**
   ```python
   lazy = grouped_df_all.loc[grouped_df_all['group']=='lazy', var].values
   rich = grouped_df_all.loc[grouped_df_all['group']=='rich', var].values
   
   t_stat_ann, p_val_ann = stats.ttest_ind(lazy, rich)
   df_ann = len(lazy) + len(rich) - 2
   d_ann = (np.mean(lazy) - np.mean(rich)) / np.sqrt((np.var(lazy) + np.var(rich)) / 2)
   ```
   - Two-sample t-test: lazy vs rich
   - Computes Cohen's d
   - Prints: t-statistic, p-value, df, Cohen's d, means, SEs

**Metrics analyzed:**
- `interference`: Retest interference (p(Rule B) at A2)
- `transfer_error_diff`: Task B switch cost (Δ accuracy)
- `generalisation_acc`: Generalization accuracy
- `summer_accuracy`: Summer feature accuracy
- `retest_error_diff`: Retest error difference
- `correct_afc`: Debrief task accuracy (humans only)

#### Step 6: Visualizations

**Combined plots (humans + ANNs):**
```python
fig, ax = figure_utils.plot_id_groups(
    data=grouped_df_all,
    grouping='group',
    group_order=['splitters', 'lumpers', 'lazy', 'rich'],
    group_names=['splitters', 'lumpers', 'lazy', 'rich'],
    var=var,
    yticks=ytick,
    ytick_labs=ytick_lab,
    ylim=ylim,
    ylab=ylab,
    colors=indiv_diff_colors,
    add_tests=0,
    p_value=np.nan,
    y_coord=np.nan
)
```

**What happens:**
1. Creates 4-group comparison plot
2. X-axis: 4 groups (splitters, lumpers, lazy, rich)
3. Y-axis: Metric value
4. Plot elements:
   - Stripplot: Individual data points
   - Pointplot: Mean ± SE with error bars
   - Connecting lines: Between human groups and between ANN groups
   - Markers: Mean points with white fill
5. **Output files:**
   - `fig4B_interference_both.png`
   - `fig4C_transfer_error_diff_both.png`
   - `fig4D_generalisation_acc_both.png`
   - `fig4E_summer_accuracy_both.png`
   - `fig4supp_retest_error_diff_both.png`

**Human-only plots:**
```python
fig, ax = figure_utils.plot_id_1group(
    data=grouped_df_all,
    grouping='group',
    group_order=['splitters', 'lumpers'],
    group_names=['splitters', 'lumpers'],
    var=var,
    ...
)
```

**What happens:**
1. Creates 2-group comparison (splitters vs lumpers only)
2. Same plot structure but only 2 groups
3. **Output files:**
   - `fig4E_summer_accuracy.png` (human only)
   - `fig4F_correct_afc.png` (human only, debrief task)

---

## Key Data Transformations Summary

### 1. Trial-Level → Aggregate Metrics

**Input**: 
- Individual trial data with columns: participant, condition, task_section, accuracy, feature_idx, block, test_trial

**Process**: 
- Group by participant and condition
- Extract specific trial windows (e.g., final 6 of A1, first 6 of B)
- Compute means over windows
- Calculate differences between windows

**Output**: 
- Participant-level metrics: transfer_error_diff, retest_error_diff, summer_accuracy, generalisation_acc

**Example:**
```
Trial data (1000 rows) 
  → Group by participant 
  → Extract windows 
  → Compute means 
  → Participant metrics (40 rows, one per participant)
```

### 2. Predictions → Von Mises Parameters

**Input**: 
- Network predictions: shape (3, n_trials, 4) - [cos_feat1, sin_feat1, cos_feat2, sin_feat2]
- Labels: shape (3, n_stim, 2, n_features)

**Process**: 
1. Convert predictions to radians: `atan2(sin, cos)`
2. Compute response angles: `winter_radians - summer_radians`
3. Wrap to [-π, π] range
4. Downsample to match human data length
5. Fit 2-component von Mises mixture model
6. Compare pure A vs pure B models (log-likelihood)

**Output**: 
- A_weight_A1, A_weight_B, A_weight_A2 (mixture weights)
- kappa_A1, kappa_B, kappa_A2 (concentration parameters)
- A_LL_B, B_LL_B, A_LL_A2, B_LL_A2 (model comparison)

**Example:**
```
Predictions (6000 values) 
  → Convert to angles (3000 values) 
  → Downsample (60 values) 
  → Fit mixture model 
  → Parameters (A_weight=0.8, kappa=5.2)
```

### 3. Hidden States → Representational Metrics

**Input**: 
- Hidden layer activations: shape (n_stimuli, dim_hidden)
- Example: (12, 50) for 12 stimuli, 50 hidden units

**Process**: 
1. **PCA dimensionality:**
   - Fit PCA to hidden states
   - Find n_components for 99% variance
   
2. **Geometric structure:**
   - Fit 2D PCA
   - Project stimuli onto 2D space
   - Visualize as connected points
   
3. **Principal angles:**
   - Fit 2D PCA to A and B separately
   - Compute inner product matrix
   - SVD to get principal angles

**Output**: 
- n_pca: Number of components needed
- hiddens_pca: 2D projections
- principal_angle_between: Angle between A and B subspaces

**Example:**
```
Hidden states (12 × 50) 
  → PCA fit 
  → 99% variance in 8 components 
  → n_pca = 8
```

### 4. Individual → Group Comparisons

**Input**: 
- Participant-level metrics DataFrame
- Columns: participant, condition, group, interference, transfer_error_diff, etc.

**Process**: 
1. **Classification:**
   - Humans: splitters vs lumpers (based on B_LL_A2 vs A_LL_A2)
   - ANNs: rich vs lazy (based on initialization regime)

2. **Statistical tests:**
   - Split by group
   - Perform t-tests
   - Compute effect sizes (Cohen's d)
   - Calculate means and SEs

3. **Visualization:**
   - Create group comparison plots
   - Show individual data points
   - Show group means with error bars
   - Add significance markers

**Output**: 
- Statistical test results (t-statistics, p-values, effect sizes)
- Publication-ready figures

**Example:**
```
40 participants 
  → Classify: 20 splitters, 20 lumpers 
  → Compare groups: t-test 
  → Effect size: d = 0.5 
  → Plot with error bars
```

---

## Data Dependencies

### Required Preprocessing Pipeline:

1. **01_preprocess_data.py**
   - **Input**: Raw CSV files in `data/participants/raw/`
   - **Output**: 
     - `data/participants/trial_df.csv` (required by all notebooks)
     - Excluded participants list

2. **02_run_simulations.py**
   - **Input**: `trial_df.csv`, configuration from `ann_experiments.json`
   - **Output**: 
     - `data/simulations/{condition_name}/sim_{participant}.npz` files
     - `data/simulations/{condition_name}/settings.json`
   - **Required for**: Notebooks 1, 2, 3 (ANN analyses)

3. **03_fit_vonmises.py**
   - **Input**: 
     - `trial_df.csv` (for humans)
     - Simulation `.npz` files (for ANNs)
   - **Output**: 
     - `data/participants/human_vonmises_fits.csv` (required by Notebooks 1, 3)
     - `data/simulations/{condition_name}_vonmises_fits.csv` (required by Notebooks 1, 3)

### Notebook Execution Order:

All notebooks can run **independently** once preprocessing is complete. However, logical order:

1. **figure2_transfer_interference.ipynb**
   - Main comparison analysis
   - Produces core publication figures
   - Requires: trial_df.csv, rich_50 simulations, von Mises fits

2. **figure3_anns.ipynb**
   - Representational analysis
   - Requires: rich_50 simulations only
   - Can run independently

3. **figure4_individual_differences.ipynb**
   - Individual differences analysis
   - Requires: trial_df.csv, rich_50 AND lazy_50 simulations, von Mises fits for both
   - Most complex data requirements

### Data File Checklist:

**For Notebook 1:**
- ✅ `data/participants/trial_df.csv`
- ✅ `data/simulations/rich_50/*.npz` files
- ✅ `data/participants/human_vonmises_fits.csv`
- ✅ `data/simulations/rich_50_vonmises_fits.csv`

**For Notebook 2:**
- ✅ `data/simulations/rich_50/*.npz` files
- ✅ `data/participants/trial_df.csv` (for reference only)

**For Notebook 3:**
- ✅ `data/participants/trial_df.csv`
- ✅ `data/participants/human_vonmises_fits.csv`
- ✅ `data/simulations/rich_50/*.npz` files
- ✅ `data/simulations/lazy_50/*.npz` files
- ✅ `data/simulations/rich_50_vonmises_fits.csv`
- ✅ `data/simulations/lazy_50_vonmises_fits.csv`

---

## Data Structure Reference

### Trial DataFrame (`trial_df.csv`)
```
Columns:
- participant: str (e.g., 'study1_near_sub1')
- condition: str ('same', 'near', 'far')
- study: int (1 or 2)
- task_section: str ('A1', 'B', 'A2', 'debrief')
- block: int (0-20)
- stimID: int (0-11)
- feature_idx: int (0=summer, 1=winter)
- accuracy: float (0-1)
- resp_error: float (radians)
- test_trial: int (0=training, 1=test)
- A_rule: float (radians)
- B_rule: float (radians)
- rule_applied: float (radians)
```

### Von Mises Fits DataFrame
```
Columns:
- participant: str
- condition: str ('near', 'far')
- study: int (humans only)
- A_weight_A1: float (0-1)
- A_weight_B: float (0-1)
- A_weight_A2: float (0-1)
- kappa_A1: float (>0)
- kappa_B: float (>0)
- kappa_A2: float (>0)
- A_LL_B: float (log-likelihood)
- B_LL_B: float (log-likelihood)
- A_LL_A2: float (log-likelihood)
- B_LL_A2: float (log-likelihood)
```

### ANN Data Structure (from `.npz` files)
```python
{
    'predictions': np.array, shape (3, n_trials, 4)
        # [cos_feat1, sin_feat1, cos_feat2, sin_feat2]
    'labels': np.array, shape (3, n_stim, 2, n_features)
        # Target labels for each stimulus
    'accuracy': np.array, shape (3, n_trials)
        # Accuracy per trial per phase
    'losses': np.array, shape (3, n_trials)
        # MSE loss per trial per phase
    'test_stim': np.array, shape (3, n_trials)
        # 0=training, 1=test trial
    'hiddens_post_phase_0': np.array, shape (12, dim_hidden)
        # Hidden states after A1 training
    'hiddens_post_phase_1': np.array, shape (12, dim_hidden)
        # Hidden states after B training
    'participant': str
        # Participant ID
}
```

---

## Summary

The three notebooks form a complete analysis pipeline:

1. **Notebook 1** compares transfer and interference between humans and ANNs at the group level
2. **Notebook 2** analyzes representational properties of ANNs (dimensionality, geometry, alignment)
3. **Notebook 3** examines individual differences, comparing human strategies (splitters/lumpers) with ANN regimes (rich/lazy)

All notebooks transform data from trial-level or simulation-level to participant-level metrics, then perform statistical comparisons and create publication-ready visualizations. The key insight is that similar patterns emerge in both humans and ANNs, suggesting common principles of learning and interference.
