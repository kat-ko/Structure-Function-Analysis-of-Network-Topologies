# Analysis of `interference_task.py`: Training Task Overview

## Quick Reference Summary

| Aspect | Details |
|--------|---------|
| **Task Type** | Catastrophic interference (3-phase: A1 → B → A2) |
| **Input Format** | One-hot encoded stimulus IDs (12 dimensions: 6 stimuli × 2 tasks) |
| **Output Format** | 4 dimensions: `[cos_feat1, sin_feat1, cos_feat2, sin_feat2]` |
| **Label Format** | 2 dimensions: `[cos(feat_val), sin(feat_val)]` (circular features) |
| **Network** | 2-layer linear (12 → hidden → 4), no bias terms |
| **Loss Function** | MSE between predicted and target [cos, sin] pairs |
| **Feature Values** | Radians (0 to 2π, typically 0.9-5.8 in data) |
| **Participants** | Each has unique feature value mappings per stimulus |
| **Training Phases** | A1: Full updates, B: Full updates, A2: Feature 1 only |

## Executive Summary

This document provides a comprehensive analysis of the interference task training process, including input/output data structures, network architecture, and participant variation.

---

## 1. Task Structure: Three-Phase Training

The task implements a **catastrophic interference** paradigm with three sequential training phases:

1. **Phase A1**: Initial training on Task A stimuli
2. **Phase B**: Training on Task B stimuli (interference phase)
3. **Phase A2**: Re-training on Task A stimuli (tests for interference/forgetting)

Each participant goes through all three phases sequentially, with the network being trained on their specific behavioral data.

---

## 2. Input Data Structure

### 2.1 Raw Data Source (`trial_df.csv`)

The raw data contains the following key columns per trial:
- `participant`: Participant identifier
- `task_section`: One of 'A1', 'B', or 'A2'
- `stimID`: Stimulus identifier (0-11, representing 6 stimuli per task × 2 tasks)
- `feature_idx`: Which feature is being probed (0 = feature 1, 1 = feature 2)
- `feat_val`: The feature value in **radians** (circular feature space)
- `index`: Trial index within the task section
- `test_trial`: Whether this is a test trial (1) or training trial (0)

### 2.2 Input Encoding: One-Hot Stimulus Representation

**Network Input Format:**
- **Shape**: `(batch_size, 12)` - One-hot encoded vector
- **Encoding**: Each stimulus ID (0-11) is represented as a one-hot vector of length 12
  - Example: Stimulus ID 3 → `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]`
- **Purpose**: The network receives only the stimulus identity, not the feature values directly

**Key Code Location:**
```python
# Lines 52-60: create_inputs_matrix()
inputs = np.zeros((length, n_stim_per_task * 2))  # 12 dimensions
inputs[index, int(row['stimID'])] = 1  # One-hot encoding
```

### 2.3 Participant-Specific Data Variation

**What Varies Between Participants:**
1. **Feature Values (`feat_val`)**: Each participant has different feature values for each stimulus
   - Feature values are in radians (circular space)
   - Each stimulus has TWO features (feature_idx 0 and 1)
   - These values determine the target labels

2. **Stimulus-to-Feature Mappings**: The mapping between stimulus IDs and their feature values is participant-specific
   - Same stimulus ID can have different feature values across participants
   - This creates different learning problems for each participant

3. **Trial Ordering**: The sequence of trials within each phase may vary

**What is Constant Across Participants:**
- Network architecture (input size: 12, output size: 4)
- Number of stimuli per task (6)
- Three-phase structure (A1 → B → A2)
- Input encoding scheme (one-hot)

---

## 3. Output Data Structure

### 3.1 Network Output Format

**Network Output:**
- **Shape**: `(batch_size, 4)` - Four-dimensional output
- **Structure**:
  - `output[:, 0:2]`: Feature 1 prediction (cos, sin)
  - `output[:, 2:4]`: Feature 2 prediction (cos, sin)

**Key Code Location:**
```python
# Lines 262-278: simpleLinearNet architecture
dim_output = 4  # 2 dimensions for each feature
```

### 3.2 Target Labels (Ground Truth)

**Label Format:**
- **Shape**: `(batch_size, 2)` - Two-dimensional label
- **Structure**: `[cos(feat_val), sin(feat_val)]`
- **Feature Selection**: Which feature to predict is determined by `feature_probe`:
  - `feature_probe == 0`: Predict Feature 1 → use `output[:, 0:2]`
  - `feature_probe == 1`: Predict Feature 2 → use `output[:, 2:4]`

**Key Code Location:**
```python
# Lines 201-219: Training loop
joined_label = torch.cat((label_x.unsqueeze(1), label_y.unsqueeze(1)), dim=1)
# joined_label = [cos(feat_val), sin(feat_val)]

if feature_probe == 0:
    loss = loss_function(out[:, :2], joined_label)  # Feature 1
elif feature_probe == 1:
    loss = loss_function(out[:, 2:4], joined_label)  # Feature 2
```

### 3.3 Label Conversion Process

**From Radians to Cartesian Coordinates:**
1. Raw data: `feat_val` in radians
2. Conversion: `label_x = cos(feat_val)`, `label_y = sin(feat_val)`
3. Purpose: Circular features are represented as points on a unit circle

**Key Code Location:**
```python
# Lines 131-133: assemble_dataset()
dataset_A1 = assemble_dataset(..., 
    np.cos(participant_training_A1['feat_val'].values), 
    np.sin(participant_training_A1['feat_val'].values))
```

---

## 4. Network Architecture

### 4.1 Architecture Overview

**Network Type**: Simple 2-layer linear network (no bias terms)

```
Input (12) → Hidden Layer → Output (4)
```

**Layer Details:**
- **Input Layer**: 12 dimensions (one-hot stimulus encoding)
- **Hidden Layer**: Configurable size (from `condition['dim_hidden']`)
- **Output Layer**: 4 dimensions (2 features × 2 dimensions each)

**Key Code Location:**
```python
# Lines 262-278: simpleLinearNet
class simpleLinearNet(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output):
        self.in_hid = nn.Linear(dim_input, dim_hidden, bias=False)
        self.hid_out = nn.Linear(dim_hidden, dim_output, bias=False)
```

### 4.2 What the Network Receives

**Per Training Step:**
1. **Input**: One-hot encoded stimulus ID (12-dim vector)
2. **Feature Probe**: Which feature to predict (0 or 1)
3. **Target Label**: `[cos(feat_val), sin(feat_val)]` for the probed feature
4. **Test Trial Flag**: Whether this is a test trial (affects weight updates)

### 4.3 What the Network Outputs

**Per Forward Pass:**
1. **Output**: 4-dimensional vector `[cos_feat1, sin_feat1, cos_feat2, sin_feat2]`
2. **Hidden State**: Hidden layer activations (for analysis)

**Prediction Extraction:**
```python
# Lines 210-216: Prediction calculation
if feature_probe == 0:
    pred_rads = math.atan2(out[:, 0], out[:, 1])  # Feature 1
elif feature_probe == 1:
    pred_rads = math.atan2(out[:, 2], out[:, 3])  # Feature 2
```

---

## 5. Training Process Flow

### 5.1 Data Processing Pipeline

**Step 1: Data Loading**
```python
df = pd.read_csv('data/participants/trial_df.csv')
# Filter to A1, B, A2 sections only
```

**Step 2: Participant-Specific Filtering**
```python
# Lines 36-43: filter_participant_data()
participant_training_A1 = filter_participant_data(df, participant, 'A1')
participant_training_B = filter_participant_data(df, participant, 'B')
participant_training_A2 = filter_participant_data(df, participant, 'A2')
```

**Step 3: Input Matrix Creation**
```python
# Lines 52-60: create_inputs_matrix()
# Converts stimulus IDs to one-hot vectors
A1_inputs = create_inputs_matrix(participant_training_A1, nStim_perTask=6)
# Shape: (n_trials, 12)
```

**Step 4: Label Preparation**
```python
# Lines 131-133: assemble_dataset()
# Convert feat_val (radians) to [cos, sin] representation
label_x = np.cos(participant_training_A1['feat_val'].values)
label_y = np.sin(participant_training_A1['feat_val'].values)
```

### 5.2 Training Loop Details

**Per Epoch:**
1. Iterate through batches from DataLoader
2. Extract: `input`, `label_x`, `label_y`, `feature_probe`, `test_stim`
3. Forward pass: `out, hid = network(input)`
4. Select output based on `feature_probe`:
   - Probe 0 → `out[:, 0:2]` (Feature 1)
   - Probe 1 → `out[:, 2:4]` (Feature 2)
5. Compute loss: MSE between selected output and `[cos(feat_val), sin(feat_val)]`
6. Backward pass (conditional on `do_update` and `test_stim`)

**Update Rules:**
- **Phase A1 & B** (`do_update=1`): Update on all training trials
- **Phase A2** (`do_update=2`): Update only when `feature_probe == 0` (Feature 1 only)

**Key Code Location:**
```python
# Lines 222-230: Conditional weight updates
if do_update == 1 and do_test==1 and test_stim.numpy() == 0:
    loss.backward()
    optimizer.step()
elif do_update == 2 and feature_probe == 0:  # A2: only Feature 1
    loss.backward()
    optimizer.step()
```

### 5.3 Metrics Tracked

**Per Training Step:**
- `indexes`: Trial indices
- `inputs`: Input vectors (one-hot)
- `labels`: Target labels `[cos, sin]`
- `probes`: Which feature was probed
- `test_stim`: Test trial flag
- `losses`: MSE loss values
- `accuracy`: Angular accuracy (1 - normalized_error)
- `predictions`: Network outputs (4-dim)
- `hiddens`: Hidden layer activations
- `embeddings`: Input-to-hidden weights
- `readouts`: Hidden-to-output weights

**Accuracy Calculation:**
```python
# Lines 154-160: compute_accuracy()
wrapped_difference = wrap_to_pi(predictions - ground_truth)
normalized_error = np.abs(wrapped_difference) / np.pi
accuracy = 1 - normalized_error
```

---

## 6. Participant Variation Analysis

### 6.1 Data Variation Sources

**1. Feature Value Differences:**
- Each participant has unique `feat_val` values for each stimulus
- These values are in radians (circular space: 0 to 2π)
- Same stimulus ID can map to different feature values across participants

**2. Stimulus-to-Feature Mapping:**
- The relationship between stimulus identity and feature values is participant-specific
- This creates different learning problems for each participant

**3. Trial Sequences:**
- The order and frequency of trials may vary between participants
- DataLoader shuffling (if enabled) further randomizes trial order

### 6.2 What Stays Constant

**Across All Participants:**
- Network architecture (12 → hidden → 4)
- Input encoding scheme (one-hot)
- Output format (4-dim: 2 features × 2 dims)
- Training phases (A1 → B → A2)
- Number of stimuli per task (6)
- Loss function (MSE)
- Optimizer (SGD)

### 6.3 Participant-Specific Processing

**Per Participant:**
1. Filter data by participant ID
2. Extract three task sections (A1, B, A2)
3. Create participant-specific input matrices
4. Generate participant-specific labels from their `feat_val` values
5. Train separate network instance on participant's data

**Key Code Location:**
```python
# Lines 419-423: Per-participant loop
for participant in participants[0:10]:
    dataset_A1, dataset_B, dataset_A2, raw_inputs, raw_labels = \
        get_datasets(df, participant, task_parameters)
    # ... train network on this participant's data
```

---

## 7. Key Data Transformations

### 7.1 Input Transformation

```
Raw Data: stimID (0-11)
    ↓
One-Hot Encoding: [0,0,0,1,0,0,0,0,0,0,0,0] (12-dim vector)
    ↓
Network Input
```

### 7.2 Label Transformation

```
Raw Data: feat_val (radians, e.g., 1.57)
    ↓
Cartesian Conversion: [cos(1.57), sin(1.57)] = [0, 1]
    ↓
Target Label for Network
```

### 7.3 Output Transformation

```
Network Output: [cos_feat1, sin_feat1, cos_feat2, sin_feat2]
    ↓
Select based on feature_probe: [cos_feat1, sin_feat1] OR [cos_feat2, sin_feat2]
    ↓
Convert to radians: atan2(sin, cos)
    ↓
Compare with ground truth feat_val
```

---

## 8. Summary: Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    RAW DATA (trial_df.csv)                  │
│  participant | task_section | stimID | feature_idx |       │
│              | feat_val (radians) | test_trial             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              PARTICIPANT-SPECIFIC FILTERING                 │
│  - Filter by participant ID                                 │
│  - Separate into A1, B, A2 sections                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    INPUT PROCESSING                         │
│  stimID → One-Hot Encoding (12-dim vector)                  │
│  Example: stimID=3 → [0,0,0,1,0,0,0,0,0,0,0,0]            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    LABEL PROCESSING                         │
│  feat_val (radians) → [cos(feat_val), sin(feat_val)]       │
│  Example: 1.57 → [0, 1]                                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    NETWORK FORWARD PASS                     │
│  Input (12) → Hidden → Output (4)                          │
│  Output: [cos_feat1, sin_feat1, cos_feat2, sin_feat2]      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    LOSS COMPUTATION                         │
│  Select output based on feature_probe:                      │
│  - Probe 0: out[:, 0:2] vs [cos_feat1, sin_feat1]          │
│  - Probe 1: out[:, 2:4] vs [cos_feat2, sin_feat2]          │
│  Loss = MSE(selected_output, target_label)                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    WEIGHT UPDATE                            │
│  Conditional on:                                            │
│  - do_update flag (1 or 2)                                  │
│  - test_stim flag (0 = training, 1 = test)                  │
│  - feature_probe (for do_update=2)                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. Important Implementation Details

### 9.1 Index Adjustment

Indices are adjusted to be continuous across phases:
- A1: indices 0 to A_length-1
- B: indices 0 to B_length-1 (offset by A_length)
- A2: indices 0 to A2_length-1 (offset by A_length + B_length)

**Key Code Location:**
```python
# Lines 111-115: adjust_indices()
A_length = len(participant_training_A1)
participant_training_B = adjust_indices(participant_training_B, A_length)
participant_training_A2 = adjust_indices(participant_training_A2, A_length + B_length)
```

### 9.2 Test Trial Handling

- Test trials (`test_trial == 1`) are used for evaluation but may not update weights
- In Phase A2, only Feature 1 updates are allowed (`do_update=2`)

### 9.3 Ordered Sweeps

After each phase, the network is evaluated on ordered inputs (sorted by feature values):
```python
# Lines 474-477, 509-511, 535-537, 561-563
ordered_inputs = np.concatenate((A_inputs[ordered_indices_A], 
                                 B_inputs[ordered_indices_B]), axis=0)
post_preds, post_hiddens = ordered_sweep(network, ordered_inputs)
```

This provides interpretable results for visualization and analysis.

---

## 10. Configuration Parameters

**Task Parameters:**
- `nStim_perTask`: 6 (stimuli per task)
- Total input dimensions: 12 (6 stimuli × 2 tasks)

**Network Parameters:**
- `dim_input`: 12
- `dim_hidden`: From condition config (e.g., 50 for "rich_50")
- `dim_output`: 4

**Training Parameters:**
- `n_epochs`: From config file
- `batch_size`: From config file
- `learning_rate`: From config file
- `gamma`: Weight initialization scale (from condition config)

---

## Conclusion

The interference task implements a three-phase learning paradigm where:
1. **Input**: One-hot encoded stimulus identities (12-dim)
2. **Output**: Four-dimensional predictions (2 features × 2 dims each)
3. **Labels**: Circular features represented as [cos, sin] pairs
4. **Variation**: Each participant has unique feature values and stimulus mappings
5. **Training**: Sequential phases (A1 → B → A2) test for catastrophic interference

The network must learn to map stimulus identities to their associated feature values, with the challenge of maintaining Task A performance after learning Task B.

