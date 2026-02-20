# How a1b2 Implements the Holton-Style Continual Learning Task in an RNN

This document explains how the **a1b2** module implements the **A1–B–A2** continual learning (transfer–interference) paradigm—in the spirit of Holton et al.—and how it trains it with both a **feedforward network (FFN)** and a **two-module RNN** (dynspec-style). It covers the task design, architecture variants, input/hidden/output handling, and how learning works across phases.

---

## 1. The A1–B–A2 Continual Learning Task

The task is a **continual learning** setup with three phases on the **same stimuli** but **different response features**:

- **A1**: Train the network to predict **feature 0** (e.g. one circular feature: cos θ₁, sin θ₁) from stimulus → response.
- **B**: Train on **feature 1** (e.g. another circular feature: cos θ₂, sin θ₂) for the same stimulus set.
- **A2**: Test (and optionally train) on **both features**: measure retention of A1 and adaptation to B, and optionally update on both feature probes.

So the network must **reuse the same input representation** across phases while switching or combining which **output feature** is trained or probed. This induces **transfer** (positive) and **interference** (negative) between A1 and B, depending on how similar the B rule is to the A rule (e.g. same / near / far in angle).

### 1.1 Task parameters (from the codebase)

- **Stimuli**: `nStim_perTask` stimuli (e.g. 6), each represented by a **one-hot-like** vector of size `nStim_perTask * 2` (e.g. **12**).
- **Output**: **4 dimensions** — two 2D unit vectors:
  - **Feature 0**: `(label_x, label_y)` = (cos θ₁, sin θ₁)
  - **Feature 1**: (cos θ₂, sin θ₂)
- So the network predicts a **4D vector** `(out[:,:2], out[:,2:4])`; each 2D part is trained or evaluated depending on **feature_probe** (0 or 1).
- **Loss**: **MSE** between predicted (cos, sin) and target (cos, sin) for the probed feature.
- **Accuracy**: circular error on the angle (1 − |wrapped(θ_pred − θ_true)| / π).

This matches the typical Holton-style setup: same input space, two response features, sequential training A1 → B → A2.

---

## 2. Architecture Variants: FFN vs Two-Module RNN

The a1b2 pipeline is **architecture-agnostic** where possible: the schedule and training loop expect any network that implements **`forward(x) -> (out, hid)`** with fixed input/output sizes. Two architectures are supported:

| Variant              | Architecture       | Config (experiments.json)     | Notes |
|----------------------|--------------------|-------------------------------|--------|
| **FFN**              | `simpleLinearNet`  | `arch: "ffn"`, `dim_hidden`   | One hidden layer, no bias. |
| **Two-module RNN**   | `TwoModuleRNNWrapper` | `arch: "two_module_rnn"`, `dim_hidden`, `sparsity`, `common_readout`; `common_input` (default **false**, true = ablation only) | Wraps Community (dynspec-style RNN). |

### 2.1 Condition definitions (experiments.json)

**File: `a1b2/models/experiments.json`**

```json
{
  "conditions": [
    {"name": "rich_50", "arch": "ffn", "gamma": 0.001, "dim_hidden": 50},
    {"name": "two_module_rnn_50", "arch": "two_module_rnn", "dim_hidden": 50, "sparsity": 1, "common_input": false, "common_readout": true},
    {"name": "two_module_rnn_50_low_sparse", "arch": "two_module_rnn", "dim_hidden": 50, "sparsity": 0.3, "common_input": false, "common_readout": true}
  ],
  "n_epochs": 100,
  "n_phase": 3,
  "learning_rate": 0.01,
  "shuffle": false,
  "batch_size": 1
}
```

So the **variants** for the RNN are:

- **two_module_rnn_50**: 2 modules, 50 hidden units each, full inter-module connectivity (`sparsity: 1`), **separate input** per module (`common_input: false`, default) and **common readout**.
- **two_module_rnn_50_low_sparse**: same but **sparse** inter-module connections (`sparsity: 0.3`).
- **common_input=true** is for **ablation only** (to validate main results); use conditions such as `two_module_rnn_50_ablation_common_input` or `two_module_rnn_50_task_routed_ablation_common_input`.

A separate **geometry** run (`train_single_schedule`) uses the same A1 → B → A2 idea but with three B conditions (same / near / far rule) and copies of the network to compare transfer/interference across B rule distance; it is FFN-only in the current script (no RNN branch in `train_single_schedule`).

In both cases the RNN uses the same **Community** model as dynspec (core + comms, masked weights), wrapped so that input/output match the A1–B–A2 interface (single-step input, 4D output, hidden state for logging).

---

## 3. Data and Input Handling

### 3.1 Dataset structure (participant data)

Data comes from **participant trial data** (e.g. `trial_df.csv`) and is split into A1, B, and A2 by `task_section`. For each phase, `get_datasets()` builds:

**File: `a1b2/data/basic_funcs.py`**

- **Inputs**: one-hot-style vectors of size **`nStim_perTask * 2`** (e.g. 12). Each row is one trial; the “which stimulus” is encoded in this vector (e.g. `create_inputs_matrix` sets one position to 1 per trial).
- **Labels**: For each stimulus, two features (two angles) are stored as (cos, sin) for feature 0 and (cos, sin) for feature 1. So we have `label_x`, `label_y` (cos/sin for the **probed** feature on that trial) and the dataset also carries the full 4D target implicitly via `raw_labels` (4 × nStim_perTask).
- **feature_probe**: 0 or 1 — which feature is trained/evaluated on this trial (feature 0 vs feature 1).
- **test_stim**: whether this trial is a test trial (no update).

**File: `a1b2/data/basic_funcs.py` (assemble_dataset)**

```python
return {
    'index': participant_data['index'].values,
    'stim_index': participant_data['stimID'].values,
    'input': inputs,                    # (n_trials, 12)
    'feature_probe': participant_data['feature_idx'].values,
    'test_stim': participant_data['test_trial'].values,
    'label_x': label_cos,                # cos of probed feature angle
    'label_y': label_sin,                # sin of probed feature angle
    ...
}
```

So each batch from `CreateParticipantDataset` gives a dict with `'input'` of shape `(batch_size, 12)` (after batching), plus `label_x`, `label_y`, `feature_probe`, etc.

### 3.2 How the RNN receives input (TwoModuleRNNWrapper)

The **FFN** receives `(batch, 12)` directly. The **two-module RNN** must present **`(seq_len, batch, input_size * n_modules)`** to the inner Community (dynspec convention). The wrapper does **not** use a multi-step sequence; it treats each trial as a **single time step** and expands the input for the two modules:

**File: `a1b2/models/two_module_rnn.py`**

```python
def forward(self, x):
    # x: (batch, input_size)
    if x.dim() == 2:
        x = x.unsqueeze(0)  # (1, batch, input_size)
    # Community expects (seq_len, batch, input_size * n_modules)
    x_expanded = x.repeat(1, 1, self.n_modules)  # (1, batch, input_size * n_modules)
    outputs, all_states = self.community(x_expanded)
    out = outputs[-1]   # last (and only) time step
    ...
    hid = all_states[-1]
    return out, hid
```

So:

- **Input to wrapper**: `x` of shape `(batch, 12)`.
- **Input to Community**: `(1, batch, 24)` — one time step, same 12-dim input **replicated** for both modules (so **common input** in the sense that both modules see the same 12-D vector).
- **No temporal sequence** is used in the default A1–B–A2 pipeline; the RNN’s “recurrence” is across trials only implicitly (same weights, but each forward is one step with no explicit hidden state carry-over between trials). So the RNN is used as a **single-step recurrent unit** per trial (state is reset each forward by PyTorch default).

So for a1b2’s continual learning script:

- **Input handling**: One vector per trial, shape `(B, 12)`; expanded to `(1, B, 24)` for Community.
- **Hidden**: The Community’s combined hidden state at that one step is returned as `hid` of shape `(batch, hidden_size * n_modules)` (e.g. `(B, 100)`) for logging/analysis; it is **not** passed as initial state to the next trial.

---

## 4. Hidden State Handling

### 4.1 Where hidden state lives

- **FFN**: `simpleLinearNet` has one hidden layer; `forward(x)` returns `(out, hid)` with `hid` of shape `(batch, dim_hidden)`.
- **Two-module RNN**: After `self.community(x_expanded)`, `all_states` has shape `(1, batch, n_modules * hidden_size)`; the wrapper returns `hid = all_states[-1]` so shape `(batch, 100)` for 2×50.

So in both cases the training loop gets **one hidden vector per sample** at the current trial.

### 4.2 No carry-over of hidden state between trials

In `train_participant_schedule`, each batch is independent:

```python
out, hid = network(input_t)
```

There is **no** code that feeds the previous trial’s `hid` as the next trial’s initial hidden state. So:

- For **FFN**: this is irrelevant (no state).
- For **RNN**: Each forward call uses the RNN’s **default initial hidden state (zero)**. So the “continual learning” is entirely in the **shared weights** across phases (A1 → B → A2), not in recurrent state across trials. The RNN’s role is to provide a **structured (modular) representation** (two modules, core + comms) that can specialize or share across the two features, similar to dynspec.

### 4.3 What is stored for analysis

The schedule stores **hiddens** per phase and per trial for later analysis (e.g. transfer, geometry, PCA):

**File: `a1b2/training/schedule.py`**

```python
metrics["hiddens"].append(hid.detach().numpy())
...
results["hiddens"][phase, :n, :] = hid[:n]
```

So you get `(n_phase, n_trials, dim_hidden)` for downstream analyses (e.g. `hiddens_post_phase_0`, `hiddens_post_phase_1` in the saved `.npz`).

---

## 5. Output Handling and Feature Probing

The network always outputs a **4D** vector: two 2D unit-vector predictions (feature 0 and feature 1). Which part is trained or evaluated is controlled by **feature_probe** and **do_update**.

### 5.1 Output layout

- **FFN**: `out = self.hid_out(hid)` → shape `(batch, 4)`.
- **Two-module RNN**: After `outputs[-1]`, shape is `(batch, 4)` when `common_readout=True` (single readout over combined state). When `common_readout=False`, the wrapper sums the two module readouts to get one 4D vector:

**File: `a1b2/models/two_module_rnn.py`**

```python
out = outputs[-1]
if not self.common_readout:
    out = out[:, : self.output_size] + out[:, self.output_size :]  # sum modules -> (batch, 4)
```

So the **training loop always sees** `out` of shape `(batch, 4)`.

### 5.2 Feature probe and loss (training loop)

**File: `a1b2/training/schedule.py`**

```python
joined_label = torch.cat((label_x.unsqueeze(1), label_y.unsqueeze(1)), dim=1)  # (B, 2)
...
if feature_probe == 0:
    loss = loss_function(out[:, :2], joined_label)
    pred_rads = math.atan2(out[:, 0].detach().numpy(), out[:, 1].detach().numpy())
    accuracy = compute_accuracy(pred_rads, radians_label)
elif feature_probe == 1:
    loss = loss_function(out[:, 2:4], joined_label)
    pred_rads = math.atan2(out[:, 2].detach().numpy(), out[:, 3].detach().numpy())
    accuracy = compute_accuracy(pred_rads, radians_label)
```

So:

- **feature_probe == 0**: loss and accuracy on **first** 2D output (feature 0: A-relevant).
- **feature_probe == 1**: loss and accuracy on **second** 2D output (feature 1: B-relevant).
- Labels `joined_label` are (cos θ, sin θ) for the **probed** feature on that trial (from `label_x`, `label_y` in the dataset, which are set per trial according to which feature is probed).

So **output handling** is: one 4D vector; slice by `feature_probe` to get the 2D prediction and compare to the 2D target for that feature.

---

## 6. Learning: Phases, Updates, and Optimizer

### 6.1 The three phases (runSchedule)

**File: `a1b2/training/schedule.py`**

```python
phases = [
    (0, trainloader_A1, 1),   # Phase 0: A1, do_update=1
    (1, trainloader_B, 1),    # Phase 1: B,  do_update=1
    (2, trainloader_A2, 2),  # Phase 2: A2, do_update=2
]
for phase, loader, do_update in phases:
    out = train_function(network, loader, n_epochs, loss_function, optimizer, do_update, do_test)
```

- **Phase 0 (A1)**: Train on A1 data; `feature_probe` in the data is 0 (feature 0). Updates every time (subject to `do_test`/test_stim below).
- **Phase 1 (B)**: Train on B data; `feature_probe` is 1 (feature 1). Same update rule.
- **Phase 2 (A2)**: Train on A2 data; trials can have **either** feature_probe 0 or 1. **do_update=2** means “update only when feature_probe==0” (see below), so in the default setup A2 trains only on feature 0 (retention of A1), while both features can still be **evaluated** (accuracy/loss logged).

So the **continual learning** schedule is: train A1 → train B → train A2 (with update rule 2). The same **network** and **optimizer** are used across all three phases; only the data loader and `do_update` change.

### 6.2 When gradients are applied (do_update and do_test)

**File: `a1b2/training/schedule.py`**

```python
if do_update == 1 and do_test == 1 and test_stim.numpy() == 0:
    loss.backward()
    optimizer.step()
elif do_update == 1 and do_test == 0:
    loss.backward()
    optimizer.step()
elif do_update == 2 and feature_probe == 0:
    loss.backward()
    optimizer.step()
```

- **do_update == 1**: Update on every (non–test) trial when do_test is 0; when do_test is 1, update only when **test_stim == 0** (i.e. not a test trial).
- **do_update == 2**: Update **only when feature_probe == 0** (A2 phase: only gradient on feature 0 output). So during A2, the network is trained only on feature 0; feature 1 is still computed and can be evaluated for transfer/interference metrics.

So **learning** in A2 is deliberately restricted to the “A” feature (feature 0) to measure retention and interference from B.

### 6.3 Optimizer and loss

- **Optimizer**: **SGD** with fixed learning rate (e.g. from `experiments.json`, typically 0.01). Same optimizer for the whole schedule (all three phases).
- **Loss**: **MSELoss** on the 2D (cos, sin) slice selected by `feature_probe`.

**File: `a1b2/training/schedule.py` (runSchedule)**

```python
optimizer = torch.optim.SGD(network.parameters(), lr=lr)
loss_function = nn.MSELoss()
```

So there is **no** separate optimizer or loss for the RNN; the same SGD + MSE setup is used for both FFN and two-module RNN. The RNN’s parameters (Community core, comms, readout) are all in `network.parameters()` and are updated by the same backward/step.

---

## 7. End-to-End Flow: From Data to Phase Results

1. **Load participant data** → split into A1, B, A2; build `dataset_A1`, `dataset_B`, `dataset_A2` with `input` (12-D), `label_x`, `label_y`, `feature_probe`, `test_stim`.
2. **Build network**: Either `simpleLinearNet(12, dim_hidden, 4)` or `TwoModuleRNNWrapper(input_size=12, output_size=4, hidden_size=50, ...)` (with `dim_hidden = 2*50` for the RNN in result arrays).
3. **runSchedule**:
   - Phase 0: `train_participant_schedule(network, trainloader_A1, ...)` → train on feature 0 (A1).
   - Phase 1: same function, `trainloader_B` → train on feature 1 (B).
   - Phase 2: same function, `trainloader_A2`, `do_update=2` → train only on feature 0 (A2), evaluate both.
4. Each training step: get batch `data` → `input_t = batch_to_torch(data['input'])` → `out, hid = network(input_t)` → choose loss slice by `feature_probe` → `loss.backward()` (if update) → `optimizer.step()`.
5. **Save** per-phase arrays: inputs, labels, predictions, hiddens, losses, accuracy, etc., for transfer/interference and geometry analyses.

---

## 8. Summary Table: Input, Hidden, Output, Learning

| Aspect | FFN | Two-module RNN |
|--------|-----|----------------|
| **Input** | `(B, 12)` | `(B, 12)` or `(T, B, 12)` → wrapper → `(T, B, 12*n_modules)` into Community |
| **Hidden** | `(B, dim_hidden)` one layer | `(B, n_modules*hidden_size)` last-step combined state; optional trajectory when T>1 |
| **Output** | `(B, 4)` | `(B, 4)` (common readout or sum of module readouts) |
| **Loss** | MSE on `out[:,:2]` or `out[:,2:4]` by feature_probe | Same |
| **Optimizer** | SGD(lr) | Same |
| **Phases** | A1 (feat 0) → B (feat 1) → A2 (update on feat 0 only) | Same |
| **Variants** | rich_10/50/200, gamma_*, lazy_50 | single_module_rnn_50, two_module_rnn_50, two_module_rnn_50_low_sparse, two_module_rnn_50_task_routed, two_module_rnn_50_task_routed_low_sparse, two_module_rnn_50_sp05, two_module_rnn_50_task_routed_sp05, two_module_rnn_50_sep_readout, + nb_steps, init_scale |

---

## 9. Experimental factors and run_id

For RNN studies the **primary comparison** is **no-module (single-module) vs two-module** architecture. All varying factors are reflected in **run_id** so that each configuration gets a unique results folder (`data/simulations/<run_id>/`).

**What varies:** task similarity (same/near/far B), architecture (single_module_rnn vs two_module_rnn), input routing (`shared` vs `task_routed`: A→module 1, B→module 2 by feature_probe), nb_steps (sequence length; >1 enables trajectory logging). Each RNN condition has an optional **_nb2** variant (e.g. `two_module_rnn_50_nb2`) with `nb_steps=2`; run_id then includes `nb2` so results do not mix with nb_steps=1 runs. Other factors: communication (sparsity, common_readout), **common_input** (default **false** = separate input per module; **true** = ablation only), cell_type (RNN/GRU), and optional init_scale.

**Run names:** New RNN runs use a **full run_id** (e.g. `two_module_rnn_50_nb1_shared_sp1_sep_cr_RNN`) so they do not overwrite legacy result folders. The run_id includes **sep** (separate input, `common_input=false`) or **ci** (common input, `common_input=true`, ablation).

**Analysis factors:** **Input scenario** = `input_routing`: `"shared"` = both modules receive the same input; `"task_routed"` = task A→module 1, task B→module 2 (different inputs per module). **Communication** = `sparsity` (inter-module connection strength, 0–1) and optionally `common_readout` (shared vs per-module readout).

**Module use (operational):** Defined from logged activity: per-module hidden magnitudes (from `hiddens_post_phase_*_per_module`, `hiddens_per_module`), and optionally trajectory when nb_steps>1. Rich/lazy is **not** assigned by init scale; see below.

**Rich/lazy (empirical):** We do not label conditions "rich" or "lazy" by initialization. Init scale (and optional recurrent rank) are experimental knobs; regime is determined **post hoc** from parameter-change norm, representation alignment (e.g. pre vs post training hiddens), and optionally kernel alignment (see e.g. arXiv:2310.08513). Keys `hiddens_pre_training` and `hiddens_post_phase_*` support these analyses.

---

## 10. Relation to Dynspec

- The **Community** model used inside `TwoModuleRNNWrapper` is the **same** dynspec-style RNN: two RNNs (core + comms) with masked weights, added hidden states, single readout (or per-module then summed). So the **internal** implementation (recurrence, communication, masks) is as in the dynspec documentation.
- The **a1b2** wrapper adapts it to the A1–B–A2 interface: **single time step** per trial, **no temporal data pipeline** (no `process_data`/temporal stacking), **fixed input 12 / output 4**, and **MSE regression** with feature probing instead of dynspec’s classification and decision (max/random over modules).
- So: **dynspec** = modular RNN + multi-step sequences + classification + decision; **a1b2 RNN** = same modular RNN, but single-step, regression, and feature-probed A1–B–A2 continual learning with the same schedule and learning logic as the FFN.

This document, together with **DYNSPEC_RNN_IMPLEMENTATION.md**, gives a complete picture of how the Holton-style continual learning task is managed in a1b2 and how the RNN variant fits in.
