# How dynspec Implements the RNN: A Detailed Explanation

This document explains how the **dynspec** package implements the recurrent neural network (RNN) in the **Community** model: architecture, input/output handling, weight masking, learning, and how it connects to data and training.

---

## 1. Overview: The Community Architecture

The main RNN is the **Community** model: a modular RNN where multiple “modules” (recurrent sub-networks) interact via **recurrent (intra-module)** and **communication (inter-module)** connections. The implementation uses PyTorch’s `nn.RNN` or `nn.GRU` and applies **masks** to their weight matrices so that:

- **Core RNN**: only **recurrent** (within-module) connections are non-zero.
- **Comms RNN**: only **inter-module** (communication) connections are non-zero.

Both stacks share the same **input → hidden** weights and are run in parallel; their outputs are **added** to get the combined hidden state, which is then passed through a **Readout** to produce task outputs.

```
                    ┌─────────────────────────────────────────────────────────┐
  input (T,B,D_in)  │  core RNN (masked: recurrence only)  →  core_out         │
       ────────────►│  comms RNN (masked: communication only) → comms_out      │
                    │  all_states = core_out + comms_out                        │
                    └─────────────────────────────────────────────────────────┘
                                                        │
                                                        ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │  Readout(all_states)  →  outputs (per module or shared)   │
                    └─────────────────────────────────────────────────────────┘
```

Below we go through each part with code and tensor shapes.

---

## 2. Input Handling

### 2.1 Data pipeline: from samples to sequences

Training/test data are processed by `process_data()` so the model receives **sequences** suitable for an RNN (`batch_first=False`).

**File: `dynspec/data_process.py`**

```python
def process_data(data, data_config):
    start_times = None
    nb_steps, noise_ratio, random_start = (
        data_config["nb_steps"],
        data_config["noise_ratio"],
        data_config["random_start"],
    )
    data, start_times = temporal_data(
        data,
        nb_steps=nb_steps,
        noise_ratio=noise_ratio,
        random_start=random_start,
    )

    data = data.transpose(1, 2)
    data = data.reshape(data.shape[0], data.shape[1], -1)

    return data, start_times
```

- **`temporal_data()`** turns each sample into a **sequence** of `nb_steps` time steps (optionally with noise and random start). It can duplicate the same content across time or add temporal noise.
- After **`transpose(1, 2)`** and **`reshape(..., -1)`**, the batch has shape:
  - **`(T, B, D)`** with `T = nb_steps`, `B = batch size`, `D = total input dimension` (flattened, e.g. `input_size * n_modules` if data is already split per module, or a single `input_size` that the model then interprets).

So the **RNN input** is always **3D**: `(seq_len, batch, input_size_effective)`.

### 2.2 What the Community expects as input

The **Community** model is built with an effective input dimension that matches this flattened layout. From `models.py`:

```python
self.core = cell_types_dict[self.cell_type](
    input_size=self.input_size * self.n_modules,
    hidden_size=self.hidden_size * self.n_modules,
    num_layers=self.n_layers,
    batch_first=False,
    ...
)
```

So the RNN core (and comms) expect:

- **Input**: `(T, B, input_size * n_modules)`  
  So each time step has a vector of size `input_size * n_modules`. If `common_input` is True, the same `input_size`-dimensional input is typically repeated/copied for each module before being passed in; the **input mask** (see below) then controls how each module sees it.

- **Batch dimension**: middle (B).
- **Time dimension**: first (T); i.e. **`batch_first=False`**.

So end-to-end:

1. Data loader gives batches of samples (e.g. images/digits).
2. `process_data()` turns them into sequences of shape `(T, B, D)`.
3. Community’s `forward(input)` receives that `(T, B, D)` with `D = input_size * n_modules` (or equivalent after reshaping in the data pipeline).

---

## 3. Core and Comms: Two RNNs With Masked Weights

The Community builds **two** RNNs with the **same layout** (same `input_size * n_modules`, `hidden_size * n_modules`, `n_layers`), then **copies** the core’s parameters into the comms and applies **different masks** so they implement different connectivity.

### 3.1 Creation and parameter sharing

**File: `dynspec/models.py`**

```python
self.core = cell_types_dict[self.cell_type](
    input_size=self.input_size * self.n_modules,
    hidden_size=self.hidden_size * self.n_modules,
    num_layers=self.n_layers,
    batch_first=False,
    bias=False,
    dropout=self.dropout,
)

self.comms = cell_types_dict[self.cell_type](
    input_size=self.input_size * self.n_modules,
    hidden_size=self.hidden_size * self.n_modules,
    num_layers=self.n_layers,
    batch_first=False,
    bias=False,
    dropout=self.dropout,
)

for n, p in self.core.named_parameters():
    getattr(self.comms, n).data = p.data
```

- **`cell_type`** is typically `"RNN"` or `"GRU"` (from `cell_types_dict`).
- **No bias** in the RNN/GRU (bias=False).
- **Parameter sharing**: after creation, **comms** is initialized from **core** by copying `.data`. So both start with the same weights; only the **parametrizations (masks)** applied later differ, so gradient updates affect the same underlying parameters, but only the masked parts are non-zero and used in the forward pass.

### 3.2 Masks: recurrence vs communication

Masks are built so that the **full** `(n_modules * hidden_size)`-dimensional state is interpreted as one vector per module (size `hidden_size`), and connections are restricted to either **within** a module (recurrence) or **between** modules (communication).

**State mask (recurrence only, block-diagonal):**

```python
def state_mask(n_modules, n_0, n_1, gru=False):
    mask = torch.eye(n_modules)
    mask = mask.repeat_interleave(n_0, 0).repeat_interleave(n_1, 1)
    if gru:
        mask = torch.concat([m for m in mask.unsqueeze(0).repeat_interleave(3, 0)])
    return mask
```

This gives a block-diagonal matrix: module `i` only receives from module `i`. For GRU, the same block structure is repeated 3 times (reset, update, new gate).

**Communication mask (inter-module, sparse):**

```python
def comms_mask(sparsity, n_modules, n_hidden, gru=False):
    comms_mask = torch.zeros((n_modules * n_hidden, n_modules * n_hidden))
    rec_mask = torch.zeros((n_modules * n_hidden, n_modules * n_hidden))

    for i, j in product(range(n_modules), repeat=2):
        if i != j:
            comms_mask[
                i * n_hidden : (i + 1) * n_hidden, j * n_hidden : (j + 1) * n_hidden
            ] = sparse_mask(sparsity, n_hidden, n_hidden)
        else:
            rec_mask[
                i * n_hidden : (i + 1) * n_hidden, j * n_hidden : (j + 1) * n_hidden
            ] = 1 - torch.eye(n_hidden)

    masks = [comms_mask, rec_mask]
    ...
```

- **`comms_mask`**: non-zero only for **off-diagonal** blocks (i ≠ j), with a **sparse** pattern inside each block (`sparse_mask(sparsity, n_hidden, n_hidden)`).
- **`rec_mask`**: non-zero only for **diagonal** blocks, and **off-diagonal within** each block (i.e. `1 - torch.eye(n_hidden)`), so recurrent self-connections within a module.

### 3.3 Applying masks via parametrization

Weights are masked using PyTorch’s **parametrization** API (`register_parametrization`). Every time the layer uses its `.weight`, the parametrization’s `forward()` is applied (element-wise product with the mask).

**Core RNN:**

```python
for n in dict(self.core.named_parameters()).copy().keys():
    if "weight_hh" in n:
        rpm(self.core, n, Masked_weight(self.rec_mask))
    elif "weight_ih" in n and n[-1] == "0":
        rpm(self.core, n, Masked_weight(self.input_mask))
    elif "weight_ih" in n and n[-1] != "0":
        rpm(self.core, n, Masked_weight(self.state_mask))
```

- **`weight_hh_*`** (hidden-to-hidden): only **recurrent** connections (`rec_mask`).
- **`weight_ih_0`** (first layer input): **input mask** (`input_mask`), which either routes a common input to all modules or keeps module-specific input routing.
- **`weight_ih_*`** for higher layers: **state mask** so only intra-module recurrence.

**Comms RNN:**

```python
for n in dict(self.comms.named_parameters()).copy().keys():
    if "weight_hh" in n:
        if n[-1] == str(self.n_layers - 1):
            rpm(self.comms, n, Masked_weight(self.comms_mask))
        else:
            rpm(self.comms, n, Masked_weight(torch.zeros_like(self.comms_mask)))
    elif "weight_ih" in n and n[-1] == "0":
        rpm(self.comms, n, Masked_weight(self.input_mask))
    elif "weight_ih" in n and n[-1] != "0":
        rpm(self.comms, n, Masked_weight(self.state_mask))
```

- **Last layer `weight_hh`**: **comms_mask** only (inter-module).
- **Other layers `weight_hh`**: **zero mask** — so only the **last** layer of the comms RNN carries inter-module connectivity; lower layers match core (recurrent) structure.
- Input and state **input** weights: same as core (`input_mask`, `state_mask`).

**Masked_weight:**

```python
class Masked_weight(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, W):
        W = W * self.mask
        return W
```

So the **effective** weight in the forward pass is always `W * mask`. Gradients flow only through the unmasked (non-zero) entries of the mask.

### 3.4 Binary communication (optional)

If `connections_config["binary"]` is True, the comms output is passed through a spike-like nonlinearity with a **surrogate gradient** so it remains differentiable:

**File: `dynspec/models.py`**

```python
class BinaryComms(nn.Module):
    def forward(self, input):
        out = super_spike(self.comms(input)[0])
        return out
```

**File: `dynspec/surrogate.py`**

```python
# Forward: step function (0 or 1)
out[input > thr] = 1.0
# Backward: surrogate gradient (fast sigmoid–style)
grad = grad_input / (SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2
```

So the **comms pathway** can be made binary in the forward pass while still allowing gradients in the backward pass.

---

## 4. Forward Pass: Combining Core and Comms

**File: `dynspec/models.py`**

```python
def forward(self, input):
    if "Cell" in self.cell_type:
        all_states, states = [], None
        for t, t_input in enumerate(input):
            states = self.core(t_input, states) + self.comms(t_input, states)
            all_states.append(states)
        all_states = torch.stack(all_states)
    else:
        core_out, comms_out = self.core(input), self.comms(input)
        all_states, final_states = (
            core_out[0] + comms_out[0],
            core_out[1] + comms_out[1],
        )

    outputs = self.readout(all_states)
    return outputs, all_states
```

- **Standard RNN/GRU** (not Cell):  
  - `self.core(input)` and `self.comms(input)` return the usual RNN tuple `(output_sequence, final_hidden)`.  
  - **Outputs** and **hidden states** are added: `core_out[0] + comms_out[0]`, `core_out[1] + comms_out[1]`.  
  - So at every time step, the **effective hidden state** is **recurrent state + communication state**.

- **Cell** variant: manual loop over time; same idea: `states = core(t_input, states) + comms(t_input, states)`.

- **`all_states`** is then the **full sequence** of combined hidden states, shape `(T, B, n_modules * hidden_size)`.

- **Readout**: `outputs = self.readout(all_states)`. So the readout gets the **whole trajectory** and produces task outputs (e.g. logits per time step or after a temporal decision; see below).

So:

- **Input**: `(T, B, input_size * n_modules)`  
- **Hidden state (combined)**: `(T, B, n_modules * hidden_size)`  
- **Outputs**: whatever the Readout returns (e.g. `(T, B, n_modules, output_size)` or `(B, output_size)` after temporal/decision steps).

---

## 5. Readout: From Hidden States to Task Outputs

The **Readout** maps the concatenated hidden state of all modules to task outputs (e.g. class logits). It can be **per-module** or **common**, and single- or multi-layer.

**File: `dynspec/models.py`**

```python
class Readout(nn.Module):
    def __init__(self, readout_config, n_modules, ag_hidden_size, out_masks=None):
        ...
        self.common_readout = readout_config.get("common_readout", False)
        self.output_size = readout_config["output_size"]
        ...
        self.layers = self.create_readout(self.output_size, self.n_hid)
        reccursive_rpm(self.layers, self.out_masks)
```

- **Input to readout**: tensor of shape `(..., n_modules * ag_hidden_size)` (e.g. `(T, B, n_modules * hidden_size)` when called with `all_states`).
- **Single linear case**:  
  - `nn.Linear(n_modules * ag_hidden_size, output_size * n_modules)` or `output_size` if common.  
- **With hidden layer** (`n_hid`):  
  - Sequential: Linear → ReLU → Linear, with dimensions set so that per-module vs common is respected.

**Forward:**

```python
def forward(self, input):
    return self.reccursive_readout(input, self.layers, self.output_size)
```

If not `common_readout`, the output is split into `n_modules` chunks and stacked so you get a tensor with a **module dimension** (e.g. `(T, B, n_modules, output_size)`), which is then used by the **decision** logic (e.g. which module “decides” or how to combine them).

So:

- **Readout input**: `all_states` — `(T, B, n_modules * hidden_size)`.  
- **Readout output**: either one vector per sample (common readout) or one vector per module per sample (and per time if T is preserved), which is then passed to the loss after **temporal** and **module** decision (see below).

---

## 6. Output and Decision Handling

The training loop does not use the raw readout output directly; it applies **temporal** and **module** decisions so that the loss is computed on a single prediction per sample (or per task).

### 6.1 From model output to loss

**File: `dynspec/training.py`**

```python
output, _ = model(data)
if decision is not None:
    output, deciding_ags = get_decision(output, *decision)
    both = decision[1] == "both"
else:
    deciding_ags = None
    both = False
...
complete_loss = get_loss(output, t_target, use_both=both)
loss = nested_mean(complete_loss)
```

So:

1. **`model(data)`** returns `(outputs, all_states)`.
2. **`get_decision(output, temporal_decision, module_decision)`** reduces the readout output (which may be `(T, n_modules, B, output_size)` or similar) to a single prediction per sample (or two for “both” tasks).
3. **`get_loss(output, t_target, ...)`** computes the loss (e.g. cross-entropy) between this decision and the task target.

### 6.2 Temporal decision

**File: `dynspec/decision.py`**

```python
def get_temporal_decision(outputs, temporal_decision):
    ...
    if temporal_decision == "last":
        outputs = outputs[-1]
    elif temporal_decision == "sum":
        outputs = torch.sum(outputs, axis=0)
    elif temporal_decision == "mean":
        outputs = torch.mean(outputs, axis=0)
```

So the **time** dimension is collapsed to one vector per sample (e.g. use last step, or mean/sum over time). Default in `get_decision` is `"last"`.

### 6.3 Module decision

**File: `dynspec/decision.py`**

```python
def get_module_decision(outputs, module_decision):
    ...
    if module_decision == "max":
        outputs, deciding_ags = max_decision(outputs)   # pick module with max confidence
    elif module_decision == "random":
        outputs, deciding_ags = random_decision(outputs)
    elif module_decision == "sum":
        outputs = outputs.sum(0)
```

So after temporal decision, the **module** dimension is reduced (e.g. take the module with maximum confidence, or sum, or random). Default is `"max"`.

So end-to-end:

- **Model output**: readout over time and modules.  
- **Temporal decision**: e.g. last time step → `(B, n_modules, output_size)` or `(B, output_size)`.  
- **Module decision**: e.g. max → `(B, output_size)` and optionally `deciding_ags`.  
- **Loss**: cross-entropy between this `(B, output_size)` and `t_target`.

---

## 7. Learning: Loss, Optimizer, and Gradients

### 7.1 Loss

**File: `dynspec/training.py`**

```python
def get_loss(output, t_target, use_both=False):
    ...
    try:
        loss = F.cross_entropy(output, t_target, reduction="none")
    except (TypeError, RuntimeError) as _:
        loss = [get_loss(o, t) for o, t in zip(output, t_target)]
```

So the task is treated as **classification**: `output` is logits, `t_target` is class indices. For multi-task (“both”), loss can be a list and then combined (e.g. via `nested_mean`).

### 7.2 Optimizer and initialization

**File: `dynspec/models.py`**

```python
def init_model(config, device=torch.device("cpu")):
    ...
    model = Community(config).to(device)
    gamma = config["optim"].pop("gamma", None)
    optimizer = torch.optim.AdamW(model.parameters(), **config["optim"])
    if gamma:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    else:
        scheduler = None
    return model, optimizer, scheduler
```

- All **Community** parameters (core, comms, readout) are trained with **AdamW**.
- Because comms weights were **copied from** core and then only **masked** (via parametrization), both core and comms share the same underlying tensors; the masks only zero out entries. So **one set of parameters** is updated, but the **effective** weights used in the forward pass are masked. That implies:
  - Recurrent connections (core) and communication connections (comms) are **trained together** from the same parameter store, with gradients only flowing through the unmasked elements.

### 7.3 Training step

**File: `dynspec/training.py`**

```python
optimizer.zero_grad()
output, _ = model(data)
...
output, deciding_ags = get_decision(output, *decision)
complete_loss = get_loss(output, t_target, use_both=both)
loss = nested_mean(complete_loss)
...
loss.backward()
optimizer.step()
```

Standard supervised loop: forward → decision → loss → backward → step. No special handling for the RNN beyond using the masked parametrizations.

---

## 8. Weights Summary

| Component        | Parameter name (concept) | Mask / role |
|-----------------|---------------------------|-------------|
| Core RNN        | `weight_ih_0`             | Input → hidden (first layer): `input_mask` (module-specific or common input routing). |
| Core RNN        | `weight_ih_*` (layers >0) | Previous hidden → current hidden (between layers): `state_mask` (block-diagonal, intra-module only). |
| Core RNN        | `weight_hh_*`             | Hidden → hidden (recurrence): `rec_mask` (block-diagonal, no inter-module). |
| Comms RNN       | Same as core initially    | Copied from core. |
| Comms RNN       | `weight_ih_*`             | Same masks as core (input and state). |
| Comms RNN       | `weight_hh_*` (last layer)| **comms_mask** (inter-module, sparse). |
| Comms RNN       | `weight_hh_*` (other)     | Zero mask (no effect). |
| Readout         | `layers`                  | Optional per-module/readout masks via `reccursive_rpm`. |

So:

- **Recurrence**: core’s `weight_hh` with `rec_mask`.  
- **Communication**: comms’ last-layer `weight_hh` with `comms_mask`.  
- **Input routing**: both use `input_mask` and (for higher layers) `state_mask`.  
- **Learning**: one optimizer over all parameters; masks are fixed (buffers), so only the unmasked weights receive gradients and are updated.

---

## 9. VanillaRNN (alternative implementation)

dynspec also defines a **VanillaRNN** that implements a simple tanh RNN **by hand** (no PyTorch RNN/GRU):

**File: `dynspec/models.py`**

```python
class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        ...
        self.weights_ih = nn.ParameterList([...])  # input → hidden per layer
        self.weights_hh = nn.ParameterList([...])  # hidden → hidden per layer
        self.bias_ih = nn.ParameterList([...])
        self.bias_hh = nn.ParameterList([...])

    def forward(self, x):
        h = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        for i in range(self.num_layers):
            h[i] = torch.tanh(
                torch.mm(x, self.weights_ih[i]) + self.bias_ih[i]
                + torch.mm(h[i - 1], self.weights_hh[i]) + self.bias_hh[i]
            )
            h[i] = self.dropout(h[i])
        return h[-1]
```

This is a classic **tanh RNN** with one hidden vector per layer, dropout, and Xavier-style init in `reset_parameters()`. The **Community** model does **not** use this class; it uses `nn.RNN` / `nn.GRU` with the masking described above. VanillaRNN is available in the codebase for reference or alternative experiments.

---

## 10. End-to-end Data and Tensor Shapes (concise)

| Stage            | Shape / meaning |
|------------------|------------------|
| Raw batch        | Depends on dataset (e.g. `(B, C, H, W)` or `(B, ...)`). |
| After `process_data` | `(T, B, D)` with `D` flattened to match `input_size * n_modules`. |
| Core / Comms in  | `(T, B, input_size * n_modules)`. |
| Core / Comms out | `(T, B, n_modules * hidden_size)` each; added to get `all_states`. |
| Readout in       | `all_states`: `(T, B, n_modules * hidden_size)`. |
| Readout out      | E.g. `(T, B, n_modules, output_size)` or `(T, B, output_size)`. |
| After temporal   | E.g. `(B, n_modules, output_size)`. |
| After module     | `(B, output_size)` (and optionally `deciding_ags`). |
| Loss             | Cross-entropy between `(B, output_size)` and `t_target`. |

This should give you a complete, implementation-level picture of how dynspec implements the RNN (Community), including input/output handling, masking, learning, and weights.
