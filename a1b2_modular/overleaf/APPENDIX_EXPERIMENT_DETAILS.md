# Appendix — Experimental details

This appendix documents the exact experimental setup used for all data
figures (`figures/accuperf.png`, `figures/reduceddim.png`,
`figures/reprgeomtry.png`). Every value below is read from the codebase
(`a1b2_modular/a1b2/...` and `a1b2_modular/scripts/...`) so that the
manuscript and the implementation cannot drift.

Source files referenced (relative to repository root):

* Experiment / global config:
  `a1b2_modular/a1b2/models/experiments.json`
* Simulation entry point:
  `a1b2_modular/scripts/02_run_simulations.py`,
  `a1b2_modular/a1b2/training/simulation.py`
* Training loop and three-phase schedule:
  `a1b2_modular/a1b2/training/schedule.py`
* Models: `a1b2_modular/a1b2/models/two_module_rnn.py`,
  `a1b2_modular/a1b2/models/community.py`,
  `a1b2_modular/a1b2/models/rnn_init.py`,
  `a1b2_modular/a1b2/models/ffn.py`
* Behavioral / mixture analysis:
  `a1b2_modular/scripts/03_fit_vonmises.py`,
  `a1b2_modular/a1b2/models/vonmises.py`,
  `a1b2_modular/a1b2/analysis/transfer_interference.py`

---

## A.1 Task and data generation

### A.1.1 Task layout

* **Task structure (Holton et al.).** Each task maps 6 plant cues to a
  circular dial, with a "summer" and a "winter" location for each cue
  related by a fixed angular offset (the task rule). Task B uses 6 new
  plant cues, with rule offset 0 (Same), $\pi/6$ (Near), or $\pi$ (Far)
  relative to A's rule.
* **Encoding.** Each stimulus is a one-hot input of dimension
  `dim_input = nStim_perTask * 2 = 12`
  (`a1b2_modular/a1b2/analysis/transfer_interference.py:78–84`,
  `a1b2_modular/a1b2/data/basic_funcs.py:74–82`).
* **Output.** 4-dimensional cosine–sine output encoding the summer
  $(\cos,\sin)$ pair (output channels 0–1) and the winter $(\cos,\sin)$
  pair (output channels 2–3). On each trial the MSE loss is computed
  only on the channel pair matching the probed season
  (`a1b2_modular/a1b2/training/schedule.py:84–93`).

### A.1.2 Schedules and runs per condition

* **Source schedules.** Each simulated network reuses the trial sequence
  of one human participant from
  `a1b2_modular/data/participants/trial_df.csv`. There are 305 unique
  participants in total (`awk` count on the CSV), distributed as:

  | Similarity condition | # schedules (and thus # networks per architecture / γ) |
  |---|---:|
  | Same | 103 |
  | Near | 101 |
  | Far  | 101 |

  These are the runs that show up as separate per-participant points in
  every Figure 2 / Figure 3 panel (one dot per simulated network).

* **Random seeding.** `set_seed(2024)` is called once at the start of
  `run_experiment`; this seeds NumPy, Python `random`, and PyTorch
  (`a1b2_modular/scripts/02_run_simulations.py:41`). Within a run, each
  network is then a fresh PyTorch instantiation drawn from the global
  RNG, so 101–103 networks per architecture × γ × similarity inherit
  distinct initial weights from the same global seed.

* **Number of architectures × γ × similarity.** For every entry in
  Figure 2 / 3 we trained: $2$ architectures (modular task-routed,
  single) $\times$ $5$ values of $\gamma$ $\times$ $3$ similarities,
  i.e. $30$ "cells". Each cell contains 101–103 trained networks
  (one per source schedule), giving roughly $3{,}050$ networks across
  all data figures.

### A.1.3 Trials per phase and per schedule

The training schedule for one network has three phases ($A1 \to B \to A2$).
Per phase, the dataloader iterates a fixed participant trial table for
`n_epochs = 100` epochs with `batch_size = 1` and `shuffle = false`
(`a1b2_modular/a1b2/models/experiments.json:5469–5474`,
`a1b2_modular/a1b2/training/simulation.py:74–77`).

The participant trial tables alternate summer (`feature_idx = 0`) and
winter (`feature_idx = 1`) probes by design. A direct count on
`trial_df.csv` for one example participant gives:

| Phase | Summer trials (feature_idx=0) | Winter trials (feature_idx=1) | Total per epoch |
|---|---:|---:|---:|
| A1 | 60 | 60 | 120 |
| B  | 60 | 60 | 120 |
| A2 | 60 | 60 | 120 |

Across all 100 epochs this becomes:

| Phase | Summer (probe=0) updates per network | Winter (probe=1) updates per network | Total per phase |
|---|---:|---:|---:|
| A1 | 6,000 | 6,000 | **12,000** |
| B  | 6,000 | 6,000 | **12,000** |
| A2 | 6,000 (update) | 6,000 (test only) | **12,000** |

Important detail for $A2$: in `runSchedule` the third phase is run with
`do_update = 2`, which inside `train_participant_schedule` updates
parameters only when `probe_val == 0` (summer trials)
(`a1b2_modular/a1b2/training/schedule.py:102–104, 245–247`). Winter
trials in $A2$ are still forward-passed and their predictions/accuracy
are logged, but they do not produce a gradient step. The transfer and
interference metrics are computed exclusively from these winter trials,
making them a held-out probe of retention.

---

## A.2 Training

### A.2.1 Optimizer

* **Algorithm.** PyTorch `torch.optim.SGD` with default arguments
  (`a1b2_modular/a1b2/training/schedule.py:215`):
  * **Learning rate.** `lr = 0.01`
    (`a1b2_modular/a1b2/models/experiments.json:5471`).
  * **Momentum.** `0.0` (PyTorch default).
  * **Weight decay.** `0.0` (PyTorch default).
  * **Dampening.** `0.0` (PyTorch default).
  * **Nesterov.** `False` (PyTorch default).
* **Loss.** `torch.nn.MSELoss` with PyTorch defaults
  (`a1b2_modular/a1b2/training/schedule.py:216`).
* **Gradient clipping.** None — there is no `clip_grad_norm_` /
  `clip_grad_value_` call anywhere in the training pipeline
  (verified by repository-wide search).

### A.2.2 Training schedule

* **Phases.** Three phases, run sequentially: $A1 \to B \to A2$.
* **Epochs per phase.** `n_epochs = 100` for every phase
  (`a1b2_modular/a1b2/models/experiments.json:5469`).
* **Batch size.** $1$ (every trial is its own SGD step).
* **Trial order.** `shuffle = false` — the trial order within each
  phase is the participant's original order, identical across epochs.
  This makes the entire training procedure deterministic given a
  fixed seed, fixed schedule, and fixed γ.
* **Optimizer state across phases.** The optimizer is **not** reset
  between phases. A single `optim.SGD(...)` instance is constructed
  before phase $A1$ and reused for $B$ and $A2$
  (`a1b2_modular/a1b2/training/schedule.py:215, 247–253`). Because
  momentum/weight-decay are zero, this has no effect (vanilla SGD is
  stateless), but it is documented for completeness.
* **Stopping criterion.** Fixed (`n_epochs = 100`); there is no
  early-stopping logic. Training proceeds for the full schedule
  regardless of accuracy or loss values.

### A.2.3 Initialization

The two architectures use different initialization paths:

* **Single-network baseline and modular RNN (recurrent components).**
  At construction, every layer uses **PyTorch's default initialization**
  for `nn.RNN(bias=False, ...)`, namely
  $\mathcal{U}\!\left(-\sqrt{1/h},\ \sqrt{1/h}\right)$ for both
  `weight_ih*` and `weight_hh*`, where $h$ is the per-module hidden
  size. The shared readout is a `nn.Linear(..., bias=...)` whose weights
  follow PyTorch's default Linear initialization
  $\mathcal{U}\!\left(-\sqrt{1/n_{\text{in}}},\ \sqrt{1/n_{\text{in}}}\right)$.
  No Xavier or orthogonal scheme is applied at construction
  (`a1b2_modular/a1b2/models/community.py:475–496`,
  `a1b2_modular/a1b2/models/two_module_rnn.py:102–117`).

* **Global γ rescaling.** Immediately after construction, every
  trainable parameter of the recurrent core (and the comms parametrized
  copy) is multiplied in place by a single scalar $\gamma$ via
  `apply_init_scale(network, scale=γ, scope="global")`
  (`a1b2_modular/a1b2/models/rnn_init.py:8–34`). The grid used in the
  paper figures is

  $$\gamma \in \{0.001,\ 0.01,\ 0.1,\ 1.0,\ 2.0\}.$$

  Larger γ → "lazy" regime (further from origin, smaller updates
  relative to weight magnitude); smaller γ → "rich" regime (closer to
  the origin, weights move proportionally more during training)
  (`overleaf/main.tex:238`).

* **Biases.** **No biases** are used anywhere on the data path in the
  recurrent models: `nn.RNN(bias=False)` is hard-coded in
  `community.py:475–491`. The shared readout (`Readout` layer) inherits
  the PyTorch default `bias=True` for `nn.Linear`, but this bias is the
  only learnable bias in the model, so reported "bias-free" applies to
  the recurrent dynamics specifically, not the linear readout.

* **Shared parameters between modules.** None — modules share **no**
  recurrent parameters. The masking machinery in `community.py:511–532`
  applies a block-diagonal mask to `weight_hh*` and the corresponding
  block mask to `weight_ih*`, so each module's recurrent and input
  weight matrices are independent within the larger weight tensor. The
  `comms` copy of the recurrent cell is initialized to share parameters
  with the `core` cell (`community.py:493–495`), but with a separate
  zero/sparse mask for inter-module connectivity. With sparsity = 0
  (no_comms, used for Figs. 2–4 of the paper), the comms mask is fully
  zero and only `core` is active.

* **Feedforward baseline (FFN, supplementary only).** The FFN
  `simpleLinearNet` uses `ex_initializer_`: hidden weights drawn
  $\mathcal{N}(0, \gamma)$ and readout weights drawn
  $\mathcal{N}(0, 10^{-3})$
  (`a1b2_modular/a1b2/models/ffn.py:25–33`). Both layers are
  `bias=False`. This applies only to the FFN comparator that does not
  appear in the paper figures.

### A.2.4 Stopping criteria and randomization

* **Stopping.** Fixed schedule, no early stopping (see §A.2.2).
* **Trial order randomization.** `shuffle = false`. Within an epoch the
  trials follow the original participant order; across epochs, the
  order repeats deterministically. The only stochasticity per network
  is the initial-weight draw (PyTorch default RNN init under the global
  seed, then γ-rescaling).

---

## A.3 Behavioral measures

### A.3.1 Accuracy

* **Per-trial accuracy (in $[0,1]$).** From the supervised cosine–sine
  pair we reconstruct the predicted angle
  $\hat\theta = \mathrm{atan2}(\hat y_{\cos}, \hat y_{\sin})$ and compare
  to the ground-truth angle $\theta$. The signed error is wrapped to
  $[-\pi, \pi]$ and converted to a normalized 0–1 score:

  $$
  \mathrm{acc} \;=\; 1 \;-\; \frac{|\,\mathrm{wrap}_{[-\pi,\pi]}(\hat\theta - \theta)\,|}{\pi}.
  $$

  `acc = 1` when $\hat\theta = \theta$; `acc = 0` when the prediction is
  $\pi$ off (`a1b2_modular/a1b2/models/ffn.py:54–60`).

* **Reported accuracy curves (Figure 2a/b).** The per-trial accuracy
  arrays are smoothed with a centered rolling mean (`min_periods=1`,
  window = 25 trials), then aggregated as participant-level mean ± SEM
  per (architecture, $\gamma$, similarity, phase, trial-index)
  (canonical notebook §3,
  `a1b2_modular/notebooks/paper_figures_size25.ipynb`).

### A.3.2 Transfer

The transfer score measures how much already-acquired computation
generalizes to the new task at the $A1 \to B$ boundary:

* **Window definition.** For each network (i.e. each schedule, each
  architecture, each γ, each similarity), use the **last 6 winter
  trials of $A1$** and the **first 6 winter trials of $B$**.
  "Winter-only" matters: those are the trials that are not the
  immediate target of the loss in $A2$ and that share the rule
  structure across the boundary
  (`a1b2_modular/notebooks/paper_figures_size25.ipynb`,
  cell "compute_metrics").
* **Per-network statistic.**

  $$
  \mathrm{Transfer} \;=\; \overline{\mathrm{acc}}_{B,\,\mathrm{winter},\,\text{first 6}}
  \;-\; \overline{\mathrm{acc}}_{A1,\,\mathrm{winter},\,\text{last 6}}.
  $$

  Negative values are typical (winter accuracy drops at the boundary
  because the rule has changed); less-negative values mean better
  transfer (less drop, i.e. more reuse of the existing computation).
* **Aggregation across networks.** Mean ± SEM across the 101–103
  networks per (architecture, γ, similarity) cell.

### A.3.3 Interference

Interference is measured from $A2$ winter responses by fitting a
two-component von Mises mixture model in
`a1b2_modular/a1b2/models/vonmises.py`:

* **Inputs to the fit.** For each network, the angular residuals on
  $A2$ winter trials are computed as

  $$
  \phi \;=\; \mathrm{wrap}_{[-\pi,\pi]}\!\bigl(\hat\theta_{\mathrm{winter}} - \hat\theta_{\mathrm{summer}}\bigr),
  $$

  where $\hat\theta_{\mathrm{summer}}$ and $\hat\theta_{\mathrm{winter}}$
  come from the network's summer/winter cosine–sine output channels on
  matched plant cues. This residual is by construction near the task A
  rule $\mu_A$ if behavior has not been overwritten, and near the task
  B rule $\mu_B$ if it has
  (`a1b2_modular/scripts/03_fit_vonmises.py:114–123`).
* **Mixture model.** Two von Mises components with **fixed means**
  $\mu_A$ and $\mu_B$ (the two task rules of that schedule) and a
  **shared concentration parameter** $\kappa$ that is fitted by
  L-BFGS-B; mixture weights $\pi_A, \pi_B$ with
  $\pi_A + \pi_B = 1$ are fit by EM. Initial conditions are swept over
  $\pi_A \in \{0.1, 0.2, \ldots, 0.9\}$ and
  $\kappa_0 \in \{1, 2.5, 5, 10, 15, 20\}$ (54 starts) and the
  best-likelihood fit is selected
  (`a1b2_modular/a1b2/models/vonmises.py:5–122`).

  $$
  p(\phi) \;=\; \pi_A \,\mathrm{vM}(\phi;\mu_A,\kappa) \;+\; \pi_B \,\mathrm{vM}(\phi;\mu_B,\kappa).
  $$

  $\pi_A$ is reported in the per-run `*_vonmises_fits.csv` as the
  column `A_weight_A2` (the mixture weight on the task-A component
  during phase $A2$).
* **Interference index.**

  $$
  \mathrm{Interference} \;=\; 1 - \pi_A^{(A2)} \;\equiv\; 1 - \mathrm{A\_weight\_A2}.
  $$

  Larger values mean more A2 winter responses are pulled toward the
  task-B rule, i.e. stronger overwriting / interference.
* **Conditions used.** The mixture is only meaningful when the rules
  differ, so interference is reported on the **near** and **far**
  conditions only (`canonical notebook §2`,
  `a1b2_modular/notebooks/paper_figures_size25.ipynb`).
* **Aggregation.** Mean ± SEM across the 101–103 networks per
  (architecture, γ, near/far) cell.

---

## A.4 Architecture details

### A.4.1 Sizes and parameter matching

| Architecture | Per-module hidden | # modules | Total recurrent state $h$ | Inter-module connectivity (paper main results) |
|---|---:|---:|---:|---|
| Modular (task-routed, no_comms) | 25 | 2 | 50 | None (sparsity = 0; the `comms` cell is masked to all-zero) |
| Single network | 50 | 1 | 50 | n/a |

Both architectures therefore have the same total hidden width
(`50`), same recurrent step count (`nb_steps = 2`), same readout
type (`common_readout = true` → a single `nn.Linear(50 → 4)`), and
the same input dimensionality (`12` one-hot). This is what we mean by
"size-matched" in the paper: a `dim_hidden = 25` two-module RNN has
the same total `h`-dimensionality as the `dim_hidden = 50` single
baseline (`a1b2_modular/a1b2/analysis/paper_single_baseline.py`).

**Parameter counts.** The two architectures are **state-matched but
not parameter-matched**. The dominant contributions are:

* Recurrent core (`nn.RNN(bias=False)`):
  * Modular (block-diagonal `weight_hh`, $25 \times 25$ per module
    plus block-diagonal `weight_ih`, $25 \times 12$ per module):
    $2 \cdot 25^2 + 2 \cdot 25 \cdot 12 = 1{,}250 + 600 = 1{,}850$
    trainable scalars.
  * Single (full $50 \times 50$ `weight_hh` plus $50 \times 12$
    `weight_ih`):
    $50^2 + 50 \cdot 12 = 2{,}500 + 600 = 3{,}100$ trainable scalars.
* Shared readout (`nn.Linear(50, 4)`): $200 + 4 = 204$ for both.

So the **single network has $\approx 1{,}250$ more trainable
parameters** than the modular task-routed RNN (extra off-diagonal
recurrent connections that are zeroed out in the modular case). We
report this as "size-matched" rather than "param-matched" in the
caption of Figure 2 if relevant. If a strictly param-matched
comparison is desired, the modular architecture is the parameter
ceiling and a smaller single network ($\approx h = 41$ for matched
recurrent count) would be required; this is not used in the paper.

### A.4.2 Biases and weight sharing

* **Recurrent core:** `nn.RNN(..., bias=False)` for both architectures
  (`community.py:475–482`). No bias on `weight_ih` or `weight_hh`.
* **Readout:** `nn.Linear(50, 4)` with PyTorch default `bias=True`.
  This is the **only learnable bias on the data path** (it is shared
  across the two task outputs).
* **Sharing across modules:** No parameters are shared across modules.
  The `weight_hh` mask (`state_mask` in `community.py`) is strictly
  block-diagonal, so each module has its own $25 \times 25$ recurrent
  matrix and $25 \times 12$ input projection (the latter is
  task-routed in the modular case: only the relevant module receives
  input on a given trial, see §A.4.3). The two modules' parameters
  are initialized independently (PyTorch default RNN init), then
  globally rescaled by γ.

### A.4.3 Task-routed input

In the modular architecture, the input on trial $t$ is routed to the
module corresponding to the currently probed feature
(`feature_probe = 0` → module 0, `feature_probe = 1` → module 1)
(`two_module_rnn.py:163–175`). Concretely, the input tensor expanded
to $(\text{seq}, \text{batch}, \text{input\_size} \cdot n_{\text{modules}})$
zeroes out the input slice for the non-active module. This is what we
mean by "task partitioning at the input": no parameters are masked
between modules; instead, on a given trial one module simply sees no
input.

### A.4.4 Recurrent depth and time-unroll

* `n_layers = 1` for the recurrent cell (`community.py`,
  `experiments.json` paper conditions).
* `nb_steps = 2` time-unroll for paper conditions (the input is
  repeated for 2 time steps; the loss is taken at the last step,
  see `a1b2_modular/a1b2/data/temporal.py:67–92` and
  `schedule.py:75–80`).

### A.4.5 No inter-module recurrent connectivity (main results)

For all paper data figures (Figs. 2, 3, 4) the modular condition is
`task_routed_no_comms_nb2`, which has `sparsity = 0`. With
`sparsity = 0`, `comms_mask` in `community.py:115–148` is the all-zero
matrix, so the `comms` cell — the only object that ever carried
inter-module connectivity in the original `Community` design — has its
`weight_hh` block-zeroed and contributes zero to the forward pass:

$$
h_{t+1} \;=\; W_{\mathrm{core}}^{\mathrm{rec}}\,h_t \;+\; W_{\mathrm{core}}^{\mathrm{in}}\,x_t,
\qquad W_{\mathrm{core}}^{\mathrm{rec}} = \mathrm{block\text{-}diag}(W_A^{\mathrm{rec}}, W_B^{\mathrm{rec}}).
$$

**Ablations not reported in the main paper figures.** These are
documented in §A.5 below.

---

## A.5 Ablations

All ablations re-use the §A.1–§A.3 protocol exactly (same trial
schedules, same γ grid where applicable, same SGD lr=0.01, same 100
epochs per phase, same MSE loss, same von Mises mixture analysis).
Only the architectural lever named in each subsection changes. Run
folders below correspond to the actual contents of
`a1b2_modular/data/simulations/`.

### A.5.1 Module-size sweep (matched single baselines)

We ran the paper's main "task-routed, no-comms modular" architecture
at four per-module hidden sizes, each with its size-matched single
network. The single network is sized so that its total state width
equals two modules:

| Per-module hidden $h_m$ | Modular total state $2 h_m$ | Matched single $h_s$ | Modular run prefix | Single run prefix |
|---:|---:|---:|---|---|
| 6  | 12  | 12  | `two_module_rnn_6_task_routed_no_comms_nb2*`  | `single_module_rnn_12_nb2*` |
| 12 | 24  | 24  | `two_module_rnn_12_task_routed_no_comms_nb2*` | `single_module_rnn_25_nb2*` ($h_s=25$ chosen as the smallest available size matching $\geq 2h_m$; see caveat below) |
| 25 | 50  | 50  | `two_module_rnn_25_task_routed_no_comms_nb2*` (paper main) | `single_module_rnn_50_nb2*` (paper main) |
| 50 | 100 | 100 | `two_module_rnn_50_task_routed_no_comms_nb2*` | `single_module_rnn_100_nb2*` |

For each size, both the modular and the single variant were trained
across the same five-point γ grid
$\{0.001,\ 0.01,\ 0.1,\ 1.0,\ 2.0\}$. The $h_m=25$ row is the only
combination shown in Figs. 2–4 of the main paper; the other rows
constitute the **size-sweep ablation** referenced in the
manuscript's discussion of size-dependence.

**Caveat: $h_m=12$ row.** The cleanest matched single baseline for two
modules of size 12 would be $h_s=24$, but no `single_module_rnn_24`
runs exist; we use the closest available width, $h_s=25$. The resulting
2-parameter mismatch (~50 extra recurrent scalars) is much smaller than
the modular vs. single off-diagonal-block gap (~$h_m^2$ scalars), so
the comparison remains state-matched within ~4 %.

**Curated notebooks for the size sweep.**
`a1b2_modular/notebooks_final/paper_figures_size{6,12,25,50}_no_comms_architecture_comparison.ipynb`
each reproduce the Fig. 2 / Fig. 3 panels for one size; the combined
view appears in
`paper_figures_sizes6_12_25_no_comms_architecture_comparison.ipynb`
and `paper_figures_size_comparison.ipynb`.

### A.5.2 Shared-modular network (block-diagonal recurrent, shared input)

The "shared modular" variant in the manuscript has the **same
block-diagonal recurrent structure** as the main task-routed modular
network — i.e. two recurrent populations with no inter-module
recurrent connections (`sparsity = 0`) — but **does not partition the
input by task**. Both modules receive the full one-hot stimulus on
every trial, so any partitioning between modules has to emerge from
training rather than being imposed at the input. The shared readout
then combines the two module states identically to the task-routed
case.

| Lever | Paper main (modular) | Shared modular (this ablation) |
|---|---|---|
| Architecture name | `two_module_rnn_<h_m>_task_routed_no_comms_nb2*` | `two_module_rnn_<h_m>_no_comms_nb2*` |
| `input_routing` (`experiments.json`) | `"task_routed"` | `"shared"` (default) |
| `sparsity` | $0$ | $0$ |
| Recurrent connectivity | block-diagonal (no inter-module comms) | block-diagonal (no inter-module comms) |
| Input pathway | gated by `feature_probe` (one module sees input per trial) | full one-hot delivered to both modules |
| Readout | shared `nn.Linear(2h_m, 4)` | shared `nn.Linear(2h_m, 4)` |
| γ grid | $\{0.001, 0.01, 0.1, 1.0, 2.0\}$ | $\{0.001, 0.01, 0.1, 1.0, 2.0\}$ |
| Similarity conditions | same / near / far | same / near / far |

Concretely, the only line that changes between the two architectures
is `condition["input_routing"]`. In code, when
`input_routing == "shared"` the wrapper repeats the same
$x_t \in \mathbb{R}^{12}$ across both modules
(`two_module_rnn.py:127`); when `task_routed`, the wrapper uses
`feature_probe` to zero-out the inactive module's input slice
(`two_module_rnn.py:163–175`).

**Run folders.** The shared-modular variant exists for all four module
sizes:

* $h_m = 6$: `two_module_rnn_6_no_comms_nb2_*_nb2_shared_sp0_sep_cr_RNN_*`
* $h_m = 12$: `two_module_rnn_12_no_comms_nb2_*_nb2_shared_sp0_sep_cr_RNN_*`
* $h_m = 25$: `two_module_rnn_25_no_comms_nb2_*_nb2_shared_sp0_sep_cr_RNN_*`
* $h_m = 50$: `two_module_rnn_50_no_comms_nb2_*_nb2_shared_sp0_sep_cr_RNN_*`

each at the five-point γ grid.

**What this ablation isolates.** Holding recurrent connectivity and
total state width constant while toggling the input from `task_routed`
to `shared` removes the architectural prior that "task A flows into
module $M_A$ only". Any difference between the two architectures in
transfer/interference or representational geometry can therefore be
attributed to the **input gating**, not to the recurrent structure or
parameter count. This is the comparison that appears in
`a1b2_modular/notebooks_final/no_comms_architecture_comparison.ipynb`
and the per-size architecture-comparison notebooks under the labels
"shared" vs. "task_routed".

### A.5.3 Inter-module sparsity sweep (size 25 only)

For the $h_m = 25$ modular network we also varied the inter-module
recurrent sparsity. The sparsity parameter controls the density of the
off-diagonal blocks of the `comms` cell's `weight_hh` (see §A.4.5):
`sparsity = 0` zeroes all comms; `sparsity = 1.0` lets every
inter-module connection exist; intermediate values produce sparse
random binary masks via `sparse_mask` (`community.py:84–112`). The
recurrent core remains block-diagonal in all cases — comms are an
*additive* contribution from a separate parametrized cell, not a
modification of the core diagonal blocks.

| Sparsity | Run prefix (shared input) | Run prefix (task-routed input) | γ grid |
|---:|---|---|---|
| 0.0 (`no_comms`)   | `two_module_rnn_25_no_comms_nb2*` (≡ shared modular, §A.5.2) | `two_module_rnn_25_task_routed_no_comms_nb2*` (paper main) | $\{0.001, 0.01, 0.1, 0.3, 1.0, 2.0\}$$^\dagger$ |
| 0.3 (`low_sparse`) | `two_module_rnn_25_low_sparse_nb2*`        | `two_module_rnn_25_task_routed_low_sparse_nb2*`        | partial, $\{0.3, 0.9\}$ available |
| 0.5 (`sp05`)       | `two_module_rnn_25_sp05_nb2*`              | `two_module_rnn_25_task_routed_sp05_nb2*`              | $\{0.001, 0.01, 0.1, 0.3, 1.0\}$ |
| 0.7 (`sp07`)       | `two_module_rnn_25_sp07_nb2*`              | `two_module_rnn_25_task_routed_sp07_nb2*`              | $\{0.3, 0.9, 1.0\}$ |
| 0.9 (`sp09`)       | `two_module_rnn_25_sp09_nb2*`              | `two_module_rnn_25_task_routed_sp09_nb2*`              | partial, $\{0.3, 0.9\}$ |
| 1.0 (full)         | `two_module_rnn_25_nb2*`                   | `two_module_rnn_25_task_routed_nb2*`                   | $\{0.3, 0.9, 1.0\}$ |

$^\dagger$ The `no_comms` and `task_routed_no_comms` cells additionally
include $\gamma = 0.3$ — used as a fine-grained mid-rich sample in the
sparsity sweep — beyond the standard five-point grid.

This grid spans the range from "no inter-module talk at all"
(`sparsity=0`, paper main) to "fully connected modular" (`sparsity=1`,
which behaves like a single network with a block-structured initial
weight pattern that decays during training).

### A.5.4 Init-scope ablation: input-pathway scaling only

By default γ rescales **all** trainable parameters of the recurrent
core after construction (`apply_init_scale(scope="global")`,
§A.2.3). A complementary ablation rescales **only** the input-to-hidden
matrices (`weight_ih*`), leaving the recurrent connectivity
`weight_hh*` and the readout at their default-PyTorch scale. The
relevant code path is the `scope="input_only"` branch in
`a1b2_modular/a1b2/models/rnn_init.py:31–34`.

* Conditions are tagged with the suffix `_input_only` in
  `experiments.json`. The corresponding run folders carry an
  `_initscopeinput_only` token in their `run_id`.
* Sizes covered: $h_m = 25$ only.
* Architectures covered: `task_routed_no_comms`, `no_comms`,
  `task_routed_sp05`, `sp05`, full-sparsity (`two_module_rnn_25_nb2`),
  `task_routed`.
* γ values covered: $\{0.001, 0.01, 0.1\}$ (the rich half of the grid;
  the lazy half $\{1.0, 2.0\}$ was not run because it would coincide
  with the global-scope variant in the relevant rich-vs-lazy comparison).

This ablation isolates whether the rich/lazy effect on representational
geometry comes from rescaling the input pathway or the recurrent
dynamics. It is reported in
`a1b2_modular/notebooks/nb_paper_figures_init_scope_ablation.ipynb`
and its `tests/` siblings.

### A.5.5 Single-network depth, cell-type, and dropout

Variants of the size-50 single network were trained to confirm the
result is not specific to a 1-layer vanilla RNN. All variants use the
same γ grid $\{0.001, 0.01, 0.1, 1.0, 2.0\}$ and the same training
schedule.

| Lever | Values run | Conditions in `experiments.json` |
|---|---|---|
| `n_layers` | 1, 2, 3 | `single_module_rnn_50_nb2`, `*_nl2`, `*_nl3` (and per-γ variants) |
| `dropout`  | 0.0, 0.1 | `*_nl2_drop0.1`, `*_nl3_drop0.1` |
| `cell_type` | `RNN`, `GRU` | `single_module_rnn_50_nb2_init0.001_gru` |
| Extra γ values | $0.0001$ | `single_module_rnn_50_nb2_init0.0001` |

Equivalent depth/cell ablations exist for the size-25 single network
(`single_module_rnn_25_nb2_*`) and for the modular network
(`two_module_rnn_25_task_routed_no_comms_nb2_init0.001_{nl2,nl3,gru,nl2_drop0.1,nl3_drop0.1}`).

### A.5.6 Common input / common readout ablations (legacy, nb_steps=1)

Earlier `nb_steps=1` runs additionally tested:

* `common_input = true` — input fan-out is replicated to all modules
  rather than masked block-diagonally (entries
  `two_module_rnn_50_ablation_common_input` and the task-routed
  counterpart).
* `common_readout = false` (suffix `_sep_readout` /
  `_sep_readout_nb2`) — each module has its own
  `nn.Linear(h_m, 4)` readout that is summed at the output, instead of
  a shared $\mathbb{R}^{2h_m \to 4}$ map.

These were used during early architecture validation but are not part
of the main figure pipeline.

### A.5.7 Summary of ablation coverage

The full ablation matrix realized on disk
(`a1b2_modular/data/simulations/`) contains:

* **Module-size sweep** at 4 sizes × 2 input routings × 5 γ values =
  40 (modular, no_comms) cells, plus 4 sizes × 5 γ values = 20 single
  baselines. Each cell has 101–103 networks (one per source schedule),
  identical to the main figure.
* **Sparsity sweep** at $h_m=25$, 6 sparsity levels × 2 input
  routings × up to 6 γ values, with 101–103 networks per cell.
* **Init-scope ablation** at $h_m=25$, 6 modular variants × 3 γ values.
* **Depth/cell-type/dropout** at $h_m \in \{25, 50\}$ for both modular
  and single, ≤3 γ values per variant.

For every ablation, the analysis pipeline is identical to the main
paper:
$\textsf{run\_simulation} \to \textsf{transfer\_interference}$ for
accuracy / transfer / `n_pcs_99` / principal angles, plus
`scripts/03_fit_vonmises.py` for the per-network mixture fits used to
compute interference.
