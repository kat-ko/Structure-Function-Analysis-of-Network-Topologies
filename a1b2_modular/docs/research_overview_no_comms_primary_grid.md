# Modular recurrent networks on a Holton-style transfer–interference task: primary grid (no communication)

This document summarizes the **a1b2_modular** line of work for external readers. The narrative focuses on the **primary experimental grid** under the **no-communication** regime, contrasting a **task-routed two-module RNN** with a **single-module baseline**, and emphasizes how **initialization scale** and **task similarity** jointly shape **representational geometry**. Communication-sparse variants exist in the codebase but are **out of scope** here.

---

## 1. Scientific aim

We study how **architecture** and **parameter scaling at initialization** interact with **task similarity** (same, near, far rule relationships between tasks A and B) to produce different **hidden-state geometry** and **behavior** in a sequential learning paradigm modeled after Holton et al.’s transfer–interference design. A central empirical theme is that **init_scale** acts as a knob on **effective dimensionality** of representations (e.g., number of principal components needed to explain most variance) and on **alignment between task-specific subspaces** (principal angles), while **hidden width** is a secondary axis whose effects on standard behavioral metrics appear comparatively modest in early sweeps (detailed factorial analysis still pending).

---

## 2. Task and behavioral protocol (high level)

Participants (human data) and models are trained on a structured **A1 → B → A2** schedule:

- **A1**: learn task A (summer and winter mappings for a set of stimuli).
- **B**: learn task B, which stands in a controlled **similarity** relationship to A (**same**, **near**, or **far**).
- **A2**: retest on task A with **asymmetric feedback**: updates follow the Holton-style rule that **summer** trials receive gradient updates while **winter** trials do not (analogous to withheld winter feedback for certain stimuli in the human protocol).

This asymmetry is important methodologically: **A2** can show **selective recovery** of supervised outputs while **winter** behavior or loss remains impaired, especially under interference. Metrics aligned with the paper include **transfer** (winter error change A1→B), **interference** from mixture fits to angular responses at retest, and **generalization** to held-out test stimuli where defined.

---

## 3. Architectures under study (no communication only)

### 3.1 Task-routed two-module RNN (main modular model)

- **Two recurrent modules** with **no inter-module communication** in the primary grid (**sparsity = 0**, i.e. modules do not exchange hidden activity; “no_comms”).
- **Input routing**: **task_routed** — on each trial, the stimulus input is routed to one module as a function of the task/feature probe, so modules can specialize by task context.
- **Readout**: typically **common readout** maps from the concatenated (or combined) hidden state to a **four-dimensional** output (summer and winter channels for 2-D circular targets).
- **Temporal processing**: **nb_steps = 2** — inputs are expanded into a short temporal sequence before the recurrent core, so representations reflect both stimulus and brief temporal integration.

### 3.2 Single-module baseline

- **One module** (`n_modules = 1`) with the same output head and training loop, serving as a **capacity-matched or protocol-matched** baseline (in many comparisons, **two modules × hidden size 25** vs **one module × hidden size 50**).
- Same loss, schedule, and A2 update rule as the modular model, so differences isolate **routing and modularity** rather than task definition.

### 3.3 What we deliberately omit in this summary

- Conditions with **communication sparsity between modules** (non-zero sparsity) and related ablations are part of the broader repository but are **not** the focus of this overview.
- Additional levers such as **input_only** initialization scope or alternative routing schemes may appear in `experiments.json` but are secondary to the **primary grid** narrative.

---

## 4. Primary factorial structure

### 4.1 Task similarity

Three **participant-matched** schedules: **same**, **near**, **far**, manipulating the geometric relationship between the A and B rules. This factor is the main **ecological** driver of **interference** and **subspace realignment** in both the original ANN work and our RNN reimplementation.

### 4.2 Initialization scale (`init_scale`)

After construction, weights are **rescaled** by a condition-specific **`init_scale`** (global scaling of trainable parameters in the default policy). This is treated as an experimental knob analogous in spirit to “rich” vs “lazy” regimes in prior work: small scales tend to encourage **lower effective rank** and different learning trajectories; larger scales can behave closer to random-feature or high-norm regimes. **Outcomes of interest** include PCA dimensionality (`n_pcs_99`, etc.) and principal angles, not only accuracy.

### 4.3 Model size (hidden width)

**Hidden size** (e.g., 6, 12, 25) varies across conditions as an **additional axis**. Early aggregates suggest **many behavioral summaries are not dominated by width** within the explored range, but **geometry** (variance captured by leading PCs, subspace overlap) has not yet been exhaustively decomposed by size × scale × similarity; this remains an **open analysis item**.

---

## 5. Representational geometry: operational definitions

### 5.1 Effective dimensionality via PCA

On **last-step hidden states** (per trial, after the recurrent update), we summarize how many components are needed to explain cumulative variance:

- **`n_pcs_99`**: smallest number of principal components such that **≥ 99%** of variance is explained (and analogously for other thresholds where computed).
- Computations can be run on **pathway-specific** activations when saved (**combined**, **core**, **comms**), though under **no_comms** the modular decomposition is most informative when contrasting **combined** geometry across architectures.

These metrics are **distinct from trajectory-based PCA** over time: they target **state geometry** at a phase-resolved snapshot.

### 5.2 Principal angles between task subspaces

We quantify **overlap between the subspaces** spanned by hidden states for **task A** vs **task B** stimuli (implementation follows the standard recipe: PCA bases for each set, SVD of their inner product, arccos of singular values). The **smallest principal angle** (often reported in degrees) summarizes how “rotated” B representations are relative to A after learning. Empirically, this quantity **co-varies with task similarity**, **init_scale**, and **architecture** (single vs routed modular).

---

## 6. Behavioral and readout metrics (brief)

- **Transfer / error differences**: winter performance change across phase boundaries, computed from stored accuracies or errors with probe-aware indexing.
- **Paper-faithful interference**: from **two-component von Mises mixture fits** to angular responses at A2 retest (e.g., complement of the estimated A-rule weight), requiring fitted parameters per run.
- **Generalization**: winter accuracy on **held-out test stimuli** in A, with a **late-A1** restriction for the strict paper definition where applicable.
- **Training loss curves**: MSE logged per trial; filtering by **`probes`** (summer vs winter) replaces brittle trial-index slicing when `nb_steps > 1`.

These are reported in **notebooks_final** and test notebooks with **Holton-style** visualization where relevant (strip/scatter of participants, means, uncertainty).

---

## 7. Implementation and reproducibility

- **Configuration**: Experimental conditions live in **`a1b2/models/experiments.json`**. Each condition specifies `arch` (`two_module_rnn` vs `single_module_rnn`), `dim_hidden`, `n_modules`, **sparsity** (0 for no_comms), **`input_routing`** (`task_routed` vs `shared`), **`common_readout`**, **`nb_steps`**, **`init_scale`**, and optional **`init_scope`**.
- **Run identity**: **`build_run_id`** hashes the salient architectural knobs into a **unique simulation folder** name so reruns and partial grids do not overwrite each other.
- **Training loop**: **`runSchedule`** implements **A1 → B → A2** with the **A2 selective-update** rule; **`train_participant_schedule`** computes probe-dependent MSE on the appropriate output head and logs **losses**, **accuracy**, **probes**, **test_stim**, and **hiddens** into compressed **`.npz`** archives per participant.
- **Analysis code**: Geometry helpers live under **`a1b2/analysis/`** (e.g., `compute_state_representation_metrics`, `compute_pca_representation_metrics`, principal-angle utilities in **`transfer_interference.py`**). Notebooks aggregate participant-level statistics and produce paper-style figures.

---

## 8. Conceptual synthesis (for discussion)

1. **Task similarity** changes the **compatible subspace** in which task B can be learned; **far** schedules force larger representational change, increasing the scope for **catastrophic interference** on components of A that are not refreshed in A2.
2. **Init_scale** modulates **intrinsic dimensionality** and learning speed, so two networks matched in width and task can occupy **different effective-dimensional regimes**, with measurable consequences for **PCA counts** and **subspace angles**.
3. **Task routing** changes **which parameters see which inputs**, altering **how interference propagates** compared with a **single shared recurrent bottleneck**, even when total parameter count is similar.
4. **Width** remains a **controlled covariate**: primary effects of interest are framed as **scale × similarity × architecture**, with width providing **robustness checks** rather than the main story until finer analyses are complete.

---

## 9. Open directions (explicit)

- Joint models of **width × init_scale × similarity** on **geometry** (not only means of behavioral metrics).
- Tighter linking of **loss-curve regimes** (summer vs winter, truncated early training windows) to **PCA and angles** at matched training counts.
- Pre-registered **primary contrasts** for the no_comms grid (task_routed vs single_module) at fixed **init_scale** tiers.

---

*Document version: written for external communication; aligns with the modular RNN implementation in **Structure-Function-Analysis-of-Network-Topologies / a1b2_modular**.*
