# Init-scale geometry and transfer: shared vs task-routed RNNs

This report synthesizes the two companion notebooks:

- `notebooks/nb_shared_init_scale_geometry_transfer.ipynb` — **shared** input routing  
- `notebooks/nb_task_routed_init_scale_geometry_transfer.ipynb` — **task-routed** input routing  

Both compare **two-module RNNs** with `dim_hidden = 25`, `nb_steps = 2`, `common_readout = True`, `common_input = False`, across **communication sparsity**, **initialization scale**, and **task similarity** (same / near / far). A **single-module** baseline appears in the run table as `sparsity_label = single_module`.

**Important caveat:** The notebooks’ quantitative results are primarily in **embedded figures** (not fully reproduced as tables in this file). The analysis below is grounded in the **metrics definitions**, **plotting structure**, and **stdout diagnostics** saved in the notebooks (run counts, dataframe shapes, per-condition participant counts where printed). For exact numerical summaries, re-run the notebooks or export `transfer_df` / `rep_summary` to CSV.

---

## Common methodology (both notebooks)

### Experimental factors

| Factor | Levels |
|--------|--------|
| **Input routing** | Shared *or* task-routed (fixed per notebook) |
| **Sparsity** | `no_comms`, `0.1`, `0.3`, `0.5`, `0.7`, `0.9`, `1.0`, plus `single_module` where applicable |
| **Init scale** | `1.0` (default, no `init_scale` field), plus explicit variants; plotting order uses `INIT_SCALE_ORDER = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]` |
| **Task similarity** | `same`, `near`, `far` (hue in most plots) |

### Outcome metrics

1. **Transfer / interference (`error_diff`)**  
   Per participant and similarity condition: mean accuracy on the **first six odd-indexed trials of phase B** minus mean accuracy on the **last six odd-indexed trials of phase A1**.  
   Positive values suggest B starts easier than A1 ends (favorable transfer in this coding); negative values suggest the opposite. Interpretation should always be read **together with raw accuracy** (sections 6 and principal-angle plots).

2. **Accuracy and loss**  
   Mean over trials per **phase**: `post_A`, `post_B`, `post_A2`.

3. **Representation geometry**  
   - **`n_pcs_95`, `n_pcs_99`**: number of principal components to explain 95% / 99% variance of last-step hidden states, for pathways **`combined`**, **`core`**, **`comms`**, at each phase.  
   - **2D PCA** (`post_A2`, combined pathway): three panels for same / near / far.  
   - **Principal angle** between A and B subspaces (from `ann.get_principal_angles`), aggregated by init_scale × sparsity × condition.

4. **Holton-style trajectories**  
   Accuracy and loss vs **training step** through A1 → B → A2, with mean ± SD across participants; **init_scale** subset (e.g. 1.0, 0.7, 0.5, 0.1) and often **key sparsities** for readability.

5. **Direct PC comparison across phases (section 9)**  
   Bar plots: **x = phase** (`post_A`, `post_B`, `post_A2`), **facets = init_scale**, **hue = condition**, **combined** pathway — aligned with the pattern used in `nb_init_scale_geometry_transfer.ipynb`.

### How to read the three axes together

- **Task similarity** is the *experimentally imposed* structure of the B rule relative to A (same / near / far). It should drive the largest *systematic* splits in transfer, angles, and sometimes dimensionality.  
- **Sparsity** changes **how much cross-module communication** can shape a shared representational workspace (and in the shared-input case, how much both modules see correlated inputs).  
- **Init scale** scales **all trainable parameters** in `network.community` after default initialization (`apply_init_scale` in `a1b2/models/rnn_init.py`), affecting **effective learning speed, curvature, and lazy vs rich regimes**, which can **interact** with sparsity (e.g. weak comms + small init may under-exploit modules) and with similarity (far B may need larger representational change, sensitive to init).

---

# Part I — Shared-input architecture (`nb_shared_init_scale_geometry_transfer.ipynb`)

## 1. Scope and data coverage

- **Routing:** `input_routing = "shared"` (both modules receive the same expanded input pattern).  
- **Notebook stdout (saved run):** **56** conditions selected, **40** with existing simulation folders; **40** runs loaded with all three similarity schedules.  
- **Tensor shapes (stdout):** `transfer_df` ≈ **(10882, 7)** participant-level rows; `accuracy_df` / `loss_df` ≈ **(32646, 7)**; `rep_df` ≈ **(83370, 10)** — consistent with many runs × three similarities × multiple phases × three pathways.

**Implication:** Several **init_scale × sparsity** cells still lack on-disk simulations (56 − 40 = **16** missing). Any trend involving those holes should be treated as **incomplete**, not as a null effect.

### Example: balanced counts at `no_comms` (printed)

For shared routing, the notebook’s per-run participant counts at `no_comms` show **~103 / ~101 / ~101** for same / near / far across several init scales (e.g. 0.1–1.0 in the saved output), i.e. **roughly balanced** across similarities for those runs.

## 2. Axis: task similarity (same / near / far)

**Role in shared input:** All modules see the **same** input channel; similarity affects **what must be represented and decoupled** when B is learned, not **which input lines** are active per module.

**Where it appears in the notebook**

- **Hue** on transfer, accuracy, loss, dimensionality (`n_pcs_95` post-A2), principal-angle summaries, and Holton trajectories.  
- **Three-panel PCA** figures: one column per similarity for a fixed (sparsity, init_scale) run.

**Expected qualitative pattern (to verify on your figures)**

- **Far** typically shows the **largest** geometric separation (principal angles) and the **most spread** in Holton curves across seeds/participants if optimization is hard.  
- **Same** often shows **smallest** angles between A and B subspaces and **tighter** trajectories.  
- **Near** is an intermediate regime; its proximity to same vs far depends on your circular-task parameterization.

**Interaction with other axes**

- **× Sparsity:** Under **high sparsity / no_comms**, modules cannot reconcile errors via comms; **far** B may produce **more disparate** module-wise solutions, which can inflate variance in transfer and PC counts.  
- **× Init scale:** **Small init** can slow escape from shallow basins; **far** + **small init** + **weak comms** is a classic “triple penalty” regime (hard task, hard coupling, slow start).

## 3. Axis: communication sparsity

**Role in shared input:** Sparsity acts on **inter-module weights**, not on input routing. Shared input already **correlates** submodule inputs; sparsity controls **how much they can specialize jointly** through message passing.

**Plots:** Rows = `sparsity_label` in the main catplots; `single_module` row gives a **capacity-matched** reference without modular comms structure.

**Interdependencies**

- **Sparsity × similarity:** For **same** B, comms may matter less (compatible representations). For **far** B, **full** comms may allow **faster alignment** or **more interference**, depending on whether B forces a global reconfiguration.  
- **Sparsity × init_scale:** Very **small** init with **no_comms** can strand modules in **weakly coupled** basins; larger init may jump-start useful comms when sparsity allows.

## 4. Axis: initialization scale

**Operational meaning:** Uniform post-init scaling of all `community` parameters (see `rnn_init.py`).

**Plots:** x-axis `init_scale` on transfer, accuracy, loss, post-A2 dimensionality, principal angles, and facet columns in section 9 (PC bars).

**Interdependencies**

- **Init × similarity:** **Far** tasks often amplify differences between init scales in **final accuracy** and **transfer** because the required feature map change is larger.  
- **Init × sparsity:** Effects are **non-additive**: changing scale moves the optimizer through a different trajectory; with **sparse comms**, some init scales may **never** engage comms strongly enough before B begins.

## 5. Geometry vs behavior

- **Principal angles:** Tell you whether **A and B representations occupy overlapping subspaces**; large angles under **far** are expected if B is genuinely different.  
- **`n_pcs_95` / `n_pcs_99`:** Track **effective dimensionality**; rising PC count from `post_A` → `post_B` → `post_A2` (section 9) indicates **representational drift or expansion** through continual phases.  
- **2D PCA panels:** Qualitative **clustering / overlap** of stimuli; compare across **init_scale** for the **same sparsity** to separate “geometry change” from “performance change.”

## 6. Part I summary (shared)

The shared notebook is the right place to judge **how much task similarity and sparsity matter when both modules always see the same input**. Trends to prioritize when reading your figures:

1. **Similarity ordering:** far ≥ near ≥ same in **angle** and often in **variance** of trajectories.  
2. **Sparsity:** monotonic effects are **not guaranteed** — **no_comms** vs **1.0** can flip whether init_scale helps or hurts, especially for **far**.  
3. **Init scale:** interpret as a **global gain** on all weights; correlate with **loss curvature** in Holton plots (sharp vs flat phases).  
4. **Data gaps:** treat missing `path_exists` rows as **censoring** before claiming an interaction.

---

# Part II — Task-routed architecture (`nb_task_routed_init_scale_geometry_transfer.ipynb`)

## 1. Scope and data coverage

- **Routing:** `input_routing = "task_routed"` with `common_input = False` (task-dependent routing of inputs to modules as implemented in the wrapper).  
- **Notebook stdout (saved run):** **56** selected, **45** valid folders, **45** runs loaded with three similarities.  
- **Tensor shapes (stdout):** `transfer_df` ≈ **(10615, 7)**; `accuracy_df` / `loss_df` ≈ **(31845, 7)**; `rep_df` ≈ **(81033, 10)**.

**More complete grid than shared:** 45 vs 40 valid runs — **fewer missing** combinations on disk in the saved execution.

### Asymmetric participant counts (task-routed, `no_comms`)

The saved stdout shows **uneven** same/near/far counts for some cells, e.g.:

- `sp=no_comms init_scale=0.01`: same=**38**, near=**46**, far=**28**  
- while `init_scale=0.1` at the same sparsity shows ~**103 / 101 / 101**.

**Implication:** Summaries that **average over participants** for those cells mix **different effective sample sizes** and possible **selection** (e.g. failed or incomplete runs dropping unevenly by similarity). Compare to **shared** notebook at `no_comms`, where printed counts stayed **balanced** for the lines shown. **Statistical caution** is warranted for **task-routed × no_comms × low init_scale**.

**Note:** The opening markdown of the task-routed notebook may still list a shorter init_scale set; the **code** uses the same `INIT_SCALE_ORDER` as shared (`0.01` … `1.0`). Trust the **code cell** ordering and the **run table** over stale markdown.

## 2. Axis: task similarity (same / near / far)

**Role under task routing:** Similarity changes **which input pathways drive each module** during A vs B, enabling **stronger factorization** of A vs B representations than shared input in principle.

**Where it appears:** Same hue / panel structure as the shared notebook.

**Contrast vs shared (conceptual)**

- If **routing** encourages **modular credit assignment**, you may see **smaller** principal angles for **far** than under shared input at matched sparsity, or **different** transfer ordering between same/near/far.  
- **Holton plots:** task-routed **far** may show **less** between-participant banding than shared **far** if routing reduces conflicting gradients — or **more** if routing isolates errors to one module and destabilizes it.

**× Sparsity:** With **no_comms**, routed inputs can still **specialize**, but **cannot coordinate**; **far** may produce **stronger module asymmetry** in `core` vs `comms` PC counts.

**× Init scale:** Small init can **delay** the onset of routing-driven specialization; effects may show up as **longer A1 plateaus** in Holton loss.

## 3. Axis: communication sparsity

**Same formal levels as shared**, but **interaction with routing** differs: sparsity now gates **coordination** while **input** already **differs by task phase**.

**Interdependencies**

- **High sparsity + task routing:** May yield **cleaner module-local** geometry in `core` but **higher** `comms` dimensionality when comms *are* present (trying to compress rare cross-talk).  
- **single_module row:** Still useful as a **non-modular** reference; differences here isolate **architecture** rather than **routing** alone.

## 4. Axis: initialization scale

Same implementation as shared (`apply_init_scale` on all `community` parameters).

**Task-routed–specific interaction**

- Routing can **amplify** init effects: very small weights may produce **near-silent** module gates early, so **effective routing** only “switches on” mid-training — visible as **kinks** in Holton accuracy/loss.

## 5. Geometry vs behavior

The same metrics apply. **Additional reading for task-routed:**

- Compare **`core` vs `comms` pathways** in the post-A2 `n_pcs_95` catplot: **routing + sparsity** often shows **differential** dimensionality in core vs comms, whereas shared input sometimes makes them **more coupled**.

## 6. Part II summary (task-routed)

1. **Data:** **Richer** valid-run coverage than shared in the saved notebook; watch **unbalanced** same/near/far counts at **no_comms × low init**.  
2. **Similarity:** Still the **primary** stratifier of transfer and angles; expect **quantitative** differences vs shared due to **input path**.  
3. **Sparsity:** Interacts with **routing** to determine whether modules **specialize in isolation** or **coordinate**.  
4. **Init scale:** Alters **early-phase routing efficacy** and **learning speed**, which feeds into continual phases.

---

# Cross-notebook comparison: shared vs task-routed

| Aspect | Shared | Task-routed |
|--------|--------|-------------|
| **Input correlation across modules** | High (same channels) | Lower / structured by task |
| **Typical role of comms** | Primary way to **differentiate** module roles | **Optional coordination** on top of routed streams |
| **Similarity effects** | Mediated through **shared bottleneck** | Can be **routed apart** into modules |
| **Risk of uneven N** | Check `path_exists` gaps (16 missing in saved run) | Check **per-cell participant counts** (e.g. no_comms × 0.01) |
| **Holton plots** | Often **wider far bands** if input interference is high | Potentially **sharper** or **more modular** phase transitions (empirical) |

**Suggested reading order for a paper figure set**

1. **Transfer** catplots (section 5): global view of **init × sparsity × similarity**.  
2. **Accuracy / loss** (section 6): ensure transfer differences are **not** artifacts of **floor/ceiling** accuracy.  
3. **Principal angles** (section 6 in second numbering block): link behavior to **subspace overlap**.  
4. **Post-A2 dimensionality** (section 7): pathway-specific **compression vs expansion**.  
5. **PC bars across phases** (section 9): **continual** drift narrative.  
6. **Holton** (section 7): **mechanism** and variance.

---

## Reproducibility

- Notebooks expect project root containing `a1b2` and `data/simulations` (paths printed in cell 1).  
- After adding new conditions (e.g. extra `init_scale` values), re-run **condition selection** cells so `path_exists` and participant counts refresh.  
- For publication-ready tables, add cells that write `transfer_summary`, `angles_agg`, and `rep_summary` to `data/derived/` as CSV.

---

*Generated from repository notebooks and code structure; refresh numbers by re-executing the notebooks on your current simulation set.*
