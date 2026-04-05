# GECCO late-breaking abstract — narrative scaffold (~2 pages)

Use this outline to draft the PDF. The **code** implements a **minimal** demonstrator (NSGA-II on input routing + Holton-style cyclic data); numbers will be **small-N** and **illustrative** unless you scale runs.

---

## 1. Problem (≈0.4 page)

- **Continual learning** trades **transfer** against **interference**. Humans and simple linear ANNs show aligned trade-offs when task rules are **Same / Near / Far** (Holton et al., 2025, *Nat Hum Behav*).
- **Representational geometry** matters: overlapping subspaces → transfer + interference; orthogonal subspaces → less interference (e.g. Johnston & Fusi-style arguments on abstraction and task structure).
- **Modularity** (structural or routed pathways) is often proposed to **segregate** representations and mitigate forgetting, but **when** it emerges and **what environmental statistics** force useful separation remain open—especially under **continual** training rather than i.i.d. multi-task batches.

**Hypothesis (LBA):** A minimum level of **environmental structure** (here: **task similarity** in a Holton-style sequence) interacts with **how inputs are allocated** across pathways. **Evolving** routing under **resource constraints** (fixed module width) makes the **transfer–interference–geometry** trade-off explicit as a **multi-objective** problem, in the spirit of **connection cost / modularity** objectives in Clune et al. (2015, *PLoS Comput Biol*).

---

## 2. Prior work (≈0.35 page)

- **Holton et al. (2025):** A1→B→A2 on a ring; twinned linear nets; von Mises interference; rich/lazy regimes.
- **Clune et al. (2015):** Multi-objective evolution (**performance + connection cost**) encourages **modular topologies** that **ease** later learning of new skills (complementary to gradient-based continual learning).
- **Johnston / Fusi (and related):** abstract vs mixed representations; compression vs binding—motivates measuring **separation** along **task-relevant** directions.
- **Your parallel line (a1b2_modular, not cited as this submission’s code):** task-routed modular RNN vs single-module baseline; **init scale** moves effective dimensionality (rich ↔ lazy); architecture effects shrink in **lazy** regimes.

---

## 3. Remaining challenges (≈0.35 page)

- Gradient methods **fix** architecture; evolution can **search** discrete **input–pathway** assignments but is **noisy** and **expensive** on full human-length schedules.
- It is unclear how **task similarity** and **routing** interact when **both** are free variables—Holton varies similarity **between** groups; here we ask whether **Pareto-optimal** routings **shift** with **Near vs Far** data drawn from the **same** trial table.
- **Scalability:** Full participant pools and long A1-B-A schedules are overkill for a **proof-of-mechanism** LBA; a **reduced cyclic** schedule (e.g. A-B-A-B-A-B) reuses trial rows to stress **repeated** context switches cheaply.

---

## 4. This submission — minimal approach (≈0.4 page)

**Task:** Holton **one-hot** (12-D) inputs and **2-D circular** targets per probed season; **loss** only on the probed output pair (MSE on cos/sin), **full supervision** on the **reduced** trial subset (LBA simplification vs asymmetric A2 in the paper).

**Schedule:** Concatenate **short** A1 and B segments per participant into **ABABAB**; pool a **few** participants per **Near / Far** condition for a single training tensor.

**Architecture:** **Two parallel pathways** (default: `Linear → tanh` per module, **not** a recurrent core over trials—cheap for the LBA) with **hard** routing: one **binary gene per input dimension** (12 genes). **Single-module** baseline with **matched width** (2× hidden per module). **Init scale** σ applied globally (evolved nets use one σ; baselines compare **low vs high** σ). Next step: `RNNCell` per module to match the modular-RNN story in prose.

**Evolution:** **NSGA-II**, two objectives: (1) **validation MSE**, (2) **−|mean routing on A stimulus slots − mean on B slots|** so that **Pareto** points trade **error** against **task-split routing** (a **structural** analogue to a cheap “modularity” pressure, inspired by Clune’s **second objective**, not a full graph-theoretic Q).

**Figure:** Scatter of **Pareto** individuals (**Near** vs **Far**) in (**routing separation**, **MSE**) space, with **single-module** stars at separation = 0 for **low/high** σ.

---

## 5. Expected “small result” + future work (≈0.15 page)

**Plausible LBA claim (if pattern holds):** Under reduced schedules, **Near** condition shows **different** Pareto sets than **Far**—e.g. easier to get **low MSE** without **strong** A/B routing separation when rules are **far** (orthogonal learning in prior work), while **Near** may **require** more extreme routing or accept higher error.

**Future work:** Recurrent **BPTT** across trials; **true** A2 masking; evolve **sparse inter-module** links (dynspec-style **Community**); larger populations; **same** condition; connect routing genes to **post-hoc** PCA / principal angles; align with **Clune-style connection cost** on **weight** graphs, not only input masks.

---

## References (short)

1. Holton, E., et al. (2025). *Nat. Hum. Behav.* https://doi.org/10.1038/s41562-025-02318-y  
2. Clune, J., et al. (2015). *PLoS Comput. Biol.* (neural modularity + connection cost).  
3. Béna & Goodman (2025). *Nat. Commun.* (modularity / specialization in RNNs; architecture reference via dynspec).  
4. Johnston, W. J. & Fusi, S. (2023). *Nat. Commun.* (abstraction / compositional representations; conceptual anchor).
