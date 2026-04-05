# What would count as a “good” experimental outcome?

This note ties **hypothesis → method → figure** so you can interpret runs from `run_lba_figure.py` or `notebooks/lba_experiments.ipynb` without moving the goalposts after the fact.

---

## 1. Hypothesis (what we claim is plausible)

**Core idea:** In **continual** exposure to related tasks, **how inputs are allocated** to pathways interacts with **task similarity** in the environment. Under a **multi-objective** search (performance vs a cheap **structural** pressure toward **task-split routing**), the **Pareto front** should **shift** with **Near vs Far** rule structure—analogous to the idea that **representational separation** (Johnston / Fusi–style abstraction discourse) and **modularity** (Clune-style second objective) become more or less **necessary** depending on whether the new task can reuse or must **orthogonalize** relative to the old one.

**Narrow operational version for this LBA:**  
When task B is **Near** task A, evolution should more often need **strong A/B routing separation** to reach **low validation MSE** on the cyclic schedule; when B is **Far**, **low MSE** should be achievable with **weaker** separation (routing can stay more entangled), because the “right” solution is less forced to share representational machinery with A.

**Explicit non-claim:** We are **not** yet claiming human alignment, full recurrent BPTT, or a full **Q**-metric graph modularity—only a **proof of mechanism** on a **reduced** Holton-style tensor and **evolved input routing**.

---

## 2. Method (what the experiment actually tests)

| Element | Role |
|--------|------|
| **Data** | Subset of real **`trial_df`** participants; **Near / Far** from the `condition` column. |
| **Schedule** | Short **A-B** segments concatenated as **ABABAB** (fast stress test; not full A2 asymmetry). |
| **Model** | **Dual pathway** encoders + **hard routing genome** (12 bits); **single-module** baseline, width-matched. |
| **Training** | Fixed **Adam** steps per genome evaluation; **MSE** on **probed** cos/sin only. |
| **Evolution** | **NSGA-II**; objectives **minimize** `(val MSE, −routing_separation)` so Pareto = trade-off **error ↔ task-split routing**. |
| **Baselines** | Single module at **low vs high init scale** (proxy for richer vs lazier feature expansion), **x = 0** on routing axis. |

**What “success” can mean here:** A **condition-dependent** change in the **set of achievable (MSE, separation)** pairs—not necessarily beating SOTA or matching full `a1b2_modular` RNNs.

---

## 3. Good outcomes (ranked, from strongest to still-publishable)

### A. **Strong** (clear “effect” for an LBA figure)

1. **Pareto shift:** The **non-dominated** cloud for **Near** lies **above and/or to the right** of **Far** in meaningful regions—e.g. for a **fixed** routing separation, **Near** requires **higher MSE**, or to hit a **target MSE**, **Near** requires **higher separation** than **Far**.  
2. **Baseline gap:** Some **Pareto** dual-path individuals **dominate** the **single-module stars** on **MSE** at **non-zero separation**, or match MSE with **strictly positive** separation (shows routing is not free lunch but can help under the second objective).  
3. **Replicability:** Same **ordering** of conditions holds across **2–3 seeds** (not identical points, same qualitative geometry).

### B. **Moderate** (honest LBA: “first empirical slice”)

1. **Histogram / population spread:** **Near** shows **heavier tail** of **high-separation** genomes in the **final population** at similar MSE, or **bimodal** MSE in **Near** but not **Far**.  
2. **Extreme points:** **Best MSE** genome under **Near** has **higher** `routing_separation` than under **Far** (single-point story, weaker than full front).  
3. **Init-scale interaction:** **Low-σ** single-module star is **much worse** than **high-σ** on **Near** but not on **Far** (links to your “lazy vs rich” narrative without overclaiming).

### C. **Null / negative** (still useful if reported cleanly)

1. **Overlapping fronts:** **Near** and **Far** Pareto sets **overlap** within sampling noise → conclusion: **at this reduced schedule / budget / architecture**, similarity **does not** strongly reshape the **routing–error** trade-off; **future work** needs recurrence, longer cycles, or larger search.  
2. **Routing irrelevant:** All good solutions have **low separation** → second objective may be **misaligned** with what reduces MSE here (routing not the right bottleneck).

---

## 4. What a figure “with an effect” should look like

### Primary figure (you already generate this)

**Axes:**  
- **x:** Task-split routing separation `|mean(routing on A slots) − mean(routing on B slots)|` ∈ [0, 1].  
- **y:** **Validation MSE** (lower better).

**Layers:**  
- **Scatter:** **Pareto** points for **Near** (e.g. blue) and **Far** (e.g. orange).  
- **Stars:** Single-module **low σ** and **high σ** at **x = 0**.

**“Effect” readout:**

| Pattern | Interpretation |
|--------|----------------|
| **Far** front is **lower** (better MSE) at **low x** than **Near**; **Near** only catches up at **higher x** | Similarity **forces** more **separation** to stay accurate—matches the operational hypothesis. |
| **Near** front **extends** to **higher x** at comparable **y** | Evolution **finds** more **modular routing** without paying MSE (interesting if Far cannot). |
| Stars sit **inside** the dual-path **Pareto hull** | **Routing evolution** finds solutions **unreachable** by a single pathway at matched width (strong but not required). |
| Two fronts **on top of each other** | **No effect** at this resolution—report as null + scale-up plan. |

### Secondary figure (notebook)

**MSE density across final population:**  
**Near** **wider** or **shifted** vs **Far** supports “harder continual niche”; identical curves support null.

---

## 5. Quick checklist before you declare “we see an effect”

- [ ] Same **participant count**, **train_steps**, and **pop/gen** for **Near** and **Far** (fair comparison).  
- [ ] At least **two seeds** for the **primary** claim.  
- [ ] **CSV** of Pareto genomes saved (notebook cell) so you can cite **example bitstrings** in text.  
- [ ] One sentence in the abstract: **directional** prediction (**Near** needs more separation **or** higher MSE at fixed separation)—so readers know what would **falsify** the claim.

---

## 6. Relation to the written LBA outline

Use **`LBA_NARRATIVE.md`** for prose; use **this file** as the **pre-registered-style** success criteria so “good outcome” is not redefined after seeing the plot.
