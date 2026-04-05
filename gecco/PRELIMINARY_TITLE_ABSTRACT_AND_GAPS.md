# GECCO / LBA — consolidated title, abstract, preliminary results, and gaps

Single reference for **submission text** plus **what is already supported** vs **still open**. Citations: adapt to venue style (e.g. Holton et al., 2025; Clune et al., 2015).

---

## Preliminary results (what we can say now)

- **Trade-off:** On smaller / Near-focused runs, Pareto fronts suggest a **performance–structure tension**: lower validation MSE on probed cos/sine outputs tends to co-occur with **less extreme task splitting** along the routing axis; **near-maximal separation** (strong A/B pathway split) tends to sit at **worse MSE**.
- **Not yet solid for a strong claim:** **Far** vs **Near** ordering **across many seeds**, the **Same** condition as a reference, and **single-module baselines** on the **same** panels need the fuller sweeps (e.g. `notebooks/lba_overnight_grid.ipynb`, overnight shell scripts). Until then, treat cross-condition comparisons as **hypothesis-generating**.

---

## Recommended titles

### Primary (ALife-oriented) — talks / LBA / workshop header

**The price of modularity: Pareto evolution of routed information flow in human-grounded modular learners**

- *Why ALife:* centers **evolution**, **modularity as a costly/beneficial structure**, and a **multi-objective phenotype** (where inputs are routed), not ML benchmark framing.
- *Human-grounded* stays in the title as one anchor phrase; methods (NSGA-II, 12-bit, baselines) go in subtitle/abstract.

**One-line subtitle (optional):** *Discrete routing genomes; cyclic sequential tasks (Near / Far / Same); validation error vs. pathway-separation objectives.*

**Short running title (program / header):** *Pareto routing in modular learners*

### Alternate ALife-friendly titles

- **Evolving where information flows: performance–separation trade-offs in modular adaptive systems**
- **Routed pathways as an evolvable phenotype: multi-objective selection on human-structured task sequences**
- **When does splitting pay off? Pareto evolution of input routing under modular architecture constraints**

### Earlier overview titles (archive)

- Who pays for separation? Evolving input routing on human continual-learning data
- Error versus separation: evolving modular routing on human continual-learning trials
- Evolving how inputs are routed under continual learning — a human-grounded Pareto view

### Detail-rich variants (only if a venue wants specificity in the title)

- **12 bits, two pathways, one Pareto front: evolving input routing on Holton-grounded continual schedules**
- **NSGA-II on 12-bit input masks: validation error trades off against task-split routing in a dual-path RNN**

### Formal / reviewer-facing (longer)

**Multi-objective evolution of discrete input routing under manipulated task similarity: a Pareto analysis on human behavioral data from a continual learning protocol**

**Concise variant:** Evolutionary multi-objective optimization of modular input routing in a similarity-controlled continual learning benchmark

### Earlier primary options (archive)

1. Evolving input routing under cyclic task structure: a Pareto view of performance versus modular separation in dual-pathway RNNs
2. Multi-objective evolution of task-specific input pathways trades validation error for routing separation in Holton-style continual schedules
3. From fixed modularity to evolvable pathways: NSGA-II on input-routing genomes links transfer-friendly sharing to measurable task separation

---

## Abstract — expert (self-contained; no specific prior study assumed; ~280 words)

Sequential and context-dependent learning force systems to balance **reusing** prior structure against **isolating** it to limit interference. Human experiments and simple neural models agree that **how related successive tasks are** shifts the **transfer–interference** trade-off. **Architecture**—modularity, and how inputs are wired in—can bias solutions toward low-dimensional shared codes (often linked to small initial weights and substantial post-training change) or toward more factorized, separation-friendly activity (often linked to larger initial scales). Across cognitive science and deep learning, **representational geometry** (subspace overlap, effective dimensionality, orthogonality) is increasingly treated as a **driver** of sequential-learning outcomes, not a mere by-product of optimization.

We link **multi-objective evolution** to that view with a compact case study. **Trial-level data** come from **published human experiments** on **blocked, sequential learning** of **two alternating rule-based tasks** over simple stimulus–response mappings; the corpus includes conditions that vary **task relatedness** (same rule, partially shared structure, and more independent rules). We derive a **cyclic A/B training stream** from subsampled participants and train small **dual-module recurrent** agents. **NSGA-II** evolves **binary genomes** that assign each input dimension to **one of two modules**; within each genome, weights are optimized by gradient descent under a **fixed training budget**. **Width-matched single-module baselines** isolate the effect of the routing choice. **Preliminary Pareto fronts** show a **performance–structure tension**: lower validation error on held-out probes tends to align with **less extreme cross-task routing split**, while **stronger separation** of A- versus B-associated inputs across modules buys **higher error**. That supports treating **sharing and specialization as coupled objectives** under sequential exposure, with viable routings often **between** maximal merge and maximal split.

The setup foregrounds **discrete routing** as an evolvable degree of freedom, alongside knobs such as **init scale**, **inter-module communication**, and **task statistics** in modular RNN work. Next steps include evolving **communication or topology** so that **routing, connectivity, and dynamics** jointly support **transfer** without **catastrophic interference** when tasks overlap only partly. Citations in the paper version can name the underlying human protocol; the abstract stands alone for readers unfamiliar with that line of work.

---

## Abstract — formal short (~240 words; venue-ready with one results sentence)

Continual learning requires managing the interaction between transfer and interference. Empirical work in humans and in matched artificial networks indicates that the geometric relationship between successive tasks modulates this interaction; representational separation can mitigate interference but is not guaranteed to co-vary with architectural modularity when connectivity is fixed before optimization. The present study examines how experimentally controlled task similarity relates to the feasible trade-off between predictive accuracy and a simple structural measure of pathway separation when input routing is treated as an evolvable discrete assignment. Trial-level data are drawn from a preregistered sequential two-task protocol on circular stimulus–response mappings (Holton et al., 2025). Each candidate solution encodes a 12-bit genome that routes each one-hot stimulus dimension exclusively to one of two parallel nonlinear encoders; a shared linear layer maps the concatenated representation to four outputs encoding cosine–sine coordinates of summer and winter targets. Parameters are optimized by gradient descent under fixed training budgets; mean squared error is evaluated on the probed output pair only. Non-dominated sorting (NSGA-II) is applied over two objectives: validation mean squared error and the negation of a task-split index defined as the absolute difference in mean routing between stimulus indices associated with task A (0–5) and task B (6–11), paralleling secondary objectives used to encourage modularity in evolutionary neuroevolution (Clune et al., 2015). Optimization uses a compressed cyclic A–B presentation schedule constructed from subsets of participants to reduce computational cost. Empirical comparisons contrast Near, Far, and Same between-task rule conditions and include width-matched single-pathway baselines under small and large weight-initialization scale. **Preliminary fronts show a trade-off between lower MSE and higher task-split routing strength; full cross-condition and multi-seed stability are still being consolidated.** The work contributes a reproducible, human-grounded experimental harness linking multi-objective evolutionary search to continual learning benchmarks, together with a stated program for extensions to recurrent architectures, asymmetric retest phases, and graph-theoretic modularity measures.

---

## Contribution bullets (submission form / cover letter)

- **Method:** Multi-objective evolution of **12-bit input routing** for a **dual-module** network on **participant-derived** cyclic A/B schedules, with **single-module** controls and configurable Near / Far / Same slices.
- **Result (preliminary):** **Non-trivial Pareto structure** between **validation MSE** and a **task-split routing score**, suggesting **coupled** rather than independent optimization of error and separation.
- **Outlook:** Use the same machinery to evolve **communication** and **pathway growth** under explicit **transfer/interference** objectives, connecting to **continual learning** and **communication-based** accounts of functional modularity.

### One-paragraph box (shorter variant)

> We introduce a **small multi-objective evolutionary** setup on **human-derived continual-learning trials**, evolving **binary input routing** between two pathways while trading **validation error** against **task-split routing strength**. We compare **Near, Far, and Same** task rules and **width-matched single-module** baselines. The goal is an **illustrative Pareto analysis** of when **structural separation** is **cheap or costly**, grounding **ALife-style modularity pressures** in a **cognitive continual-learning** protocol and outlining **recurrent** and **graph-modularity** extensions.

---

## Earlier working titles (archive)

- *Pareto fronts of error and routing: multi-objective evolution of modular input pathways on human continual-learning trials*
- *Who pays for separation? Evolving task-split routing under near versus far rules in a human-grounded continual benchmark*
- *Multi-objective evolution of input routing under task similarity in a human-grounded continual learning benchmark*
- *When does selection pressure for modular routing pay off? A Pareto view of continual learning with evolved pathways*
- *Evolving modular input pathways for transfer–interference trade-offs under near and far task rules*

---

## What we still need for a **concrete** paper (checklist)

Mark each item when you have an artifact (figure, table, log path, seed list).

### Essential (minimum credible short paper)

| # | Result | How you get it | Pass criterion |
|---|--------|----------------|----------------|
| 1 | **Pareto + baselines** for **Near** and **Far** on the **same** training budget | `python -m gecco.experiments.run_lba_figure` or `notebooks/lba_experiments.ipynb` | One clear panel; legend identifies conditions and single-module stars |
| 2 | **Seed stability** (≥2–3 seeds) | `scripts/overnight/` or `notebooks/lba_overnight_grid.ipynb` | **Directional** claim holds or you **report null** |
| 3 | **Hyperparameters + data slice** in text | Copy from `CONFIG` / CLI into Methods | Reproducible: `trial_df` source, `n` participants, cyclic pattern, pop/gen/steps |
| 4 | **One sentence falsifier** | From `EXPECTED_OUTCOMES_AND_FIGURES.md` | Reader knows what would **contradict** the hypothesis |

### Strongly recommended (harder to dismiss as noise)

| # | Result | Why |
|---|--------|-----|
| 5 | **Hand-crafted task routing** baseline (A-slots→mod0, B-slots→mod1) | Shows GA is not only rediscovering the obvious split |
| 6 | **Same** condition on the same plot | Anchors “no rule change” reference |
| 7 | **Init-scale** sweep for **evolved** nets aligned with **single-module** σ | Fair comparison at matched σ (optional third star) |
| 8 | **CSV of Pareto genomes** | `lba_overnight_grid.ipynb` → `pareto_points.csv` / manifest |

### For a **stronger** ALife / continual-learning claim (future work or appendix)

| # | Result | Why it advances the field |
|---|--------|----------------------------|
| 9 | **Recurrent** module (`RNNCell` / dynspec `Community`) | Continual **temporal** credit, not per-trial Markov |
| 10 | **Full or partial A2** masking (Holton retest) | Matches **interference** definition in the cognitive paper |
| 11 | **Connection cost on weights** (Clune-style) not only input mask | Closer to **topological** modularity |
| 12 | **Population diversity** / lineage stats | Classic ALife: dynamics of **open-ended** search |
| 13 | **Cross-condition transfer** of best genome | Does routing **generalize** across draws of participants? |

---

## How this **advances ALife** (specific, not generic)

1. **Multi-objective evolutionary search on a human-anchored task** — Most ALife neuroevolution uses **hand-built** environments; here the **data distribution** comes from a **preregistered human continual-learning paradigm**, so “environmental complexity” is **operationally defined** (Near / Far / Same), not only a designer’s toy.

2. **Structure is the evolving object** — Rather than optimizing weights only, the **input graph** (which features each pathway sees) is part of the **phenotype**. That aligns with **modularity-as-emergent** narratives (Clune et al.) but applied to **continual** exposure patterns (**cyclic A/B**), i.e. **adaptation over a trajectory** of contexts.

3. **Pareto fronts as scientific output** — ALife often reports **best-of-run** fitness; reporting **trade-off surfaces** between **performance** and **separation** makes **plasticity costs** explicit—relevant when “survival” requires both **learning new** and **not destroying old**.

4. **Honest nulls** — If Near/Far **do not** separate the fronts at this scale, the contribution becomes **methodological**: a **cheap co-evolutionary testbed** for continual modularity hypotheses, with a clear **scaling** roadmap (RNN, full schedule, graph metrics).

---

## How this **advances continual learning** (specific)

1. **Similarity-conditioned structure** — CL literature often studies **architectures** or **regularizers** with **fixed** task boundaries; here **task similarity** (Near/Far/Same) is an **experimental factor** paired with **search over routing**, connecting to **subspace** and **interference** stories.

2. **Evolution as complement to SGD** — A **non-gradient** way to ask **which inductive biases** (routing) are **compatible** with **low error** under **repeated** task blocks—useful when **discrete** structure is hard to relax continuously.

3. **Bridge to cognitive metrics** — Positions future work to tie **routing Pareto** to **transfer** and **interference** metrics on the **same** `trial_df`, unifying **evolved** and **gradient** learners on **one** benchmark.

---

## File cross-references in this repo

- Narrative sections: `LBA_NARRATIVE.md`
- Interpreting plots: `EXPECTED_OUTCOMES_AND_FIGURES.md`
- Overnight runs: `scripts/overnight/README.md`, `figures/overnight/`
- Large grid notebook: `notebooks/lba_overnight_grid.ipynb`
- Upstream context: `REFERENCE_UPSTREAM_PROJECTS.md`
