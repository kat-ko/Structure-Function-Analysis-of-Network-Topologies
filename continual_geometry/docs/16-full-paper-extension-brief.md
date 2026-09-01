# 16 — Full-Paper Extension: Experimental Brief

```
Target:   Full paper (ICLR or ICML — venue not yet fixed; do not optimise for
          either deadline). Multiple people working; correctness is the binding
          constraint, not schedule.
Standing: Every rule in AGENTS.md remains in force. Pre-committed readings before
          arms exist. Measured cost before any run over ~30 min. Off-design
          results never enter results/phase1/.
Status:   Specification. Nothing here is authorised to run until §0 is closed
          and each axis has a written pre-commit.
Amended:  2026-09-01 — eight decisions in §Amendments; original wording below
          is unchanged so the disagreement stays visible. Adopted order is
          in the amendment, not in §8.
```

**Amendments (2026-09-01).** The eight gaps named in the reading of this brief
are closed in §Amendments at the end of this file. Do not treat the original
§0–§8 wording as current for those eight items.

---

## 0. Two prerequisites, before any new axis

### 0.1 The learning-rate law must be settled first

Every existing result uses `η = lr0 · γ₀² · (N/N_base)`. `00` §4.3 already records
that Atanasov et al. give `η* ∝ γ^(2/L)` for `γ ≫ 1`, and that the quadratic law
is therefore misspecified in the rich regime. `lr_scaling="corrected"` is
implemented and has never been run.

This blocks the depth axis specifically: **`L` appears in the exponent, so adding
a hidden layer changes the LR law.** Depth and LR-law would be confounded from
the first arm.

**Run first:** the corrected-LR arm at `L = 2`, over the full registered γ grid,
all four conditions, at the design point. This produces the like-for-like
reference that every subsequent axis is compared against.

**Pre-commit before running.** Three outcomes, all informative:
- Channel composition unchanged under corrected LR → the quadratic law was
  harmless here, existing results stand, and this becomes an appendix robustness
  arm.
- Composition changes at `γ ≥ 1` only → consistent with Atanasov's stated
  regime; the paper reports both laws and states which is primary.
- Composition changes throughout → the γ sweep is partly an LR sweep. That is a
  finding about the parameterisation and it must be reported as one.

### 0.2 A held-out arrangement set, reserved now

`lr0 = 5.0`, `target_loss = 0.05`, the γ grid and the measurement budget were all
chosen against arrangements that then produced the reported results. That is
ordinary practice and it is also a weakness nobody has to accept.

**Reserve a set of arrangement seeds that are touched by no design decision, no
pilot, no hyperparameter choice, and no exploratory analysis.** Every headline
result in the full paper is additionally reported on that set. Record the
reserved seeds in a committed file before any new arm runs, so the reservation is
verifiable rather than asserted.

This is cheap and very few papers do it. It is the strongest available answer to
"were these numbers tuned into existence."

---

## 1. Direct dimensionality control — highest priority

**Why.** Reviewer mobw on the prior paper: the causal claim is more accurately
read as "the regime, which co-varies with dimensionality, gates the effect." That
objection now lands harder, because one of the three attribution channels *is*
dimension, and γ moves dimensionality and optimisation dynamics together. This is
the single item between the current work and that reviewer's stated bar for a
main-conference paper.

**Design.** Low-rank linear projection between hidden layer and readout,
rank `r ∈ {2, 4, 8, 16, N}`, crossed with the γ grid. Full 2×2 in the sense that
matters: rank at fixed γ, and γ at fixed rank.

**Implementation constraints.**
- The projection must not change the parameterisation. Verify that hidden-weight
  initialisation remains bitwise identical across γ *and* across rank, as `I6`
  requires. Add a unit test.
- Rank is applied to the *readout path*, so the measured representation `h` is
  unchanged in dimension — but its task-relevant dimension is constrained. State
  explicitly in the spec which of the two is intended; they are different
  experiments.
- `lr0` and `target_loss` were pinned at `r = N`. Confirm every rank reaches the
  loss target; if not, `lr0` needs re-pinning per rank and that must be recorded
  as a design change, not absorbed silently.

**Pre-commit.**
- Channel reorganisation reproduces under rank control at fixed γ → dimensionality
  is doing the work, and the attribution is licensed as a claim about
  representational dimension.
- Reorganisation tracks γ regardless of rank → the driver is optimisation
  dynamics; the paper says so and the dimension channel is described as
  co-varying rather than causal.
- Mixed → report the interaction; do not pick a reading afterwards.

---

## 2. Depth

**Why.** The dominant generalizability doubt, and a specific mechanistic
hypothesis. Sixteen random Gaussian centres in `d = 150` under one ReLU are near
separable at initialisation (α ≈ 0.31 before training). The network is
re-arranging an adequate geometry rather than building one — which is plausibly
*why* the utility channel dominates. With depth, features must be constructed,
and the dimension channel should carry more.

**Design.** One additional hidden layer, `L = 3`. Not a sweep. One extra depth is
sufficient to test whether the result is a property of a single ReLU.

**Implementation constraints.**
- The μP parameterisation must be extended to `L = 3` and the base-width
  equivalence re-derived. `docs/reference/parameterization-derivation.md` is
  human-owned; the extension needs the same treatment.
- The LR law exponent changes with `L`. Use the outcome of §0.1.
- Report **capacity at initialisation** at both depths. If it is already lower at
  `L = 3`, that is the scarcity hypothesis being confirmed at init, before any
  training, and it is worth its own line.
- Width must be chosen so that parameter count is comparable, or the comparison
  is depth-plus-capacity. State which.

**Pre-commit.**
- Channel composition invariant to depth → the result is not an artefact of a
  single nonlinearity, and the claim strengthens.
- Dimension channel carries more at `L = 3` → the scarcity hypothesis is
  confirmed, and the paper states that channel composition depends on whether the
  representation must build features or only re-arrange them. This is the more
  interesting outcome and it must not be reported as a failure of the original
  result.
- Composition changes unpredictably → report; do not interpret post hoc.

---

## 3. Scope as a manipulated axis

**Why.** The Line 2 audit finds that scope — the breadth of experience grouped
under one task label — is specified by essentially no benchmark, and its Figure 1
("one stream, several partitions") is a thought experiment because on real
benchmarks the partition *is* the dataset. In this design it need not be.

**Primary design — granularity of the cut.** Fix a sequence of dichotomies, then
partition it three ways: one dichotomy per task (fine), two jointly (medium), four
jointly (coarse). Same underlying experience, three observer-imposed cuts.

Requires a multi-output readout for the coarse conditions. Keep the shared
readout; add outputs, do not add task conditioning.

**Fallback design — breadth per task.** A task is a dichotomy over a subset
`S ⊆ {1..16}`, `|S| ∈ {4, 8, 16}`. Keeps the scalar readout entirely. Use this if
the multi-output version introduces confounds that cannot be controlled (the loss
scale changes with output count; check this before choosing).

**Constraint from Line 2 itself.** That work argues reporting is insufficient — a
property varied without being shown to matter has not been probed. So the scope
axis must be accompanied by a demonstration that scope changes something for this
learner, not merely that it was varied.

**Pre-commit.**
- Forgetting differs by partition at matched underlying stream → the audit's
  central claim is demonstrated rather than argued, and it is a headline result.
- No difference → scope is not consequential for this learner in this regime,
  which is a substantive negative and must be reported as one. It would also
  bound the audit's own claim, which is worth saying plainly.

---

## 4. Recurrence in the stream

**Why.** The sharpest available test of the paper's central interpretive claim.
If retention is a surplus that relaxes toward a baseline, re-encountering an
earlier dichotomy should be **faster to re-learn than to learn first** — a savings
effect. If forgetting is overwriting, re-learning costs what first learning cost.

**Design.** Re-present task 0's dichotomy at a later position in the stream, after
several intervening tasks. Measure steps-to-target-loss on the re-encounter
against steps-to-target on the original, and the capacity trajectory across the
re-encounter.

**Implementation constraints.**
- Matched-loss stopping makes steps-to-target the natural savings measure. Record
  it explicitly; it is currently logged but not analysed.
- Control for position in the stream: a task at position 12 may be faster for
  reasons unrelated to having been seen before. **The control is a novel
  dichotomy at the same position**, matched for similarity to its predecessor.
- Line 2 lists recurrence as an exposure property, and notes it belongs to
  exposure when the schedule returns to a region and to change when the process
  repeats. This is the former. State it.

**Pre-commit.**
- Re-learning is faster than first learning, with the position control → savings;
  the surplus reading is supported by a behavioural measure independent of the
  geometry.
- Equal → the surplus reading is not supported behaviourally, and the paper says
  the geometric and behavioural pictures diverge. That is reportable and it is
  the outcome that would most change the framing.

---

## 5. Continual-learning baselines — appendix, but treated properly

**Why, revised.** Not a benchmark comparison — the contribution is a measurement,
not a method, so "does modularity beat EWC" is a question this paper does not
ask. But EWC and replay *change the geometry by construction*: EWC constrains
directions via a Fisher penalty, replay re-presents old input distributions. So
the decomposition can say **which channel each method protects.**

That is an application of the instrument rather than a defence against a
criticism, and it is a genuine result. It also pre-empts the objection at low
cost.

**Design.** One γ (rich, where forgetting is largest and the channels are most
distinguishable), all four similarity conditions, three arms: none / EWC /
replay. Report the attribution table for each.

**Implementation constraints.**
- EWC: diagonal Fisher, penalty strength swept over at least three values, since
  a single λ chosen badly is not a baseline.
- Replay: buffer size stated as a fraction of the task; the Line 2 replay case
  turned on objective and number of passes, so record both.
- Both interact with the invariants: EWC adds a penalty term to the loss, replay
  changes what "one task" means for the matched-loss stopping rule. Neither may
  silently alter the stopping criterion. State how each is handled.
- These are off-design. Separate results directory, not pooled, off-design guard
  applies.

**Pre-commit.** Which channel each method protects is the measurement; there is
no pass/fail. But state in advance that a null — both methods reduce forgetting
without differentiating the channels — is a reportable outcome and not a reason
to expand the sweep.

---

## 6. Naturalistic arm

**Why.** The one criticism the Line 2 framing reframes but does not remove.
Lowest priority and the most likely to consume disproportionate effort.

**Design.** Per `docs/10-cifar-pilot-spec.md`. `P = 10` changes the estimation
regime, so the first job is establishing the estimator behaves there at all —
noise floors, whether the `M^-0.45` compression law applies, pairwise vs `full_P`.
One measured arm before any projection.

**Scope honestly.** Raw pixels into the same MLP is a *lower bound* on
generalisation: if the decomposition works there it likely works on learned
features; if it fails, learned features are not ruled out. Say this in the spec
and in the paper.

**Constraint.** Feature similarity cannot be set independently of class identity
on real images, so the 2×2 does not transfer. This arm tests the readout axis
only, and must be scoped that way — building four corners a different way and
comparing them under the same labels would be the worst instance of the
mismatched-population fault in the appendix.

---

## 7. Two additions not requested

### 7.1 A formal interaction test

Reviewer mobw asked for one on the prior paper's headline and it was not
provided. The headline here is γ × condition on channel composition. With the
sampling-unit machinery already built, this is straightforward, and it closes the
last outstanding statistical criticism.

Specify the model and the unit before running it: the unit is the unique
(arrangement, init) draw, not the file.

### 7.2 Theory, at the level the reviews asked for

Reviewer mobw #7: *"even a toy linear-network analysis deriving the regime in
which task subspaces must compete for limited directions would strengthen the
contribution considerably."*

Two tractable targets, neither requiring new experiments:

- **Why the utility channel dominates when representational capacity exceeds task
  demand.** The scarcity hypothesis in §2 is currently a verbal argument. If it
  can be stated for a linear or single-ReLU model, it becomes a prediction that
  the depth arm tests rather than an interpretation of the depth arm's result.
- **Why the two similarity axes act separately on centre geometry.** Hiratani
  derives their interaction behaviourally in a linear teacher-student model.
  Whether their approximate additivity on `ρ_c` follows from that setup, or is
  independent of it, is a question with a derivable answer.

This is not an experimental item and it does not block anything. It is the
difference between a descriptive paper and one the reviews would call
main-conference.

---

## 8. Order

1. §0.1 corrected-LR reference, and §0.2 reserved arrangements — both before anything else
2. §1 rank control
3. §2 depth
4. §3 scope, §4 recurrence (independent; can proceed in parallel)
5. §5 CL baselines, §7.1 interaction test
6. §6 naturalistic arm
7. §7.2 theory throughout, not at the end

Each axis: written pre-commit before arms exist, measured cost before launch,
separate results directory, and a report before the next axis starts.

## 9. What does not change

The sampling-unit discipline (unique n, arrangement vs initialisation, population
table). The resolution gates and the floors. The artifact-family appendix. The
inventory and prose-pinning. The scope audit. Nothing in `results/phase1/` is
regenerated or added to.

Every new axis inherits the same question the last two weeks established as
primary: **at what sampling unit is this finding true?**

---

## Amendments (2026-09-01)

Authoritative decisions closing the eight gaps in the reading of this brief.
The sections above are the original text. Where an original sentence would now
be false, the replacement is here; the original is not silently edited.

### A1. The γ cut is γ > 1, not γ ≥ 1

Original §0.1: "Composition changes at `γ ≥ 1` only".

**Amended.** The code is authoritative: `lr_scaling="corrected"` fires only when
`gamma_0 > 1` (`src/models/parameterization.py`). At γ = 1 the two laws are
identical. **γ = 1 is a negative control**: any composition change there is not
the LR law and must be diagnosed, not absorbed.

### A2. Unique-n throughout, no copy inflation

Original §0.1: "over the full registered γ grid, all four conditions".

**Amended.** That meant coverage, not a second Phase 1 of 960 files.
§0.1 is **6 γ × 4 conditions × 8 unique arrangements = 192 arms**, with
`stream_id` drawing the arrangement under the fixed RNG for every *new* axis.
The same unique-n=8 rule applies to every new axis. No stream_id copies.

### A3. Old vs reserved arrangements, two named populations

Original §0.2 reserved "arrangement seeds" without naming the comparator
population. Running §0.1 on the reserved set would make the LR comparison a
mismatched-population comparison.

**Amended.** Two populations, recorded in `results/reserved_arrangements.json`:

- **Old arrangements.** The unique n=8 of the registered grid. On that grid,
  arrangement was keyed by `paired_init(seed)["stream"]`, not by `stream_id`.
  Identifiers are init seeds 0–7. **This is the LR comparator**: corrected-LR
  versus quadratic, to choose the primary law. Reproducing these arrangements
  requires `arrangement_source="legacy_seed"`, not `stream_rng`.
- **Reserved arrangements.** Eight `stream_id`s under `pipeline.stream_rng`
  (salt 20260817), in a block that has never been used as a stream id.
  **Every new axis** uses this set, plus a duplicate of the headline
  measurement under the primary law as the not-tuned report.

**Rule, in the file itself:** the reserved set never informs `lr0`, the rank
grid, or the stopping criterion. A unit test fails if a reserved id appears in
any design-decision artifact.

### A4. Missed loss under corrected LR is the manipulation, not a failure

At γ = 10, L = 2, corrected gives η ∝ γ¹ against γ² — a 100× smaller
rich-regime learning rate. Many rich arms will need far more steps and some
may not reach 0.05 within 20k.

**Pre-commit:** `lr0` is not re-pinned after seeing a miss. Record misses as
an outcome; if a large fraction miss, that is a finding about the corrected
law's practicality at this loss target, reported as such. If re-pinning
becomes necessary it is a design change with its own entry, decided before
the arms and not after. `lr0` and `target_loss` stay at the registered
5.0 / 0.05.

### A5. Rank is on the readout path; first slice; same miss rule

Original §1 listed `r ∈ {2, 4, 8, 16, N}` and allowed silent re-pinning.

**Amended.** Rank is a low-rank projection on the **readout path**, so
`h ∈ ℝ^N` is unchanged and GLUE's ambient dimension is untouched.
Hidden-width reduction is a different experiment and is not this one.

First slice: **γ ∈ {0.03, 1, 10} × r ∈ {4, 16, N}**, one measured arm, then
expand on the result. Rank waits on the primary LR law from §0.1.

If a rank cannot reach target at the `lr0` pinned on r = N, that rank is
reported as not reaching target rather than re-pinned, unless re-pinning is
decided in advance for all ranks uniformly.

### A6. Depth sequential arms stay blocked; init-α moves to the front

Original §2 required the μP extension before any depth work, and left width
choice unstated.

**Amended.** μP at L = 3 is human-derived; `parameterization-derivation.md` is
human-owned and unwritten; `parameterization.md` already flags μP+1/√L as out
of scope and not human-verified. An agent-derived L = 3 parameterisation is
what §8.1 forbids acting on. **Sequential depth arms wait on that derivation.**

**Init-α is not blocked.** Capacity at initialisation at L = 2 versus L = 3,
no training, no stream. A forward pass does not require the μP derivation.
If α at init is already lower at L = 3, the scarcity hypothesis is confirmed
before any sequential arm exists. Do it as soon as a second hidden layer
exists in the model code. Training at L = 3 must refuse until the derivation
exists.

**Width: match parameter count, and state it.** Matching N would confound
depth with capacity, which collides with §1.

### A7. Scope: decide the readout before the pre-commit

Original §3 named granularity primary and the loss-scale confound as a
check-before-choosing, but did not require the decision before the pre-commit.

**Amended.** Granularity remains primary. The readout decision comes **before**
the scope pre-commit. Check the loss-scale confound first — matched-loss
stopping is calibrated at one output, and the coarse conditions change output
count. If the confound cannot be controlled cleanly, fall back to breadth
`|S|`, which keeps the scalar readout and the instrument identical at the
cost of changing GLUE's P. Do not decide this after seeing arms.

### A8. Recurrence position control, specified

Original §4: "a novel dichotomy at the same position, matched for similarity
to its predecessor."

**Amended.** The control is a **novel dichotomy at the same stream index,
matched on `s_r` to its own predecessor**, in the same condition corner, on
the same arrangement. Comparison: re-presented task-0 dichotomy at index k,
versus a fresh dichotomy at index k with the same readout-similarity to the
task at k−1. Same arrangement seed for both, drawn from the reserved set.
That isolates "seen before" from "position in the stream" and from
"similarity to what precedes it."

### A9. Two reorderings, adopted in full

Original §8 queued the §7.1 interaction test behind CL baselines and left
§7.2 theory "throughout" without an owner or a start.

**Amended order, start now:**

1. Commit the reserved-arrangement file (zero compute).
2. Run the interaction test on existing unique-n=8 data. Model and unit
   specified before running. Unit is the unique (arrangement, init) draw, not
   the file. Report before starting §0.1.
3. Write the §0.1 pre-commit, then measure one arm, then launch. 192 arms,
   unique n=8, **old** arrangements, γ > 1 as the cut with γ = 1 as negative
   control. Report the miss rate alongside the composition result.
4. In parallel: second hidden layer in the model code and the init-α check
   (L = 2 vs L = 3, forward pass only, parameter-count matched).
5. In parallel: scope readout confound — does matched-loss stopping survive
   a multi-output readout, and how does loss scale with output count? This
   decides granularity versus breadth, and it must land before the scope
   pre-commit is written.

§7.1 runs on existing unique-n=8 data, needs no new arms, and is the one
statistical object the reviewer explicitly asked for.

§7.2 theory runs in parallel from step 1, owned by a person rather than
queued. It is not blocked by any experiment. One of its two targets — why
utility dominates when capacity exceeds demand — would turn the depth arm
from an interpretation into a test of a prediction.

**Not yet:** rank (waits on the primary LR law), depth *arms* (wait on the
human μP derivation), scope and recurrence arms (wait on their pre-commits),
CL baselines, CIFAR.

**Human-owned, still outstanding:** the two verifications; the estimator
paragraph; the μP-at-L=3 derivation (the gate on the depth axis).
