# Architectural Ablation Notes

This document consolidates motivations, implementation details, interim findings, limitations, and next-step ideas from the ongoing architectural-ablation work around the two-module RNN in `a1b2_modular`.

It is meant as a working research note, not a polished methods section.

## Scope

The current architectural questions are centered on the following levers:

- explicit inter-module communication sparsity (`no_comms`, `0.5`, `1.0`)
- input routing (`shared` vs `task_routed`)
- initialization scale (`init_scale`)
- initialization scope (`global` vs `input_only`)
- readout coupling (`common_readout=True` vs a future separated-readout control)

The notebook most directly associated with the current line of investigation is:

- `a1b2_modular/tests/nb_comms_pathway_probe.ipynb`

Related comparison notebooks:

- `a1b2_modular/notebooks_final/paper_figures_size_comparison.ipynb`
- `a1b2_modular/notebooks_final/paper_figures_size25_no_comms_architecture_comparison.ipynb`
- `a1b2_modular/notebooks_final/paper_figures_size50_no_comms_architecture_comparison.ipynb`
- `a1b2_modular/tests/paper_figures_init_scope_global_vs_input_only.ipynb`

## Architecture recap

The two-module RNN uses a `Community` module with two recurrent streams:

- `core`: module-internal recurrence
- `comms`: masked communication recurrence

At inference, the shared sequence sent to the readout is the sum of the two streams:

- `sequence = core_out[0] + comms_out[0]`

The `comms` stream recurrent hidden-to-hidden weights are masked by `comms_mask`; this is the explicit communication mechanism manipulated by the sparsity setting.

This creates at least three possible routes for coordination between modules:

1. shared input (`input_routing="shared"`)
2. shared readout / shared output loss (`common_readout=True`)
3. explicit recurrent communication via `comms_mask`

This distinction is central for interpreting ablation results.

## Main motivations

The ablation work was motivated by several observations:

1. Behavioral and representational comparisons suggested surprisingly weak differences between `no_comms` and denser communication in several paper-style conditions.
2. That raised the question of whether:
   - communication was not actually being used,
   - the implementation was ineffective,
   - or alternative coordination routes (shared readout, scale effects) were dominating.
3. In particular, high initialization scale seemed to amplify communication-related effects, raising the possibility that the effect was at least partly scale-induced rather than communication-specific.

## Current notebook logic

### Participant matching

The probe notebook was extended to:

- match participants across sparsity conditions
- later balance matched participants across similarity regimes
- compute contrasts separately for:
  - `same`
  - `near`
  - `far`

This was necessary because naive sorted-ID selection biased the sample toward `far` participants.

### Temporal alignment

The probe reconstructs the input using the same `nb_steps` logic as training:

- if `nb_steps > 1`, `temporal_data(..., noise_ratio=None)` is used
- therefore the probe uses the same recurrent depth as training rather than a single-step shortcut

This was an important correction: initial probe versions under-estimated recurrent effects by effectively using `seq_len = 1`.

## Current ablation tests

### Test A: readout ablation

The first ablation removes the `comms` contribution before the readout by feeding only the `core` sequence into the readout.

Interpretation:

- broad test of whether the model uses the comms stream at all
- does not isolate whether the effect comes from recurrent communication specifically

### Test B: mask-based comms ablation

The second ablation keeps the `comms` stream present but zeros `community.comms_mask` during the forward pass, thereby disabling the recurrent communication hidden-to-hidden weights while leaving the rest of the stream intact.

Interpretation:

- narrower test of whether the recurrent communication mechanism itself matters
- more directly addresses the question of explicit communication, not just generic reliance on an auxiliary stream

### Why both tests matter

The two tests answer different questions:

- Test A: "Does the comms pathway matter at all?"
- Test B: "Does the recurrent communication mask matter specifically?"

Reading them together is more informative than either alone.

## Metrics currently used

The notebook tracks both global and task-relevant effects.

### Representation / output sensitivity

- `mean_l2_delta_logits_*`
- `mean_l2_delta_task_logits_*`

These measure how much the output moves under ablation.

### Behavioral proxy

- `acc_full`
- `acc_ablate_readout`
- `acc_ablate_mask`
- `acc_delta_*_minus_full`

The task-relevant output pair is selected using the probe label:

- probe 0 -> output dims `0:2`
- probe 1 -> output dims `2:4`

Accuracy is then computed as a thresholded angular agreement proxy on the selected output head.

### Sparsity contrasts

For both tests, direct contrasts are computed as:

- `sp=1.0` minus `no_comms`

and summarized by:

- `input_routing`
- `init_scale`
- `similarity_regime`

## Current findings

### 1. Low/moderate init, task_routed

Across `same`, `near`, and `far`, low/moderate init (`0.001`, `0.01`, `0.1`) shows:

- near-zero task-logit deltas
- near-zero behavioral deltas
- both for readout ablation and mask ablation

Interpretation:

- communication appears largely unnecessary in this regime
- and this is not merely a `far`-sampling artifact

### 2. Low/moderate init, shared

With `shared` input, low/moderate init shows:

- positive task-logit effects from communication
- much weaker and less consistent behavioral effects
- mask-based effects often as large as or larger than readout-ablation effects

Interpretation:

- communication contributes to internal/task-relevant computation
- but behavioral consequences remain weak in the main low-init regime
- effects are not well explained as pure readout artifacts

### 3. High init (`init_scale = 2`)

At high init, both `shared` and `task_routed` show substantially larger effects.

Interpretation:

- communication becomes functionally important in a high-gain / high-dynamics regime
- mask-based ablation being large indicates that recurrent communication itself is implicated, not just the existence of the comms stream

### 4. Similarity-regime dependence

After stratifying by `same`, `near`, `far`:

- low/moderate init effects are weakly regime-dependent at most
- high-init effects show clearer regime structure

Interim interpretation:

- similarity-dependent communication effects emerge more clearly in the high-init regime than in the lower-init regime

## Important limitations

### 1. Communication-specific vs scale-specific ambiguity

The high-init result is real, but not fully isolated from a broader scale-induced dynamical regime change.

Current evidence shows:

- the effect is not just a trivial readout artifact
- but it may still reflect communication operating inside a higher-gain regime, rather than a communication-only principle

### 2. Shared readout as an alternative coordination route

The current paper-style runs use `common_readout=True`, which means modules may coordinate during learning via shared output gradients even if explicit communication is weak.

Therefore, the current ablations test usage at inference, but do not fully separate:

- explicit recurrent communication
- versus coordination induced by shared readout during training

### 3. Behavioral proxy remains local

Current accuracy measures are based on the probe batch and a thresholded angular proxy, not a full end-to-end task evaluation.

### 4. Missing cells in the grid

Some sparsity/init combinations are missing, especially:

- `0.5` at some init values
- some higher-init variants

This limits clean monotonicity claims.

### 5. Architectural controls not yet crossed

The current notebook does not yet jointly compare:

- `common_readout=True` vs `False`
- `global` vs `input_only`
- readout ablation vs mask ablation

across the same full matched grid.

## Input-only vs global scaling

This is currently one of the most important unresolved interpretive axes.

### Why it matters

If the strong high-init communication effect mainly reflects general dynamical amplification, then similar effects might appear when only some parameter subsets are scaled.

In particular, `input_only` scaling is useful because it asks:

- does stronger input drive alone reproduce the apparent communication effect?
- or is scaling of recurrent/comms-related parameters required?

### Current interpretation

The existing ablation data suggest that high-init communication effects are not purely readout-stream artifacts.

However, they do not yet rule out the possibility that:

- high init generally increases representational mobility / recurrent sensitivity
- and communication becomes important as one consequence of that regime shift

### Why `input_only` is not arbitrary

An `input_only` comparison is a meaningful control because it probes where the scale effect enters:

- if `input_only` reproduces the effect, the story shifts toward general scale sensitivity
- if `global` is much stronger than `input_only`, that supports a recurrent-communication-specific role

## Readout separation / `common_readout=False`

This is another major control that remains conceptually important.

### Motivation

With `common_readout=True`, both modules can contribute to the same output and receive shared gradient structure.

That creates a possible coordination route even without strong recurrent communication.

### What a separated-readout control would ask

If modules are forced to decide more independently at the output layer, then:

- do communication effects remain?
- do they become stronger?
- or do they disappear?

### Likely interpretations

If communication effects remain under separated readout:

- stronger evidence for explicit recurrent communication as a necessary mechanism

If communication effects collapse:

- shared readout may have been doing much of the coordination work

If communication effects increase:

- the shared readout may have been masking an otherwise necessary communication role

### Why this is especially relevant in `a1b2`

In `a1b2`, the task structure already includes modular output subspaces and input-routing manipulations, so a separated-readout control would not be identical in meaning to dynspec. It would still be scientifically useful, but should be treated as an architectural comparison, not as a pure ablation.

## Working conclusions at this stage

### Safe conclusions

- In `task_routed`, low/moderate-init models appear to rely very little on communication.
- In `shared`, communication affects internal/task-relevant representations more than behavior in the low-init regime.
- At high init, communication becomes much more consequential.
- The second, mask-based test suggests that when communication matters, recurrent communication itself is implicated.

### Not yet fully isolated

- whether the high-init effect is mostly communication-specific or mostly scale-induced
- whether shared readout is providing an alternative coordination mechanism during learning

## Recommended next steps

### Priority 1: scale controls

- compare `global` vs `input_only` scaling under the same ablation logic
- ideally on the same routing/init/regime grid

### Priority 2: readout coupling controls

- compare `common_readout=True` vs `False`
- especially for `shared` input

### Priority 3: combined decomposition

Longer term, the most informative design would cross:

- scaling scope (`global` vs `input_only`)
- readout structure (`common_readout=True` vs `False`)
- communication sparsity
- readout ablation vs mask ablation

### Priority 4: cleaner behavioral evaluation

- complement the current proxy with a fuller task evaluation
- verify that conclusions based on `acc_delta_*` survive under less local metrics

## Suggested wording for current cautious interpretation

One conservative summary consistent with the present evidence is:

> Communication effects are real but highly regime-dependent. In the main low-init regime, task-routed models appear to rely little on communication, whereas shared-input models show modest communication-sensitive changes in internal/task-relevant computation with weak behavioral consequences. In the high-init regime, communication becomes much more functionally relevant, and mask-based ablations suggest this dependence is tied to recurrent communication weights rather than merely to a generic auxiliary stream. However, additional controls are needed to determine how much of this high-init effect is communication-specific versus a broader scale-induced shift in network dynamics, and how much coordination may instead be mediated by shared readout during learning.

## Open TODO checklist

- [ ] Run the ablation notebook with the final intended participant cap and confirm stability of conclusions.
- [ ] Add an `input_only` vs `global` scaling comparison under the same ablation framework.
- [ ] Add a `common_readout=False` control condition.
- [ ] Decide whether winner-take-all or another separated-readout decision rule is appropriate for `a1b2`.
- [ ] Compare readout-ablation and mask-ablation results directly in the same summary plots/tables for paper-facing interpretation.
- [ ] Strengthen behavioral evaluation beyond the current local accuracy proxy.
- [ ] Document clearly which conclusions belong to low/moderate init versus high-init only.

