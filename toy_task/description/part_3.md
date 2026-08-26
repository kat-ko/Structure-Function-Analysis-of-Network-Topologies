# Experimental Specification

## Part III — Continual Learning Protocol, Network Architectures and Training Procedure

### Status

This section specifies the complete continual learning protocol, neural network architectures, optimization procedure and experimental variables. Combined with Parts I and II, this defines the full experimental benchmark independently of any implementation language.

---

# 1. Continual Learning Protocol

The benchmark follows a three-phase continual learning curriculum.

[
\mathcal{T}(0)
\rightarrow
\mathcal{T}(s)
\rightarrow
\mathcal{T}(0)
]

These phases are referred to as

* Phase A1
* Phase B
* Phase A2

The reference task is identical in A1 and A2.

Only the task mapping changes during Phase B.

No other aspect of the environment changes.

---

# 2. Phase A1

The network is initialized from random weights.

Training proceeds exclusively on

[
\mathcal T(0).
]

Objectives

* learn stable object representations,
* converge on the reference task,
* establish baseline representational geometry.

No measurements of transfer or interference are made during this phase.

Representations extracted at the end of A1 serve as the reference for later analyses.

---

# 3. Phase B

Training continues from the final weights obtained after A1.

Weights are **not reset**.

The task changes from

[
\mathcal T(0)
]

to

[
\mathcal T(s).
]

The stimulus manifold remains unchanged.

The network therefore experiences

* identical objects,
* identical observations,
* identical sampling,

while receiving different supervision.

Measurements performed during Phase B include

* learning speed,
* forward transfer,
* representational drift,
* representational dimensionality,
* representational overlap.

---

# 4. Phase A2

Training continues directly from Phase B.

Weights again are **not reset**.

The original task

[
\mathcal T(0)
]

is restored.

Measurements include

* recovery speed,
* interference,
* representational recovery,
* comparison with original representations from A1.

---

# 5. Balanced Exposure

Every object receives identical exposure.

Within one epoch

every object is sampled equally often.

No object receives preferential training.

No task-dependent sampling is permitted.

This removes exposure as a confounding variable.

---

# 6. Training Order

The benchmark deliberately removes participant-specific schedules.

Within each epoch

objects are sampled in randomized order.

Sampling is balanced.

Every epoch therefore contains exactly one presentation of each object (or an equal number of presentations if multiple repetitions per epoch are chosen).

A new random ordering is generated every epoch.

This preserves stochastic gradient optimization while maintaining equal exposure.

---

# 7. Number of Epochs

The benchmark should initially use a fixed number of epochs per phase.

Training should **not** terminate using convergence criteria.

Reason

Adaptive stopping would produce different amounts of learning across

* architectures,
* learning regimes,
* similarity conditions.

Using a fixed computational budget ensures direct comparability.

Initial recommendation

100 epochs per phase.

This value may be adjusted after pilot experiments.

---

# 8. Training Samples

Each presentation consists of

latent object

↓

observation generation

↓

noise addition

↓

network input

↓

continuous target

Training data are generated online.

No dataset is stored.

Every presentation is an independent noisy observation of one persistent object identity.

---

# 9. Neural Network Inputs

Input dimensionality

[
d_x=8.
]

Input vectors correspond only to observations.

The network never receives

* latent variables,
* object indices,
* task identifiers.

Task identity must therefore be inferred solely from sequential learning.

---

# 10. Neural Network Outputs

Output dimensionality

[
d_y=2.
]

The network predicts

[
(\hat y_x,\hat y_y).
]

Loss is computed against the target vector

[
(\cos\theta,\sin\theta).
]

---

# 11. Dense Recurrent Architecture

The baseline model consists of

Input

↓

single recurrent layer

↓

linear readout

The recurrent layer is identical to the previous study.

Recommendations

* single Elman RNN,
* tanh activation,
* hidden dimension identical to previous work,
* linear output layer.

The architecture intentionally remains shallow.

No additional hierarchy is introduced.

---

# 12. Modular Recurrent Architecture

The modular network follows the previous implementation.

Input

↓

parallel recurrent modules

↓

shared readout

Each module possesses independent recurrent dynamics.

Modules do not communicate internally in the initial benchmark.

The shared linear readout combines module activations into the final prediction.

The objective is to isolate the effect of structural modularity while keeping every other architectural property unchanged.

---

# 13. Architectural Constraints

To ensure fair comparison

the following quantities should be matched across architectures.

* total hidden units,
* recurrent activation,
* optimizer,
* initialization procedure,
* output layer,
* parameter scale (where applicable).

Only connectivity differs.

---

# 14. Hidden State Handling

The benchmark follows the previous recurrent implementation.

Hidden states are reset between object presentations.

Recurrence therefore operates within individual stimulus presentations rather than across the complete sequence.

Continual learning occurs through weight updates rather than persistent hidden states.

This simplifies interpretation of representational changes.

---

# 15. Temporal Input Structure

Each observation may optionally be repeated for multiple recurrent time steps.

Example

[
x
\rightarrow
x
\rightarrow
x
]

The loss is computed only at the final time step.

This matches the previous experimental implementation.

The exact sequence length should remain constant throughout the benchmark.

---

# 16. Loss Function

Mean squared error.

[
\mathcal L
==========

||y-\hat y||^2.
]

The continuous output representation naturally supports regression.

No classification loss is required.

---

# 17. Optimizer

To maintain continuity with the previous study

the optimizer should remain unchanged.

Initial implementation

* SGD,
* identical learning rate,
* identical momentum,
* identical weight decay.

No optimizer tuning is performed between conditions.

---

# 18. Learning Regime Manipulation

The benchmark adopts the same initialization manipulation as the previous work.

Let

[
\gamma
]

denote the initialization scale.

After standard weight initialization

all trainable recurrent parameters are multiplied by

[
\gamma.
]

Different values of

[
\gamma
]

produce different learning regimes ranging from rich to lazy.

No other hyperparameters change.

---

# 19. Experimental Variables

The complete experimental design consists of

Architecture

×

Initialization Scale

×

Task Similarity

×

Random Seed

Specifically

Architecture

* Dense
* Modular

Learning regime

* selected values of γ

Task similarity

* selected values of s

Random seeds

* independent environments.

No additional variables are manipulated in the initial benchmark.

---

# 20. Random Seeds

Every random seed generates

* new latent object identities,
* new observation mapping,
* new network initialization,
* new training order.

Within one seed

these quantities remain fixed throughout

A1,

B,

A2.

Each seed therefore represents one independent realization of the benchmark environment.

---

# 21. Experimental Pipeline

For each seed

generate environment

↓

initialize network

↓

train A1

↓

store weights and representations

↓

continue training on B

↓

store weights and representations

↓

continue training on A2

↓

store weights and representations

↓

repeat for all architectures

↓

repeat for all γ

↓

repeat for all similarity values

The complete benchmark therefore differs from standard continual learning benchmarks in one critical respect.

The environment is regenerated only between seeds.

Within an experiment, the environment remains completely fixed.

Consequently, any observed changes in performance or representation arise exclusively from sequential changes in the task mapping rather than changes in the stimulus distribution. This distinction is fundamental to the interpretation of all subsequent analyses.
