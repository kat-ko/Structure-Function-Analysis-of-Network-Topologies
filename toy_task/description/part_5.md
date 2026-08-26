# Experimental Specification

## Part V — Implementation Plan, Validation Strategy, Ablations and Future Extensions

### Status

This section defines the practical implementation strategy, validation procedure, debugging order and planned extensions. The objective is to minimize engineering complexity while ensuring that every stage of implementation can be verified independently before proceeding.

---

# 1. Implementation Philosophy

The benchmark should be implemented incrementally.

Each implementation stage should answer one scientific or technical question before introducing additional complexity.

No component should be added until the previous stage has been validated.

The objective is not to implement the complete benchmark immediately, but to establish a reproducible experimental platform whose behaviour is understood at every stage.

---

# 2. Development Stages

## Stage 1 — Environment

Implement

* latent object generation,
* observation generation,
* target generation,
* similarity parameter,
* task switching.

Validation

Visualize

* latent objects,
* observed objects,
* output targets.

Verify

* objects remain fixed,
* only targets change,
* task similarity behaves as expected.

No neural network is required.

---

## Stage 2 — Learning Task

Implement

single recurrent network.

Validation

Train only

[
\mathcal T(0).
]

Verify

* convergence,
* stable representations,
* reproducible learning curves.

No continual learning.

No modularity.

No learning regime manipulation.

---

## Stage 3 — Continual Learning

Implement

A1

↓

B

↓

A2.

Validation

Measure

* transfer,
* interference.

Confirm

task switching

functions correctly.

Representational analyses

remain optional

during this stage.

---

## Stage 4 — Representation Extraction

Implement

representation recording.

Validation

Verify

representation matrices

are

stable,

consistent,

reproducible.

Generate

initial PCA plots.

Inspect

representation trajectories

through learning.

---

## Stage 5 — Modular Architecture

Replace

dense RNN

with

modular RNN.

Everything else

remains unchanged.

Validation

Verify

identical learning

on

Task A

before comparing

continual learning.

---

## Stage 6 — Learning Regimes

Introduce

initialization scaling

γ.

Validation

Confirm

effective dimensionality

changes

with γ

as expected.

Only after this stage

should

complete experiments

begin.

---

# 3. Verification Checklist

Before running experiments

verify

Environment

✓

latent identities fixed

✓

observation mapping fixed

✓

balanced sampling

✓

noise generation

✓

continuous targets

✓

task similarity

implemented correctly.

---

Network

✓

correct hidden size

✓

correct output size

✓

correct loss

✓

hidden state reset

✓

optimizer

✓

learning rate.

---

Continual Learning

✓

weights preserved

between phases

✓

tasks switch correctly

✓

stimuli remain unchanged.

---

Representations

✓

correct extraction

✓

correct object ordering

✓

consistent storage.

---

# 4. Expected Behaviour

Before

Task B

all networks

should converge

to

low prediction error.

Immediately after

switching

to

Task B

performance

should depend on

similarity.

Higher similarity

should produce

higher transfer.

Lower similarity

should produce

lower transfer.

Immediately after

returning

to

Task A

performance

should depend on

interference.

Higher similarity

is expected

to preserve

more previous knowledge.

These expectations

should hold

independently

of architecture.

Architecture

is expected

to modify

the magnitude

or mechanism

of these effects.

---

# 5. Failure Modes

The benchmark

may fail

for several reasons.

---

Task too simple

Indicators

* immediate convergence,
* no transfer differences,
* identical representations.

Possible solution

Increase

representational richness

without changing

overall philosophy.

---

Task too difficult

Indicators

* failure to learn,
* unstable optimization,
* noisy representations.

Possible solution

Reduce

latent dimensionality

or

simplify

target mapping.

---

Representations

remain identical

across conditions.

Possible causes

* task similarity

insufficient,

* hidden dimension

too large,

* learning regime

too lazy.

---

# 6. Planned Ablations

The benchmark

is intentionally minimal.

Additional complexity

should only be introduced

after

baseline behaviour

is understood.

---

Ablation 1

Two correlated outputs.

Purpose

Determine

whether

Holton-style

multi-output supervision

contributes

to

representational richness.

---

Ablation 2

Non-rigid task transformations.

Replace

rigid output rotation

with

continuous task deformation.

Purpose

Determine

whether

representation organization

depends on

task deformation

rather than

simple rotation.

---

Ablation 3

More structured latent manifolds.

Replace

Gaussian latent identities

with

prototype families

or

hierarchical latent structures.

Purpose

Investigate

whether

latent structure

changes

representational allocation.

---

Ablation 4

Feature-routed modular architecture.

Replace

shared routing

with

feature-specific routing.

Purpose

Investigate

how

routing constraints

modify

representation organization.

---

Ablation 5

Hierarchical architectures.

Introduce

multiple recurrent layers.

Purpose

Study

interaction between

hierarchical organization

and

functional modularity.

This extension

lies beyond

the scope

of the present benchmark.

---

# 7. Reproducibility

Every experiment

must record

Random seed

Architecture

Initialization scale

Task similarity

Optimizer settings

Training duration

Noise level

Object identities

Observation mapping

Model parameters

Representation checkpoints.

No experiment

should depend

on

implicit defaults.

---

# 8. Computational Cost

The benchmark

is intentionally designed

to be

computationally inexpensive.

Compared to

participant-matched

training schedules,

the benchmark

should permit

substantially more

random seeds,

hyperparameter sweeps,

representation analyses,

pilot experiments,

and

ablation studies.

Fast iteration

is considered

an explicit design objective.

---

# 9. Benchmark Limitations

The benchmark

does not attempt

to model

human learning.

It does not

represent

real-world perception.

It does not

benchmark

continual learning algorithms.

It intentionally

removes

many complexities

present in

natural environments.

These simplifications

are deliberate.

The objective

is to isolate

one scientific question

under

controlled conditions.

---

# 10. Scientific Interpretation

Positive result

Similarity-dependent

representational organization

reproduces.

Interpretation

The phenomenon

does not depend

on

participant schedules

or

task-specific details

of previous work.

Instead,

it emerges

from

more general principles

of

task similarity,

learning dynamics,

and

network structure.

---

Negative result

No systematic effects

are observed.

Interpretation

The previous findings

may depend on

additional properties

of the original task,

such as

multi-output structure,

richer task geometry,

or

other forms

of representational complexity.

This immediately

motivates

the planned ablations.

---

# 11. Long-Term Research Program

The benchmark

is intended

to become

the first member

of a family

of increasingly complex

continual learning environments.

Possible future directions

include

* richer task transformations,
* hierarchical architectures,
* structured latent manifolds,
* compositional tasks,
* multiple sequential tasks,
* evolving task structures,
* adaptive modularity,
* emergence of specialization.

The underlying philosophy,

however,

remains unchanged.

A fixed environment,

a continuously parameterized task family,

and explicit representational analyses

provide the foundation

for studying

how structural priors

shape continual learning.

---

# 12. Final Specification Summary

The benchmark consists of

* eight persistent latent object identities,
* fixed nonlinear observation generation,
* continuous two-dimensional output targets,
* a continuously parameterized family of task mappings,
* balanced A1→B(s)→A2 continual learning,
* dense and modular single-layer recurrent networks,
* rich and lazy learning regimes,
* systematic behavioural and representational analyses.

The benchmark is deliberately minimal.

Every component included serves the single objective of isolating how continuous changes in task similarity influence representational organization under different structural priors.

This benchmark constitutes the baseline platform.

Future work should increase complexity only after the mechanisms operating within this controlled setting are understood.
