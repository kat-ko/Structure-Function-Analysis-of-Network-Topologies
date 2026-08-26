# Experimental Specification

## Part II — Mathematical Formulation, Stimulus Manifold and Task Family

### Status

This section defines the mathematical objects comprising the benchmark. The objective is to specify the environment independently of any particular neural network architecture. The environment should be completely defined before considering learning algorithms or representational analyses.

---

# 1. Mathematical Objects

The benchmark consists of four independent mathematical components.

## Stimulus manifold

[
\mathcal{X}
]

defines the space of observations presented to the learner.

---

## Object identities

[
\mathcal{O}={o_1,o_2,\ldots,o_N}
]

defines a finite collection of persistent objects that exist throughout the entire experiment.

Object identities never change.

---

## Task family

[
\mathcal{T}(s)
]

defines a continuous family of mappings

[
\mathcal{T}(s):\mathcal{X}\rightarrow\mathcal{Y}
]

parameterized by a single similarity variable

[
s.
]

---

## Learning protocol

The continual learning curriculum

[
\mathcal{T}(0)
\rightarrow
\mathcal{T}(s)
\rightarrow
\mathcal{T}(0)
]

defines the sequential learning history.

No other component changes.

---

# 2. Latent Object Manifold

The benchmark assumes that every stimulus corresponds to an underlying latent object.

The learner never observes the latent representation directly.

Instead, observations are generated from latent object identities through a fixed observation function.

Formally,

[
z_i\in\mathbb{R}^{d_z}
]

denotes the latent representation of object

[
o_i.
]

The latent manifold remains fixed throughout the experiment.

---

## Proposed dimensionality

Initial implementation

[
d_z=4
]

This value is intentionally small.

The objective is not to produce high-dimensional sensory data but to define a compact latent world from which observations are generated.

Future work may investigate larger latent spaces.

---

# 3. Number of Objects

The benchmark contains

[
N=8
]

persistent object identities.

Reasons

* sufficient diversity for representational analyses,
* direct visualization remains possible,
* computationally inexpensive,
* conceptually similar to previous object-based paradigms.

Object identities are indexed

[
o_1,\ldots,o_8.
]

These identities remain fixed throughout

* A1,
* B,
* A2.

Objects are never added or removed.

---

# 4. Latent Object Generation

For every experimental seed

generate

[
N
]

latent identities

[
z_i
]

from

[
z_i
\sim
\mathcal N(0,I).
]

Generation occurs once at the beginning of the experiment.

The same latent identities are used throughout the complete continual learning protocol.

Each random seed therefore defines one independent environment.

---

# 5. Observation Space

The learner does not receive latent variables.

Instead,

observations are generated through

[
x=g(z).
]

where

[
g
]

is a fixed nonlinear observation function.

---

## Design Rationale

This separates

latent object identity

from

sensory observation.

Consequently,

the learner must organize representations from observations rather than memorizing latent coordinates.

This better reflects natural learning environments while remaining computationally simple.

---

# 6. Observation Function

Initial implementation

latent representation

↓

fixed random projection

↓

tanh nonlinearity

↓

observation vector

Formally,

[
x
=

\tanh(Wz+b)
]

where

* (W) is randomly initialized,
* (b) is randomly initialized,
* neither is trainable.

The observation function is generated once per random seed and remains fixed thereafter.

---

## Observation dimensionality

Initial implementation

[
d_x=8.
]

Thus,

latent space

[
\mathbb R^4
]

is embedded into

[
\mathbb R^8.
]

The network therefore receives an eight-dimensional continuous feature vector.

---

# 7. Observation Noise

Every presentation of object

[
o_i
]

produces

[
x_i
===

g(z_i)
+\epsilon.
]

where

[
\epsilon
\sim
\mathcal N(0,\sigma^2I).
]

Noise serves two purposes.

* prevents exact memorization,
* produces repeated but non-identical observations.

---

## Noise usage

Noise is added

during training only.

Evaluation is performed using

noise-free observations.

This ensures that transfer and interference measurements reflect learned object representations rather than sampling variability.

---

# 8. Stimulus Distribution

Every object is sampled with equal probability.

No curriculum is imposed within individual phases.

No object receives preferential exposure.

The stimulus distribution therefore remains constant throughout

A1,

B,

A2.

---

# 9. Output Space

The benchmark initially uses continuous regression targets.

Output dimensionality

[
d_y=2.
]

Targets lie on the unit circle.

Representing outputs continuously has several advantages.

* continuous task transformations,
* direct control of similarity,
* compatibility with mean squared error,
* avoidance of discontinuous class changes.

---

# 10. Target Representation

Each object

[
o_i
]

is associated with one angular target

[
\theta_i.
]

The corresponding output vector is

[
y_i
===

(\cos\theta_i,
\sin\theta_i).
]

---

## Initial angle assignment

Objects are distributed uniformly around the circle.

For

[
N=8
]

objects,

angles become

[
0^\circ,
45^\circ,
90^\circ,
135^\circ,
180^\circ,
225^\circ,
270^\circ,
315^\circ.
]

Reasons

* equal angular spacing,
* symmetric output geometry,
* reproducible analyses,
* elimination of unnecessary randomness.

---

# 11. Definition of the Reference Task

The reference task

[
\mathcal T(0)
]

maps every object observation

[
x_i
]

to

its assigned target vector

[
y_i.
]

Formally,

[
\mathcal T(0):
x_i
\mapsto
(\cos\theta_i,
\sin\theta_i).
]

This mapping defines both

A1

and

A2.

---

# 12. Continuous Task Family

The benchmark defines a family of tasks

[
\mathcal T(s).
]

The stimulus manifold remains unchanged.

Only target vectors change.

For similarity parameter

[
s,
]

Task

[
\mathcal T(s)
]

becomes

[
x_i
\mapsto
(\cos(\theta_i+s),
\sin(\theta_i+s)).
]

Every object experiences the same transformation.

Consequently,

the relationship between objects remains constant while the task changes continuously.

---

# 13. Similarity Parameter

Task similarity is defined intrinsically by

[
s.
]

Interpretation

[
s=0
]

identical tasks.

Small

[
s
]

high similarity.

Intermediate

[
s
]

partial similarity.

Large

[
s
]

low similarity.

No empirical estimation of similarity is required.

The benchmark therefore contains an explicit ground-truth notion of task similarity.

---

# 14. Initial Similarity Values

The initial benchmark should evaluate several values of

[
s.
]

Suggested values

[
0,
\frac{\pi}{12},
\frac{\pi}{6},
\frac{\pi}{4},
\frac{\pi}{3},
\frac{\pi}{2},
\pi.
]

These provide a continuous spectrum from identical to maximally different tasks.

The exact grid may later be adjusted depending on observed learning dynamics.

---

# 15. Invariances

The following quantities remain invariant across all task transformations.

* object identities,
* latent manifold,
* observation mapping,
* stimulus statistics,
* sampling distribution,
* output dimensionality,
* network architecture,
* optimizer,
* training procedure.

Only

[
\mathcal T(s)
]

changes.

---

# 16. Working Assumptions

The benchmark currently assumes that a rigid transformation of the output manifold is sufficient to induce meaningful representational reorganization.

This assumption is intentionally treated as a working hypothesis rather than an established fact.

If the initial benchmark fails to produce informative representational dynamics,

future work may replace rigid rotations with richer task deformations while preserving every other component of the benchmark.

Beginning with the mathematically simplest task family provides the clearest baseline against which subsequent extensions can be evaluated.
