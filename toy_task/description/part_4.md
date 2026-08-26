# Experimental Specification

## Part IV — Evaluation Metrics, Representational Analyses and Experimental Outputs

### Status

This section specifies every quantity measured during the benchmark. The benchmark is designed to analyze not only behavioral performance but also the evolution of internal representations throughout continual learning.

The primary scientific object is the hidden representation rather than the final task accuracy.

---

# 1. Evaluation Philosophy

The benchmark distinguishes between two levels of analysis.

Behavioral level

* learning
* transfer
* interference

Representational level

* geometry
* dimensionality
* drift
* reuse
* specialization

Behavioral measurements determine whether continual learning occurs.

Representational analyses attempt to explain why.

---

# 2. Evaluation Timepoints

Measurements are collected throughout training rather than only after convergence.

Minimum extraction points

Before training

End of A1

Beginning of B

Regular intervals throughout B

End of B

Beginning of A2

Regular intervals throughout A2

End of A2

For initial implementation,

representations should be extracted every

5–10 epochs

plus

all phase transitions.

This allows reconstruction of representational trajectories.

---

# 3. Evaluation Dataset

All analyses use

noise-free observations.

The benchmark therefore evaluates

object identities

rather than individual noisy presentations.

Each object is evaluated exactly once.

Consequently,

representation matrices have dimensions

Number of Objects

×

Hidden Dimension.

For

N=8

objects,

one representation matrix is

[
8\times H
]

where

H

denotes hidden dimensionality.

---

# 4. Behavioural Metrics

The benchmark measures four behavioural quantities.

---

## Training Performance

Average loss

per epoch

during

A1,

B,

A2.

Purpose

Verify successful learning.

Determine optimization dynamics.

---

## Learning Speed

Loss

or prediction error

as a function of training epoch.

Purpose

Determine

learning efficiency

under different

architectures,

learning regimes,

similarity conditions.

---

## Forward Transfer

Immediately after switching

from

A1

to

B,

measure

performance

before substantial retraining.

Transfer quantifies

how much previously learned knowledge

supports

the new task.

Higher transfer indicates

greater representational reuse.

---

## Interference

Immediately after switching

from

B

back to

A2,

measure

performance

on the original task.

Interference quantifies

how much knowledge

was disrupted

during Phase B.

Lower interference indicates

greater preservation

of previous representations.

---

# 5. Representation Extraction

Representations are extracted

from the recurrent hidden layer.

Output-layer activations

are not analyzed

except where explicitly stated.

For every object

the final hidden activation

is stored.

These activations define

the representational state

of the network

at one training epoch.

---

# 6. Representation Matrix

At every extraction point

construct

[
R
=

\begin{bmatrix}
h_1\
h_2\
\vdots\
h_N
\end{bmatrix}
]

where

[
h_i
]

is the hidden representation

of object

[
o_i.
]

Every representational analysis

operates on

R.

---

# 7. Principal Component Analysis

Perform PCA

on

R.

Objectives

Visualize

representation geometry.

Observe

task-dependent

reorganization.

Determine

whether

representations

cluster,

rotate,

or separate.

Outputs

2-dimensional projection

3-dimensional projection

variance explained

principal axes.

---

# 8. Effective Dimensionality

Estimate representational dimensionality

using the same procedure

as the previous study.

Possible measures include

participation ratio

or

number of principal components

required to explain

a specified proportion

of variance.

This analysis

connects directly

to the previous work

on rich versus lazy learning.

---

# 9. Principal Angles

Compute principal angles

between representation subspaces.

Comparisons include

A1

vs

B

A1

vs

A2

B

vs

A2

Objectives

Measure

subspace reuse

versus

subspace separation.

Determine

whether

modularity

promotes

greater representational specialization.

---

# 10. Representational Similarity Analysis

Construct

representational similarity matrices

using

hidden representations.

Compute

pairwise similarities

between

object representations.

Possible similarity measures

cosine similarity

or

correlation.

Compare

similarity matrices

between

training phases

and

architectures.

Purpose

Measure

global organization

of object representations.

---

# 11. Centered Kernel Alignment (Optional)

If computational resources permit,

compute

CKA

between

representation matrices.

Purpose

Compare

global representational organization

independently

of

linear transformations.

This analysis

may complement

principal angles.

---

# 12. Representational Drift

Define representational drift

as the change

in hidden representation

of one object

through learning.

For every object

compute

distance

between

A1

B

A2

representations.

Possible measures

Euclidean distance

cosine distance

subspace projection.

Purpose

Determine

whether

objects

return

to their original representation

after relearning.

---

# 13. Object-Level Analysis

Representations

should be analyzed

both

globally

and

per object.

Questions include

Do all objects

drift similarly?

Do some objects

remain stable?

Does similarity

affect

all objects equally?

Does modularity

stabilize

particular object representations?

---

# 14. Phase Comparisons

Every representational analysis

should compare

A1

↓

B

↓

A2

rather than

analyzing

individual phases

independently.

The benchmark

is designed

to study

representation dynamics,

not

static representations.

---

# 15. Expected Phenomena

The benchmark

is intended

to detect

changes in

Representational overlap

Representational separation

Representational reuse

Representational recovery

Representational drift

Representational dimensionality

Subspace organization.

These constitute

the primary outcomes

of the benchmark.

Behavioural performance

serves primarily

to contextualize

representational changes.

---

# 16. Statistical Analysis

Every experimental condition

should be repeated

across multiple random seeds.

For every metric

report

mean

standard deviation

confidence interval

where appropriate.

Architectures

should be compared

using identical seeds

to reduce environmental variability.

---

# 17. Data Storage

For every experiment

store

Training losses

Prediction errors

Model checkpoints

Hidden representations

Output predictions

Representational analyses

Configuration parameters

Random seed

Task similarity

Initialization scale

Architecture

This enables

reproducibility

and

post hoc analyses.

---

# 18. Visualization

Minimum figures

Learning curves

Transfer

Interference

PCA trajectories

Representational similarity matrices

Principal angle plots

Effective dimensionality

Representational drift

Each visualization

should compare

Dense

vs

Modular

across

γ

and

task similarity.

---

# 19. Primary Hypothesis Tests

The benchmark

is designed

to evaluate

the following hypotheses.

H1

Representational organization

changes continuously

with task similarity.

H2

Rich

and

lazy

learning

produce

different

representational organizations.

H3

Structural modularity

changes

how

representations

are allocated

across sequential tasks.

H4

Transfer

and

interference

are explained

by

representational organization

rather than

architecture alone.

---

# 20. Success Criteria

The benchmark should be considered successful if it satisfies three conditions.

First,

all architectures

successfully learn

the reference task.

Second,

continuous variation

of

task similarity

produces

systematic differences

in

transfer

interference

or

representational organization.

Third,

the benchmark

reveals measurable differences

between

dense

and

modular

architectures

under at least some

learning regimes

or similarity conditions.

If these criteria are met,

the benchmark provides a suitable experimental platform for investigating structural influences on continual learning and can serve as the foundation for larger-scale studies involving richer task families, hierarchical architectures and more complex notions of task structure.
