# Experimental Specification

## Part I — Scientific Objective, Design Philosophy and Experimental Scope

### Status

This document specifies a synthetic continual learning benchmark designed to investigate how structural priors influence representational organization during sequential learning. The document is intended as an implementation specification. Every design choice should be sufficiently explicit that an independent researcher can implement the benchmark without knowledge of previous publications or discussions.

The benchmark is intentionally minimal. It is not intended as a general continual learning benchmark or a cognitive model. Instead, it provides a controlled environment for studying mechanisms underlying transfer, interference, and representational organization.

---

# 1. Scientific Motivation

Continual learning studies often evaluate algorithms on benchmark datasets in which task similarity, task structure, input statistics, exposure schedules, and curriculum are simultaneously modified. Consequently, it is often unclear which experimental factor is responsible for observed differences in transfer or forgetting.

The purpose of the present benchmark is to isolate one specific factor:

> How does continuous variation in task similarity influence representational organization under different structural priors?

The benchmark therefore minimizes all sources of variation except the task mapping itself.

---

# 2. Scientific Question

The benchmark is designed to answer the following question.

Given

* identical stimulus identities,
* identical exposure,
* identical curriculum,
* identical optimization,
* identical model capacity,

how does changing only the similarity between two sequential tasks affect

* learning,
* transfer,
* interference,
* representational geometry,

and how do these effects depend on

* network architecture,
* learning regime?

---

# 3. Central Hypothesis

Internal representations are expected to reorganize continuously as task similarity changes.

Structural priors are hypothesized to influence

* representational reuse,
* representational separation,
* representational drift,

rather than directly determining task performance.

Transfer and interference are treated as consequences of representational organization.

Formally,

Architecture × Learning Regime × Task Similarity

↓

Representational Organization

↓

Transfer / Interference

The benchmark is designed to isolate this causal pathway.

---

# 4. Design Philosophy

The benchmark follows five design principles.

## Principle 1 — Task similarity is an independent variable

Task similarity is explicitly defined by the experimenter.

It is not estimated from performance after training.

A single continuous parameter controls the relationship between sequential tasks.

No other aspect of the experiment changes.

---

## Principle 2 — Fixed stimulus manifold

The environment remains constant throughout learning.

The same object identities are presented during every phase.

Only the task mapping changes.

This separates

environment change

from

task change.

---

## Principle 3 — Controlled learning history

Every architecture receives

* identical observations,
* identical order,
* identical exposure,
* identical optimization.

Performance differences therefore arise only from

* architecture,
* learning regime,
* task similarity.

---

## Principle 4 — Observable representations

Each stimulus possesses a persistent identity.

Representations can therefore be tracked

* before learning,
* during learning,
* after task switches,
* after relearning.

This enables direct analysis of representational dynamics.

---

## Principle 5 — Minimal experimental complexity

Every component included in the benchmark must directly contribute to answering the scientific question.

Complexity originating from

* human participant schedules,
* dataset-specific preprocessing,
* perception,
* benchmark engineering,

is intentionally removed.

---

# 5. Scope

The benchmark is designed for mechanistic investigation.

It is not intended to

* maximize benchmark difficulty,
* achieve state-of-the-art continual learning performance,
* model human behavior,
* compare continual learning algorithms.

Instead, the benchmark should provide a controlled experimental platform for understanding how network structure organizes representations under sequential learning.

---

# 6. Experimental Variables

The benchmark contains three independent variables.

## Architecture

Network topology.

Initially

* dense recurrent network,
* modular recurrent network.

No other architectural differences are introduced.

---

## Learning Regime

Initialization scale controlling rich versus lazy learning dynamics.

The exact implementation follows the previous experimental protocol.

No additional optimization changes are introduced.

---

## Task Similarity

A single continuous parameter

s

controls the similarity between sequential tasks.

Task similarity is therefore mathematically defined rather than empirically inferred.

---

# 7. Controlled Variables

The following quantities remain fixed across every experiment.

* stimulus identities
* latent object manifold
* observation mapping
* curriculum
* optimizer
* learning rate
* batch size
* number of recurrent units
* number of recurrent layers
* training duration
* exposure frequency
* evaluation procedure
* representational analyses

Changing any of these would introduce additional experimental factors and therefore complicate interpretation.

---

# 8. Benchmark Philosophy

The benchmark should be viewed as a parameterized family of tasks rather than a single task.

Let

X

denote the stimulus manifold.

Let

T(s)

denote a task parameterized by similarity parameter s.

The continual learning protocol becomes

T(0)

↓

T(s)

↓

T(0)

where

* T(0) represents the reference task,
* T(s) represents a continuously transformed task,
* s = 0 corresponds to identical tasks,
* increasing s corresponds to progressively less similar tasks.

The stimulus manifold remains invariant throughout learning.

Only the mapping

T(s)

changes.

This distinction is fundamental.

The benchmark studies continual adaptation to changing task requirements rather than changing environments.

---

# 9. Experimental Objectives

The benchmark has five objectives.

Objective 1

Determine whether similarity-dependent structural effects reproduce under a controlled synthetic task.

Objective 2

Measure how representational organization changes continuously with task similarity.

Objective 3

Determine whether learning regime influences representational allocation.

Objective 4

Determine whether structural priors influence representational reuse or representational separation.

Objective 5

Provide a minimal experimental platform that can later be extended toward richer task families, hierarchical architectures and more complex notions of task structure.

---

# 10. Expected Scientific Contribution

The benchmark is intended to contribute

* a mathematically controlled family of continual learning tasks,
* explicit manipulation of task similarity,
* reproducible representational analyses,
* controlled comparison of structural priors,
* a foundation for future studies on task structure, compositionality and hierarchy.

The benchmark is intentionally designed so that additional complexity can be introduced incrementally without changing the underlying experimental philosophy.
