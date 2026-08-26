# Benchmark Specification Patch

This document contains corrections and additions that should be merged
into Parts I--V of the specification.

## 1. Freeze benchmark definition

The following choices are considered fixed for the baseline
implementation and are **not** experimental variables.

-   Number of latent objects: `N = 8`
-   Latent dimensionality: `d_z = 4`
-   Observation dimensionality: `d_x = 8`
-   Output dimensionality: `d_y = 2`
-   Single-layer Elman RNN
-   Dense vs. modular architectures only
-   A1 → B(s) → A2 curriculum
-   Continuous output regression
-   Fixed epoch budget
-   Balanced exposure
-   Rich/lazy manipulation through initialization scale γ
-   Existing representational analyses

Any modifications (multiple outputs, hierarchical architectures, task
deformation, richer latent manifolds, etc.) are reserved for later
ablations.

------------------------------------------------------------------------

## 2. Observation model (replace Part II §§5--7)

The observation function is

\[ x=`\tanh`{=tex}(Wz+b)+`\epsilon`{=tex}. \]

where

-   (W`\in`{=tex}`\mathbb{R}`{=tex}\^{8`\times4`{=tex}})
-   (W\_{ij}`\sim`{=tex}`\mathcal `{=tex}N(0,1/d_z))
-   (b=`\mathbf0`{=tex})
-   \(W\) and (b) are sampled once per random seed and never updated.

Noise

\[ `\epsilon`{=tex}`\sim`{=tex}`\mathcal `{=tex}N(0,`\sigma`{=tex}\^2I)
\]

with

-   σ = 0.05 during training,
-   σ = 0 during evaluation.

Each seed generates

-   latent objects,
-   observation mapping,
-   training order,
-   network initialization.

These remain fixed throughout A1, B and A2.

------------------------------------------------------------------------

## 3. Target generation

Object base angles follow the patch specification. **Default (`angle_mode=even`):**
evenly spaced on the ring at 45° separation. **Legacy ablation (`angle_mode=random`):**
angles derived from latent draws `θ_i = atan2(z_{i,1}, z_{i,0})` (median min-separation
~4°; see nb09).

For object i (even mode)

\[ `\theta`{=tex}\_i=`\frac{2\pi(i-1)}{N}`{=tex} \]

giving

0°,45°,90°,135°,180°,225°,270°,315°.

Task A

\[ y_i=(`\cos`{=tex}`\theta`{=tex}\_i,`\sin`{=tex}`\theta`{=tex}\_i). \]

Task B(s)

\[
y_i=(`\cos`{=tex}(`\theta`{=tex}\_i+s),`\sin`{=tex}(`\theta`{=tex}\_i+s)).
\]

Similarity grid is fixed to

\[ s`\in`{=tex} `\left`{=tex}{ 0, `\pi`{=tex}/12, `\pi`{=tex}/6,
`\pi`{=tex}/4, `\pi`{=tex}/3, `\pi`{=tex}/2, `\pi`{=tex}
`\right`{=tex}}. \]

------------------------------------------------------------------------

## 4. Training schedule

One epoch consists of exactly one presentation of every object.

Procedure

1.  Randomly permute the eight object identities.
2.  Generate one noisy observation for each object.
3.  Perform one SGD update per observation.

Therefore

-   8 updates per epoch,
-   100 epochs per phase,
-   800 updates per phase,
-   2400 updates per complete A1→B→A2 experiment.

Training uses balanced exposure only.

No curriculum.

No class imbalance.

------------------------------------------------------------------------

## 5. Optimizer (replace abstract wording)

Optimizer

-   SGD
-   learning rate = 0.01
-   momentum = 0.0
-   weight decay = 0.0
-   batch size = 1
-   no learning-rate schedule
-   no gradient clipping

Optimizer state is preserved across phase transitions.

Weights are never reset.

------------------------------------------------------------------------

## 6. Architecture specification

Dense network

-   Input dimension = 8
-   Hidden dimension = 50
-   One Elman recurrent layer
-   tanh activation
-   Linear output layer (50→2)

Modular network

-   Two recurrent modules
-   25 hidden units per module
-   No recurrent inter-module connections
-   Shared linear output layer (50→2)

Hidden state

-   reset after every stimulus presentation
-   sequence length = 3 repeated inputs
-   loss computed only at final timestep.

------------------------------------------------------------------------

## 7. Behavioural metrics

Forward transfer

Evaluate loss on Task B immediately after switching from A1 and before
any parameter update on B.

Interference

Evaluate loss on Task A immediately after switching back from B and
before any parameter update during A2.

Learning curves

Store average training loss after every epoch.

------------------------------------------------------------------------

## 8. Representation extraction

Extract hidden representations

-   before training,
-   every 10 epochs,
-   immediately before every phase transition,
-   immediately after every phase transition,
-   final epoch of every phase.

Store

-   latent vectors,
-   observations,
-   hidden representations,
-   output predictions,
-   targets.

------------------------------------------------------------------------

## 9. Design rationale

The benchmark intentionally adopts the mathematically simplest
continuously parameterized task family.

The rigid target rotation is not assumed to be the unique or correct
definition of task similarity.

Instead, it serves as the baseline from which richer task deformations
can later be introduced.

The objective of the first study is to determine whether
similarity-dependent structural effects reproduce under this minimal
controlled setting before increasing task complexity.
