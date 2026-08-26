# Late-Breaking Abstract Outline (Revised)

## Working Title

**Hidden Assumptions About Tasks in Continual Learning**

Alternative:

**Tasks as Discretizations of Experience in Continual Learning**

## Target Positioning

This should read as a concise work-in-progress perspective paper rather than a finished taxonomy or formal theory paper. The contribution is conceptual: it argues that the notion of a task in continual learning often compresses several distinct properties of a sequential learning problem, proposes a preliminary decomposition of those properties, and motivates a larger follow-up framework paper.

## Core Message

Continual learning research is organized around tasks, but the notion of a task often conflates multiple distinct properties of a sequential learning problem. In many settings, what is called a task reflects at least three layers at once: the structure of the underlying experience stream, the way an observer discretizes that stream into units, and the information about that discretization that is available to the learner. As a result, benchmark comparisons can unintentionally mix together problem structure, protocol design, and learner observability. A better starting point is to treat tasks as researcher-defined discretizations of structured experience.

## Review-Safe Thesis

The paper should make a strong conceptual claim without sounding overstated. A good target formulation is:

> We argue that the notion of a task in continual learning often bundles together several conceptually distinct properties of sequential learning problems, and that separating these properties can clarify what current benchmarks do and do not allow us to compare.

This is stronger than "benchmarks are inconsistent" but safer than claiming the entire field has fundamentally misunderstood its basic unit of analysis.

## Recommended 2-4 Page Structure

### 1. Introduction and Problem Statement (0.5-0.75 page)

Goal:

- establish why the notion of a task matters in continual learning
- show that the field relies on it without always separating what is bundled into it
- motivate why this matters scientifically, not just terminologically

Key points:

- continual learning is usually framed as learning across a sequence of tasks
- what counts as a task varies across settings, but the deeper issue is that "task" often bundles distinct ingredients of the learning problem
- without separating those ingredients, differences in forgetting, transfer, replay, or modularity are hard to attribute cleanly
- this submission is a first step toward a more principled framework

Suggested scientific motivation:

> Without distinguishing these assumptions, it remains difficult to tell whether reported differences in continual-learning performance arise from the algorithm itself, from the structure of the experience stream, or from what information about that structure is revealed to the learner.

Suggested closing sentence for the section:

> We argue that many benchmark differences should be understood not simply as differences between tasks, but as differences in how structured experience is discretized, presented, and revealed to the learner.

### 2. What Do People Mean by "Task"? (0.5-0.75 page)

Goal:

- make the critique concrete with a few representative uses of the word `task`
- show that benchmarks are evidence for a broader conceptual ambiguity, not the whole story

Recommended examples:

- Split MNIST / Split CIFAR: task as class partition
- Permuted MNIST: task as input transformation or distributional regime
- Continual World or LIBERO: task as goal or skill configuration
- task-based versus task-free Split MNIST: similar broad experience family, different learner access and transition structure

Important restraint:

- keep this section illustrative, not exhaustive
- use benchmarks as compact evidence for different meanings of "task"
- explicitly say that benchmark properties are often protocol-dependent

Suggested takeaway:

> The point is not only that benchmarks differ, but that the term "task" is being used to refer to different kinds of units across sequential learning settings.

### 3. Preliminary Framework: Separating Three Layers (0.5-0.75 page)

Goal:

- introduce the cleanest version of the framework
- keep the number of moving parts small
- make the decomposition memorable enough for a reviewer to retain after one read

Recommended framing:

1. **Experience structure**  
   What regularities exist in the stream itself, such as similarity, change, recurrence, and exposure.

2. **Task discretization**  
   How an observer or benchmark designer segments and aggregates continuous experience into units treated as distinct tasks. This includes segmentation, boundary placement, granularity, and scope.

3. **Learner access**  
   Which aspects of that partition are revealed to the learner, such as task identity or boundary signals.

Important note:

- if space is tight, keep `scope` inside task discretization rather than promoting it to a separate top-level axis
- continuity/reset can be mentioned briefly as a domain-sensitive property that matters especially in interactive settings
- state explicitly that the listed experience dimensions are examples rather than a complete taxonomy

Suggested diagram:

`underlying experience -> task discretization -> benchmark protocol / learner access`

Optional extension if it helps clarity:

`underlying experience -> discretization -> benchmark protocol -> learner observations`

### 4. What This Decomposition Makes Distinguishable (0.5-0.75 page)

Goal:

- explain what becomes newly visible once the three layers are separated
- connect back to existing CL concerns

Possible claims:

- two settings may use similar task labels but differ in learner access
- two protocols may use the same discretization but differ in exposure or change
- observed forgetting may reflect benchmark structure or observability assumptions rather than algorithmic weakness alone
- claims about modularity, replay, or transfer should be interpreted relative to experience structure rather than task labels alone

Simple comparative examples to include:

- same discretization, different learner access
- same learner access, different exposure structure
- same broad benchmark family, different assumptions about what counts as a task

This section should sound cautious:

- emphasize conditional interpretation
- avoid overclaiming predictive power at this stage

Good framing sentence:

> The value of the decomposition is not that it already predicts all continual-learning outcomes, but that it makes explicit which aspects of a sequential learning problem are being varied, held fixed, or conflated.

### 5. Work-in-Progress Contribution and Outlook (0.25-0.5 page)

Goal:

- make the status of the work explicit
- explain what is already contributed and what remains future work

Suggested points:

- this late-breaking abstract introduces a preliminary decomposition of what is often treated simply as a "task"
- the benchmark comparison is illustrative rather than final
- future work will refine definitions, test the framework empirically, and ask whether these dimensions help predict forgetting, transfer, replay value, and architecture sensitivity

Good closing sentence:

> If continual learning is to compare methods across genuinely different sequential settings, the field may need a language for characterizing experience structure before it can reliably generalize conclusions across benchmark-defined tasks.

## Recommended Figures or Tables

Use at most one major visual element in the short version.

Best options:

1. **Preferred:** a simplified schematic showing  
   `underlying experience -> task discretization -> learner access / protocol`,  
   with 1-2 annotations showing what changes at each layer

2. **Alternative:** the task-based versus task-free Split MNIST figure, if it is used specifically to separate change in the stream from boundary observability

3. **Optional small table:** a very small benchmark illustration with only 3-4 rows and 2-3 carefully chosen columns

Avoid in the short version:

- large benchmark taxonomies
- tables with many categorical judgments that are hard to defend in a short paper

## Safer Benchmark Table Strategy

If you include a table, use columns like:

- benchmark family
- typical task unit
- what varies over time
- learner access fixed by benchmark or by protocol?

Avoid, unless carefully qualified:

- hard yes/no claims for task identity or boundaries when these are protocol-dependent
- broad similarity ratings like `high`, `medium`, `low` without defining the level of similarity

## Literature Priorities for the Short Version

Keep only the literature needed to support the main argument.

### Essential

- `van2022three` for standard CL scenarios
- `van_de_Ven_2025` for task-based versus task-free framing
- `hiratani2024disentangling`, `lin2023theory`, `li2025optimal`, or `bell2022effect` for evidence that task relations and order matter
- `ostapenko2022continual` and/or `thede2024reflecting` for why modern pretrained/foundation-model settings complicate standard CL interpretations

### Optional if space allows

- `mendez2022reuse`
- `ramasesh2020anatomy`
- one compositionality citation

### Probably postpone for this version

- broader psychology framing
- too many neighboring literatures
- detailed benchmark survey across every subfield
- long discussion of foundation models beyond brief motivation

## Writing Constraints for This Submission

- present this as a perspective and work in progress
- do not promise a complete taxonomy
- do not let the benchmark table carry the full argument
- prefer one strong conceptual through-line over breadth
- keep the tone careful and literature-grounded
- lead with the decomposition of `task`, not just with benchmark inconsistency
- use benchmarks as evidence, not as the paper's sole organizing principle
- keep the strongest claims phrased as clarifications, decompositions, and distinctions rather than definitive predictive theory

## Suggested Through-Line

The paper should roughly feel like this:

1. continual learning talks about tasks as if they were stable units
2. in practice, the term `task` often bundles multiple distinct properties of sequential learning problems
3. a useful first step is to separate experience structure, task discretization, and learner access
4. doing so clarifies what current benchmarks are actually varying
5. this can support cleaner interpretation now and more formal empirical work later

## Concrete Next Drafting Step

Turn this outline into a short paper skeleton with:

1. title
2. 150-200 word abstract
3. five short sections matching the outline above
4. one figure or one compact table
5. a short final paragraph explicitly marking the framework as preliminary and work in progress
