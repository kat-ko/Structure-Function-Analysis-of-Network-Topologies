AI Chatbot geenrated overviews of the project


I would pitch it as follows:


**Hidden Assumptions About Tasks in Continual Learning**

Continual learning research is fundamentally organized around the concept of a *task*: algorithms are evaluated on sequences of tasks, benchmarks are constructed from tasks, and performance is measured through transfer and forgetting between tasks. Despite its central role, however, the field lacks a principled characterization of what a task actually is. Existing benchmarks implicitly encode very different assumptions about task boundaries, similarity, recurrence, distribution shift, and learner information, yet these assumptions are rarely made explicit or systematically compared.

This project argues that tasks should not be viewed as fundamental entities, but rather as **researcher-defined discretizations of an underlying experience stream**. From this perspective, many seemingly different continual learning settings can be understood as different ways of segmenting and sampling continuous experience. Rather than asking whether an algorithm works on "Task A then Task B," we ask what structural properties of the underlying experience are actually responsible for transfer, interference, and forgetting.

The paper develops a conceptual framework that identifies the hidden dimensions underlying task formulations—for example, latent similarity between experiences, the scope of variation grouped within a task, temporal patterns of change, and exposure structure. These dimensions provide a common vocabulary for describing continual learning problems independent of any particular benchmark.

Using this framework, the paper reinterprets classical benchmarks such as SplitMNIST, PermutedMNIST, Domain Incremental, and streaming continual learning as different points within a shared design space rather than fundamentally different problem classes. It also connects naturally to modern foundation models and large language models, where learning increasingly occurs over heterogeneous continuous data streams and task boundaries are often imposed only during evaluation.

Ultimately, the goal is not to propose a new continual learning algorithm, but to establish a taxonomy of experience structure that clarifies benchmark assumptions, enables more meaningful comparisons across methods, and provides a foundation for designing future continual learning benchmarks and theories.

---

I would summarize the contribution in a single sentence as:

> **This paper asks whether continual learning should be defined in terms of *tasks* at all, or whether tasks are better understood as one particular way of discretizing structured experience.**

I think that sentence captures the central intellectual move of the project. It makes clear that the contribution is conceptual rather than algorithmic, while hinting at why the work is relevant to both classical continual learning and emerging foundation-model settings.




## Project Summary

Continual learning research has produced a large body of work on catastrophic forgetting, replay, regularization, modular architectures, parameter isolation, and dynamic networks. However, these methods are typically evaluated on benchmark suites in which "tasks" are predefined by the experimenter and presented according to fixed protocols. While benchmarks differ substantially in how they construct tasks and sequential experiences, these design choices are rarely described using a common language. As a result, conclusions about forgetting, transfer, architectural design, or stability-plasticity are often compared across experimental settings that differ in important but largely implicit ways.

This project asks a simple question:

> **What exactly is a "task" in continual learning, and which assumptions about tasks are embedded in current benchmarks?**

Our central premise is that many benchmark tasks should be viewed as **researcher-defined discretizations of a richer underlying experience stream**, rather than as fundamental units of learning themselves. Similar learning problems can be partitioned into tasks in multiple valid ways, and these choices interact with properties of the experience stream—such as structural similarity, temporal ordering, recurrence, drift, and continuity—as well as with the information available to the learner, such as task identity or boundary signals.

The project aims to disentangle these factors by proposing a conceptual framework that separates:

* the **structure of experience** (e.g., similarity, exposure, change, continuity),
* the **discretization of experience into tasks** (e.g., what constitutes a task and at what granularity),
* the **information available to the learner** (e.g., known task identities or boundaries),

and studies how these interact with properties of the learning system to produce phenomena such as transfer, interference, catastrophic forgetting, abstraction, specialization, and stability-plasticity trade-offs.

The first outcome will be a perspective paper that surveys how existing continual learning benchmarks implicitly make different assumptions about these dimensions, identifies where current terminology conflates distinct concepts, and proposes a common vocabulary for describing sequential learning settings. Rather than introducing a new continual learning algorithm, the paper seeks to provide a conceptual foundation that allows researchers to more precisely characterize benchmark design, interpret experimental findings, and understand which conclusions generalize across settings.

The longer-term goal is to develop this conceptual framework into a more formal theory of sequential learning, informed by continual learning, transfer learning, curriculum learning, cognitive science, computational neuroscience, and dynamical systems. Ultimately, we hope this will support the development of benchmark families that systematically manipulate properties of experience, enabling more mechanistic investigations into when and why different continual learning methods succeed or fail.

---

## Short Pitch (≈30 seconds)

Current continual learning research relies heavily on predefined benchmark tasks, yet the notion of a "task" itself is rarely examined. Different benchmarks implicitly make different assumptions about task identity, similarity, boundaries, ordering, and environmental change, making it difficult to compare results across studies. We are developing a conceptual framework that treats tasks as discretizations of continuous experience and explicitly separates properties of the experience stream, task construction, and learner information. By providing a common language for describing sequential learning settings, the project aims to improve benchmark design, clarify when existing findings generalize, and enable more mechanistic research into transfer, interference, and lifelong learning.
