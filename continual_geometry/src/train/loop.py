"""Sequential training over a task stream (`00` §3, `01` Phase 1).

One task = one balanced dichotomy `y_t` over the `P` manifolds of that task's
arrangement. Targets are `±1` per manifold, shared by every point of the manifold,
so the network must map a whole manifold to one side — which is what makes
manifold capacity the right readout.

Boundaries are where everything is measured. The loop records performance on the
current task, on every past task (retained), and on a held-out probe dichotomy
never trained on; geometry is measured separately by `src/analysis/` from the
representations this loop exposes, so that the expensive estimator can run at a
Tier-2 subset of boundaries without re-training.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..models import MODULES, TwoModuleNet


@dataclass(frozen=True)
class TrainConfig:
    """`target_accuracy` is not optional bookkeeping — see `TaskRecord.converged`."""

    steps_per_task: int = 2000
    batch_size: int | None = None      # None = full batch
    loss_tol: float = 0.0              # >0 enables early stop on train loss
    target_accuracy: float | None = 0.98
    record_every: int = 50


@dataclass
class TaskRecord:
    task: int
    final_loss: float
    steps_taken: int
    train_accuracy: float
    loss_curve: list[float]
    weight_change: dict[str, float]
    converged: bool = True

    @property
    def usable_for_forgetting(self) -> bool:
        """A task the network never learned cannot meaningfully be forgotten.

        Retained-capacity and forgetting numbers for a non-converged task confound
        forgetting with under-training, so any run containing one is invalid for
        H1/H2 and must be reported, not silently averaged in.
        """
        return self.converged


@dataclass
class BoundaryRecord:
    """Measured after finishing task `t`. Indices are absolute task indices."""

    after_task: int
    current_accuracy: float
    retained_accuracy: dict[int, float]
    probe_accuracy: float
    weight_change: dict[str, float]
    mean_retained_accuracy: float = field(init=False)

    def __post_init__(self) -> None:
        vals = list(self.retained_accuracy.values())
        self.mean_retained_accuracy = float(np.mean(vals)) if vals else float("nan")


def flatten_task(points: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(P, M, d)` manifolds + `(P,)` dichotomy → `(P·M, d)` inputs and `(P·M,)` targets."""
    pts = np.asarray(points, dtype=np.float64)
    P, M, d = pts.shape
    y = np.asarray(y, dtype=np.float64)
    if y.shape != (P,):
        raise ValueError(f"dichotomy must be ({P},); got {y.shape}")
    return pts.reshape(P * M, d), np.repeat(y, M)


def accuracy(model: TwoModuleNet, X: np.ndarray, y: np.ndarray) -> float:
    """Sign agreement. `f = 0` counts as wrong, so accuracy at init is 0, not 0.5."""
    pred = model.forward(X)
    return float(np.mean(np.sign(pred) == np.sign(y)))


def manifold_accuracy(model: TwoModuleNet, points: np.ndarray, y: np.ndarray) -> float:
    X, target = flatten_task(points, y)
    return accuracy(model, X, target)


def train_task(
    model: TwoModuleNet,
    points: np.ndarray,
    y: np.ndarray,
    cfg: TrainConfig,
    rng: np.random.Generator,
    *,
    task_index: int = 0,
) -> TaskRecord:
    """SGD on one task, in place. Returns the record; the model is mutated."""
    X, target = flatten_task(points, y)
    n = X.shape[0]
    curve: list[float] = []
    loss = float("nan")
    step = 0
    reached = False
    for step in range(1, cfg.steps_per_task + 1):
        if cfg.batch_size is None or cfg.batch_size >= n:
            loss = model.sgd_step(X, target)
        else:
            idx = rng.choice(n, size=cfg.batch_size, replace=False)
            loss = model.sgd_step(X[idx], target[idx])
        if step % cfg.record_every == 0 or step == 1:
            curve.append(loss)
            if cfg.target_accuracy is not None and accuracy(model, X, target) >= cfg.target_accuracy:
                reached = True
                break
        if cfg.loss_tol > 0 and loss < cfg.loss_tol:
            break
    final_acc = accuracy(model, X, target)
    return TaskRecord(
        task=task_index,
        final_loss=loss,
        steps_taken=step,
        train_accuracy=final_acc,
        loss_curve=curve,
        weight_change={m: model.weight_change(m) for m in MODULES},
        converged=(
            True if cfg.target_accuracy is None
            else bool(reached or final_acc >= cfg.target_accuracy)
        ),
    )


def run_stream(
    model: TwoModuleNet,
    task_points: list[np.ndarray],
    task_dichotomies: np.ndarray,
    cfg: TrainConfig,
    rng: np.random.Generator,
    *,
    probe_y: np.ndarray | None = None,
    probe_points: np.ndarray | None = None,
) -> tuple[list[TaskRecord], list[BoundaryRecord]]:
    """Train tasks in order, measuring at every boundary.

    `task_points[t]` is `(P, M, d)` for task `t` — a separate array per task
    because feature similarity `s_f` is imposed by redrawing the arrangement.

    Retained accuracy is evaluated on each past task's **own** arrangement and
    dichotomy, with the network as it stands. No readout is refit, so this is
    forgetting of the trained solution, not of the representation; the probe is
    what isolates the representation, and it gets a freshly trained readout in
    `src/analysis/` rather than here.

    Check `all(t.converged for t in tasks)` before using a run for H1/H2. Under a
    fixed step budget, later tasks are *harder* than the first — they start from a
    solution to a different dichotomy and must overwrite it — so a budget tuned on
    task 0 silently leaves later tasks unlearned, and their apparent "forgetting"
    is under-training.
    """
    ys = np.asarray(task_dichotomies)
    if len(task_points) != ys.shape[0]:
        raise ValueError(f"{len(task_points)} arrangements vs {ys.shape[0]} dichotomies")

    tasks: list[TaskRecord] = []
    boundaries: list[BoundaryRecord] = []
    for t, pts in enumerate(task_points):
        tasks.append(train_task(model, pts, ys[t], cfg, rng, task_index=t))
        boundaries.append(
            BoundaryRecord(
                after_task=t,
                current_accuracy=manifold_accuracy(model, pts, ys[t]),
                retained_accuracy={
                    j: manifold_accuracy(model, task_points[j], ys[j]) for j in range(t)
                },
                probe_accuracy=(
                    manifold_accuracy(
                        model,
                        pts if probe_points is None else probe_points,
                        probe_y,
                    )
                    if probe_y is not None
                    else float("nan")
                ),
                weight_change={m: model.weight_change(m) for m in MODULES},
            )
        )
    return tasks, boundaries
