"""
Minimal task target helper for dynspec-style training. Full task definitions live in dynspec.tasks.
"""
import torch


def get_task_target(target, task, n_classes_per_digit=None):
    """Return target for the given task. Stub: returns target as-is for generic use."""
    if isinstance(task, list):
        return [get_task_target(target, t, n_classes_per_digit) for t in task]
    return target
