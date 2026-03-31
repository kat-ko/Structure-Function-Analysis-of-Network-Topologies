"""
Deterministic run_id from condition for separate result folders.
"""
from typing import Any


def build_run_id(condition: dict) -> str:
    """
    Build a deterministic string from the condition so that any change in
    a varying factor produces a different folder name.

    For backward compatibility: when condition has only the legacy keys
    (name, arch, dim_hidden, and for RNN: sparsity, common_input, common_readout),
    the run_id equals condition["name"] so existing folder names stay valid.
    """
    name = condition.get("name", "unknown")
    arch = condition.get("arch", "ffn")

    # FFN: only name, arch, dim_hidden, gamma matter for uniqueness
    if arch == "ffn":
        dim_hidden = condition.get("dim_hidden", 50)
        gamma = condition.get("gamma", 0.01)
        # Backward compat: if condition is exactly the legacy shape, use name only
        if _is_legacy_ffn(condition):
            return name
        return f"{name}_h{dim_hidden}_g{gamma}"

    # RNN (single_module_rnn or two_module_rnn)
    n_modules = condition.get("n_modules", 2 if arch == "two_module_rnn" else 1)
    dim_hidden = condition.get("dim_hidden", 50)
    nb_steps = condition.get("nb_steps", 1)
    input_routing = condition.get("input_routing", "shared")
    sparsity = condition.get("sparsity", 1.0)
    common_input = condition.get("common_input", False)
    common_readout = condition.get("common_readout", True)
    cell_type = condition.get("cell_type", "RNN")
    init_scale = condition.get("init_scale")
    init_scope = condition.get("init_scope", condition.get("init_policy", "global"))

    # Backward compat: legacy two_module_rnn conditions get run_id == name
    if _is_legacy_rnn(condition, arch):
        return name

    parts = [name]
    parts.append(f"nb{nb_steps}")
    parts.append(input_routing)
    parts.append(f"sp{sparsity}")
    parts.append("ci" if common_input else "sep")
    parts.append("cr" if common_readout else "pr")
    parts.append(cell_type)
    if init_scale is not None:
        parts.append(f"init{init_scale}")
        if init_scope != "global":
            parts.append(f"initscope{init_scope}")
    return "_".join(parts)


def _is_legacy_ffn(condition: dict) -> bool:
    """True if condition looks like legacy FFN (no extra keys)."""
    allowed = {"name", "arch", "dim_hidden", "gamma"}
    return condition.get("arch") == "ffn" and set(condition.keys()) <= allowed


def _is_legacy_rnn(condition: dict, arch: str) -> bool:
    """True if condition looks like legacy RNN (no nb_steps, input_routing, etc.)."""
    if arch not in ("two_module_rnn", "single_module_rnn"):
        return False
    # When common_input is False, always use full run_id so new runs don't overwrite legacy folders
    if condition.get("common_input", True) is False:
        return False
    # Legacy: no new levers set (or all at default)
    if condition.get("nb_steps", 1) != 1:
        return False
    if condition.get("input_routing", "shared") != "shared":
        return False
    if condition.get("cell_type", "RNN") != "RNN":
        return False
    if "init_scale" in condition:
        return False
    return True
