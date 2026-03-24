"""
Deterministic run_id from condition for separate result folders.
"""
from typing import Any


def effective_sim_schema_version(condition: dict) -> int:
    """
    Schema 1: legacy NPZ keys only.
    Schema 2+: optional during-training core/comms capture (__s2 run_id suffix, etc.).
    """
    v = int(condition.get("sim_schema_version", 1))
    if condition.get("log_during_training", False):
        v = max(v, 2)
    return v


def _apply_run_id_suffix(run_id: str, condition: dict) -> str:
    """Append __s{N} and optional run_id_suffix for versioned / A-B run folders."""
    schema = effective_sim_schema_version(condition)
    parts = []
    if schema >= 2:
        parts.append(f"s{schema}")
    extra = condition.get("run_id_suffix")
    if extra is not None and str(extra).strip() != "":
        parts.append(str(extra).strip().strip("_"))
    if not parts:
        return run_id
    return f"{run_id}__{'_'.join(parts)}"


def build_run_id(condition: dict) -> str:
    """
    Build a deterministic string from the condition so that any change in
    a varying factor produces a different folder name.

    For backward compatibility: when condition has only the legacy keys
    (name, arch, dim_hidden, and for RNN: sparsity, common_input, common_readout),
    the run_id equals condition["name"] so existing folder names stay valid.

    When sim_schema_version >= 2 or log_during_training is True, appends __s2
    (or __s3, ...) so enhanced runs do not overwrite legacy folders.
    Optional condition["run_id_suffix"] adds another segment after the schema token.
    """
    base = _build_run_id_base(condition)
    return _apply_run_id_suffix(base, condition)


def _build_run_id_base(condition: dict) -> str:
    name = condition.get("name", "unknown")
    arch = condition.get("arch", "ffn")

    if arch == "ffn":
        dim_hidden = condition.get("dim_hidden", 50)
        gamma = condition.get("gamma", 0.01)
        if _is_legacy_ffn(condition):
            return name
        return f"{name}_h{dim_hidden}_g{gamma}"

    n_modules = condition.get("n_modules", 2 if arch == "two_module_rnn" else 1)
    dim_hidden = condition.get("dim_hidden", 50)
    nb_steps = condition.get("nb_steps", 1)
    input_routing = condition.get("input_routing", "shared")
    sparsity = condition.get("sparsity", 1.0)
    common_input = condition.get("common_input", False)
    common_readout = condition.get("common_readout", True)
    cell_type = condition.get("cell_type", "RNN")
    init_scale = condition.get("init_scale")

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
    return "_".join(parts)


def _is_legacy_ffn(condition: dict) -> bool:
    """True if condition looks like legacy FFN (no extra keys)."""
    allowed = {"name", "arch", "dim_hidden", "gamma"}
    return condition.get("arch") == "ffn" and set(condition.keys()) <= allowed


def _is_legacy_rnn(condition: dict, arch: str) -> bool:
    """True if condition looks like legacy RNN (no nb_steps, input_routing, etc.)."""
    if arch not in ("two_module_rnn", "single_module_rnn"):
        return False
    if condition.get("common_input", True) is False:
        return False
    if condition.get("nb_steps", 1) != 1:
        return False
    if condition.get("input_routing", "shared") != "shared":
        return False
    if condition.get("cell_type", "RNN") != "RNN":
        return False
    if "init_scale" in condition:
        return False
    return True
