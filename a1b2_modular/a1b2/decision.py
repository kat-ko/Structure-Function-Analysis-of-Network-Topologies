import torch


def random_decision(outputs, p=0.5):
    """Randomly choose between two possible outputs."""
    batchs = outputs.shape[1]
    device = outputs.device
    deciding_modules = torch.rand(batchs).to(device) < p
    mask = torch.einsum(
        "ab, a -> ab", torch.ones_like(outputs[0]), deciding_modules
    ).bool()
    outputs = torch.where(mask, outputs[0, ...], outputs[1, ...])
    return outputs, deciding_modules


def max_decision(outputs):
    """Get the readout from module that produced the maximum overall value."""
    device = outputs.device
    n_modules = outputs.shape[0]
    _, deciding_ags = torch.max(
        torch.stack([torch.max(outputs[i, ...], axis=-1)[0] for i in range(n_modules)]),
        axis=0,
    )
    mask_1 = deciding_ags.unsqueeze(0).unsqueeze(-1).expand_as(outputs)
    mask_2 = torch.einsum(
        "b, b... -> b...",
        torch.arange(n_modules).to(device),
        torch.ones_like(outputs),
    )
    mask = mask_1 == mask_2
    return (outputs * mask).sum(0), deciding_ags


def get_temporal_decision(outputs, temporal_decision):
    """Given outputs and temporal_decision, return selected outputs (last/sum/mean or time index)."""
    try:
        deciding_ts = int(temporal_decision)
        outputs = outputs[deciding_ts]
    except (ValueError, TypeError):
        if temporal_decision == "last":
            outputs = outputs[-1]
        elif temporal_decision == "sum":
            outputs = torch.sum(outputs, axis=0)
        elif temporal_decision == "mean":
            outputs = torch.mean(outputs, axis=0)
        elif temporal_decision is None or temporal_decision == "none":
            return outputs
        else:
            raise ValueError(
                'temporal decision not recognized, try "last", "sum" or "mean", or time_step ("0", "-1")'
            )
    return outputs


def get_module_decision(outputs, module_decision):
    """Select outputs by module (int, "max", "random", "sum", "self", "both", "all", "none")."""
    try:
        deciding_ags = int(module_decision)
        outputs = outputs[deciding_ags]
        deciding_ags = torch.ones(outputs.shape[0]) * deciding_ags
    except ValueError:
        if module_decision == "max":
            outputs, deciding_ags = max_decision(outputs)
        elif module_decision == "random":
            outputs, deciding_ags = random_decision(outputs)
        elif module_decision == "sum":
            outputs = outputs.sum(0)
            deciding_ags = None
        elif module_decision == "self":
            assert len(outputs.shape) > 3
            outputs = [out[ag] for ag, out in enumerate(outputs)]
            deciding_ags = None
            try:
                outputs = torch.stack(outputs)
            except TypeError:
                pass
        elif module_decision in ["both", "all", "none", None]:
            deciding_ags = None
        else:
            raise ValueError(
                'Deciding module not recognized, try "0", "1", "max", "random", "both" or "sum"'
            )
    return outputs, deciding_ags


def get_decision(outputs, temporal_decision="last", module_decision="max"):
    """Combine temporal and module decision; returns (decision_tensor, deciding_ags)."""
    if isinstance(outputs, list):
        decs = [get_decision(out, temporal_decision, module_decision) for out in outputs]
        outputs = [dec[0] for dec in decs]
        deciding_ags = [dec[1] for dec in decs]
        return torch.stack(outputs, -3), deciding_ags
    else:
        outputs = get_temporal_decision(outputs, temporal_decision)
        try:
            if len(outputs.shape) == 2:
                return outputs, None
        except AttributeError:
            pass
        for ag_decision in module_decision.split("_"):
            outputs, deciding_ags = get_module_decision(outputs, ag_decision)
        return outputs.squeeze(), deciding_ags
