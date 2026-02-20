import torch


class SurrGradSpike(torch.autograd.Function):
    """Spiking nonlinearity with surrogate gradient (Zenke & Ganguli 2018)."""
    scale = 100.0

    @staticmethod
    def forward(ctx, input, thr=0):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > thr] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad


super_spike = SurrGradSpike.apply
