import torch
from torch import nn


def get_model_size_str(model: nn.Module) -> str:
    """
    Get the size of a model in MB, and the number of parameters by dtype.
    """
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return "(Model has no parameters)"

    emoji = "🧊" if device.type == "cpu" else "🔥"

    total_size = sum(p.numel() * p.element_size() for p in model.parameters())
    total_size_mb = total_size / (1024**2)

    # count params by dtype
    fp32 = sum(p.numel() for p in model.parameters() if p.dtype == torch.float32)
    bf16 = sum(p.numel() for p in model.parameters() if p.dtype == torch.bfloat16)
    other = sum(
        p.numel()
        for p in model.parameters()
        if p.dtype not in [torch.float32, torch.bfloat16]
    )

    return f"{total_size_mb:.2f} MB @ {emoji} {device} [{fp32/1e6:.1f} M fp32, {bf16/1e6:.1f} M bf16, {other/1e6:.1f} M other]"
