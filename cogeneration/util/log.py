import logging
from datetime import datetime
from typing import Optional

import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_only


def rank_zero_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    )
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def log_tensor_info(tensor: torch.Tensor, name: Optional[str]):
    """Util to log in `forward` method of PyTorch module without upsetting dyanmo."""
    with open("tensor_info.log", "a") as f:
        if isinstance(tensor, torch.Tensor):
            f.write(
                f"[{datetime.now().strftime('%H %M %S')}] {name}: {tensor.shape} {tensor.dtype} {tensor.device}\n"
            )
        else:
            f.write(f"[{datetime.now().strftime('%H %M %S')}] {name}: {type(tensor)}\n")
