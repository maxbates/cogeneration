import os
from functools import wraps
from time import time
from typing import List, Optional, Union

import GPUtil
import torch

from cogeneration.util.log import rank_zero_logger

logger = rank_zero_logger(__name__)


def get_available_device(device_limit: int) -> List[Union[int, str]]:
    device_ids = GPUtil.getAvailable(order="memory", limit=device_limit)

    # support being on a Mac, which doesn't have a GPU that GPUtil picks up
    # Lightning also only supports 1 device for MPS
    # TODO(train) - validate and check explicitly for MPS devices
    if len(device_ids) == 0:
        device_ids = [0]

    return device_ids


def print_timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        logger.info(f"func:{f.__name__} args:[{args}, {kw}] took: {te-ts:2.4f} sec")
        return result

    return wrap


def setup_cuequivariance_env(kernels_enabled: bool = True, enable_bf16: bool = True):
    """
    Set up environment variables for optimal cuEquivariance kernel performance.
    Should be called early in training/inference scripts.

    Args:
        kernels_enabled: Whether CUDA kernels are enabled
        enable_bf16: Whether to enable bf16 precision for kernels
    """
    if not kernels_enabled or not torch.cuda.is_available():
        return

    # Enable bf16 precision for cuEquivariance kernels
    if enable_bf16:
        os.environ["CUQUI_ENABLE_BF16"] = "1"

    # Enable other cuEquivariance optimizations
    os.environ["CUQUI_ENABLE_FAST_MATH"] = "1"
    os.environ["CUQUI_ENABLE_TENSOR_CORES"] = "1"

    # Set memory pool allocation for better performance
    os.environ["CUQUI_MEMORY_POOL"] = "1"
