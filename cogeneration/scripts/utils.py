from functools import wraps
from time import time
from typing import List, Union

import GPUtil

from cogeneration.util.log import rank_zero_logger

logger = rank_zero_logger(__name__)


def get_available_device(device_limit: int) -> List[Union[int, str]]:
    device_ids = GPUtil.getAvailable(order="memory", limit=device_limit)

    # support being on a Mac, which doesn't have a GPU that GPUtil picks up
    # Lightning also only supports 1 device for MPS
    # TODO - validate and check explicitly for MPS devices
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
