import gc
import os
import signal
from functools import wraps
from time import time
from typing import List, Union

import GPUtil
import psutil
import torch
from pytorch_lightning.callbacks import Callback

from cogeneration.util.log import rank_zero_logger

# internal utils logger
_logger = rank_zero_logger(__name__)


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
        _logger.info(f"func:{f.__name__} args:[{args}, {kw}] took: {te-ts:2.4f} sec")
        return result

    return wrap


class MemoryMonitorCallback(Callback):
    """
    PyTorch Lightning callback to monitor memory usage.
    Triggered on-demand via SIGUSR1 signal, or optionally every N steps.

    Usage:
        # Add to callbacks in train.py (signal-only mode)
        callbacks.append(MemoryMonitorCallback())

        # Or with periodic logging
        callbacks.append(MemoryMonitorCallback(log_every_n_steps=1000))

        # Trigger from command line:
        kill -SIGUSR1 <PID>
    """

    def __init__(self, log_every_n_steps: int = None, enable_signal: bool = True):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self._signal_triggered = False

        # Register signal handler for on-demand profiling
        if enable_signal:
            signal.signal(signal.SIGUSR1, self._signal_handler)
            _logger.info(
                f"📞 Memory monitoring registered on SIGUSR1 (PID {os.getpid()}). "
                f"Trigger: kill -SIGUSR1 {os.getpid()}"
            )

    def _signal_handler(self, _signum, _frame):
        """Handle SIGUSR1 signal to trigger memory profiling."""
        _logger.info("SIGUSR1 received - triggering memory dump")
        self._signal_triggered = True

    def _get_memory_stats(self):
        """Collect comprehensive memory statistics."""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()

        stats = {
            "system_rss_gb": mem_info.rss / 1024**3,
            "system_vms_gb": mem_info.vms / 1024**3,
            "gc_objects": len(gc.get_objects()),
        }

        # GPU memory if available
        if torch.cuda.is_available():
            stats["cuda_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
            stats["cuda_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
            stats["cuda_max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3

        return stats

    def _log_memory(self, trainer, prefix="debug"):
        """Log memory stats and optionally dump detailed info."""
        stats = self._get_memory_stats()

        # Log to trainer
        for key, value in stats.items():
            trainer.lightning_module.log(
                f"{prefix}/{key}",
                value,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )

        # Detailed logging
        _logger.info(
            f"Memory Stats - RSS: {stats['system_rss_gb']:.2f}GB, "
            f"GC Objects: {stats['gc_objects']:,}"
            + (
                f", CUDA Allocated: {stats.get('cuda_allocated_gb', 0):.2f}GB, "
                f"CUDA Reserved: {stats.get('cuda_reserved_gb', 0):.2f}GB"
                if torch.cuda.is_available()
                else ""
            )
        )

        return stats

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log memory on signal or periodically if configured."""
        should_log = self._signal_triggered

        # Optional periodic logging
        if self.log_every_n_steps is not None:
            should_log = should_log or (
                trainer.global_step > 0
                and trainer.global_step % self.log_every_n_steps == 0
            )

        if should_log:
            self._log_memory(trainer, prefix="memory")

            # If signal triggered, dump extra debug info
            if self._signal_triggered:
                _logger.info("=" * 80)
                _logger.info("MEMORY DUMP (SIGUSR1 triggered)")
                _logger.info("=" * 80)

                # Force garbage collection
                collected = gc.collect()
                _logger.info(f"Garbage collected {collected} objects")

                # Log top object types
                import collections

                obj_types = collections.Counter(
                    type(obj).__name__ for obj in gc.get_objects()
                )
                _logger.info("Top 10 object types:")
                for obj_type, count in obj_types.most_common(10):
                    _logger.info(f"  {obj_type}: {count:,}")

                _logger.info("=" * 80)
                self._signal_triggered = False
