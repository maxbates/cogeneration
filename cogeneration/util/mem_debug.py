import logging
import os
import signal
import time
import tracemalloc
from typing import Callable, Optional

import objgraph
import psutil
import torch

_logger = logging.getLogger(__name__)


def _safe_get_cuda_megabytes() -> float:
    if torch is None:
        return 0.0
    try:
        if getattr(torch, "cuda", None) and torch.cuda.is_available():
            return float(torch.cuda.memory_reserved() / 1e6)
    except Exception:
        return 0.0
    return 0.0


def _safe_get_mps_megabytes() -> float:
    if torch is None:
        return 0.0
    try:
        has_mps = hasattr(torch, "mps") and getattr(torch.backends, "mps", None)
        if has_mps and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return float(torch.mps.driver_allocated_memory() / 1e6)  # type: ignore[attr-defined]
    except Exception:
        return 0.0
    return 0.0


def register_memory_debugger(
    sig: signal.Signals | int = signal.SIGUSR2,
    log_path: Optional[str] = None,
    top_n: int = 50,
    tracemalloc_frames: int = 25,
) -> Callable[[], None]:
    """Register a signal handler that dumps memory/object debug info.

    On receiving ``sig`` (default: SIGUSR2), the handler writes a snapshot to
    ``log_path`` or to ``/tmp/memory_debug_<PID>.log`` if not provided.

    Returns a callable that will unregister the handler and restore the previous one.
    """

    # Start tracemalloc if not already active
    try:
        if not tracemalloc.is_tracing():
            tracemalloc.start(tracemalloc_frames)
    except Exception:
        # Some Python builds may not support is_tracing; best-effort start
        try:
            tracemalloc.start(tracemalloc_frames)
        except Exception:
            pass

    process = psutil.Process(os.getpid())
    pid = process.pid
    output_path = log_path or f"/tmp/memory_debug_{pid}.log"

    last_snapshot: Optional[tracemalloc.Snapshot] = None

    def _handler(signum, frame):  # noqa: ARG001 - signature required by signal
        _logger.info("[memdebug] received signal...")

        nonlocal last_snapshot
        try:
            rss_mb = float(process.memory_info().rss / (1024 * 1024))
            child_rss_mb = sum(
                float(child.memory_info().rss / (1024 * 1024))
                for child in process.children()
            )
        except Exception:
            rss_mb = 0.0
            child_rss_mb = 0.0

        try:
            snapshot = tracemalloc.take_snapshot()
        except Exception:
            snapshot = None

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        try:
            with open(output_path, "a") as f:
                # Header line with system/GPU memory
                cuda_mb = _safe_get_cuda_megabytes()
                mps_mb = _safe_get_mps_megabytes()
                f.write(
                    f"\n[{timestamp}] RSS={rss_mb:.0f}MB children_RSS={child_rss_mb:.0f}MB CUDA={cuda_mb:.0f}MB MPS={mps_mb:.0f}MB\n"
                )

                # Allocation statistics
                if snapshot is not None:
                    f.write(f"[{timestamp}] Top allocations (lineno):\n")
                    for stat in snapshot.statistics("lineno")[:top_n]:
                        f.write(str(stat) + "\n")

                    if last_snapshot is not None:
                        f.write(f"\n[{timestamp}] Diff since last snapshot:\n")
                        for stat in snapshot.compare_to(last_snapshot, "lineno")[
                            :top_n
                        ]:
                            f.write(str(stat) + "\n")
                    last_snapshot = snapshot

                # Object growth and counts
                try:
                    f.write(f"\n[{timestamp}] Object growth:\n")
                    f.write(str(objgraph.growth(limit=top_n)) + "\n")
                except Exception:
                    pass

                f.write("\n-------------------------------------\n")
        except Exception as e:  # pragma: no cover - best-effort debug util
            _logger.exception("Memory debug dump failed: %s", e)
        else:
            _logger.info("[memdebug] wrote %s", output_path)

    prev = signal.getsignal(sig)
    signal.signal(sig, _handler)

    # Helpful debug usage message
    try:
        sig_name = sig.name if isinstance(sig, signal.Signals) else str(sig)
    except Exception:
        sig_name = str(sig)
    _logger.debug(
        "Registered memory debugger on %s for PID %s. Send: `kill -%s %s` to dump to %s",
        sig_name,
        pid,
        sig_name.replace("SIG", ""),
        pid,
        output_path,
    )

    def _unregister() -> None:
        try:
            signal.signal(sig, prev)  # type: ignore[arg-type]
        except Exception:
            pass

    return _unregister
