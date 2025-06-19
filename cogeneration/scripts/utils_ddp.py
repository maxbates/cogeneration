import os
from dataclasses import dataclass

from torch import distributed as dist


def setup_ddp(
    trainer_strategy="auto",  # config.experiment.trainer.strategy
    accelerator="cuda",  # config.experiment.trainer.accelerator
    rank: str = "0",  # rank / device, when multiple threads
    world_size: str = "1",
):
    """
    Sets up DistributedDataParallel if Pytorch Lightning won't do it for us (i.e. strategy != "ddp").
    Primarily for use to debug, e.g. on a Mac using CPUs or MPS.

    Required for distributed data loaders.

    TODO(train) - properly use from env / set up Pytorch Lightning DDP environment variables
    """

    # Initialize the process group for distributed training
    # These environment variables are required for PyTorch DDP / by Lightning
    # TODO(train) - consider specifying externally, e.g. for real distributed training, if want to support outside lightning.
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    # TODO(train) - validate config-specified devices are available, esp when consider folding devices
    os.environ["WORLD_SIZE"] = world_size
    os.environ["NODE_RANK"] = rank

    if accelerator == "mps":
        # Pytorch doesn't support all the ops we need on MPS, ensure fallback enabled
        # Can't seem to enable it here, so need to set it in the environment
        assert (
            os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
        ), "MPS fallback not enabled and not all ops supported on MPS. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` in environment."

        # Lightning does not support DDP with MPS accelerator, i.e. on a Mac
        # However, DataLoaders use DDP, so we need to initialize it
        if trainer_strategy == "auto":
            # TODO(train) - ensure each thread has its own rank / device if multiple threads
            local_rank = int(rank)
            world_size = int(world_size)

            # Make sure not already initialized (mostly an issue in tests)
            if dist.is_initialized():
                return

            # In theory, we might want to support other backends, e.g. `nccl` if using GPUs
            # In practice, pytorch lightning will do this for us.
            # So we specify `gloo` which works with CPUs and MPS.
            dist.init_process_group(
                backend="gloo", rank=local_rank, world_size=world_size
            )


@dataclass
class DDPInfo:
    """Helper to get DDP information and handles DDP not setup."""

    node_id: int
    local_rank: int
    rank: int
    world_size: int

    @classmethod
    def from_env(cls):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        try:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            node_id = rank // world_size
        except:
            # Not using DDP
            rank = 0
            world_size = 1
            node_id = 0

        return cls(
            node_id=node_id, local_rank=local_rank, rank=rank, world_size=world_size
        )
