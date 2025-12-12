"""
Lightweight profiling script to identify bottlenecks in forward/backward pass.
uses `torch.profiler` on single batch to profile forward and backward pass.
"""

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from cogeneration.config.base import Config
from cogeneration.dataset.test_utils import MockDataloader
from cogeneration.models.loss_calculator import BatchLossCalculator
from cogeneration.models.model import FlowModel
from cogeneration.type.task import DataTask

# triangle attention kernel
torch.backends.cuda.matmul.allow_tf32 = True


def profile_model(
    num_res: int = 256,
    batch_size: int = 4,
    num_warmup: int = 3,
    num_profile: int = 5,
    use_autocast: bool = True,
    enable_esm: bool = False,
):
    """Profile forward and backward pass."""

    print(f"\n{'='*60}")
    print(f"Profiling: batch_size={batch_size}, num_res={num_res}")
    print(f"Precision: {'bfloat16 (autocast)' if use_autocast else 'float32'}")
    print(f"ESM Combiner: {'enabled' if enable_esm else 'disabled'}")
    print(f"{'='*60}\n")

    # Setup
    cfg = Config().interpolate()
    cfg.model.esm_combiner.enabled = enable_esm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = FlowModel(cfg.model).to(device)
    model.train()

    # create mock batch
    mock_dataloader = MockDataloader(
        batch_size=batch_size,
        sample_lengths=[num_res] * batch_size,
        task=DataTask.inpainting,
        corrupt=True,
    )
    batch = next(iter(mock_dataloader))
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }

    # Warmup
    print("Warming up...")
    for _ in range(num_warmup):
        pred = None
        if use_autocast:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                pred = model(batch)
        else:
            pred = model(batch)
        loss_calc = BatchLossCalculator(cfg=cfg, batch=batch, pred=pred)
        losses, aux = loss_calc.calculate()
        losses.train_loss.mean().backward()
        model.zero_grad()

    torch.cuda.synchronize()

    # Profile
    print("\nProfiling...\n")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(num_profile):
            if use_autocast:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    pred = model(batch)
            else:
                pred = model(batch)
            loss_calc = BatchLossCalculator(cfg=cfg, batch=batch, pred=pred)
            losses, aux = loss_calc.calculate()
            losses.train_loss.mean().backward()
            model.zero_grad()
            torch.cuda.synchronize()

    # Print results
    print("\n" + "=" * 60)
    print("TOP CUDA OPERATIONS BY TIME")
    print("=" * 60)
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=20,
        )
    )

    print("\n" + "=" * 60)
    print("TOP OPERATIONS BY MEMORY")
    print("=" * 60)
    print(
        prof.key_averages().table(
            sort_by="self_cuda_memory_usage",
            row_limit=15,
        )
    )

    # Export trace for detailed analysis
    precision_str = "bf16" if use_autocast else "fp32"
    esm_str = "_esm" if enable_esm else ""
    trace_path = f"profile_b{batch_size}_n{num_res}_{precision_str}{esm_str}.json"
    prof.export_chrome_trace(trace_path)
    print(f"\n✅ Chrome trace saved to: {trace_path}")
    print("   Open in chrome://tracing or https://ui.perfetto.dev")

    # Memory summary
    if torch.cuda.is_available():
        print(f"\n{'='*60}")
        print("GPU MEMORY SUMMARY")
        print(f"{'='*60}")
        print(f"Peak allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print(f"Peak reserved:  {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_res", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_warmup", type=int, default=3)
    parser.add_argument("--num_profile", type=int, default=5)
    parser.add_argument(
        "--no_autocast",
        action="store_true",
        help="Disable bfloat16 autocast (use full fp32 precision)",
    )
    parser.add_argument(
        "--enable_esm",
        action="store_true",
        help="Enable ESM combiner (disabled by default)",
    )
    args = parser.parse_args()

    profile_model(
        num_res=args.num_res,
        batch_size=args.batch_size,
        num_warmup=args.num_warmup,
        num_profile=args.num_profile,
        use_autocast=not args.no_autocast,
        enable_esm=args.enable_esm,
    )
