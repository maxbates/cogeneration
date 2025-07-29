import tempfile
import time
from pathlib import Path

import pytest
import torch

from cogeneration.config.base import Config
from cogeneration.data.const import aatype_to_seq
from cogeneration.data.tools.boltz_runner import BoltzRunner
from cogeneration.dataset.test_utils import create_pdb_dataloader
from cogeneration.type.batch import BatchProp as bp


@pytest.mark.skip
class TestBenchmarkBoltzRunner:
    """
    Benchmark tests comparing different BoltzRunner approaches.

    TL;DR:
    Boltz prediction approaches are fairly similar, as by default inference dataloader is single item batches.
    - batched native approaches are similar (fasta vs batch features)
    - fold_batch_sequential is ~ 1.2-1.5x slower
    - fold_fasta_subprocess is ~ 1.1x slower
    """

    def test_benchmark_runner_methods(self):
        """Benchmark fold_batch, fold_fasta_native and fold_fasta_subprocess."""
        cfg = Config().interpolate()
        boltz_cfg = cfg.folding.boltz

        # Parameters
        num_batches = 5

        # Create dataloader and collect batches
        dataloader = create_pdb_dataloader(cfg=cfg, training=True)
        batch_iter = iter(dataloader)
        batches = []
        protein_counter = 0
        for _ in range(num_batches):
            batch = next(batch_iter)
            aatypes = batch[bp.aatypes_1]
            chain_idx = batch.get(bp.chain_idx, torch.zeros_like(aatypes))
            batch_size = aatypes.shape[0]

            # Generate protein IDs and sequences
            protein_ids = [f"protein_{protein_counter + i}" for i in range(batch_size)]
            sequences = [
                aatype_to_seq(
                    aatypes[i].detach().cpu().numpy(),
                    (
                        chain_idx[i].detach().cpu().numpy()
                        if chain_idx is not None
                        else None
                    ),
                )
                for i in range(batch_size)
            ]
            protein_counter += batch_size

            batches.append(
                {
                    "aatypes": aatypes,
                    "chain_idx": chain_idx,
                    "protein_ids": protein_ids,
                    "sequences": sequences,
                }
            )

        results = {}

        # log batches
        print("BATCH STATS")
        for i, batch in enumerate(batches):
            print(f"Batch {i}: {batch['aatypes'].shape}")

        # --- fold_batch benchmark ---
        print("BENCHMARKING fold_batch")
        runner_native = BoltzRunner(cfg=boltz_cfg)
        start_time = time.time()
        total_predictions = 0
        for bd in batches:
            try:
                prediction_set = runner_native.fold_batch(
                    aatypes=bd["aatypes"],
                    chain_idx=bd["chain_idx"],
                    protein_ids=bd["protein_ids"],
                )
                total_predictions += len(prediction_set)
            except Exception as e:
                print(f"fold_batch failed: {e}")
                continue
        elapsed = time.time() - start_time
        results["fold_batch"] = {"time": elapsed, "num_results": total_predictions}

        # --- fold_batch_sequential benchmark ---
        print("BENCHMARKING fold_batch_sequential")
        start_time = time.time()
        total_predictions_single = 0
        for bd in batches:
            # Iterate through each structure individually
            batch_size = bd["aatypes"].shape[0]
            for i in range(batch_size):
                try:
                    prediction_set = runner_native.fold_batch(
                        aatypes=bd["aatypes"][i : i + 1],
                        chain_idx=(
                            bd["chain_idx"][i : i + 1]
                            if bd["chain_idx"] is not None
                            else None
                        ),
                        protein_ids=[bd["protein_ids"][i]],
                    )
                    total_predictions_single += len(prediction_set)
                except Exception as e:
                    print(f"fold_batch_sequential failed: {e}")
                    continue
        elapsed_single = time.time() - start_time
        results["fold_batch_sequential"] = {
            "time": elapsed_single,
            "num_results": total_predictions_single,
        }

        # --- fold_fasta_native benchmark ---
        print("BENCHMARKING fold_fasta_native")
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fasta_path = tmp_path / "input.fasta"
            with open(fasta_path, "w") as fasta_f:
                for bd in batches:
                    for pid, seq in zip(bd["protein_ids"], bd["sequences"]):
                        fasta_f.write(f">{pid}\n{seq}\n")
            output_dir = tmp_path / "native_output"
            output_dir.mkdir(parents=True, exist_ok=True)

            start_time = time.time()
            try:
                df_native = runner_native.fold_fasta_native(
                    fasta_path=fasta_path, output_dir=output_dir
                )
                num_results_native = len(df_native)
            except Exception as e:
                print(f"fold_fasta_native failed: {e}")
                num_results_native = 0
            elapsed_native = time.time() - start_time
            results["fold_fasta_native"] = {
                "time": elapsed_native,
                "num_results": num_results_native,
            }

        # --- fold_fasta_subprocess benchmark ---
        print("BENCHMARKING fold_fasta_subprocess")
        boltz_cfg_sub = boltz_cfg.clone()
        boltz_cfg_sub.run_native = False
        runner_sub = BoltzRunner(cfg=boltz_cfg_sub)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fasta_path = tmp_path / "input.fasta"
            with open(fasta_path, "w") as fasta_f:
                for bd in batches:
                    for pid, seq in zip(bd["protein_ids"], bd["sequences"]):
                        fasta_f.write(f">{pid}\n{seq}\n")
            output_dir = tmp_path / "subprocess_output"
            output_dir.mkdir(parents=True, exist_ok=True)

            start_time = time.time()
            try:
                df_sub = runner_sub.fold_fasta_subprocess(
                    fasta_path=fasta_path, output_dir=output_dir
                )
                num_results_sub = len(df_sub)
            except Exception as e:
                print(f"fold_fasta_subprocess failed: {e}")
                num_results_sub = 0
            elapsed_sub = time.time() - start_time
            results["fold_fasta_subprocess"] = {
                "time": elapsed_sub,
                "num_results": num_results_sub,
            }

        # --- Print Benchmark Results ---
        print("BENCHMARK RESULTS\n")
        for method, data in results.items():
            total_time = data["time"]
            num_results = data["num_results"]
            print(f"{method}:")
            print(f"  Total time: {total_time:.2f} seconds")
            print(f"  Results generated: {num_results}")
            if num_results > 0:
                print(
                    f"  Average time per structure: {total_time / num_results:.2f} seconds\n"
                )

        # --- Batch Statistics ---
        total_structures = sum(len(bd["protein_ids"]) for bd in batches)
        print("BATCH STATISTICS")
        print(f"Total batches processed: {len(batches)}")
        print(f"Total structures: {total_structures}")
        for idx, bd in enumerate(batches):
            print(f"  Batch {idx}: {bd['aatypes'].shape}")
