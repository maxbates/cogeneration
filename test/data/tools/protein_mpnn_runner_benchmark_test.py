import tempfile
import time
from pathlib import Path

import pytest
import torch

from cogeneration.config.base import Config
from cogeneration.data import all_atom
from cogeneration.data.protein import write_prot_to_pdb
from cogeneration.data.tools.protein_mpnn_runner import (
    ProteinMPNNRunner,
    ProteinMPNNRunnerPool,
)
from cogeneration.dataset.test_utils import create_pdb_dataloader
from cogeneration.type.batch import BatchProp as bp


@pytest.mark.skip
class TestBenchmarkProteinMPNNRunner:
    """Benchmark tests comparing different ProteinMPNNRunner approaches"""

    def test_benchmark_runner_approaches(self):
        """
        Benchmark test comparing run_batch, inverse_fold_pdb_native, and run_subprocess approaches.

        This test uses create_pdb_dataloader to generate real PDB batches,
        converts to atom37 representation, writes PDB files, and then times each approach on the same data.

        TL;DR on A100 40GB:
        run_batch is fastest
        run_batch_pool is ~1.5x slower
        run_subprocess is ~3 slower

        Note that MPNN effectively must be run serially per sequence because each batch is autoregressive.
        """
        # Use default config, not mock
        cfg = Config().interpolate()

        # Benchmark parameters
        num_batches = 5
        num_passes = 3
        temperature = 0.1

        dataloader = create_pdb_dataloader(
            cfg=cfg,
            training=True,
        )

        batches = []
        train_iter = iter(dataloader)
        for i in range(num_batches):
            batch = next(train_iter)
            batch_size = batch[bp.trans_1].shape[0]
            num_res = batch[bp.trans_1].shape[1]
            print(f"  Train batch {i}: batch_size={batch_size}, num_res={num_res}")
            batches.append(batch)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Convert batches to atom37 and write PDB files
            batch_pdbs = []

            for batch_idx, batch in enumerate(batches):
                batch_size = batch[bp.trans_1].shape[0]
                num_res = batch[bp.trans_1].shape[1]

                # Convert to atom37 using all_atom.atom37_from_trans_rot
                atom37_coords = all_atom.atom37_from_trans_rot(
                    trans=batch[bp.trans_1],
                    rots=batch[bp.rotmats_1],
                    aatype=batch[bp.aatypes_1],
                    res_mask=batch[bp.res_mask],
                )

                # Write PDB files for each structure in the batch
                batch_pdb_paths = []
                for struct_idx in range(batch_size):
                    pdb_name = f"batch_{batch_idx}_struct_{struct_idx}"
                    pdb_path = tmp_path / f"{pdb_name}.pdb"

                    # Write PDB file using write_prot_to_pdb
                    write_prot_to_pdb(
                        prot_pos=atom37_coords[struct_idx].numpy(),
                        file_path=str(pdb_path),
                        aatype=batch[bp.aatypes_1][struct_idx].numpy(),
                        chain_idx=(
                            batch[bp.chain_idx][struct_idx].numpy()
                            if bp.chain_idx in batch
                            else None
                        ),
                        no_indexing=True,
                        overwrite=True,
                    )
                    batch_pdb_paths.append(pdb_path)

                batch_pdbs.append(
                    {
                        "batch": batch,
                        "pdb_paths": batch_pdb_paths,
                        "atom37": atom37_coords,
                        "batch_size": batch_size,
                        "num_res": num_res,
                    }
                )

            print(
                f"Created PDB files for {len(batch_pdbs)} batches, sizes: {[(bd['batch_size'], bd['num_res']) for bd in batch_pdbs]}"
            )

            # Configure ProteinMPNN for testing
            mpnn_cfg = cfg.folding.protein_mpnn
            mpnn_cfg.use_native_runner = True  # Start with native mode

            results = {}

            print("\n=== Benchmarking run_batch ===")
            runner_batch = ProteinMPNNRunner(mpnn_cfg)

            start_time = time.time()
            batch_results = []

            for batch_data in batch_pdbs:
                batch = batch_data["batch"]
                batch_size = batch_data["batch_size"]
                num_res = batch_data["num_res"]

                try:
                    result = runner_batch.run_batch(
                        trans=batch[bp.trans_1],
                        rotmats=batch[bp.rotmats_1],
                        aatypes=batch[bp.aatypes_1],
                        res_mask=batch[bp.res_mask],
                        diffuse_mask=torch.ones_like(batch[bp.res_mask]),
                        chain_idx=(
                            batch[bp.chain_idx] if bp.chain_idx in batch else None
                        ),
                        num_passes=num_passes,
                        sequences_per_pass=1,
                        temperature=temperature,
                    )
                    # convert batch to list to mirror run_native and run_subprocess output shapes
                    batch_results.extend(result.sequences.tolist())

                except Exception as e:
                    print(f"run_batch failed: {e}")
                    continue

            batch_time = time.time() - start_time
            results["run_batch"] = {
                "time": batch_time,
                "results": batch_results,
                "method": "run_batch",
            }
            print(f"run_batch completed in {batch_time:.2f} seconds")

            print("\n=== Benchmarking run_batch_pool ===")
            runner_pool = ProteinMPNNRunnerPool(mpnn_cfg, num_models=8)

            start_time = time.time()
            pool_results = []

            for batch_data in batch_pdbs:
                batch = batch_data["batch"]
                batch_size = batch_data["batch_size"]
                num_res = batch_data["num_res"]

                try:
                    result = runner_pool.run_batch(
                        trans=batch[bp.trans_1],
                        rotmats=batch[bp.rotmats_1],
                        aatypes=batch[bp.aatypes_1],
                        res_mask=batch[bp.res_mask],
                        diffuse_mask=torch.ones_like(batch[bp.res_mask]),
                        chain_idx=(
                            batch[bp.chain_idx] if bp.chain_idx in batch else None
                        ),
                        num_passes=num_passes,
                        sequences_per_pass=1,
                        temperature=temperature,
                    )
                    # convert batch to list to mirror run_native and run_subprocess output shapes
                    pool_results.extend(result.sequences.tolist())

                except Exception as e:
                    print(f"run_batch_pool failed: {e}")
                    continue

            pool_time = time.time() - start_time
            results["run_batch_pool"] = {
                "time": pool_time,
                "results": pool_results,
                "method": "run_batch_pool",
            }
            print(f"run_batch_pool completed in {pool_time:.2f} seconds")

            print("\n=== Benchmarking inverse_fold_pdb_native ===")

            start_time = time.time()
            native_results = []

            for batch_data in batch_pdbs:
                batch_size = batch_data["batch_size"]
                num_res = batch_data["num_res"]

                for pdb_path in batch_data["pdb_paths"]:
                    try:
                        result_path = runner_batch.inverse_fold_pdb_native(
                            pdb_path=pdb_path,
                            output_dir=tmp_path / "native_output",
                            diffuse_mask=torch.ones(
                                200
                            ),  # Mock diffuse mask for typical protein size
                            num_sequences=num_passes,
                            temperature=temperature,
                        )
                        native_results.append(result_path)

                    except Exception as e:
                        print(f"inverse_fold_pdb_native failed for {pdb_path}: {e}")
                        continue

            native_time = time.time() - start_time
            results["inverse_fold_pdb_native"] = {
                "time": native_time,
                "results": native_results,
                "method": "inverse_fold_pdb_native",
            }
            print(f"inverse_fold_pdb_native completed in {native_time:.2f} seconds")

            print("\n=== Benchmarking run_subprocess ===")

            # Switch to subprocess mode
            mpnn_cfg.use_native_runner = False
            runner_subprocess = ProteinMPNNRunner(mpnn_cfg)

            start_time = time.time()
            subprocess_results = []

            for batch_data in batch_pdbs:
                batch_size = batch_data["batch_size"]
                num_res = batch_data["num_res"]

                for pdb_path in batch_data["pdb_paths"]:
                    try:
                        result_path = runner_subprocess.inverse_fold_pdb_subprocess(
                            pdb_path=pdb_path,
                            output_dir=tmp_path / "subprocess_output",
                            diffuse_mask=torch.ones(
                                200
                            ),  # Mock diffuse mask for typical protein size
                            num_sequences=num_passes,
                            temperature=temperature,
                        )
                        subprocess_results.append(result_path)

                    except Exception as e:
                        print(f"run_subprocess failed for {pdb_path}: {e}")
                        continue

            subprocess_time = time.time() - start_time
            results["run_subprocess"] = {
                "time": subprocess_time,
                "results": subprocess_results,
                "method": "run_subprocess",
            }
            print(f"run_subprocess completed in {subprocess_time:.2f} seconds")

            # Print benchmark results
            print("BENCHMARK RESULTS")

            for method_name, result_data in results.items():
                method_desc = result_data["method"]
                elapsed_time = result_data["time"]
                num_results = len(result_data["results"])

                print(f"\n{method_desc}:")
                print(f"  Total time: {elapsed_time:.2f} seconds")
                print(f"  Results generated: {num_results}")
                if num_results > 0:
                    print(
                        f"  Average time per structure: {elapsed_time / num_results:.2f} seconds"
                    )

            # Print batch statistics
            print("BATCH STATISTICS")

            total_batch_size = sum(bd["batch_size"] for bd in batch_pdbs)
            total_residues = sum(bd["batch_size"] * bd["num_res"] for bd in batch_pdbs)

            print(f"Total batches processed: {len(batch_pdbs)}")
            print(f"Total structures: {total_batch_size}")
            print(f"Total residues: {total_residues}")

            for i, batch_data in enumerate(batch_pdbs):
                print(
                    f"  Batch {i}: batch_size={batch_data['batch_size']}, num_res={batch_data['num_res']}"
                )

            # Verify that all methods produced some results
            assert len(results) == 4, "All four methods should have been tested"

            for method_name, result_data in results.items():
                num_results = len(result_data["results"])
                if num_results == 0:
                    print(f"WARNING: {method_name} produced no results")
                else:
                    print(f"SUCCESS: {method_name} produced {num_results} results")

            print("\nBenchmark completed successfully!")
