import pytest
import torch

from cogeneration.data.interpolant import Interpolant
from cogeneration.data.residue_constants import restype_order_with_x, restypes_with_x
from cogeneration.dataset.test_utils import create_single_item_batch
from cogeneration.models.loss_calculator import BatchLossCalculator
from cogeneration.type.batch import BatchProp as bp
from cogeneration.type.batch import feats_to_prediction
from cogeneration.type.task import DataTask


class TestLossCalculator:
    @pytest.mark.parametrize(
        "hot_spot_case,expected_loss_condition",
        [
            ("no_hot_spots", "zero"),
            ("interface_hot_spot", "zero"),
            ("outside_interface_hot_spot", "non_zero"),
        ],
    )
    def test_hot_spots_loss(
        self, mock_cfg, pdb_2qlw_processed_feats, hot_spot_case, expected_loss_condition
    ):
        clean_batch = create_single_item_batch(pdb_2qlw_processed_feats)

        interpolant = Interpolant(mock_cfg.interpolant)
        noisy_feats = interpolant.corrupt_batch(
            clean_batch,
            task=DataTask.hallucination,
        )
        prediction = feats_to_prediction(clean_batch, mock_cfg)

        res_idx = clean_batch[bp.res_idx]  # (B, N)
        aatypes = clean_batch[bp.aatypes_1]  # (B, N)

        # Initialize hot spots mask (all False by default)
        batch_size, num_res = clean_batch[bp.res_mask].shape
        hot_spots_mask = torch.zeros((batch_size, num_res), dtype=torch.bool)

        if hot_spot_case == "no_hot_spots":
            # No hot spots marked - should result in zero loss
            pass

        elif hot_spot_case == "interface_hot_spot":
            # Mark HIS at pos 47 (res_idx 48 now, residue 46 in raw PDB, which starts from -1)
            interface_res_idx = 47

            # confirm that residue is histidine
            assert (
                res_idx[0, interface_res_idx] == 48
            ), f"Expected residue index 48, got {res_idx[0, interface_res_idx]}"
            assert (
                aatypes[0, interface_res_idx].item() == restype_order_with_x["H"]
            ), f"Residue should be histidine, got {aatypes[0, interface_res_idx].item()} ({restypes_with_x[aatypes[0, interface_res_idx].item()]})"

            hot_spots_mask[0, interface_res_idx] = True

        elif hot_spot_case == "outside_interface_hot_spot":
            # Mark SER at pos 69 (res_idx 70 now, residue 68 in raw PDB, which starts from -1)
            outside_res_idx = 69

            # confirm that residue is serine
            assert (
                res_idx[0, outside_res_idx] == 70
            ), f"Expected residue index 70, got {res_idx[0, outside_res_idx]}"
            assert (
                aatypes[0, outside_res_idx].item() == restype_order_with_x["S"]
            ), f"Residue 70 should be serine, got {aatypes[0, outside_res_idx].item()} ({restypes_with_x[aatypes[0, outside_res_idx].item()]})"

            hot_spots_mask[0, outside_res_idx] = True

        # Add hot spots to the noisy batch
        noisy_feats[bp.hot_spots] = hot_spots_mask

        calculator = BatchLossCalculator(
            cfg=mock_cfg,
            batch=noisy_feats,
            pred=prediction,
        )

        hot_spot_loss = calculator.loss_hot_spots()

        # Check expected loss conditions
        if expected_loss_condition == "zero":
            assert torch.allclose(
                hot_spot_loss, torch.zeros_like(hot_spot_loss), atol=1e-6
            ), f"Expected zero loss for {hot_spot_case}, got {hot_spot_loss}"
        elif expected_loss_condition == "non_zero":
            assert torch.any(
                hot_spot_loss > 1.0
            ), f"Expected non-zero loss for {hot_spot_case}, got {hot_spot_loss}"
