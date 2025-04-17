import pytest
import torch

from cogeneration.config.base import DataTaskEnum
from cogeneration.data.interpolant import Interpolant
from cogeneration.data.noise_mask import centered_gaussian
from cogeneration.data.rigid import batch_align_structures, batch_center_of_mass


class TestInterpolant:
    """Test suite for the Interpolant class."""

    def test_batch_ot(self, mock_cfg_uninterpolated):
        """
        Test the _batch_ot method of the Interpolant class.
        This method should:
        1. Compute optimal transport between two batches of translations
        2. Re-center both batches
        3. Return the OT mapping of trans_0 to trans_1
        """
        mock_cfg_uninterpolated.data.task = DataTaskEnum.inpainting
        mock_cfg = mock_cfg_uninterpolated.interpolate()

        device = torch.device("cpu")
        interpolant = Interpolant(mock_cfg.interpolant)
        interpolant.set_device(device)

        num_batch, num_res = 2, 10

        res_mask = torch.ones(num_batch, num_res, device=device)

        # Set some residues to be fixed
        diffuse_mask = torch.ones(num_batch, num_res, device=device)
        diffuse_mask[:, 2:5] = 0
        # ensure diffuse_mask is actually a mask
        assert diffuse_mask.sum() > 0 and (diffuse_mask == 0).sum() > 0

        trans_1 = centered_gaussian(*res_mask.shape, device=device)
        assert torch.allclose(
            batch_center_of_mass(trans_1), torch.zeros(num_batch, 3), atol=0.1
        )

        # Add a translation to trans_0 to ensure it is off-center
        trans_0 = centered_gaussian(*res_mask.shape, device=device)
        trans_0_offset = torch.tensor([3.0, 5.0, 0.0], device=device)
        trans_0 += trans_0_offset
        assert torch.allclose(batch_center_of_mass(trans_0), trans_0_offset, atol=0.1)

        # Pre-compute the alignment, ensure centered
        trans_0_aligned_centered, _, _ = batch_align_structures(
            trans_0, trans_1, mask=res_mask, center=True
        )
        trans_0_aligned_centered.reshape(num_batch, num_res, 3)
        assert torch.allclose(
            batch_center_of_mass(trans_0_aligned_centered),
            torch.zeros(num_batch, 3),
            atol=0.1,
        )

        result = interpolant._batch_ot(trans_0, trans_1, res_mask=res_mask, center=True)

        # ensure centered
        assert result.shape == trans_1.shape
        assert torch.allclose(
            batch_center_of_mass(result), torch.zeros(num_batch, 3), atol=0.1
        )

        # Check that the result has lower total distance than the original noisy batch
        original_distances = torch.linalg.norm(trans_0 - trans_1, dim=-1)
        result_distances = torch.linalg.norm(result - trans_1, dim=-1)
        assert torch.sum(result_distances) <= torch.sum(original_distances)
