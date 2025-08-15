import logging
import os
from pathlib import Path

import pytest
import torch

from cogeneration.data.potentials import (
    FKSteeringTrajectory,
    FKStepMetric,
    PotentialField,
)
from cogeneration.data.trajectory_save import (
    save_potential_logits_traj,
    write_fk_steering_energy_traj,
)


class TestWriteFKSteeringTraj:
    def test_write_fk_steering_traj(self, tmp_path):
        """Test that write_fk_steering_traj creates a plot file with dummy FK steering data."""
        # Create dummy FK steering trajectory with 10 steps and 4 particles
        num_batch = 1
        num_res = 10
        num_particles = 4
        num_steps = 10

        # Create dummy metrics for each step
        metrics = []
        for step in range(num_steps):
            # Create dummy values that vary over time to make interesting plots
            energy = [0.5 + 0.1 * step + 0.05 * i for i in range(num_particles)]
            log_G = [0.2 * step + 0.1 * i for i in range(num_particles)]
            log_G_delta = [0.1 * (-1) ** step + 0.02 * i for i in range(num_particles)]
            weights = [
                1.0 / num_particles + 0.01 * i * (-1) ** step
                for i in range(num_particles)
            ]
            # Normalize weights to sum to 1
            weight_sum = sum(weights)
            weights = [w / weight_sum for w in weights]
            effective_sample_size = 3.5 - 0.1 * step  # Decreasing ESS over time
            keep = list(range(num_particles))  # All particles kept

            # guidance just saves the logits, they are converted when plotting
            guidance = PotentialField(
                trans=torch.randn(num_particles, num_res, 3),
                rotmats=torch.randn(num_particles, num_res, 3),
                logits=torch.randn(num_particles, num_res, 21),  # assume masking
            )

            metric = FKStepMetric(
                step=step,
                energy=energy,
                log_G=log_G,
                log_G_delta=log_G_delta,
                weights=weights,
                effective_sample_size=effective_sample_size,
                keep=keep,
                guidance=guidance,
            )
            metrics.append(metric)

        # Create FK steering trajectory
        fk_traj = FKSteeringTrajectory(
            num_batch=num_batch,
            num_particles=num_particles,
            metrics=metrics,
        )

        # Write the trajectory plot
        output_path = tmp_path / "test_fk_steering_traj.png"
        write_fk_steering_energy_traj(fk_traj, str(output_path))

        # Verify the file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0  # File should have content

        # Write the logits trajectory animation
        aa_traj = torch.randint(low=0, high=21, size=(num_steps, num_res))
        logits_anim_path = save_potential_logits_traj(
            metrics=fk_traj.metrics,
            sample_aa_traj=aa_traj.numpy(),
            motif_mask=None,
            output_dir=str(tmp_path),
        )

        assert logits_anim_path is not None
        assert os.path.exists(logits_anim_path)

        # Log the path for manual inspection
        print(f"FK steering trajectory plot saved to: {output_path}")
