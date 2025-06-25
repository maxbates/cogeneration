import logging
import os
from pathlib import Path

import pytest

from cogeneration.data.potentials import FKSteeringTrajectory, FKStepMetric
from cogeneration.data.trajectory_save import write_fk_steering_traj


class TestWriteFKSteeringTraj:
    def test_write_fk_steering_traj(self, tmp_path):
        """Test that write_fk_steering_traj creates a plot file with dummy FK steering data."""
        # Create dummy FK steering trajectory with 10 steps and 4 particles
        num_steps = 10
        num_particles = 4

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

            metric = FKStepMetric(
                step=step,
                energy=energy,
                log_G=log_G,
                log_G_delta=log_G_delta,
                weights=weights,
                effective_sample_size=effective_sample_size,
                keep=keep,
            )
            metrics.append(metric)

        # Create FK steering trajectory
        fk_traj = FKSteeringTrajectory(
            num_batch=1,
            num_particles=num_particles,
            metrics=metrics,
        )

        # Write the trajectory plot
        output_path = tmp_path / "test_fk_steering_traj.png"
        write_fk_steering_traj(fk_traj, str(output_path))

        # Verify the file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0  # File should have content

        # Log the path for manual inspection
        logging.info(f"FK steering trajectory plot saved to: {output_path}")
        print(f"FK steering trajectory plot saved to: {output_path}")
