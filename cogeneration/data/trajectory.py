import os
from typing import Dict

import numpy as np

from cogeneration.data.protein import write_prot_to_pdb


def save_traj(
    sample: np.ndarray,
    bb_prot_traj: np.ndarray,
    x0_traj: np.ndarray,
    diffuse_mask: np.ndarray,
    output_dir: str,
    aa_traj: np.ndarray = None,
    clean_aa_traj: np.ndarray = None,
    write_trajectories: bool = True,
) -> Dict[str, str]:
    """Writes final sample and reverse diffusion trajectory.

    Args:
        bb_prot_traj: [noisy_T, N, 37, 3] atom37 sampled diffusion states.
            T is number of time steps. First time step is t=eps,
            i.e. bb_prot_traj[0] is the final sample after reverse diffusion.
            N is number of residues.
        x0_traj: [clean_T, N, 37, 3] atom37 predictions of clean data at each time step.
        res_mask: [N] residue mask.
        diffuse_mask: [N] which residues are diffused.
        output_dir: where to save samples.
        aa_traj: [noisy_T, N] amino acids (0 - 20 inclusive).
        clean_aa_traj: [clean_T, N] amino acids (0 - 20 inclusive).
        write_trajectories: bool Whether to also write the trajectories as well
                                 as the final sample

    Returns:
        Dictionary with paths to saved samples.
            'sample_path': PDB file of final state of reverse trajectory.
            'traj_path': PDB file os all intermediate diffused states.
            'x0_traj_path': PDB file of C-alpha x_0 predictions at each state.
        b_factors are set to 100 for diffused residues
        residues if there are any.
    """

    # Write sample.
    diffuse_mask = diffuse_mask.astype(bool)
    sample_path = os.path.join(output_dir, "sample.pdb")
    prot_traj_path = os.path.join(output_dir, "bb_traj.pdb")
    x0_traj_path = os.path.join(output_dir, "x0_traj.pdb")

    # Use b-factors to specify which residues are diffused.
    b_factors = np.tile((diffuse_mask * 100)[:, None], (1, 37))

    noisy_traj_length, num_res, _, _ = bb_prot_traj.shape
    clean_traj_length = x0_traj.shape[0]
    assert sample.shape == (num_res, 37, 3)
    assert bb_prot_traj.shape == (noisy_traj_length, num_res, 37, 3)
    assert x0_traj.shape == (clean_traj_length, num_res, 37, 3)

    if aa_traj is not None:
        assert aa_traj.shape == (noisy_traj_length, num_res)
        assert clean_aa_traj is not None
        assert clean_aa_traj.shape == (clean_traj_length, num_res)

    sample_path = write_prot_to_pdb(
        sample,
        sample_path,
        b_factors=b_factors,
        no_indexing=True,
        aatype=aa_traj[-1] if aa_traj is not None else None,
    )
    if write_trajectories:
        prot_traj_path = write_prot_to_pdb(
            bb_prot_traj,
            prot_traj_path,
            b_factors=b_factors,
            no_indexing=True,
            aatype=aa_traj,
        )
        x0_traj_path = write_prot_to_pdb(
            x0_traj,
            x0_traj_path,
            b_factors=b_factors,
            no_indexing=True,
            aatype=clean_aa_traj,
        )
    return {
        "sample_path": sample_path,
        "traj_path": prot_traj_path,
        "x0_traj_path": x0_traj_path,
    }
