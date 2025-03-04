import os
from typing import Dict, Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from data.residue_constants import restypes_with_x

from cogeneration.data.protein import write_prot_to_pdb


def save_logits_traj(
    logits_traj: npt.NDArray,
    aa_traj: npt.NDArray,
    diffuse_mask: npt.NDArray,
    output_dir: str,
    animation_step_time_sec: float = 0.1,
) -> str:
    """
    Writes logits trajectory as a heatmap animation, one frame per step in the trajectory.
    Returns the path to the animation.

    TODO - highlight diffuse_mask somehow
    """

    # Create output directory if it doesn't exist.
    os.makedirs(output_dir, exist_ok=True)

    # Create figure and axis.
    fig, ax = plt.subplots()
    ax.set_title("Logits trajectory")
    ax.set_xlabel("Residue")
    ax.set_ylabel("Amino acid")
    ax.set_xticks(np.arange(logits_traj.shape[1]))
    ax.set_xticklabels(np.arange(logits_traj.shape[1]))
    ax.set_yticks(np.arange(21))
    ax.set_yticklabels(restypes_with_x)

    # Show logits as green heatmap
    im = ax.imshow(logits_traj[0].T, cmap="Greens", vmin=0, vmax=1)

    # Create animation.
    def update(frame):
        im.set_data(logits_traj[frame].T)

        # border cells in black for the current aa in aa_traj
        for i in range(logits_traj.shape[1]):
            ax.add_patch(
                plt.Rectangle(
                    (i - 0.5, aa_traj[frame, i] - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="black",
                    lw=1,
                )
            )

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=logits_traj.shape[0],
        interval=animation_step_time_sec * 1000,
    )

    # Save animation.
    anim_path = os.path.join(output_dir, "logits_traj.gif")
    anim.save(anim_path)
    plt.close(fig)

    return anim_path


def save_traj(
    sample: npt.NDArray,
    bb_prot_traj: npt.NDArray,
    x0_traj: npt.NDArray,
    diffuse_mask: npt.NDArray,
    output_dir: str,
    aa_traj: Optional[npt.NDArray] = None,
    clean_aa_traj: Optional[npt.NDArray] = None,
    clean_logits_traj: Optional[npt.NDArray] = None,
    write_trajectories: bool = True,
) -> Dict[str, str]:
    """Writes final sample and reverse diffusion trajectory.

    Args:
        sample: [N, 37, 3] atom37 final sample.
        bb_prot_traj: [noisy_T, N, 37, 3] atom37 sampled diffusion states.
            T is number of time steps. First time step is t=eps,
            i.e. bb_prot_traj[0] is the final sample after reverse diffusion.
            N is number of residues.
        x0_traj: [clean_T, N, 37, 3] atom37 predictions of clean data at each time step.
        res_mask: [N] residue mask.  # TODO add argument
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
        b_factors are set to 100 for diffused residues residues if there are any.
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

    clean_logits_traj_path = None
    if clean_logits_traj is not None:
        clean_logits_traj_path = save_logits_traj(
            clean_logits_traj,
            clean_aa_traj,
            diffuse_mask,
            output_dir,
        )

    return {
        "sample_path": sample_path,
        "traj_path": prot_traj_path,
        "x0_traj_path": x0_traj_path,
        "logits_traj_path": clean_logits_traj_path,
    }
