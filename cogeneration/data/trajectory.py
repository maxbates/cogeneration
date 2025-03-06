import os
from dataclasses import dataclass
from typing import Dict, Optional, Union

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from cogeneration.data.protein import write_prot_to_pdb
from cogeneration.data.residue_constants import restypes_with_x


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


@dataclass
class SavedTrajectory:
    """
    Struct that tracks file saved by `save_trajectory()`.
    Entires should be None if not written.
    """

    sample_pdb_path: str
    traj_path: Optional[str] = None
    x0_traj_path: Optional[str] = None
    aa_traj_fasta_path: Optional[str] = None
    logits_traj_path: Optional[str] = None


def save_trajectory(
    sample_name: Union[int, str],
    sample: npt.NDArray,
    bb_prot_traj: npt.NDArray,
    x0_traj: npt.NDArray,
    diffuse_mask: npt.NDArray,
    output_dir: str,
    aa_traj: Optional[npt.NDArray] = None,
    model_aa_traj: Optional[npt.NDArray] = None,
    model_logits_traj: Optional[npt.NDArray] = None,
    write_trajectories: bool = True,
) -> SavedTrajectory:
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
        model_aa_traj: [clean_T, N] amino acids (0 - 20 inclusive).
        model_logits_traj: [clean_T, N, 21] logits for each amino acid, from model
        write_trajectories: bool Whether to also write the trajectories as well
                                 as the final sample

    Returns:
        Dictionary with paths to saved samples:
            'sample_pdb_path': PDB file of final state of reverse trajectory.
        And if `write_trajectories == True`:
            'traj_path': PDB file os all intermediate diffused states.
            'x0_traj_path': PDB file of C-alpha x_0 predictions at each state.
            'aa_traj_fasta_path': Fasta file of amino acid sequence at each state of trajectory.
            'logits_traj_path': GIF animation of logits trajectory, if provided.
    """

    diffuse_mask = diffuse_mask.astype(bool)
    sample_pdb_path = os.path.join(output_dir, "sample.pdb")

    noisy_traj_length, num_res, _, _ = bb_prot_traj.shape
    model_traj_length = x0_traj.shape[0]
    assert sample.shape == (num_res, 37, 3)
    assert bb_prot_traj.shape == (noisy_traj_length, num_res, 37, 3)
    assert x0_traj.shape == (model_traj_length, num_res, 37, 3)

    if aa_traj is not None:
        assert aa_traj.shape == (noisy_traj_length, num_res)
        assert model_aa_traj is not None
        assert model_aa_traj.shape == (model_traj_length, num_res)

    # Use b-factors to specify which residues are diffused.
    b_factors = np.tile((diffuse_mask * 100)[:, None], (1, 37))

    sample_pdb_path = write_prot_to_pdb(
        sample,
        file_path=sample_pdb_path,
        b_factors=b_factors,
        no_indexing=True,
        aatype=aa_traj[-1] if aa_traj is not None else None,
    )

    if not write_trajectories:
        return SavedTrajectory(
            sample_pdb_path=sample_pdb_path,
        )

    prot_traj_path = os.path.join(output_dir, "bb_traj.pdb")
    x0_traj_path = os.path.join(output_dir, "x0_traj.pdb")
    # These file paths gated by files being provided
    aa_traj_fasta_path = None
    model_logits_traj_path = None

    prot_traj_path = write_prot_to_pdb(
        bb_prot_traj,
        file_path=prot_traj_path,
        b_factors=b_factors,
        no_indexing=True,
        aatype=aa_traj,
    )
    x0_traj_path = write_prot_to_pdb(
        x0_traj,
        file_path=x0_traj_path,
        b_factors=b_factors,
        no_indexing=True,
        aatype=model_aa_traj,
    )

    # Write amino acids trajectory, if provided.
    if aa_traj is not None:
        aa_traj_fasta_path = os.path.join(output_dir, "aa_traj.fasta")
        num_steps = aa_traj.shape[0]
        with open(aa_traj_fasta_path, "w") as f:
            for i in range(num_steps):
                f.write(f">step{i}\n")
                f.write("".join([restypes_with_x[aa] for aa in aa_traj[i]]))

    # Animate logits trajectory, if provided.
    if model_logits_traj is not None:
        model_logits_traj_path = save_logits_traj(
            model_logits_traj,
            model_aa_traj,
            diffuse_mask,
            output_dir,
        )

    return SavedTrajectory(
        sample_pdb_path=sample_pdb_path,
        traj_path=prot_traj_path,
        x0_traj_path=x0_traj_path,
        aa_traj_fasta_path=aa_traj_fasta_path,
        logits_traj_path=model_logits_traj_path,
    )
