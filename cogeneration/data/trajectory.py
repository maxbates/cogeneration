import os
from dataclasses import dataclass
from typing import Optional, Union

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D

from cogeneration.data.protein import write_prot_to_pdb
from cogeneration.data.residue_constants import restypes_with_x
from cogeneration.type.metrics import OutputFileName


@dataclass
class SavedTrajectory:
    """
    Struct that tracks file saved by `save_trajectory()`.
    Entires should be None if not written.
    """

    sample_pdb_path: str  # OutputFileName.sample_pdb
    traj_path: Optional[str] = None  # OutputFileName.bb_traj_pdb
    x0_traj_path: Optional[str] = None  # OutputFileName.x0_traj_pdb
    aa_traj_fasta_path: Optional[str] = None  # OutputFileName.aa_traj_fa
    logits_traj_path: Optional[str] = None  # OutputFileName.logits_traj_anim
    traj_panel_path: Optional[str] = None  # OutputFileName.traj_panel_anim


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
    """

    # Create output directory if it doesn't exist.
    os.makedirs(output_dir, exist_ok=True)

    # Create figure and axis.
    fig, ax = plt.subplots(figsize=(10, int(logits_traj.shape[1] * 0.25)))
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

        # Clear previous patches
        for patch in reversed(ax.patches):
            patch.remove()

        # Border cells in black for the current aa in aa_traj
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

        # Highlight diffuse_mask cells in blue
        for i in range(logits_traj.shape[1]):
            if diffuse_mask[i]:
                ax.add_patch(
                    plt.Rectangle(
                        (i - 0.5, 0 - 0.5),  # Highlight entire column
                        1,
                        logits_traj.shape[2],
                        fill=True,
                        edgecolor="blue",
                        facecolor="blue",
                        alpha=0.3,
                    )
                )

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=logits_traj.shape[0],
        interval=animation_step_time_sec * 1000,
    )

    # Save animation
    anim_path = os.path.join(output_dir, OutputFileName.logits_traj_anim)
    anim.save(anim_path)
    plt.close(fig)

    return anim_path


def animate_trajectories(
    bb_prot_traj: npt.NDArray,  # [noisy_T, N, 37, 3]
    aa_traj: npt.NDArray,  # [noisy_T, N]
    x0_traj: npt.NDArray,  # [clean_T, N, 37, 3]  (clean_T = noisy_T-1)
    model_aa_traj: npt.NDArray,  # [clean_T, N]
    output_dir: str,
):
    """
    Animates the protein and model trajectories side by side.
    2 x 2 grid with AA sequences on top and structures on bottom.
    TODO also plot logits trajectory
    """

    os.makedirs(output_dir, exist_ok=True)

    # `model_traj` is one step shorter than `protein_traj`
    num_timesteps, num_res = bb_prot_traj.shape[:2]
    assert x0_traj.shape[0] == num_timesteps - 1
    # take every 5th frame, min 20 frames, by default to speed up animation drawing
    timesteps = np.arange(0, num_timesteps, min(5, max(1, int(num_timesteps / 20))))

    # residue names
    letters = list(restypes_with_x)
    letters[20] = "-"  # UNK

    # residue colors

    # helper – make a lighter shade by linear-mixing with white
    def tint(hex_code, f=0.35):
        r, g, b = to_rgb(hex_code)
        return (1 - f) * np.array([r, g, b]) + f * np.ones(3)

    # base hues for each class
    NEG = "#3778bf"  # blue
    POS = "#d62728"  # red
    POL = "#2ca02c"  # green
    NON = "#7f7f7f"  # grey

    aa_color = {
        # negative (blue shades)
        "D": to_rgb(NEG),  # Asp
        "E": tint(NEG, 0.4),  # Glu
        # positive (red shades)
        "K": to_rgb(POS),  # Lys
        "R": tint(POS, 0.35),  # Arg
        "H": tint(POS, 0.55),  # His
        # polar uncharged (green shades)
        "N": to_rgb(POL),  # Asn
        "Q": tint(POL, 0.35),  # Gln
        "S": tint(POL, 0.50),  # Ser
        "T": tint(POL, 0.65),  # Thr
        "C": tint(POL, 0.20),  # Cys
        "Y": tint(POL, 0.80),  # Tyr
        "W": tint(POL, 0.10),  # Trp
        # non-polar / hydrophobic (grey shades)
        "A": to_rgb(NON),  # Ala
        "V": tint(NON, 0.25),  # Val
        "L": tint(NON, 0.10),  # Leu
        "I": tint(NON, 0.40),  # Ile
        "M": tint(NON, 0.55),  # Met
        "F": tint(NON, 0.70),  # Phe
        "P": tint(NON, 0.85),  # Pro
        "G": tint(NON, 0.95),  # Gly
        # unknown
        "-": (1.0, 1.0, 1.0),  # white
    }
    # build the (21, 3) RGB array in the canonical order
    res_colors = np.array([aa_color[ltr] for ltr in letters])

    # camera limits by inspecting c-alphas
    ca_all = bb_prot_traj[:, :, 1, :].reshape(-1, 3)  # CA = atom index 1
    centre = ca_all.mean(0)
    half_range = (ca_all - centre).ptp(0).max() / 2

    def _set_limits(ax):
        ax.set_xlim(centre[0] - half_range, centre[0] + half_range)
        ax.set_ylim(centre[1] - half_range, centre[1] + half_range)
        ax.set_zlim(centre[2] - half_range, centre[2] + half_range)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_box_aspect([1, 1, 1])  # lock cube
        ax.view_init(elev=25, azim=45)

    # Fig determined ahead of time
    dpi = 100
    square_px = 24  # side-length of an AA square
    fig_w_in = max(6, min(30, num_res * square_px / dpi))
    seq_h_in = square_px / dpi
    struct_h_in = fig_w_in * 0.45
    fig_h_in = seq_h_in + struct_h_in
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi, constrained_layout=False)

    # grid: 1 row (seq) + 1 row (structures) / 2 cols
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[seq_h_in * dpi, struct_h_in * dpi],
        wspace=0.12,
        hspace=0.03,
    )
    ax_seq_l = fig.add_subplot(gs[0, 0])
    ax_seq_r = fig.add_subplot(gs[0, 1])
    ax_str_l = fig.add_subplot(gs[1, 0], projection="3d")
    ax_str_r = fig.add_subplot(gs[1, 1], projection="3d")
    ax_seq_l.set_axis_off()
    ax_seq_r.set_axis_off()
    _set_limits(ax_str_l)
    _set_limits(ax_str_r)

    # tighten outside padding
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01, wspace=0.08)

    def plot_seq(ax, seq):
        ax.cla()
        ax.set_axis_off()

        for i, r in enumerate(seq):
            ax.add_patch(
                Rectangle(
                    (i, 0),
                    2,
                    3,
                    facecolor=res_colors[r],
                    edgecolor="white",
                    lw=0.5,
                    alpha=0.5,
                )
            )
            ax.text(
                i + 0.5,
                0.5,
                letters[r],
                ha="center",
                va="center",
                fontsize=10,
                color="k",
            )
        ax.set_xlim(0, num_res)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")

    def plot_structure(ax, positions):
        ax.cla()
        _set_limits(ax)

        # color chains by residue position using Spectrum colormap
        pos_idx = np.arange(positions.shape[0]) / (positions.shape[0] - 1)
        pos_colors = plt.get_cmap("Spectral")(pos_idx)

        # plot each backbone atom, c-alpha emphasized
        N, CA, CB = positions[:, 0, :], positions[:, 1, :], positions[:, 2, :]
        ax.scatter(
            N[:, 0], N[:, 1], N[:, 2], c=pos_colors, s=12, depthshade=True, alpha=0.25
        )
        ax.scatter(CA[:, 0], CA[:, 1], CA[:, 2], c=pos_colors, s=20, depthshade=True)
        ax.scatter(
            CB[:, 0],
            CB[:, 1],
            CB[:, 2],
            c=pos_colors,
            s=12,
            depthshade=True,
            alpha=0.25,
        )

    def update(frame):
        t = float(frame) / (num_timesteps - 1)
        fig.suptitle(f"t = {t:.2f}", fontsize=9, y=0.99)

        plot_seq(ax_seq_l, aa_traj[frame])
        plot_structure(ax_str_l, bb_prot_traj[frame])

        if frame > 0:
            plot_seq(ax_seq_r, model_aa_traj[frame - 1])
            plot_structure(ax_str_r, x0_traj[frame - 1])
        else:
            ax_seq_r.cla()
            ax_seq_r.set_axis_off()
            ax_str_r.cla()
            _set_limits(ax_str_r)

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=list(timesteps.astype(int)),
        interval=100,
        repeat_delay=1000,  # linger on final frame
        blit=False,
    )

    # Save animation
    anim_path = os.path.join(output_dir, OutputFileName.traj_panel_anim)
    anim.save(anim_path, dpi=dpi)
    plt.close(fig)

    return anim_path


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
    sample_pdb_path = os.path.join(output_dir, OutputFileName.sample_pdb)

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

    prot_traj_path = os.path.join(output_dir, OutputFileName.bb_traj_pdb)
    x0_traj_path = os.path.join(output_dir, OutputFileName.x0_traj_pdb)
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
        aa_traj_fasta_path = os.path.join(output_dir, OutputFileName.aa_traj_fa)
        num_steps = aa_traj.shape[0]
        with open(aa_traj_fasta_path, "w") as f:
            for i in range(num_steps):
                f.write(f">step{i}\n")
                f.write("".join([restypes_with_x[aa] for aa in aa_traj[i]]) + "\n")

    # Animate logits trajectory, if provided.
    if model_logits_traj is not None:
        model_logits_traj_path = save_logits_traj(
            model_logits_traj,
            model_aa_traj,
            diffuse_mask=diffuse_mask,
            output_dir=output_dir,
        )

    traj_panel_path = None
    if (
        model_aa_traj is not None
        and model_logits_traj is not None
        and aa_traj_fasta_path is not None
    ):
        traj_panel_path = animate_trajectories(
            bb_prot_traj,
            aa_traj,
            x0_traj,
            model_aa_traj,
            output_dir=output_dir,
        )

    return SavedTrajectory(
        sample_pdb_path=sample_pdb_path,
        traj_path=prot_traj_path,
        x0_traj_path=x0_traj_path,
        aa_traj_fasta_path=aa_traj_fasta_path,
        logits_traj_path=model_logits_traj_path,
        traj_panel_path=traj_panel_path,
    )
