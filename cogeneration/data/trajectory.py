import logging
import math
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from mpl_toolkits.mplot3d.art3d import Path3DCollection

from cogeneration.data.protein import write_prot_to_pdb
from cogeneration.data.residue_constants import restypes_with_x
from cogeneration.type.metrics import OutputFileName

"""
Blitting-friendly trajectory visualization.
Most of the complexity here is to avoid redrawing the entire figure on each frame.
Artists are created once (`_init` methods), and then updated in plate (`_update` methods).
"""

# quiet down the matplotlib logs
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
logging.getLogger("matplotlib.animation").setLevel(logging.WARNING)


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
    logits_traj_path: Optional[str] = None  # OutputFileName.logits_traj_{gif/mp4}
    traj_panel_path: Optional[str] = None  # OutputFileName.traj_panel_{gif/mp4}


def _get_anim_writer() -> Tuple[str, matplotlib.animation.AbstractMovieWriter]:
    if animation.writers.is_available("ffmpeg"):
        return "mp4", animation.FFMpegWriter(fps=10, bitrate=1800)
    elif animation.writers.is_available("imagemagick"):
        return "gif", animation.ImageMagickWriter(fps=10)
    else:
        return "gif", animation.PillowWriter(fps=10)


def _subsample_timesteps(num_timesteps: int, up_to: int = 50) -> List[int]:
    """Subsample timesteps to limit animation size/time."""
    timesteps = np.arange(0, num_timesteps, math.ceil(num_timesteps / up_to))

    # ensure we include the final timestep
    if timesteps[-1] != num_timesteps - 1:
        timesteps = np.append(timesteps, num_timesteps - 1)

    return list(timesteps.astype(int))


def _get_letters_res_colors() -> Tuple[List[str], npt.NDArray]:
    """Get residue names in order (letters) and their colors."""
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

    return letters, res_colors


def _rgba_from_logits(logits: npt.NDArray) -> npt.NDArray:
    """
    logits: (N_res, S) where S = 20 or 21
    returns a (S, N_res, 4) RGBA image with row-wise fixed hue + value-based alpha
    """
    letters, res_colors = _get_letters_res_colors()

    assert (
        logits.shape[1] == 21 or logits.shape[1] == 20
    ), "Logits must be of shape (N_res, 20) or (N_res, 21)"
    # base RGB for each aa (21, 1, 3) → broadcast across columns
    rgb = np.repeat(res_colors[:, None, :], logits.T.shape[1], axis=1)
    # clip / scale to [0,1]
    alpha = np.clip(logits.T, 0.0, 1.0)[..., None]  # (S, L, 1)
    return np.concatenate([rgb, alpha], axis=-1)  # (S, L, 4)


def _init_logits_heatmap(
    ax: plt.Axes,
    logits: npt.NDArray,  # (N, S) where S = 20 or 21
    title: Optional[str] = "",
) -> matplotlib.image.AxesImage:
    rgba = _rgba_from_logits(logits)  # (21, L, 4)
    im = ax.imshow(rgba, origin="lower", interpolation="none")  # 1-time image

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel("Residue")
    ax.set_ylabel("Amino-acid")

    N, S = logits.shape
    ax.set_xticks(np.arange(0, N))
    labels = [str(i) if (i % 5 == 0) else "" for i in range(N)]  # every 5th
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(S))
    ax.set_yticklabels(restypes_with_x[:S])

    return im


def _init_logits_borders(
    ax: plt.Axes,
    logits_traj: npt.NDArray,  # (T, N, S) where S = 20 or 21
    aa_traj: npt.NDArray,  # (T, N)
    diffuse_mask: npt.NDArray,  # (N,)
) -> List[Rectangle]:
    """
    Draws borders around each residue in the logits heatmap, black for current res or red if fixed.
    x stays fixed, y will be updated each frame
    """
    num_timesteps, num_res, num_tokens = logits_traj.shape
    rects = []
    for i in range(num_res):
        edge = "black" if diffuse_mask[i] else "red"
        r = plt.Rectangle(
            (i - 0.5, aa_traj[0, i] - 0.5), 1, 1, fill=False, edgecolor=edge, lw=0.5
        )
        ax.add_patch(r)
        rects.append(r)
    return rects


def save_logits_traj(
    logits_traj: npt.NDArray,  # (T, N, S) where S = 20 or 21
    aa_traj: npt.NDArray,  # (T, N)
    diffuse_mask: npt.NDArray,  # (N,)
    output_dir: str,
    animation_interval_ms: float = 100,
) -> str:
    """
    Writes logits trajectory as a heatmap animation, one frame per step in the trajectory.
    Returns the path to the animation.
    """

    # Create output directory if it doesn't exist.
    os.makedirs(output_dir, exist_ok=True)

    # Create figure and axis.
    num_timesteps, num_res = logits_traj.shape[:2]
    fig, ax = plt.subplots(figsize=(int(num_res * 0.15), 5), dpi=100)

    # create image to update, draw frame 1
    im = _init_logits_heatmap(ax, logits=logits_traj[0], title="Logits trajectory")
    rects = _init_logits_borders(
        ax, logits_traj=logits_traj, aa_traj=aa_traj, diffuse_mask=diffuse_mask
    )

    def update(frame):
        im.set_data(_rgba_from_logits(logits_traj[frame]))

        # move rectangles to new AA row
        for i, r in enumerate(rects):
            r.set_y(aa_traj[frame, i] - 0.5)

        return (im, *rects)

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=_subsample_timesteps(num_timesteps, up_to=50),
        interval=animation_interval_ms,
        blit=True,
    )

    # Save animation
    ext, writer = _get_anim_writer()
    if ext == "gif":
        anim_path = os.path.join(output_dir, OutputFileName.logits_traj_gif)
    elif ext == "mp4":
        anim_path = os.path.join(output_dir, OutputFileName.logits_traj_mp4)
    else:
        raise ValueError(f"Unknown animation format: {ext}")
    anim.save(anim_path, writer=writer)
    plt.close(fig)

    return anim_path


def _init_seq_artists(
    ax: plt.Axes,
    num_res: int,
    letters: List[str],
    res_colors: npt.NDArray,
) -> Tuple[Sequence[Rectangle], Sequence[Text]]:
    """Create one rectangle + text per residue; return them for later updates."""
    ax.set_axis_off()
    rects: List[Rectangle] = []
    texts: List[matplotlib.text.Text] = []

    for i in range(num_res):
        rect = Rectangle(
            (i, 0),
            1.0,
            1.0,
            facecolor=res_colors[20],  # unknown to start
            edgecolor="white",
            lw=0.5,
            alpha=0.5,
        )
        ax.add_patch(rect)
        rects.append(rect)

        txt = ax.text(
            i + 0.5, 0.5, "", ha="center", va="center", fontsize=10, color="k"
        )
        texts.append(txt)

    ax.set_xlim(0, num_res)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    return rects, texts


def _update_seq_artists(
    seq: npt.NDArray,  # (N,)
    rects: Sequence[Rectangle],
    texts: Sequence[matplotlib.text.Text],
    letters: List[str],
    res_colors: npt.NDArray,
) -> Tuple[Rectangle, ...]:
    """Mutate rect/text colors + labels in place and return artists."""
    for i, aa_idx in enumerate(seq):
        rects[i].set_facecolor(res_colors[int(aa_idx)])
        texts[i].set_text(letters[int(aa_idx)])
    return tuple(rects) + tuple(texts)


def plot_seq(ax: Optional[Axes], seq: npt.NDArray):
    """
    Plot sequence as line of boxes with AA inside.
    Expects ax with appropriate ratio.
    """
    if ax is None:
        seq_len = len(seq)
        fig = plt.figure(figsize=(seq_len * 0.15, 1), dpi=100)
        ax = fig.add_subplot(111)

    ax.cla()
    ax.set_axis_off()

    letters, res_colors = _get_letters_res_colors()

    rects, texts = _init_seq_artists(
        ax, num_res=len(seq), letters=letters, res_colors=res_colors
    )
    _update_seq_artists(
        seq, rects=rects, texts=texts, letters=letters, res_colors=res_colors
    )

    return rects, texts


@dataclass
class CameraLimits:
    """Helper for 3D scenes to fix camera limits."""

    center: npt.NDArray
    width: float

    @classmethod
    def from_bb_traj(cls, bb_traj: npt.NDArray) -> "CameraLimits":
        # determine camera limits by inspecting c-alphas
        ca_all = bb_traj[:, :, 1, :].reshape(-1, 3)  # CA = atom index 1
        center = ca_all.mean(0)  # 3D
        width = (ca_all - center).ptp(0).max() / 2
        return cls(center=center, width=width)

    def set_limits(self, ax):
        ax.set_xlim(self.center[0] - self.width, self.center[0] + self.width)
        ax.set_ylim(self.center[1] - self.width, self.center[1] + self.width)
        ax.set_zlim(self.center[2] - self.width, self.center[2] + self.width)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_box_aspect([1, 1, 1])  # lock cube
        ax.view_init(elev=25, azim=45)


def _get_structure_cmap(N: int, diffuse_mask: Optional[npt.NDArray]):
    _GREY = (0.6, 0.6, 0.6, 1.0)
    cmap = plt.get_cmap("Spectral")(np.linspace(0, 1, N))
    if diffuse_mask is not None:
        cmap[~diffuse_mask] = _GREY
    return cmap


def _init_structure_artists(
    ax: Axes,
    positions: npt.NDArray,  # (N, 3_atoms, 3)
    diffuse_mask: npt.NDArray,  # (N,)
    limits: CameraLimits,
) -> Tuple[Path3DCollection, Path3DCollection, Path3DCollection]:
    """Create three scatter collections (N, CA, CB) once and return them. Draws first frame."""
    limits.set_limits(ax)

    colors = _get_structure_cmap(positions.shape[0], diffuse_mask)

    N, CA, CB = positions[:, 0, :], positions[:, 1, :], positions[:, 2, :]

    scat_N: Path3DCollection = ax.scatter(
        N[:, 0], N[:, 1], N[:, 2], c=colors, s=12, depthshade=True, alpha=0.25
    )
    scat_CA: Path3DCollection = ax.scatter(
        CA[:, 0], CA[:, 1], CA[:, 2], c=colors, s=20, depthshade=True
    )
    scat_CB: Path3DCollection = ax.scatter(
        CB[:, 0], CB[:, 1], CB[:, 2], c=colors, s=12, depthshade=True, alpha=0.25
    )
    return scat_N, scat_CA, scat_CB


def _update_structure_artists(
    positions: npt.NDArray,
    scats: Tuple[Path3DCollection, Path3DCollection, Path3DCollection],
) -> Tuple[Path3DCollection, Path3DCollection, Path3DCollection]:
    """In‑place update of the scatter collections using public setters."""
    N, CA, CB = positions[:, 0, :], positions[:, 1, :], positions[:, 2, :]

    # update N
    scats[0].set_offsets(np.c_[N[:, 0], N[:, 1]])
    scats[0].set_3d_properties(N[:, 2], zdir="z")

    # update CA
    scats[1].set_offsets(np.c_[CA[:, 0], CA[:, 1]])
    scats[1].set_3d_properties(CA[:, 2], zdir="z")

    # update CB
    scats[2].set_offsets(np.c_[CB[:, 0], CB[:, 1]])
    scats[2].set_3d_properties(CB[:, 2], zdir="z")

    return scats


def plot_structure(
    ax: Optional[Axes],
    positions: npt.NDArray,
    diffuse_mask: npt.NDArray,
    limits: Optional[CameraLimits] = None,
):
    """
    Plots translation positions in 3D, using camera limits `center` and `width`
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 12), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
    ax.cla()

    if limits is None:
        limits = CameraLimits.from_bb_traj(positions[None])  # (N, 3, 3) -> (1, N, 3, 3)

    return _init_structure_artists(
        ax=ax, positions=positions, diffuse_mask=diffuse_mask, limits=limits
    )


def animate_trajectories(
    prot_structure_traj: npt.NDArray,  # [noisy_T, N, 37, 3]
    prot_aa_traj: npt.NDArray,  # [noisy_T, N]
    model_structure_traj: npt.NDArray,  # [clean_T, N, 37, 3]  (clean_T = noisy_T-1)
    model_aa_traj: npt.NDArray,  # [clean_T, N]
    model_logits_traj: npt.NDArray,  # [clean_T, N, S] (S = 20, 21)
    diffuse_mask: npt.NDArray,  # [N]
    output_dir: str,
    animation_max_frames: int,
):
    """
    Animates the protein and model trajectories side by side.
    3 x 2 grid with logits, AA sequences on top and structures on bottom, protein on left, model pred on right.
    """
    os.makedirs(output_dir, exist_ok=True)

    # `model_traj` is one step shorter than `protein_traj`
    num_timesteps, num_res = prot_structure_traj.shape[:2]
    assert model_structure_traj.shape[0] == num_timesteps - 1

    # Determine backbone 3D camera limits
    camera_limits = CameraLimits.from_bb_traj(prot_structure_traj)

    # Fig determined ahead of time
    dpi = 100
    square_px = 20  # side-length of an AA square
    fig_w_in = max(6, min(30, num_res * square_px / dpi))
    seq_h_in = square_px / dpi
    struct_h_in = fig_w_in * 0.45
    logits_h_in = seq_h_in * 20
    fig_h_in = logits_h_in + seq_h_in + struct_h_in
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi, constrained_layout=False)

    # grid: 1 row (logits) + 1 row (seq) + 1 row (structures) / 2 cols (protein, model)
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[logits_h_in * dpi, seq_h_in * dpi, struct_h_in * dpi],
        wspace=0.02,
        hspace=0.01,
    )
    # ax_logits_prot = fig.add_subplot(gs[0, 0]) # (not used)
    ax_logits_model = fig.add_subplot(gs[0, 1])
    ax_sequence_prot = fig.add_subplot(gs[1, 0])
    ax_sequence_model = fig.add_subplot(gs[1, 1])
    ax_structure_prot = fig.add_subplot(gs[2, 0], projection="3d")
    ax_structure_model = fig.add_subplot(gs[2, 1], projection="3d")

    # tighten outside padding
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    # initialise artists
    logits_im = _init_logits_heatmap(ax_logits_model, model_logits_traj[0])
    logits_rects = _init_logits_borders(
        ax_logits_model,
        logits_traj=model_logits_traj,
        aa_traj=prot_aa_traj,
        diffuse_mask=diffuse_mask,
    )

    letters, res_colors = _get_letters_res_colors()
    seq_rects_l, seq_texts_l = _init_seq_artists(
        ax_sequence_prot, num_res, letters=letters, res_colors=res_colors
    )
    seq_rects_r, seq_texts_r = _init_seq_artists(
        ax_sequence_model, num_res, letters=letters, res_colors=res_colors
    )

    scat_l = _init_structure_artists(
        ax_structure_prot,
        positions=prot_structure_traj[0],
        diffuse_mask=diffuse_mask,
        limits=camera_limits,
    )
    scat_r = _init_structure_artists(
        ax_structure_model,
        positions=model_structure_traj[0],
        diffuse_mask=diffuse_mask,
        limits=camera_limits,
    )

    dynamic_artists = (
        tuple(seq_rects_l)
        + tuple(seq_texts_l)
        + tuple(seq_rects_r)
        + tuple(seq_texts_r)
        + scat_l
        + scat_r
        + (logits_im,)
        + tuple(logits_rects)
    )

    def update(timestep):
        # timestep in protein trajectory == timestep - 1 in model trajectory
        t = float(timestep) / (num_timesteps - 1)
        fig.suptitle(f"t = {t:.2f}", fontsize=9, y=0.99)

        _update_seq_artists(
            prot_aa_traj[timestep],
            rects=seq_rects_l,
            texts=seq_texts_l,
            letters=letters,
            res_colors=res_colors,
        )
        _update_structure_artists(prot_structure_traj[timestep], scats=scat_l)

        if timestep > 0:
            _update_seq_artists(
                model_aa_traj[timestep - 1],
                rects=seq_rects_r,
                texts=seq_texts_r,
                letters=letters,
                res_colors=res_colors,
            )
            _update_structure_artists(model_structure_traj[timestep - 1], scats=scat_r)
            logits_im.set_data(_rgba_from_logits(model_logits_traj[timestep - 1]))
            for i, r in enumerate(logits_rects):
                r.set_y(model_aa_traj[timestep - 1, i] - 0.5)
        else:
            pass  # blank on t=0 timestep

        return dynamic_artists

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=_subsample_timesteps(num_timesteps, up_to=animation_max_frames),
        interval=100,
        repeat_delay=1000,  # linger on final frame
        blit=True,
    )

    # Save animation
    ext, writer = _get_anim_writer()
    if ext == "gif":
        anim_path = os.path.join(output_dir, OutputFileName.traj_panel_gif)
    elif ext == "mp4":
        anim_path = os.path.join(output_dir, OutputFileName.traj_panel_mp4)
    else:
        raise ValueError(f"Unknown animation format: {ext}")
    anim.save(anim_path, writer=writer, dpi=dpi)
    plt.close(fig)

    return anim_path


def save_trajectory(
    sample_name: Union[int, str],
    sample_atom37: npt.NDArray,  # (N, 37, 3)
    protein_structure_traj: npt.NDArray,  # (noisy_T, N, 37, 3)
    model_structure_traj: npt.NDArray,  # (clean_T, N, 37, 3)
    diffuse_mask: npt.NDArray,  # (N,)
    output_dir: str,
    protein_aa_traj: Optional[npt.NDArray] = None,  # (noisy_T, N)
    model_aa_traj: Optional[npt.NDArray] = None,  # (clean_T, N)
    model_logits_traj: Optional[npt.NDArray] = None,  # (clean_T, N, S)
    write_trajectories: bool = True,
    write_animations: bool = True,
    animation_max_frames: int = 50,
) -> SavedTrajectory:
    """
    Writes final sample and reverse diffusion trajectory.

    Args:
        sample_atom37: [N, 37, 3] atom37 final sample.
        protein_structure_traj: [noisy_T, N, 37, 3] atom37 sampled diffusion states.
            T is number of time steps. First time step is t=eps,
            i.e. bb_prot_traj[0] is the final sample after reverse diffusion.
            N is number of residues.
        model_structure_traj: [clean_T, N, 37, 3] atom37 predictions of clean data at each time step.
        diffuse_mask: [N] which residues are diffused.
        output_dir: where to save samples.
        protein_aa_traj: [noisy_T, N] amino acids (0 - S inclusive where S = 20 or 21).
        model_aa_traj: [clean_T, N] amino acids (0 - S inclusive where S = 20 or 21).
        model_logits_traj: [clean_T, N, S] logits for each amino acid, from model
        write_trajectories: bool Whether to also write the PDB trajectories
        write_animations: bool Whether to create animation of the trajectory (slow, ~10-15s for 50 frames)
        animation_max_frames: int Max number of frames of all timesteps to include in animation.

    Returns:
        SavedTrajectory with paths to saved samples:
            'sample_pdb_path': PDB file of final state of reverse trajectory.
        And if `write_trajectories == True`:
            'traj_path': PDB file os all intermediate diffused states.
            'x0_traj_path': PDB file of C-alpha x_0 predictions at each state.
            'aa_traj_fasta_path': Fasta file of amino acid sequence at each state of trajectory.
            'logits_traj_path': GIF animation of logits trajectory, if provided.
        And if `write_animations == True`:
            `traj_panel_path`: GIF animation of the trajectory.
    """

    start_time = time.time()
    logger = logging.getLogger(__name__)

    def log_time(msg):
        elapsed_time = time.time() - start_time
        logger.debug(f"sample {sample_name} {msg}: {elapsed_time:.2f} seconds")

    # ensure directory exists
    os.makedirs(output_dir, exist_ok=True)

    diffuse_mask = diffuse_mask.astype(bool)
    sample_pdb_path = os.path.join(output_dir, OutputFileName.sample_pdb)

    noisy_traj_length, num_res, _, _ = protein_structure_traj.shape
    model_traj_length = model_structure_traj.shape[0]
    assert sample_atom37.shape == (num_res, 37, 3)
    assert protein_structure_traj.shape == (noisy_traj_length, num_res, 37, 3)
    assert model_structure_traj.shape == (model_traj_length, num_res, 37, 3)

    if protein_aa_traj is not None:
        assert protein_aa_traj.shape == (noisy_traj_length, num_res)
        assert model_aa_traj is not None
        assert model_aa_traj.shape == (model_traj_length, num_res)

    # Use b-factors to specify which residues are diffused.
    b_factors = np.tile((diffuse_mask * 100)[:, None], (1, 37))

    sample_pdb_path = write_prot_to_pdb(
        sample_atom37,
        file_path=sample_pdb_path,
        b_factors=b_factors,
        no_indexing=True,
        aatype=protein_aa_traj[-1] if protein_aa_traj is not None else None,
    )

    if not write_trajectories:
        return SavedTrajectory(
            sample_pdb_path=sample_pdb_path,
        )

    prot_traj_path = os.path.join(output_dir, OutputFileName.bb_traj_pdb)
    x0_traj_path = os.path.join(output_dir, OutputFileName.x0_traj_pdb)
    prot_traj_path = write_prot_to_pdb(
        protein_structure_traj,
        file_path=prot_traj_path,
        b_factors=b_factors,
        no_indexing=True,
        aatype=protein_aa_traj,
    )
    x0_traj_path = write_prot_to_pdb(
        model_structure_traj,
        file_path=x0_traj_path,
        b_factors=b_factors,
        no_indexing=True,
        aatype=model_aa_traj,
    )

    log_time("structure trajectory PDBs")

    # These file paths gated by files being provided
    aa_traj_fasta_path = None
    model_logits_traj_path = None
    traj_panel_path = None

    # Write amino acids trajectory, if provided.
    if protein_aa_traj is not None:
        aa_traj_fasta_path = os.path.join(output_dir, OutputFileName.aa_traj_fa)
        num_steps = protein_aa_traj.shape[0]
        with open(aa_traj_fasta_path, "w") as f:
            for i in range(num_steps):
                f.write(f">step{i}\n")
                f.write(
                    "".join([restypes_with_x[aa] for aa in protein_aa_traj[i]]) + "\n"
                )

    if (
        write_animations
        and protein_aa_traj is not None
        and model_aa_traj is not None
        and model_logits_traj is not None
    ):
        traj_panel_path = animate_trajectories(
            prot_structure_traj=protein_structure_traj,
            prot_aa_traj=protein_aa_traj,
            model_structure_traj=model_structure_traj,
            model_aa_traj=model_aa_traj,
            model_logits_traj=model_logits_traj,
            diffuse_mask=diffuse_mask,
            output_dir=output_dir,
            animation_max_frames=animation_max_frames,
        )
        log_time("trajectory animation")

    return SavedTrajectory(
        sample_pdb_path=sample_pdb_path,
        traj_path=prot_traj_path,
        x0_traj_path=x0_traj_path,
        aa_traj_fasta_path=aa_traj_fasta_path,
        logits_traj_path=model_logits_traj_path,
        traj_panel_path=traj_panel_path,
    )
