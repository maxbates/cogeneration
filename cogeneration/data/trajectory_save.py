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
import torch
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from mpl_toolkits.mplot3d.art3d import Path3DCollection

from cogeneration.data.potentials import (
    FKSteeringTrajectory,
    FKStepMetric,
    PotentialField,
)
from cogeneration.data.protein import write_prot_to_pdb
from cogeneration.data.residue_constants import restypes_with_x
from cogeneration.type.metrics import OutputFileName
from cogeneration.util.log import rank_zero_logger

"""
Trajectory visualization.
Most of the complexity here is to avoid redrawing the entire figure on each frame.
Artists are created once (`_init` methods), and then updated in plate (`_update` methods).
"""

logger = rank_zero_logger(__name__)


@dataclass
class SavedTrajectory:
    """
    Struct that tracks file saved by `save_trajectory()`.
    Entires should be None if not written.
    """

    sample_pdb_path: str  # OutputFileName.sample_pdb
    sample_pdb_backbone_path: str  # OutputFileName.sample_pdb_backbone
    sample_traj_path: Optional[str] = None  # OutputFileName.sample_traj_pdb
    model_pred_traj_path: Optional[str] = None  # OutputFileName.model_pred_traj_pdb
    aa_traj_fasta_path: Optional[str] = None  # OutputFileName.aa_traj_fa
    logits_traj_path: Optional[str] = None  # OutputFileName.logits_traj_{gif/mp4}
    traj_panel_path: Optional[str] = None  # OutputFileName.traj_panel_{gif/mp4}
    fk_steering_traj_path: Optional[str] = None  # OutputFileName.fk_steering_traj_png
    fk_steering_potential_logits_path: Optional[str] = (
        None  # OutputFileName.fk_steering_potential_logits_{gif/mp4}
    )


def _get_anim_writer() -> Tuple[str, matplotlib.animation.AbstractMovieWriter]:
    if animation.writers.is_available("ffmpeg"):
        return "mp4", animation.FFMpegWriter(
            fps=10,
            codec="libx264",
            extra_args=[
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ],
        )
    elif animation.writers.is_available("imagemagick"):
        return "gif", animation.ImageMagickWriter(fps=10)
    else:
        return "gif", animation.PillowWriter(fps=10)


def _subsample_timesteps(
    num_timesteps: int, up_to: int = 50, take_last: int = 1
) -> List[int]:
    """Subsample timesteps to limit animation size/time."""
    timesteps = np.arange(0, num_timesteps, math.ceil(num_timesteps / up_to))

    # take last N timesteps
    if take_last > 1:
        timesteps = np.concatenate(
            [timesteps, np.arange(num_timesteps - take_last, num_timesteps)]
        )

    # regardless of take_last, ensure we include the final timestep
    if timesteps[-1] != num_timesteps - 1:
        timesteps = np.append(timesteps, num_timesteps - 1)

    # Ensure unique and sorted
    timesteps = np.unique(timesteps)
    timesteps = np.sort(timesteps)

    return list(timesteps.astype(int))


def _get_letters_res_colors() -> Tuple[List[str], npt.NDArray]:
    """Get residue names in order (letters) and their colors."""
    # residue names
    letters = list(restypes_with_x)
    letters[20] = "-"  # UNK

    # residue colors

    # helper – make a lighter shade by linear-mixing with white
    def tint(hex_code, f: float):
        r, g, b = to_rgb(hex_code)
        return (1 - f) * np.array([r, g, b]) + f * np.ones(3)

    # base hues for each class
    NEG = "#3778bf"  # blue
    POS = "#d62728"  # red
    POL = "#2ca02c"  # green
    NON = "#c040c0"  # purple

    aa_color = {
        # negative (blue shades)
        "D": tint(NEG, 0.2),  # Asp
        "E": tint(NEG, 0.4),  # Glu
        # positive (red shades)
        "K": tint(POS, 0.2),  # Lys
        "R": tint(POS, 0.4),  # Arg
        "H": tint(POS, 0.8),  # His (positive-ish)
        # polar uncharged (green shades)
        "N": tint(POL, 0.1),  # Asn
        "Q": tint(POL, 0.4),  # Gln
        "S": tint(POL, 0.5),  # Ser
        "T": tint(POL, 0.6),  # Thr
        "C": tint(POL, 0.3),  # Cys
        "Y": tint(POL, 0.7),  # Tyr
        "W": tint(POL, 0.2),  # Trp
        # non-polar / hydrophobic (grey shades)
        "A": tint(NON, 0.8),  # Ala is like a default -> light
        "V": tint(NON, 0.2),  # Val
        "L": tint(NON, 0.1),  # Leu
        "I": tint(NON, 0.3),  # Ile
        "M": tint(NON, 0.4),  # Met
        "F": tint(NON, 0.5),  # Phe
        "P": tint(NON, 0.6),  # Pro
        "G": tint(NON, 0.9),  # Gly is like a default -> light
        # unknown
        "-": (0.9, 0.9, 0.9),  # light grey
    }
    # build the (21, 3) RGB array in the canonical order
    res_colors = np.array([aa_color[ltr] for ltr in letters])

    return letters, res_colors


def _softmax(x: npt.NDArray, axis: int = -1) -> npt.NDArray:
    """
    Numerically stable softmax for numpy arrays.

    Args:
        x: input array
        axis: axis over which to apply softmax (typically the class dimension)
    Returns:
        probabilities with the same shape as x, summing to 1 along `axis`.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def _rgba_from_logits(logits: npt.NDArray, softmax_logits: bool = True) -> npt.NDArray:
    """
    logits: (N_res, S) where S = 20 or 21
    returns a (S, N_res, 4) RGBA image with row-wise fixed hue + value-based alpha

    If `softmax_logits` is True, applies softmax over the last dimension before
    mapping to alpha; otherwise uses the raw values clipped to [0, 1].
    """
    letters, res_colors = _get_letters_res_colors()

    assert (
        logits.shape[1] == 21 or logits.shape[1] == 20
    ), "Logits must be of shape (N_res, 20) or (N_res, 21)"
    # base RGB for each aa (21, 1, 3) → broadcast across columns
    rgb = np.repeat(res_colors[:, None, :], logits.T.shape[1], axis=1)
    # convert to alpha in [0,1]
    if softmax_logits:
        probs = _softmax(logits, axis=-1)  # (N, S)
        alpha = probs.T[..., None]
    else:
        alpha = np.clip(logits.T, 0.0, 1.0)[..., None]  # (S, L, 1)
    return np.concatenate([rgb, alpha], axis=-1)  # (S, L, 4)


def _init_logits_heatmap(
    ax: plt.Axes,
    logits: npt.NDArray,  # (N, S) where S = 20 or 21
    title: Optional[str] = "",
    softmax_logits: bool = True,
) -> matplotlib.image.AxesImage:
    rgba = _rgba_from_logits(logits, softmax_logits=softmax_logits)  # (S, L, 4)
    im = ax.imshow(rgba, origin="lower", interpolation="none")  # 1-time image

    if title is not None:
        suffix = " (softmax)" if softmax_logits else ""
        ax.set_title(f"{title}{suffix}")

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
    motif_mask: Optional[npt.NDArray] = None,  # (N,)
) -> List[Rectangle]:
    """
    Draws borders around each residue in the logits heatmap.
    x stays fixed, y will be updated each frame.

    Also draws a line under the logits to indicate diffused / motif residues.
    """
    num_timesteps, num_res, num_tokens = logits_traj.shape

    rects = []
    for i in range(num_res):
        edge = "black"
        r = plt.Rectangle(
            (i - 0.5, aa_traj[0, i] - 0.5), 1, 1, fill=False, edgecolor=edge, lw=0.5
        )
        ax.add_patch(r)
        rects.append(r)

        if motif_mask is not None:
            if motif_mask[i]:
                # draw a black line underneath for fixed positions
                ax.add_patch(
                    Rectangle(
                        xy=(i - 0.5, -1),  # under logits
                        width=1.0,
                        height=0.3,
                        facecolor=(0.0, 0.0, 0.0, 0.8),
                        lw=0,
                    )
                )

    if motif_mask is not None:
        ax.set_ylim(-1, num_tokens)

    return rects


def _save_logits_traj(
    logits_traj: npt.NDArray,  # (T, N, S) where S = 20 or 21
    aa_traj: npt.NDArray,  # (T, N)
    motif_mask: Optional[npt.NDArray],  # (N,)
    file_path: str,
    writer: matplotlib.animation.AbstractMovieWriter,
    animation_interval_ms: float = 100,
    softmax_logits: bool = True,
) -> str:
    """
    Writes logits trajectory as a heatmap animation, one frame per step in the trajectory, up to 50 frames.
    Returns the path to the animation.
    """
    # Create figure and axis.
    num_timesteps, num_res = logits_traj.shape[:2]
    fig, ax = plt.subplots(figsize=(int(num_res * 0.15), 5), dpi=100)

    # create image to update, draw frame 1
    im = _init_logits_heatmap(
        ax,
        logits=logits_traj[0],
        title="Logits trajectory",
        softmax_logits=softmax_logits,
    )
    rects = _init_logits_borders(
        ax, logits_traj=logits_traj, aa_traj=aa_traj, motif_mask=motif_mask
    )

    def update(frame):
        im.set_data(
            _rgba_from_logits(logits_traj[frame], softmax_logits=softmax_logits)
        )

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

    anim.save(file_path, writer=writer)
    plt.close(fig)

    return file_path


def save_logits_traj(
    logits_traj: npt.NDArray,  # (T, N, S) where S = 20 or 21
    aa_traj: npt.NDArray,  # (T, N)
    motif_mask: Optional[npt.NDArray],  # (N,)
    output_dir: str,
    animation_interval_ms: float = 100,
    softmax_logits: bool = True,
) -> str:
    """Write sampled logits trajectory"""
    ext, writer = _get_anim_writer()

    if ext == "gif":
        anim_path = os.path.join(output_dir, OutputFileName.logits_traj_gif)
    elif ext == "mp4":
        anim_path = os.path.join(output_dir, OutputFileName.logits_traj_mp4)
    else:
        raise ValueError(f"Unknown animation format: {ext}")

    os.makedirs(output_dir, exist_ok=True)

    return _save_logits_traj(
        logits_traj=logits_traj,
        aa_traj=aa_traj,
        motif_mask=motif_mask,
        file_path=anim_path,
        writer=writer,
        animation_interval_ms=animation_interval_ms,
        softmax_logits=softmax_logits,
    )


def save_potential_logits_traj(
    fk_traj: FKSteeringTrajectory,
    sample_aa_traj: Optional[npt.NDArray],  # (T, N)
    motif_mask: Optional[npt.NDArray],  # (N,)
    output_dir: str,
    animation_interval_ms: float = 100,
    softmax_logits: bool = True,
) -> Optional[str]:
    """Write FK steering potential logits trajectory if logits guidance present.

    Returns path to saved animation, or None if logits guidance not present.
    """
    # Check if any metric has logits guidance
    if (
        fk_traj is None
        or fk_traj.metrics is None
        or len(fk_traj.metrics) == 0
        or sample_aa_traj is None
    ):
        return None

    first_guidance = fk_traj.metrics[0].guidance
    if first_guidance is None or first_guidance.logits is None:
        return None

    # Determine per-step best particle lineage for this sample (assumes batch is already sliced to 1)
    k_indices = fk_traj.best_particle_lineage(batch_idx=0)

    # Build logits trajectory across FK resampling steps for that lineage
    fk_logits_guidance: List[npt.NDArray] = []
    for m, k_idx in zip(fk_traj.metrics, k_indices):
        if m.guidance is None or m.guidance.logits is None:
            continue
        logits_t = m.guidance.logits  # (K, N, S) or (N, S)
        if logits_t.dim() == 3:
            logits_t = logits_t[int(k_idx)]
        logits_np = logits_t.detach().cpu().to(torch.float32).numpy()  # (N, S)
        fk_logits_guidance.append(logits_np)

    if len(fk_logits_guidance) == 0:
        return None

    logits_traj = np.stack(fk_logits_guidance, axis=0)  # (T, N, S)

    # Build AA trajectory across FK resampling steps
    aa_traj = np.stack(
        [sample_aa_traj[m.step] for m in fk_traj.metrics], axis=0
    )  # (T, N)

    # Determine output path and writer
    ext, writer = _get_anim_writer()
    if ext == "gif":
        anim_path = os.path.join(
            output_dir, OutputFileName.fk_steering_potential_logits_gif
        )
    elif ext == "mp4":
        anim_path = os.path.join(
            output_dir, OutputFileName.fk_steering_potential_logits_mp4
        )
    else:
        raise ValueError(f"Unknown animation format: {ext}")

    os.makedirs(output_dir, exist_ok=True)

    return _save_logits_traj(
        logits_traj=logits_traj,
        aa_traj=aa_traj,
        motif_mask=motif_mask,
        file_path=anim_path,
        writer=writer,
        animation_interval_ms=animation_interval_ms,
        softmax_logits=softmax_logits,
    )


def _init_seq_artists(
    ax: plt.Axes,
    num_res: int,
    motif_mask: Optional[npt.NDArray],  # (N,)
) -> Tuple[Sequence[Rectangle], Sequence[Text]]:
    """
    Create one rectangle + text per residue; return them for later updates.
    """
    ax.set_axis_off()
    rects: List[Rectangle] = []
    texts: List[matplotlib.text.Text] = []

    for i in range(num_res):
        rect = Rectangle(
            (i, 0),
            1.0,
            1.0,
            facecolor=(1.0, 1.0, 1.0),  # white to start
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

        # draw a black line underneath for fixed positions
        if motif_mask is not None:
            if motif_mask[i]:
                ax.add_patch(
                    Rectangle(
                        xy=(i, -0.3),  # at bottom
                        width=1.0,
                        height=0.2,
                        facecolor=(0.0, 0.0, 0.0, 0.8),
                        lw=0,
                    )
                )

    ax.set_xlim(0, num_res)
    if motif_mask is not None:
        ax.set_ylim(-0.3, 1)
    else:
        ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    return rects, texts


def _update_seq_artists(
    seq: npt.NDArray,  # (N,)
    rects: Sequence[Rectangle],
    texts: Sequence[matplotlib.text.Text],
) -> Tuple[Rectangle, ...]:
    """Mutate rect/text colors + labels in place and return artists."""
    letters, res_colors = _get_letters_res_colors()
    for i, aa_idx in enumerate(seq):
        rects[i].set_facecolor(res_colors[int(aa_idx)])
        texts[i].set_text(letters[int(aa_idx)])
    return tuple(rects) + tuple(texts)


def plot_seq(ax: Optional[Axes], seq: npt.NDArray, motif_mask: Optional[npt.NDArray]):
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

    rects, texts = _init_seq_artists(ax, num_res=len(seq), motif_mask=motif_mask)
    _update_seq_artists(seq, rects=rects, texts=texts)

    return rects, texts


@dataclass
class CameraLimits:
    """Helper for 3D scenes to fix camera limits."""

    center: npt.NDArray
    width: float

    @classmethod
    def from_bb_traj(
        cls, bb_traj: npt.NDArray, max_allowed_width=100.0
    ) -> "CameraLimits":
        # determine camera limits by inspecting c-alphas
        ca_all = bb_traj[:, :, 1, :].reshape(-1, 3)  # CA = atom index 1
        center = ca_all.mean(0)  # 3D
        width = (ca_all - center).ptp(0).max() / 2

        if width > max_allowed_width:
            logger.warning(
                f"Camera width is very large ({width:.1f}Å). "
                f"Is the structure reasonable? Setting width to {max_allowed_width}Å."
            )
            width = max_allowed_width

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
        cmap[~diffuse_mask.astype(bool)] = _GREY
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

    # Matplotlib 3D scatter updates require setting _offsets3d
    scats[0]._offsets3d = (N[:, 0], N[:, 1], N[:, 2])
    scats[1]._offsets3d = (CA[:, 0], CA[:, 1], CA[:, 2])
    scats[2]._offsets3d = (CB[:, 0], CB[:, 1], CB[:, 2])

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
    sample_structure_traj: npt.NDArray,  # [noisy_T, N, 37, 3]
    sample_aa_traj: npt.NDArray,  # [noisy_T, N]
    model_structure_traj: npt.NDArray,  # [clean_T, N, 37, 3]  (clean_T = noisy_T-1)
    model_aa_traj: npt.NDArray,  # [clean_T, N]
    model_logits_traj: npt.NDArray,  # [clean_T, N, S] (S = 20, 21)
    diffuse_mask: npt.NDArray,  # [N]
    motif_mask: Optional[npt.NDArray],  # [N]
    output_dir: str,
    animation_max_frames: int,
    animation_take_last_frames: int,
    softmax_logits: bool = True,
    fk_steering_traj: Optional[FKSteeringTrajectory] = None,
):
    """
    Animates the protein and model trajectories side by side.
    3 x 2 grid with logits, AA sequences on top and structures on bottom, protein on left, model pred on right.
    """
    os.makedirs(output_dir, exist_ok=True)

    # `model_traj` is one step shorter than `sample_traj`
    num_timesteps, num_res = sample_structure_traj.shape[:2]
    assert model_structure_traj.shape[0] == num_timesteps - 1

    # Determine backbone 3D camera limits
    camera_limits = CameraLimits.from_bb_traj(sample_structure_traj)

    # Fig determined ahead of time
    dpi = 100
    square_px = 20  # side-length of an AA square
    fig_w_in = max(6, min(30, num_res * square_px / dpi))
    seq_h_in = square_px / dpi
    struct_h_in = fig_w_in * 0.45
    logits_h_in = seq_h_in * 20
    fig_h_in = logits_h_in + seq_h_in + struct_h_in

    # Force even pixel dimensions to satisfy yuv420p requirements
    fig_w_px = int(round(fig_w_in * dpi))
    fig_h_px = int(round(fig_h_in * dpi))
    if fig_w_px % 2 != 0:
        fig_w_px += 1
    if fig_h_px % 2 != 0:
        fig_h_px += 1
    fig_w_in = fig_w_px / dpi
    fig_h_in = fig_h_px / dpi
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi, constrained_layout=False)

    # grid: 1 row (logits) + 1 row (seq) + 1 row (structures) / 2 cols (protein, model)
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[logits_h_in * dpi, seq_h_in * dpi, struct_h_in * dpi],
        wspace=0.02,
        hspace=0.01,
    )
    ax_logits_guidance = fig.add_subplot(gs[0, 0])
    ax_logits_model = fig.add_subplot(gs[0, 1])
    ax_sequence_prot = fig.add_subplot(gs[1, 0])
    ax_sequence_model = fig.add_subplot(gs[1, 1])
    ax_structure_prot = fig.add_subplot(gs[2, 0], projection="3d")
    ax_structure_model = fig.add_subplot(gs[2, 1], projection="3d")

    # tighten outside padding but reserve bottom space for figure-level title
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.06)
    title_text = fig.text(0.5, 0.02, "", ha="center", va="bottom", fontsize=20)

    # Prepare FK guidance logits (if provided) for the lineage of the best particle across steps
    guidance_map = {}
    if fk_steering_traj is not None and len(fk_steering_traj.metrics) > 0:
        k_indices = fk_steering_traj.best_particle_lineage(batch_idx=0)
        for m, k_idx in zip(fk_steering_traj.metrics, k_indices):
            if m.guidance is None or m.guidance.logits is None:
                continue
            logits_t = m.guidance.logits  # (K, N, S) or (N, S)
            if logits_t.dim() == 3:
                logits_t = logits_t[int(k_idx)]
            logits_np = logits_t.detach().cpu().to(torch.float32).numpy()  # (N, S)
            guidance_map[int(m.step)] = logits_np
    has_guidance = len(guidance_map) > 0
    guidance_steps: List[int] = sorted(guidance_map.keys()) if has_guidance else []

    # initialise artists
    guidance_im = None
    guidance_rects: List[Rectangle] = []
    if has_guidance:
        # pick an initial logits just to create artists; hide if not at timestep 0
        initial_step = 0 if 0 in guidance_map else sorted(guidance_map.keys())[0]
        initial_logits = guidance_map[initial_step]
        guidance_im = _init_logits_heatmap(
            ax_logits_guidance,
            initial_logits,
            title="Guidance logits",
            softmax_logits=softmax_logits,
        )
        guidance_rects = _init_logits_borders(
            ax_logits_guidance,
            logits_traj=initial_logits[None, ...],
            aa_traj=sample_aa_traj,
            motif_mask=motif_mask,
        )
        if initial_step != 0:
            guidance_im.set_visible(False)
            for r in guidance_rects:
                r.set_visible(False)
            ax_logits_guidance.set_axis_off()
    else:
        # nothing to show on guidance panel
        ax_logits_guidance.set_axis_off()

    logits_im = _init_logits_heatmap(
        ax_logits_model,
        model_logits_traj[0],
        title="Model logits",
        softmax_logits=softmax_logits,
    )
    logits_rects = _init_logits_borders(
        ax_logits_model,
        logits_traj=model_logits_traj,
        aa_traj=sample_aa_traj,
        motif_mask=motif_mask,
    )

    seq_rects_l, seq_texts_l = _init_seq_artists(
        ax_sequence_prot, num_res, motif_mask=motif_mask
    )
    seq_rects_r, seq_texts_r = _init_seq_artists(
        ax_sequence_model, num_res, motif_mask=motif_mask
    )

    scat_l = _init_structure_artists(
        ax_structure_prot,
        positions=sample_structure_traj[0],
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
        + ((guidance_im,) if guidance_im is not None else tuple())
        + tuple(guidance_rects)
        + (logits_im,)
        + tuple(logits_rects)
        + (title_text,)
    )

    def update(timestep):
        # timestep in protein trajectory == timestep - 1 in model trajectory
        t = float(timestep) / (num_timesteps - 1)
        title_text.set_text(f"t = {t:.2f}")

        _update_seq_artists(
            sample_aa_traj[timestep],
            rects=seq_rects_l,
            texts=seq_texts_l,
        )
        _update_structure_artists(sample_structure_traj[timestep], scats=scat_l)

        if timestep > 0:
            _update_seq_artists(
                model_aa_traj[timestep - 1],
                rects=seq_rects_r,
                texts=seq_texts_r,
            )
            _update_structure_artists(model_structure_traj[timestep - 1], scats=scat_r)
            logits_im.set_data(
                _rgba_from_logits(
                    model_logits_traj[timestep - 1], softmax_logits=softmax_logits
                )
            )
            for i, r in enumerate(logits_rects):
                r.set_y(model_aa_traj[timestep - 1, i] - 0.5)
        else:
            pass  # blank on t=0 timestep

        # Update FK guidance logits: show latest available guidance <= current timestep, else hide
        if has_guidance and guidance_im is not None:
            # find latest guidance step not exceeding current timestep
            display_step = None
            for s in guidance_steps:
                if s <= timestep:
                    display_step = s
                else:
                    break
            if display_step is not None:
                ax_logits_guidance.set_axis_on()
                guidance_im.set_data(
                    _rgba_from_logits(
                        guidance_map[display_step], softmax_logits=softmax_logits
                    )
                )
                guidance_im.set_visible(True)
                for i, r in enumerate(guidance_rects):
                    r.set_y(sample_aa_traj[timestep, i] - 0.5)
                    r.set_visible(True)
            else:
                guidance_im.set_visible(False)
                for r in guidance_rects:
                    r.set_visible(False)
                ax_logits_guidance.set_axis_off()

        return dynamic_artists

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=_subsample_timesteps(
            num_timesteps,
            up_to=animation_max_frames,
            take_last=animation_take_last_frames,
        ),
        interval=100,
        repeat_delay=1000,  # linger on final frame
        blit=False,
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


def write_fk_steering_energy_traj(
    fk_steering_traj: FKSteeringTrajectory,
    file_path: str,
):
    """
    Write FK steering particle energy trajectory as a multi-panel plot showing particle metrics over time.

    Args:
        fk_steering_traj: FKSteeringTrajectory containing metrics for each step
        file_path: Path to save the plot (PNG format)
    """
    assert (
        fk_steering_traj.num_steps >= 1
    ), "FK steering trajectory must have at least one step"

    steps = [metric.step for metric in fk_steering_traj.metrics]
    num_particles = fk_steering_traj.num_particles

    # Prepare data arrays
    energy = np.array(
        [metric.energy for metric in fk_steering_traj.metrics]
    )  # (num_steps, num_particles)
    log_G = np.array([metric.log_G for metric in fk_steering_traj.metrics])
    log_G_delta = np.array([metric.log_G_delta for metric in fk_steering_traj.metrics])
    weights = np.array([metric.weights for metric in fk_steering_traj.metrics])
    ess = np.array(
        [metric.effective_sample_size for metric in fk_steering_traj.metrics]
    )  # (num_steps,)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("FK Steering Trajectory", fontsize=14)

    # Color palette for particles
    colors = plt.cm.tab10(np.linspace(0, 1, num_particles))

    # Plot 1: Energy
    ax = axes[0, 0]
    for i in range(num_particles):
        ax.plot(
            steps,
            energy[:, i],
            color=colors[i],
            alpha=0.7,
            linewidth=1.5,
            label=f"Particle {i}" if i < 5 else None,
        )  # Limit legend entries
    ax.set_xlabel("Step")
    ax.set_ylabel("Energy")
    ax.set_title("Energy per Particle")
    ax.grid(True, alpha=0.3)
    if num_particles <= 5:
        ax.legend()

    # Plot 2: Log G
    ax = axes[0, 1]
    for i in range(num_particles):
        ax.plot(steps, log_G[:, i], color=colors[i], alpha=0.7, linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Log G")
    ax.set_title("Log G per Particle")
    ax.grid(True, alpha=0.3)

    # Plot 3: Log G Delta
    ax = axes[1, 0]
    for i in range(num_particles):
        ax.plot(steps, log_G_delta[:, i], color=colors[i], alpha=0.7, linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Log G Delta")
    ax.set_title("Log G Delta per Particle")
    ax.grid(True, alpha=0.3)

    # Plot 4: Effective Sample Size and Weights
    ax = axes[1, 1]

    # Plot effective sample size as a bold line
    ax.plot(
        steps, ess, color="red", linewidth=3, label="Effective Sample Size", alpha=0.8
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Effective Sample Size", color="red")
    ax.tick_params(axis="y", labelcolor="red")
    ax.grid(True, alpha=0.3)

    # Create second y-axis for weights
    ax2 = ax.twinx()
    for i in range(num_particles):
        ax2.plot(
            steps,
            weights[:, i],
            color=colors[i],
            alpha=0.5,
            linewidth=1,
            linestyle="--",
            label=f"Weight {i}" if i < 3 else None,
        )
    ax2.set_ylabel("Weights", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2[:3], labels1 + labels2[:3], loc="upper right")

    ax.set_title("Effective Sample Size & Weights")

    plt.tight_layout()
    plt.savefig(file_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_trajectory(
    sample_name: Union[int, str],
    sample_atom37: npt.NDArray,  # (N, 37, 3)
    sample_structure_traj: npt.NDArray,  # (noisy_T, N, 37, 3)
    model_structure_traj: npt.NDArray,  # (clean_T, N, 37, 3)
    diffuse_mask: npt.NDArray,  # (N,)
    motif_mask: Optional[npt.NDArray],  # (N,)
    chain_idx: npt.NDArray,  # (N,)
    res_idx: npt.NDArray,  # (N,)
    output_dir: str,
    sample_aa_traj: Optional[npt.NDArray] = None,  # (noisy_T, N)
    model_aa_traj: Optional[npt.NDArray] = None,  # (clean_T, N)
    model_logits_traj: Optional[npt.NDArray] = None,  # (clean_T, N, S)
    fk_steering_traj: Optional[FKSteeringTrajectory] = None,
    write_trajectories: bool = True,
    write_animations: bool = True,
    animation_max_frames: int = 50,
    animation_take_last_frames: int = 1,
    softmax_logits: bool = True,
) -> SavedTrajectory:
    """
    Writes final sample and reverse diffusion trajectory.

    Args:
        sample_atom37: [N, 37, 3] atom37 final sample.
        sample_structure_traj: [noisy_T, N, 37, 3] atom37 sampled diffusion states.
            T is number of time steps. First time step is t=eps, final step is t=1.
            N is number of residues.
        model_structure_traj: [clean_T, N, 37, 3] atom37 predictions of clean data at each time step.
        diffuse_mask: [N] which residues are diffused.
        motif_mask: [N] (inpainting only) which residues are motifs.
        output_dir: where to save samples.
        sample_aa_traj: [noisy_T, N] amino acids (0 - S inclusive where S = 20 or 21).
        model_aa_traj: [clean_T, N] amino acids (0 - S inclusive where S = 20 or 21).
        model_logits_traj: [clean_T, N, S] logits for each amino acid, from model
        fk_steering_traj: Optional[FKSteeringTrajectory] FK steering trajectory for this batch member.
        write_trajectories: bool Whether to also write the PDB trajectories
        write_animations: bool Whether to create animation of the trajectory (slow, ~10-15s for 50 frames)
        animation_max_frames: int Max number of frames of all timesteps to include in animation.
        animation_take_last_frames: int Number of final frames to include in animation (in addition to animation_max_frames)
    Returns:
        SavedTrajectory with paths to saved samples:
            sample_pdb_path: PDB file of final structure
            sample_pdb_backbone_path: PDB of final backbone
        And if `write_trajectories == True`:
            sample_traj_path: PDB file os all intermediate states
            model_pred_traj_path: PDB file of t=1 model predictions at each state
            aa_traj_fasta_path: Fasta file of amino acid sequence at each state
        And if `write_animations == True`:
            traj_panel_path: animation of the trajectory, model and sample side by side.
    """

    start_time = time.time()
    logger = rank_zero_logger(__name__)

    def log_time(msg):
        elapsed_time = time.time() - start_time
        logger.debug(f"sample {sample_name} {msg}: {elapsed_time:.2f} seconds")

    # ensure directory exists
    os.makedirs(output_dir, exist_ok=True)

    sample_pdb_path = os.path.join(output_dir, OutputFileName.sample_pdb)
    sample_pdb_backbone_path = os.path.join(
        output_dir, OutputFileName.sample_pdb_backbone
    )

    noisy_traj_length, num_res, _, _ = sample_structure_traj.shape
    model_traj_length = model_structure_traj.shape[0]
    assert sample_atom37.shape == (num_res, 37, 3)
    assert sample_structure_traj.shape == (noisy_traj_length, num_res, 37, 3)
    assert model_structure_traj.shape == (model_traj_length, num_res, 37, 3)

    if sample_aa_traj is not None:
        assert sample_aa_traj.shape == (noisy_traj_length, num_res)
        assert model_aa_traj is not None
        assert model_aa_traj.shape == (model_traj_length, num_res)

    # Use b-factors to specify which residues are diffused.
    b_factor_mask = motif_mask if motif_mask is not None else diffuse_mask
    b_factor_mask = b_factor_mask.astype(bool)
    b_factors = np.tile((b_factor_mask * 100)[:, None], (1, 37))

    sample_pdb_path = write_prot_to_pdb(
        sample_atom37,
        file_path=sample_pdb_path,
        b_factors=b_factors,
        no_indexing=True,
        aatype=sample_aa_traj[-1] if sample_aa_traj is not None else None,
        chain_idx=chain_idx,
        res_idx=res_idx,
        backbone_only=False,
        # ensure UNK converted to ALA, e.g. for ProteinMPNN
        convert_unk_to_alanine=True,
    )
    sample_pdb_backbone_path = write_prot_to_pdb(
        sample_atom37,
        file_path=sample_pdb_backbone_path,
        b_factors=b_factors,
        no_indexing=True,
        aatype=sample_aa_traj[-1] if sample_aa_traj is not None else None,
        chain_idx=chain_idx,
        res_idx=res_idx,
        backbone_only=True,
        convert_unk_to_alanine=True,
    )

    if not write_trajectories:
        return SavedTrajectory(
            sample_pdb_path=sample_pdb_path,
            sample_pdb_backbone_path=sample_pdb_backbone_path,
        )

    sample_traj_path = os.path.join(output_dir, OutputFileName.sample_traj_pdb)
    model_pred_traj_path = os.path.join(output_dir, OutputFileName.model_pred_traj_pdb)
    sample_traj_path = write_prot_to_pdb(
        sample_structure_traj,
        file_path=sample_traj_path,
        b_factors=b_factors,
        no_indexing=True,
        aatype=sample_aa_traj,
        chain_idx=chain_idx,
        res_idx=res_idx,
    )
    model_pred_traj_path = write_prot_to_pdb(
        model_structure_traj,
        file_path=model_pred_traj_path,
        b_factors=b_factors,
        no_indexing=True,
        aatype=model_aa_traj,
        chain_idx=chain_idx,
        res_idx=res_idx,
    )

    log_time("structure trajectory PDBs")

    # These file paths gated by files being provided
    aa_traj_fasta_path = None
    model_logits_traj_path = None
    traj_panel_path = None
    fk_steering_energy_traj_path = None
    fk_steering_potential_logits_path = None

    # Write amino acids trajectory, if provided.
    if sample_aa_traj is not None:
        aa_traj_fasta_path = os.path.join(output_dir, OutputFileName.aa_traj_fa)
        num_steps = sample_aa_traj.shape[0]
        with open(aa_traj_fasta_path, "w") as f:
            for i in range(num_steps):
                f.write(f">step{i}\n")
                f.write(
                    "".join([restypes_with_x[aa] for aa in sample_aa_traj[i]]) + "\n"
                )

    if fk_steering_traj is not None:
        # Plot FK steering energies, ESS, etc.
        fk_steering_energy_traj_path = os.path.join(
            output_dir, OutputFileName.fk_steering_energy_traj_png
        )
        write_fk_steering_energy_traj(
            fk_steering_traj, file_path=fk_steering_energy_traj_path
        )

        # Plot logits guidance
        if write_animations:
            fk_steering_potential_logits_path = save_potential_logits_traj(
                fk_traj=fk_steering_traj,
                sample_aa_traj=sample_aa_traj,
                motif_mask=motif_mask,
                output_dir=output_dir,
                softmax_logits=softmax_logits,
            )

    # Trajectory panel animation
    if (
        write_animations
        and sample_aa_traj is not None
        and model_aa_traj is not None
        and model_logits_traj is not None
    ):
        traj_panel_path = animate_trajectories(
            sample_structure_traj=sample_structure_traj,
            sample_aa_traj=sample_aa_traj,
            model_structure_traj=model_structure_traj,
            model_aa_traj=model_aa_traj,
            model_logits_traj=model_logits_traj,
            diffuse_mask=diffuse_mask,
            motif_mask=motif_mask,
            output_dir=output_dir,
            animation_max_frames=animation_max_frames,
            animation_take_last_frames=animation_take_last_frames,
            softmax_logits=softmax_logits,
            fk_steering_traj=fk_steering_traj,
        )
        log_time("trajectory animation")

    return SavedTrajectory(
        sample_pdb_path=sample_pdb_path,
        sample_pdb_backbone_path=sample_pdb_backbone_path,
        sample_traj_path=sample_traj_path,
        model_pred_traj_path=model_pred_traj_path,
        aa_traj_fasta_path=aa_traj_fasta_path,
        logits_traj_path=model_logits_traj_path,
        traj_panel_path=traj_panel_path,
        fk_steering_traj_path=fk_steering_energy_traj_path,
        fk_steering_potential_logits_path=fk_steering_potential_logits_path,
    )
