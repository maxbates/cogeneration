import functools
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Tuple

import numpy as np
import torch
from matplotlib import animation as animation
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import proj3d
from tqdm import tqdm

from cogeneration.data import all_atom
from cogeneration.data.const import MASK_TOKEN_INDEX
from cogeneration.data.residue_constants import restypes_with_x
from cogeneration.util.log import rank_zero_logger
from varco.config import (
    VarcoHazardConfig,
    VarcoInterpolantAATypesCouplerConfig,
    VarcoInterpolantConfig,
    VarcoInterpolantTransCouplerConfig,
)
from varco.data import (
    DataBatch,
    DataCorrupted,
    ModelPrediction,
    SampleTrajectory,
    Trajectory,
)
from varco.interpolant import TreeInterpolant

logger = rank_zero_logger("viz")


@dataclass
class TrajectoryFrame:
    """Visualization-ready frame data, converted to numpy.

    This is a single-sample (no batch dimension) representation suitable for
    plotting. All tensors are numpy arrays on CPU.
    """

    trans: np.ndarray  # (P, 3) CA positions
    rotmats: np.ndarray  # (P, 3, 3) rotation matrices
    aatypes: np.ndarray  # (P,) amino acid indices 0-20
    motif_mask: np.ndarray  # (P,) bool
    valid_mask: np.ndarray  # (P,) bool
    t: float  # timestep value
    remaining_insertions: Optional[np.ndarray] = None  # (P,) or None
    atom37: Optional[np.ndarray] = None  # (P, 37, 3) pre-computed if needed

    def get_backbone_positions(self, only_alpha_carbons: bool) -> np.ndarray:
        """Return backbone positions for valid residues.

        Args:
            only_alpha_carbons: If True, return (n_valid, 3) CA positions.
                If False, return (n_valid, 3, 3) N/CA/C positions.
                Requires atom37 to have been computed at construction time.
        """
        valid = self.valid_mask
        if only_alpha_carbons:
            return self.trans[valid]  # (n_valid, 3)
        else:
            if self.atom37 is None:
                raise ValueError(
                    "atom37 not available. Set include_atom37=True when creating frame."
                )
            # N=0, CA=1, C=2 in atom37 ordering
            return self.atom37[valid][:, [0, 1, 2], :]  # (n_valid, 3, 3)

    @classmethod
    def from_data_corrupted(
        cls,
        data: DataCorrupted,
        batch_idx: int,
        include_atom37: bool = False,
    ) -> "TrajectoryFrame":
        """Extract single batch item from DataCorrupted, convert to numpy.

        Args:
            data: The DataCorrupted containing batched samples.
            batch_idx: Which batch item to extract.
            include_atom37: If True, compute and store atom37 representation.
                This is faster than computing it later since the data is still
                on the original device (potentially GPU).
        """
        atom37 = None
        if include_atom37:
            # Compute atom37 on device before moving to CPU
            atom37_batch = data.to_atom37()  # (B, P, 37, 3)
            atom37 = atom37_batch[batch_idx].cpu().numpy()

        return cls(
            trans=data.trans_t[batch_idx].cpu().numpy(),
            rotmats=data.rotmats_t[batch_idx].cpu().numpy(),
            aatypes=data.aatypes_t[batch_idx].cpu().numpy(),
            motif_mask=data.motif_mask[batch_idx].cpu().numpy(),
            valid_mask=data.valid_mask[batch_idx].cpu().numpy(),
            t=data.t[batch_idx].item(),
            remaining_insertions=(
                data.remaining_insertions[batch_idx].cpu().numpy()
                if data.remaining_insertions is not None
                else None
            ),
            atom37=atom37,
        )

    @classmethod
    def from_model_prediction(
        cls,
        pred: ModelPrediction,
        sample: DataCorrupted,
        batch_idx: int,
        split_hazard: VarcoHazardConfig,
        include_atom37: bool = False,
    ) -> "TrajectoryFrame":
        """Extract single batch item from ModelPrediction, convert to numpy.

        Uses the sample for motif_mask, valid_mask, and t since predictions
        are made for a given sample state.

        Args:
            pred: The ModelPrediction containing batched predictions.
            sample: The corresponding DataCorrupted sample (for metadata).
            batch_idx: Which batch item to extract.
            split_hazard: Hazard config used to convert time-independent split mass
                to expected remaining insertions at time t.
            include_atom37: If True, compute and store atom37 representation.
        """
        atom37 = None
        if include_atom37:
            # Compute atom37 from prediction on device before moving to CPU
            atom37_batch = all_atom.atom37_from_trans_rot(
                trans=pred.pred_trans_1,
                rots=pred.pred_rotmats_1,
                torsions=None,
                aatype=pred.pred_aatype_logits.argmax(dim=-1),
                res_mask=sample.valid_mask.float(),
                unknown_to_alanine=True,
            )
            atom37 = atom37_batch[batch_idx].cpu().numpy()

        # Model predicts time-independent insertion mass M, but visualization expects
        # remaining insertions R_t at the current time t: R_t = M * S(t).
        t_val = float(sample.t[batch_idx].item())
        S_t = float(TreeInterpolant.compute_hazard_survival(t_val, split_hazard))
        remaining_insertions = (
            (pred.pred_split_mass[batch_idx].clamp_min(0.0) * S_t).cpu().numpy()
        )

        return cls(
            trans=pred.pred_trans_1[batch_idx].cpu().numpy(),
            rotmats=pred.pred_rotmats_1[batch_idx].cpu().numpy(),
            aatypes=pred.pred_aatype_logits[batch_idx].argmax(dim=-1).cpu().numpy(),
            motif_mask=sample.motif_mask[batch_idx].cpu().numpy(),
            valid_mask=sample.valid_mask[batch_idx].cpu().numpy(),
            t=t_val,
            remaining_insertions=remaining_insertions,
            atom37=atom37,
        )


@dataclass
class PlotPanel:
    """Manages matplotlib artists for one sequence+structure visualization panel.

    A panel consists of a sequence bar (2D axes) and a 3D structure view.
    This class encapsulates the artists needed to render a TrajectoryFrame.
    """

    ax_seq: plt.Axes
    ax_3d: Any  # Axes3D
    seq_artists: (
        tuple  # (rectangles, texts, motif_rects, letters, colors, positions_per_row)
    )
    scatter_artist: Any  # PathCollection3D
    letter_artists: Optional[List]
    title_prefix: str
    max_atoms: int
    only_alpha_carbons: bool
    color_by: str

    @classmethod
    def create(
        cls,
        ax_seq: plt.Axes,
        ax_3d: Any,
        max_seq_len: int,
        max_atoms: int,
        trans_min: np.ndarray,
        trans_max: np.ndarray,
        only_alpha_carbons: bool,
        color_by: str,
        show_residue_letters: bool,
        title_prefix: str = "",
    ) -> "PlotPanel":
        """Create all artists for this panel."""
        seq_artists = cls._create_sequence_artists(ax_seq, max_seq_len)
        scatter_artist = cls._create_3d_scatter_artist(
            ax_3d, max_atoms, trans_min, trans_max, only_alpha_carbons, color_by
        )
        letter_artists = (
            cls._create_3d_residue_letter_artists(ax_3d, max_seq_len)
            if show_residue_letters
            else None
        )
        return cls(
            ax_seq=ax_seq,
            ax_3d=ax_3d,
            seq_artists=seq_artists,
            scatter_artist=scatter_artist,
            letter_artists=letter_artists,
            title_prefix=title_prefix,
            max_atoms=max_atoms,
            only_alpha_carbons=only_alpha_carbons,
            color_by=color_by,
        )

    def update(self, frame: TrajectoryFrame) -> None:
        """Update all artists with new frame data."""
        rectangles, texts, motif_rects, letters, colors, _ = self.seq_artists
        valid = frame.valid_mask

        # Update sequence bar
        self._update_sequence_bar(
            rectangles,
            texts,
            motif_rects,
            letters,
            colors,
            frame.aatypes[valid],
            frame.motif_mask[valid],
        )

        # Update 3D scatter
        backbone_pos = frame.get_backbone_positions(self.only_alpha_carbons)
        remaining_ins = (
            frame.remaining_insertions[valid]
            if frame.remaining_insertions is not None
            else None
        )
        self._update_3d_scatter(
            self.scatter_artist,
            self.ax_3d,
            backbone_pos,
            frame.motif_mask[valid],
            frame.aatypes[valid],
            self.max_atoms,
            frame.t,
            self.only_alpha_carbons,
            remaining_ins,
            self.color_by,
        )

        # Update title with prefix
        n_res = valid.sum()
        title = f"{self.title_prefix}t = {frame.t:.2f} (N={n_res})"
        self.ax_3d.set_title(title)

        # Update residue letters if enabled
        if self.letter_artists is not None:
            ca_pos = frame.trans[valid]
            self._update_3d_residue_letter_artists(
                self.letter_artists,
                self.ax_3d,
                ca_pos,
                frame.aatypes[valid],
            )

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _aa_letters_and_colors() -> Tuple[Tuple[str, ...], np.ndarray]:
        """Returns (letters, colors) for 21 amino acid types + X. Cached."""
        letters = list(restypes_with_x)
        letters[20] = "-"  # UNK

        def tint(rgb, f):  # mix with white
            return tuple((1 - f) * c + f for c in rgb)

        NEG = (0.22, 0.47, 0.75)  # blue
        POS = (0.84, 0.15, 0.16)  # red
        POL = (0.17, 0.63, 0.17)  # green
        NON = (0.75, 0.25, 0.75)  # purple
        aa_map = {
            # negative
            "D": tint(NEG, 0.2),
            "E": tint(NEG, 0.4),
            # positive
            "K": tint(POS, 0.2),
            "R": tint(POS, 0.4),
            "H": tint(POS, 0.8),
            # polar
            "N": tint(POL, 0.1),
            "Q": tint(POL, 0.4),
            "S": tint(POL, 0.5),
            "T": tint(POL, 0.6),
            "C": tint(POL, 0.3),
            "Y": tint(POL, 0.7),
            "W": tint(POL, 0.2),
            # non-polar
            "A": tint(NON, 0.8),
            "V": tint(NON, 0.2),
            "L": tint(NON, 0.1),
            "I": tint(NON, 0.3),
            "M": tint(NON, 0.4),
            "F": tint(NON, 0.5),
            "P": tint(NON, 0.6),
            "G": tint(NON, 0.9),
            "-": (0.9, 0.9, 0.9),
        }
        colors = np.array([aa_map[ltr] for ltr in letters])
        return tuple(letters), colors

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _aa_listed_colormap() -> ListedColormap:
        """A categorical colormap for amino acid indices (0..20)."""
        _, colors = PlotPanel._aa_letters_and_colors()
        return ListedColormap(colors, name="aa")

    @staticmethod
    def _create_sequence_artists(
        ax: plt.Axes, max_len: int, positions_per_row: int = 175
    ):
        """Pre-create all artists needed for sequence visualization."""
        letters, colors = PlotPanel._aa_letters_and_colors()
        num_rows = math.ceil(max_len / positions_per_row)

        box_width = 1.0
        box_height = 1.0
        row_spacing = 0.3
        row_height = box_height + row_spacing

        ax.set_xlim(-0.1, positions_per_row + 0.1)
        total_height = num_rows * row_height
        ax.set_ylim(-total_height - 0.1, 0.5)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        rectangles = []
        texts = []
        motif_rects = []

        for i in range(max_len):
            row = i // positions_per_row
            col = i % positions_per_row
            y_base = -row * row_height
            x_pos = col * box_width

            rect = plt.Rectangle(
                (x_pos, y_base - box_height),
                box_width,
                box_height,
                facecolor=colors[0],
                edgecolor="white",
                lw=0.5,
            )
            ax.add_patch(rect)
            rectangles.append(rect)

            text = ax.text(
                x_pos + box_width / 2,
                y_base - box_height / 2,
                letters[0],
                ha="center",
                va="center",
                fontsize=8,
                color="k",
            )
            texts.append(text)

            motif_rect = plt.Rectangle(
                (x_pos, y_base - box_height - 0.25),
                box_width,
                0.15,
                facecolor="black",
                lw=0,
            )
            ax.add_patch(motif_rect)
            motif_rects.append(motif_rect)

        return rectangles, texts, motif_rects, letters, colors, positions_per_row

    @staticmethod
    def _update_sequence_bar(
        rectangles,
        texts,
        motif_rects,
        letters,
        colors,
        aatypes: np.ndarray,
        motif_mask: np.ndarray,
    ):
        """Update pre-created artists with new sequence data."""
        n = len(aatypes)
        for i in range(len(rectangles)):
            if i < n:
                aa_idx = int(aatypes[i]) if aatypes[i] < len(letters) else 20
                rectangles[i].set_facecolor(colors[aa_idx])
                rectangles[i].set_visible(True)
                texts[i].set_text(letters[aa_idx])
                texts[i].set_visible(True)
                motif_rects[i].set_visible(bool(motif_mask[i]))
            else:
                rectangles[i].set_visible(False)
                texts[i].set_visible(False)
                motif_rects[i].set_visible(False)

    @staticmethod
    def _create_3d_scatter_artist(
        ax: plt.Axes,
        max_atoms: int,
        trans_min: np.ndarray,
        trans_max: np.ndarray,
        only_alpha_carbons: bool = False,
        color_by: Literal["position", "sequence"] = "position",
    ):
        """Pre-create a 3D scatter artist with max_atoms capacity."""
        dummy_pos = np.zeros((max_atoms, 3))
        dummy_colors = np.zeros(max_atoms)
        dummy_sizes = np.ones(max_atoms) * 40.0

        scat = ax.scatter(
            dummy_pos[:, 0],
            dummy_pos[:, 1],
            dummy_pos[:, 2],
            c=dummy_colors,
            cmap=(
                PlotPanel._aa_listed_colormap()
                if color_by == "sequence"
                else "Spectral"
            ),
            vmin=0,
            vmax=1,
            s=dummy_sizes,
            depthshade=True,
            alpha=0.75,
        )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=25, azim=45)
        ax.set_xlim(trans_min[0], trans_max[0])
        ax.set_ylim(trans_min[1], trans_max[1])
        ax.set_zlim(trans_min[2], trans_max[2])

        return scat

    @staticmethod
    def _create_3d_residue_letter_artists(
        ax: plt.Axes,
        max_len: int,
        fontsize: float = 8.0,
    ):
        """Pre-create per-residue text artists."""
        texts = []
        for _ in range(max_len):
            text = ax.text2D(
                0.0,
                0.0,
                "",
                transform=ax.transData,
                ha="center",
                va="center",
                fontsize=fontsize,
                color="k",
                alpha=0.9,
            )
            text.set_visible(False)
            texts.append(text)
        return texts

    @staticmethod
    def _update_3d_residue_letter_artists(
        texts,
        ax: plt.Axes,
        ca_pos: np.ndarray,
        aatypes: np.ndarray,
    ) -> None:
        """Update pre-created artists with new residue letters/positions."""
        letters, _ = PlotPanel._aa_letters_and_colors()
        n = len(aatypes)

        x2, y2 = None, None
        if n > 0:
            x2, y2, _ = proj3d.proj_transform(
                ca_pos[:, 0],
                ca_pos[:, 1],
                ca_pos[:, 2],
                ax.get_proj(),
            )

        for i in range(len(texts)):
            if i < n:
                aa_idx = int(aatypes[i]) if aatypes[i] < len(letters) else 20
                texts[i].set_text(letters[aa_idx])
                texts[i].set_position((float(x2[i]), float(y2[i])))
                texts[i].set_visible(True)
            else:
                texts[i].set_visible(False)

    @staticmethod
    def _update_3d_scatter(
        scat,
        ax: plt.Axes,
        backbone_pos: np.ndarray,
        motif_alive: np.ndarray,
        aatypes_alive: Optional[np.ndarray],
        max_atoms: int,
        t_val: float,
        only_alpha_carbons: bool = False,
        remaining_insertions_alive: Optional[np.ndarray] = None,
        color_by: Literal["position", "sequence"] = "position",
    ):
        """Update pre-created 3D scatter artist with backbone atoms."""
        n_res = backbone_pos.shape[0] if backbone_pos.size > 0 else 0

        if n_res > 0:
            if only_alpha_carbons:
                n_atoms = n_res
                flat_pos = backbone_pos

                padded_pos = np.zeros((max_atoms, 3))
                padded_pos[:n_atoms] = flat_pos

                color_idx = np.zeros(max_atoms)
                if color_by == "sequence":
                    if aatypes_alive is None:
                        raise ValueError(
                            "aatypes_alive is required for color_by='sequence'"
                        )
                    color_idx[:n_atoms] = aatypes_alive
                else:
                    color_idx[:n_atoms] = np.arange(n_res)

                if remaining_insertions_alive is not None:
                    is_anchor = remaining_insertions_alive > 0
                    base_sizes = np.where(
                        is_anchor,
                        30.0 + 10.0 * remaining_insertions_alive,
                        20.0,
                    )
                else:
                    base_sizes = np.full(n_res, 20.0)
                motif_factor = np.where(motif_alive, 0.6, 1.0)
                sizes = np.zeros(max_atoms)
                sizes[:n_atoms] = base_sizes * motif_factor
            else:
                n_atoms = n_res * 3
                flat_pos = backbone_pos.reshape(-1, 3)

                padded_pos = np.zeros((max_atoms, 3))
                padded_pos[:n_atoms] = flat_pos

                color_idx = np.zeros(max_atoms)
                if color_by == "sequence":
                    if aatypes_alive is None:
                        raise ValueError(
                            "aatypes_alive is required for color_by='sequence'"
                        )
                    color_idx[:n_atoms] = np.repeat(aatypes_alive, 3)
                else:
                    res_colors = np.repeat(np.arange(n_res), 3)
                    color_idx[:n_atoms] = res_colors

                if remaining_insertions_alive is not None:
                    is_anchor = remaining_insertions_alive > 0
                    ca_sizes = np.where(
                        is_anchor,
                        30.0 + 10.0 * remaining_insertions_alive,
                        20.0,
                    )
                else:
                    ca_sizes = np.full(n_res, 20.0)
                base_sizes = np.zeros(n_atoms)
                base_sizes[0::3] = 10.0
                base_sizes[1::3] = ca_sizes
                base_sizes[2::3] = 10.0

                motif_expanded = np.repeat(motif_alive, 3)
                motif_factor = np.where(motif_expanded, 0.6, 1.0)
                sizes = np.zeros(max_atoms)
                sizes[:n_atoms] = base_sizes * motif_factor

            scat._offsets3d = (padded_pos[:, 0], padded_pos[:, 1], padded_pos[:, 2])
            scat.set_array(color_idx)
            if color_by == "sequence":
                scat.set_cmap(PlotPanel._aa_listed_colormap())
                _, colors = PlotPanel._aa_letters_and_colors()
                scat.set_clim(-0.5, float(colors.shape[0] - 1) + 0.5)
            else:
                scat.set_cmap("Spectral")
                scat.set_clim(0, max(n_res - 1, 1))
            scat.set_sizes(sizes)
        else:
            scat.set_sizes(np.zeros(max_atoms))

        ax.set_title(f"t = {t_val:.2f} (N={n_res})")


class BranchingFlowVisualizer:
    def __init__(
        self,
        sigma: Optional[float] = 1.0,
    ):
        self.interpolant = TreeInterpolant(
            cfg=VarcoInterpolantConfig(
                trans_coupler=VarcoInterpolantTransCouplerConfig(noise_scale=sigma),
                aatypes_coupler=VarcoInterpolantAATypesCouplerConfig(drift_temp=sigma),
            ),
        )

    @staticmethod
    def _get_anim_writer() -> Tuple[str, animation.AbstractMovieWriter]:
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
        if animation.writers.is_available("imagemagick"):
            return "gif", animation.ImageMagickWriter(fps=10)
        return "gif", animation.PillowWriter(fps=10)

    def _plot_trajectory_frames(
        self,
        frame_lists: List[List[TrajectoryFrame]],
        panel_titles: List[str],
        out_dir: str,
        filename: str,
        only_alpha_carbons: bool,
        show_residue_letters: bool,
        color_by: str,
        max_cols: int = 2,
    ) -> str:
        """Shared implementation for plotting trajectory frames.

        Args:
            frame_lists: List of frame lists, one per panel column.
                Each inner list contains TrajectoryFrames for that column.
                Caller is responsible for downsampling before calling.
            panel_titles: Title prefix for each panel column (e.g., "Sample: ", "Prediction: ").
            out_dir: Output directory for the animation.
            filename: Base filename (without extension).
            only_alpha_carbons: If True, show only CA atoms.
            show_residue_letters: If True, overlay AA letters on 3D view.
            color_by: "position" or "sequence".
            max_cols: Maximum columns in the grid.

        Returns:
            Path to the saved animation file.
        """
        num_panels = len(frame_lists)
        num_frames = len(frame_lists[0])

        # Validate frame lists have same length
        for i, frames in enumerate(frame_lists):
            if len(frames) != num_frames:
                raise ValueError(
                    f"Frame list {i} has {len(frames)} frames, expected {num_frames}"
                )

        # Compute global limits and max sequence length across all frames in all lists
        trans_min = np.full(3, np.inf)
        trans_max = np.full(3, -np.inf)
        max_seq_len = 0
        for frames in frame_lists:
            for frame in frames:
                valid_trans = frame.trans[frame.valid_mask]
                if valid_trans.shape[0] > 0:
                    trans_min = np.minimum(trans_min, valid_trans.min(axis=0))
                    trans_max = np.maximum(trans_max, valid_trans.max(axis=0))
                max_seq_len = max(max_seq_len, frame.valid_mask.sum())

        # Setup figure layout
        num_cols = min(num_panels, max_cols)
        num_rows = math.ceil(num_panels / num_cols)

        fig = plt.figure(figsize=(10 * num_cols, 12 * num_rows))
        gs = fig.add_gridspec(
            num_rows * 2,
            num_cols,
            height_ratios=[2, 10] * num_rows,
            hspace=0.02,
            wspace=0.05,
        )
        fig.subplots_adjust(
            left=0.03, right=0.97, bottom=0.03, top=0.95, wspace=0.05, hspace=0.05
        )

        # Create panels
        max_atoms = max_seq_len if only_alpha_carbons else max_seq_len * 3
        panels: List[PlotPanel] = []
        for i in range(num_panels):
            row, col = divmod(i, num_cols)
            ax_seq = fig.add_subplot(gs[row * 2, col])
            ax_3d = fig.add_subplot(gs[row * 2 + 1, col], projection="3d")

            panel = PlotPanel.create(
                ax_seq=ax_seq,
                ax_3d=ax_3d,
                max_seq_len=max_seq_len,
                max_atoms=max_atoms,
                trans_min=trans_min,
                trans_max=trans_max,
                only_alpha_carbons=only_alpha_carbons,
                color_by=color_by,
                show_residue_letters=show_residue_letters,
                title_prefix=panel_titles[i] if i < len(panel_titles) else "",
            )
            panels.append(panel)

        # Save animation
        ext, writer = self._get_anim_writer()
        anim_path = os.path.join(out_dir, f"{filename}.{ext}")
        logger.info(f"💾 Saving trajectory animation to {anim_path}")

        with writer.saving(fig, anim_path, dpi=100):
            for frame_idx in tqdm(
                range(num_frames), desc="trajectory frames", leave=False
            ):
                for panel_idx, panel in enumerate(panels):
                    frame = frame_lists[panel_idx][frame_idx]
                    panel.update(frame)
                writer.grab_frame()

        plt.close(fig)
        return anim_path

    def plot_trajectory(
        self,
        traj: Trajectory,
        out_dir: Optional[str] = None,
        filename: str = "trajectory",
        max_frames: Optional[int] = 50,
        max_samples: int = 2,
        max_cols: int = 2,
        only_alpha_carbons: bool = True,
        show_residue_letters: bool = True,
        color_by: Literal["auto", "position", "sequence"] = "auto",
    ) -> str:
        """Plot a trajectory animation showing multiple batch samples.

        Args:
            traj: Trajectory containing samples to plot.
            out_dir: Output directory (defaults to temp directory).
            filename: Base filename for the animation.
            max_frames: Maximum frames to render (downsamples if exceeded).
            max_samples: Maximum batch samples to show.
            max_cols: Maximum columns in the grid.
            only_alpha_carbons: If True, show only CA atoms (faster).
            show_residue_letters: If True, overlay AA letters on 3D view.
            color_by: 'auto' (infer), 'position' (chain index), or 'sequence' (aatype).

        Returns:
            Path to the saved animation file.
        """
        if out_dir is None:
            out_dir = tempfile.mkdtemp()
        os.makedirs(out_dir, exist_ok=True)

        if not traj.samples:
            raise ValueError("Trajectory has no samples to plot")

        if color_by not in {"auto", "position", "sequence"}:
            raise ValueError(
                f"Invalid color_by={color_by!r}; expected 'auto', 'position', or 'sequence'"
            )
        if color_by == "auto":
            color_by = "sequence" if show_residue_letters else "position"

        num_batch = traj.samples[0].trans_t.shape[0]
        num_plots = min(num_batch, max_samples)
        num_total_frames = len(traj.samples)

        # Compute frame indices BEFORE converting to TrajectoryFrame
        if max_frames is not None and num_total_frames > max_frames:
            sample_indices = np.linspace(0, num_total_frames - 1, max_frames, dtype=int)
        else:
            sample_indices = np.arange(num_total_frames)

        # Convert only the needed frames to TrajectoryFrames
        frame_lists: List[List[TrajectoryFrame]] = []
        for batch_idx in range(num_plots):
            frames = [
                TrajectoryFrame.from_data_corrupted(
                    data=traj.samples[idx],
                    batch_idx=batch_idx,
                    include_atom37=not only_alpha_carbons,
                )
                for idx in sample_indices
            ]
            frame_lists.append(frames)

        return self._plot_trajectory_frames(
            frame_lists=frame_lists,
            panel_titles=[""] * num_plots,
            out_dir=out_dir,
            filename=filename,
            only_alpha_carbons=only_alpha_carbons,
            show_residue_letters=show_residue_letters,
            color_by=color_by,
            max_cols=max_cols,
        )

    def plot_sampling_trajectory(
        self,
        traj: SampleTrajectory,
        batch_idx: int = 0,
        out_dir: Optional[str] = None,
        filename: str = "sampling_trajectory",
        max_frames: Optional[int] = 50,
        only_alpha_carbons: bool = True,
        show_residue_letters: bool = True,
        color_by: Literal["auto", "position", "sequence"] = "auto",
    ) -> str:
        """Plot sample and model prediction side-by-side for one batch item.

        Args:
            traj: SampleTrajectory containing samples and predictions.
            batch_idx: Which batch item to visualize.
            out_dir: Output directory (defaults to temp directory).
            filename: Base filename for the animation.
            max_frames: Maximum frames to render (downsamples if exceeded).
            only_alpha_carbons: If True, show only CA atoms (faster).
            show_residue_letters: If True, overlay AA letters on 3D view.
            color_by: 'auto' (infer), 'position' (chain index), or 'sequence' (aatype).

        Returns:
            Path to the saved animation file.
        """
        if out_dir is None:
            out_dir = tempfile.mkdtemp()
        os.makedirs(out_dir, exist_ok=True)

        if not traj.samples:
            raise ValueError("Trajectory has no samples to plot")
        if not traj.pred:
            raise ValueError("SampleTrajectory has no predictions to plot")

        if color_by not in {"auto", "position", "sequence"}:
            raise ValueError(
                f"Invalid color_by={color_by!r}; expected 'auto', 'position', or 'sequence'"
            )
        if color_by == "auto":
            color_by = "sequence" if show_residue_letters else "position"

        num_total_frames = len(traj.samples)

        # Compute frame indices BEFORE converting to TrajectoryFrame
        if max_frames is not None and num_total_frames > max_frames:
            sample_indices = np.linspace(0, num_total_frames - 1, max_frames, dtype=int)
        else:
            sample_indices = np.arange(num_total_frames)

        # Convert only the needed samples to TrajectoryFrames
        sample_frames = [
            TrajectoryFrame.from_data_corrupted(
                data=traj.samples[idx],
                batch_idx=batch_idx,
                include_atom37=not only_alpha_carbons,
            )
            for idx in sample_indices
        ]

        # Convert predictions (clamping index to available predictions)
        # Use the sample corresponding to the prediction index, not the frame index,
        # since sequence lengths can change during sampling (insertions/deletions)
        num_preds = len(traj.pred)
        pred_frames = [
            TrajectoryFrame.from_model_prediction(
                pred=traj.pred[min(idx, num_preds - 1)],
                sample=traj.samples[min(idx, num_preds - 1)],
                batch_idx=batch_idx,
                split_hazard=self.interpolant.cfg.sampling.split_hazard,
                include_atom37=not only_alpha_carbons,
            )
            for idx in sample_indices
        ]

        return self._plot_trajectory_frames(
            frame_lists=[sample_frames, pred_frames],
            panel_titles=["Sample: ", "Prediction: "],
            out_dir=out_dir,
            filename=filename,
            only_alpha_carbons=only_alpha_carbons,
            show_residue_letters=show_residue_letters,
            color_by=color_by,
            max_cols=2,
        )

    def visualize_corruption(
        self,
        batch: DataBatch,
        out_dir: Optional[str] = None,
        times: Optional[List[float]] = None,
        only_alpha_carbons: bool = False,  # faster; skips to_atom37
        filename: str = "corruption",
        coupled: bool = True,
        seed: Optional[int] = None,
    ) -> str:
        """Create a corruption trajectory and plot it.

        If coupled=True, generates a time-coupled trajectory from a single sampled set of
        domain couplings (anchors + creation states), rather than sampling each timepoint
        marginal independently.
        """
        self.interpolant.set_device(batch.trans_1.device)
        self.interpolant.seed_all(seed)
        if times is None:
            times = list(np.linspace(0.0, 1.0, 50))
        times = sorted(times)

        num_batch = batch.trans_1.shape[0]
        device = batch.trans_1.device
        tree = batch.tree.to(device)
        min_t = float(self.interpolant.min_t)
        times = [float(np.clip(t, min_t, 1.0 - min_t)) for t in times]

        # Define consistent base samples for the whole trajectory (in aligned space)
        trans_0 = self.interpolant.translation_coupler.sample_base(
            motif_mask=tree.motif_mask,
            x1=tree.broadcast_to_leaves(batch.trans_1.to(device), fill_value=0),
            device=device,
        )
        rotmats_0 = self.interpolant.rotation_coupler.sample_base(
            motif_mask=tree.motif_mask,
            x1=tree.broadcast_to_leaves(
                batch.rotmats_1.to(device), fill_value=torch.eye(3, device=device)
            ),
            device=device,
        )
        aatypes_0 = self.interpolant.aatypes_coupler.sample_base(
            motif_mask=tree.motif_mask,
            x1=tree.broadcast_to_leaves(
                batch.aatypes_1.to(device), fill_value=MASK_TOKEN_INDEX
            ),
            device=device,
        )

        traj, _ = self.interpolant.corrupt_trajectory(
            batch=batch,
            times=times,
            trans_0=trans_0,
            rotmats_0=rotmats_0,
            aatypes_0=aatypes_0,
        )
        return self.plot_trajectory(
            traj=traj,
            out_dir=out_dir,
            filename=filename,
            only_alpha_carbons=only_alpha_carbons,
        )
