"""
Script which consumes defined datasets and generates a report of statistics and plots about them.

Generates a multi-page PDF and individual PNGs for each plot under
`reports/dataset_descriptions/<run_id>/` in the project root.

Example:
    python cogeneration/dataset/scripts/describe_dataset.py

You can use hydra to override datasets, filtering, etc.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
from omegaconf import OmegaConf

from cogeneration.config.base import Config, DatasetTrimMethod
from cogeneration.dataset.datasets import BaseDataset, DATASET_KEY
from cogeneration.dataset.filterer import DatasetFilterer
from cogeneration.type.dataset import MetadataColumn as mc
from cogeneration.type.dataset import RedesignColumn


@dataclass
class DatasetDescriber:
    cfg: Config

    def __post_init__(self):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)

        # Resolve output locations
        project_root = Path(self.cfg.shared.project_root).resolve()
        self.output_dir = (
            project_root / "reports" / "dataset_descriptions" / self.cfg.shared.id
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_path = self.output_dir / "dataset_report.pdf"

        self.log.info(f"Writing dataset description to {self.output_dir}...")

        # Load raw metadata across all configured datasets (no filtering applied)
        self.metadata = BaseDataset.load_datasets(
            dataset_cfg=self.cfg.dataset,
            logger=self.log,
        )

        # Filtering utility configured from cfg
        self.filterer = DatasetFilterer(
            cfg=self.cfg.dataset.filter,
            modeled_trim_method=self.cfg.dataset.modeled_trim_method,
        )

        # Seaborn theme
        sns.set_theme(style="whitegrid", context="talk")

    # Utilities

    @property
    def modeled_length_col(self) -> mc:
        return self.cfg.dataset.modeled_trim_method.to_dataset_column()

    def _save_fig(self, fig: plt.Figure, name: str, pdf: PdfPages) -> None:
        self.log.info(f"Saving figure {name}...")
        png_path = self.output_dir / f"{name}.png"
        fig.savefig(png_path, bbox_inches="tight", dpi=200)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    def _get_train_test_split_series(self, df: pd.DataFrame) -> pd.Series:
        cutoff = pd.to_datetime(self.cfg.dataset.test_date_cutoff, errors="coerce")
        dates = pd.to_datetime(
            df.get(mc.date, pd.Series(index=df.index)), errors="coerce"
        )
        is_test = (dates >= cutoff) & (~dates.isna())
        # Treat missing dates as train
        split = np.where(is_test, "Test", "Train")
        return pd.Series(split, index=df.index)

    # Plotters

    def plot_num_entries_per_database(self, df: pd.DataFrame) -> plt.Figure:
        counts = df[DATASET_KEY].value_counts().reset_index()
        counts.columns = ["dataset", "count"]
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=counts, x="dataset", y="count", ax=ax, color="#4472C4")
        ax.set_title("Number of entries per dataset")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Entries")
        ax.bar_label(ax.containers[0], fmt="%d")
        ax.tick_params(axis="x", rotation=30)
        return fig

    def plot_percent_filtered_per_database(self, raw_df: pd.DataFrame) -> plt.Figure:
        rows: List[Tuple[str, int, int, float]] = []
        for dataset_name, sub_df in raw_df.groupby(DATASET_KEY):
            try:
                filtered_df = self.filterer.filter_metadata(sub_df)
            except Exception:
                filtered_df = sub_df
            raw_n = len(sub_df)
            filt_n = len(filtered_df)
            pct_filtered = 0.0 if raw_n == 0 else 100.0 * (raw_n - filt_n) / raw_n
            rows.append((str(dataset_name), raw_n, filt_n, pct_filtered))

        stats = pd.DataFrame(
            rows, columns=["dataset", "raw", "filtered", "pct_filtered"]
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=stats, x="dataset", y="pct_filtered", ax=ax, color="#ED7D31")
        ax.set_title("Percent filtered per dataset")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Filtered (%)")
        ax.bar_label(ax.containers[0], fmt="%.1f%%")
        ax.tick_params(axis="x", rotation=30)
        return fig

    def plot_filter_reasons_stacked(self, raw_df: pd.DataFrame) -> plt.Figure:
        filters_order = list(self.filterer.all_filters.keys())
        rows: List[Dict[str, object]] = []

        for dataset_name, sub_df in raw_df.groupby(DATASET_KEY):
            total = len(sub_df)
            if total == 0:
                continue
            df_current = sub_df
            for fname in filters_order:
                try:
                    before = len(df_current)
                    df_after = self.filterer.all_filters[fname](df_current)
                    after = len(df_after)
                except Exception:
                    before = len(df_current)
                    after = before
                    df_after = df_current
                removed = max(0, before - after)
                frac = 0.0 if total == 0 else removed / total
                rows.append(
                    {"dataset": str(dataset_name), "reason": fname, "fraction": frac}
                )
                df_current = df_after
            kept = len(df_current)
            rows.append(
                {
                    "dataset": str(dataset_name),
                    "reason": "kept",
                    "fraction": (0.0 if total == 0 else kept / total),
                }
            )

        stats = pd.DataFrame(rows)
        reasons_order = filters_order + ["kept"]
        datasets = sorted(stats["dataset"].unique().tolist())

        fig, ax = plt.subplots(figsize=(12, 7))
        bottom = np.zeros(len(datasets))
        for reason in reasons_order:
            vals = []
            for d in datasets:
                sub = stats[(stats["dataset"] == d) & (stats["reason"] == reason)][
                    "fraction"
                ]
                vals.append(float(sub.sum()) if len(sub) > 0 else 0.0)
            ax.bar(datasets, vals, bottom=bottom, label=reason)
            bottom += np.array(vals)

        ax.set_title("Filter reasons per dataset (fractions)")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Fraction")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="Reason", bbox_to_anchor=(1.02, 1), loc="upper left")
        return fig

    def plot_sequence_length_histogram(self, df: pd.DataFrame) -> plt.Figure:
        # Cap lengths at 10k and use log-spaced bins; show cap as ">10k"
        sub = df.dropna(subset=[mc.seq_len]).copy()
        cap_value = 10000
        # avoid zeros for log scale; also cap very long lengths
        vals = sub[mc.seq_len].astype(float)
        vals = vals.clip(lower=1.0)
        sub["seq_len_capped"] = np.minimum(vals, cap_value)
        # choose log-spaced bins
        min_len = float(np.nanmax([1.0, sub["seq_len_capped"].min()]))
        bins = np.logspace(np.log10(min_len), np.log10(cap_value), num=50)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(
            data=sub,
            x="seq_len_capped",
            hue=DATASET_KEY,
            multiple="stack",
            bins=bins,
            edgecolor=None,
            ax=ax,
        )
        ax.set_xscale("log")
        ax.set_xlim(left=min_len, right=cap_value)

        def _tick_fmt(x, pos):
            if abs(x - cap_value) < 1e-6:
                return ">10k"
            try:
                return f"{int(x):d}"
            except Exception:
                return f"{x:.0f}"

        ax.xaxis.set_major_formatter(FuncFormatter(_tick_fmt))
        ax.set_title("Sequence length histogram (log scale)")
        ax.set_xlabel("Sequence length (AA)")
        ax.set_ylabel("Count")
        return fig

    def plot_seq_vs_modeled_indep_len(self, df: pd.DataFrame) -> plt.Figure:
        if mc.modeled_indep_seq_len not in df.columns:
            # Fallback to modeled_seq_len
            modeled_col = mc.modeled_seq_len
        else:
            modeled_col = mc.modeled_indep_seq_len

        sub = df.dropna(subset=[mc.seq_len, modeled_col])
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.scatterplot(
            data=sub,
            x=mc.seq_len,
            y=modeled_col,
            hue=DATASET_KEY,
            s=15,
            alpha=0.25,
            ax=ax,
        )
        ax.set_title("Sequence length vs. trimmed independent seq length")
        ax.set_xlabel("Sequence length (AA)")
        ax.set_ylabel(
            "Modeled independent length (AA)"
            if modeled_col == mc.modeled_indep_seq_len
            else "Modeled length (AA)"
        )
        ax.legend(loc="best", fontsize=9)
        return fig

    def plot_secondary_structure_densities(
        self, df: pd.DataFrame
    ) -> Optional[plt.Figure]:
        needed = [mc.helix_percent, mc.coil_percent, mc.strand_percent]
        if any(col not in df.columns for col in needed):
            return None
        pairs: List[Tuple[mc, mc, str]] = [
            (mc.helix_percent, mc.coil_percent, "Helix vs. Coil"),
            (mc.helix_percent, mc.strand_percent, "Helix vs. Beta"),
            (mc.coil_percent, mc.strand_percent, "Coil vs. Beta"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
        for ax, (xcol, ycol, title) in zip(axes, pairs):
            sub = df.dropna(subset=[xcol, ycol])
            # Optional downsampling for speed on very large datasets
            max_points = 200000
            if len(sub) > max_points:
                sub = sub.sample(n=max_points, random_state=123)
            sns.histplot(
                data=sub,
                x=xcol,
                y=ycol,
                bins=25,
                binrange=((0, 1), (0, 1)),
                cbar=True,
                cmap="mako",
                ax=ax,
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(title)
            ax.set_xlabel(str(xcol))
            ax.set_ylabel(str(ycol))
        fig.suptitle("Secondary structure densities", y=1.02)
        return fig

    def plot_date_histogram_with_split(self, df: pd.DataFrame) -> Optional[plt.Figure]:
        if mc.date not in df.columns:
            return None
        sub = df.copy()
        sub["split"] = self._get_train_test_split_series(sub)
        sub[mc.date] = pd.to_datetime(sub[mc.date], errors="coerce")
        sub = sub.dropna(subset=[mc.date])

        # Compute proportions
        split_counts = (
            sub["split"]
            .value_counts(normalize=True)
            .reindex(["Train", "Test"])
            .fillna(0)
        )
        train_pct = split_counts.get("Train", 0) * 100
        test_pct = split_counts.get("Test", 0) * 100
        legend_title = f"Train ({train_pct:.1f}%) vs Test ({test_pct:.1f}%)"

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(data=sub, x=mc.date, hue="split", multiple="stack", bins=50, ax=ax)
        ax.set_yscale("log")
        ax.set_title("Structure dates (Train/Test split)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Count")
        ax.legend(title=legend_title)
        return fig

    def plot_length_by_monomer_vs_multimer(
        self, df: pd.DataFrame
    ) -> Optional[plt.Figure]:
        if mc.num_chains not in df.columns:
            return None
        modeled_col = self.modeled_length_col
        sub = df.dropna(subset=[modeled_col, mc.num_chains]).copy()
        sub["assembly"] = np.where(
            sub[mc.num_chains].astype(float) <= 1, "Monomer", "Multimer"
        )
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.boxplot(data=sub, x="assembly", y=modeled_col, ax=ax)
        ax.set_yscale("log")
        ax.set_title("Modeled length by assembly")
        ax.set_xlabel("Assembly")
        ax.set_ylabel("Modeled length (AA)")
        return fig

    def plot_non_residue_entity_fractions(
        self, df: pd.DataFrame
    ) -> Optional[plt.Figure]:
        needed = [
            mc.num_metal_atoms,
            mc.num_small_molecules,
            mc.num_nucleic_acid_polymers,
            mc.num_other_polymers,
            mc.num_chains,
        ]
        if any(col not in df.columns for col in needed):
            return None

        sub = df.copy()
        sub["assembly"] = np.where(
            sub[mc.num_chains].astype(float) <= 1, "Monomer", "Multimer"
        )
        presence_cols: Dict[str, mc] = {
            "Metals": mc.num_metal_atoms,
            "Small molecules": mc.num_small_molecules,
            "Nucleic acids": mc.num_nucleic_acid_polymers,
            "Other polymers": mc.num_other_polymers,
        }
        rows: List[Dict[str, object]] = []
        for assembly, sdf in sub.groupby("assembly"):
            for label, col in presence_cols.items():
                if col not in sdf.columns:
                    frac = np.nan
                else:
                    frac = float((sdf[col].fillna(0).astype(float) > 0).mean())
                rows.append({"assembly": assembly, "entity": label, "fraction": frac})
        frac_df = pd.DataFrame(rows)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=frac_df, x="entity", y="fraction", hue="assembly", ax=ax)
        ax.set_title("Fraction with non-residue entities")
        ax.set_xlabel("Entity type")
        ax.set_ylabel("Fraction")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=20)
        return fig

    def plot_redesigns_scatter(self, df: pd.DataFrame) -> List[plt.Figure]:
        if RedesignColumn.rmsd not in df.columns:
            return []
        modeled_col = self.modeled_length_col
        sub = df.dropna(subset=[modeled_col, RedesignColumn.rmsd]).copy()
        figs: List[plt.Figure] = []

        # Length vs RMSD
        fig1, ax1 = plt.subplots(figsize=(9, 6))
        sns.scatterplot(
            data=sub,
            x=modeled_col,
            y=RedesignColumn.rmsd,
            hue=DATASET_KEY,
            s=15,
            alpha=0.25,
            ax=ax1,
        )
        ax1.axhline(
            self.cfg.redesign.rmsd_good,
            color="red",
            linestyle="--",
            label=f"RMSD good = {self.cfg.redesign.rmsd_good}",
        )
        ax1.set_title("Redesigns: length vs RMSD")
        ax1.set_xlabel("Modeled length (AA)")
        ax1.set_ylabel("RMSD (Å)")
        ax1.legend(loc="best")
        figs.append(fig1)

        # Length vs TM-score
        if RedesignColumn.tm_score in df.columns:
            sub_tm = df.dropna(subset=[modeled_col, RedesignColumn.tm_score]).copy()
            fig2, ax2 = plt.subplots(figsize=(9, 6))
            sns.scatterplot(
                data=sub_tm,
                x=modeled_col,
                y=RedesignColumn.tm_score,
                hue=DATASET_KEY,
                s=15,
                alpha=0.25,
                ax=ax2,
            )
            ax2.set_title("Redesigns: length vs TM-score")
            ax2.set_xlabel("Modeled length (AA)")
            ax2.set_ylabel("TM-score")
            ax2.set_ylim(0, 1)
            figs.append(fig2)

        return figs

    def run(self) -> None:
        with PdfPages(self.pdf_path) as pdf:
            # 1) Basic counts
            fig = self.plot_num_entries_per_database(self.metadata)
            self._save_fig(fig, "01_num_entries_per_dataset", pdf)

            # 2) Percent filtered per dataset
            fig = self.plot_percent_filtered_per_database(self.metadata)
            self._save_fig(fig, "02_percent_filtered_per_dataset", pdf)

            # 2b) Reasons for filtering (stacked fractions)
            fig = self.plot_filter_reasons_stacked(self.metadata)
            self._save_fig(fig, "02b_filter_reasons_stacked", pdf)

            # 3) Sequence length histogram
            fig = self.plot_sequence_length_histogram(self.metadata)
            self._save_fig(fig, "03_sequence_length_histogram", pdf)

            # 4) seq_len vs modeled independent len
            fig = self.plot_seq_vs_modeled_indep_len(self.metadata)
            self._save_fig(fig, "04_seq_vs_modeled_indep", pdf)

            # 5) Secondary structure density plots
            fig = self.plot_secondary_structure_densities(self.metadata)
            if fig is not None:
                self._save_fig(fig, "05_secondary_structure_densities", pdf)

            # 6) Date histogram with split
            fig = self.plot_date_histogram_with_split(self.metadata)
            if fig is not None:
                self._save_fig(fig, "06_date_histogram_split", pdf)

            # 7) Length by monomer vs multimer
            fig = self.plot_length_by_monomer_vs_multimer(self.metadata)
            if fig is not None:
                self._save_fig(fig, "07_length_by_assembly", pdf)

            # 8) Non-residue entity fractions by assembly
            fig = self.plot_non_residue_entity_fractions(self.metadata)
            if fig is not None:
                self._save_fig(fig, "08_non_residue_entity_fractions", pdf)

            # 9) Redesigns: length vs RMSD, TM-score
            for i, fig_i in enumerate(
                self.plot_redesigns_scatter(self.metadata), start=1
            ):
                self._save_fig(fig_i, f"09_redesigns_scatter_{i:02d}", pdf)

        self.log.info(f"Wrote dataset description to {self.pdf_path}")


@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(cfg: Config) -> None:
    config = cfg if isinstance(cfg, Config) else OmegaConf.to_object(cfg)
    config = config.interpolate()

    describer = DatasetDescriber(cfg=config)
    describer.run()


if __name__ == "__main__":
    main()
