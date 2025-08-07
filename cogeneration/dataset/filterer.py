import logging
from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from cogeneration.config.base import (
    DatasetConfig,
    DatasetFilterConfig,
    DatasetTrimMethod,
)
from cogeneration.dataset.process_pdb import DataError
from cogeneration.type.dataset import BestRedesignColumn
from cogeneration.type.dataset import MetadataColumn as mc
from cogeneration.type.dataset import MetadataCSVRow, MetadataDataFrame


def _log_filter(stage: str) -> Callable[[Callable], Callable]:
    """
    Decorator logs how many rows remain after `stage`.
    Requires `self._log` to be defined.
    """

    def deco(fn: Callable) -> Callable:
        def wrapper(self, df: MetadataDataFrame) -> MetadataDataFrame:
            before = len(df)

            if before == 0:
                self._log.debug(f"Skipping {stage} on empty DataFrame")
                return df

            try:
                df = fn(self, df)  # type: ignore[arg-type]
                if len(df) != before:
                    self._log.debug(f"{before} -> {len(df)} after {stage}")
            except Exception as e:
                self._log.error(f"Error during filter {stage}: {e}")
                raise e

            return df

        # keep original name and docstring
        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper

    return deco


@dataclass
class DatasetFilterer:
    """Filter PDB‑metadata DataFrames according to *dataset_cfg*."""

    cfg: DatasetFilterConfig
    modeled_trim_method: DatasetTrimMethod
    _log: logging.Logger = None

    def __post_init__(self):
        if self._log is None:
            self._log = logging.getLogger("DatasetFilterer")
            self._log.setLevel(logging.DEBUG)

    @property
    def modeled_length_col(self) -> mc:
        """Get appropriate column for modeled length depending on trimming method"""
        return self.modeled_trim_method.to_dataset_column()

    @_log_filter("oligomeric filter")
    def _oligomeric_filter(self, df: MetadataDataFrame) -> MetadataDataFrame:
        if self.cfg.oligomeric is not None:
            df = df[df[mc.oligomeric_detail].isin(self.cfg.oligomeric)]
        if self.cfg.oligomeric_skip is not None:
            df = df[~df[mc.oligomeric_detail].isin(self.cfg.oligomeric_skip)]
        return df

    @_log_filter("num_chains filter")
    def _num_chains_filter(self, df: MetadataDataFrame) -> MetadataDataFrame:
        if self.cfg.num_chains is not None:
            df = df[df[mc.num_chains].isin(self.cfg.num_chains)]
        if self.cfg.num_chains_skip is not None:
            df = df[~df[mc.num_chains].isin(self.cfg.num_chains_skip)]
        return df

    @_log_filter("length filter")
    def _length_filter(self, df: MetadataDataFrame) -> MetadataDataFrame:
        return df[
            (df[self.modeled_length_col] >= self.cfg.min_num_res)
            & (df[self.modeled_length_col] <= self.cfg.max_num_res)
        ]

    @_log_filter("% unknown filter")
    def _max_percent_unknown_filter(self, df: MetadataDataFrame) -> MetadataDataFrame:
        return df[
            (df[mc.seq_len] / df[self.modeled_length_col])
            >= self.cfg.max_percent_residues_unknown
        ]

    @_log_filter("max coil filter")
    def _max_coil_filter(self, df: MetadataDataFrame) -> MetadataDataFrame:
        return df[df[mc.coil_percent] <= self.cfg.max_coil_percent]

    @_log_filter("rog filter")
    def _rog_filter(self, df: MetadataDataFrame) -> MetadataDataFrame:
        if self.cfg.rog_quantile > 0.999:
            return df

        y_quant = pd.pivot_table(
            df,
            values=mc.radius_gyration,
            index=self.modeled_length_col,
            aggfunc=lambda x: np.quantile(x, self.cfg.rog_quantile),
        )
        x_quant = y_quant.index.to_numpy()
        y_quant = y_quant.radius_gyration.to_numpy()

        poly = PolynomialFeatures(degree=4, include_bias=True)
        model = LinearRegression().fit(poly.fit_transform(x_quant[:, None]), y_quant)

        max_len = df[self.modeled_length_col].max()
        preds = model.predict(poly.fit_transform(np.arange(max_len)[:, None])) + 0.1
        cutoffs = df[self.modeled_length_col].map(lambda x: preds[x - 1])
        return df[df[mc.radius_gyration] < cutoffs]

    @_log_filter("pLDDT filter")
    def _plddt_filter(self, df: MetadataDataFrame) -> MetadataDataFrame:
        # not used in the public multiflow codebase; only applies to synthetic structures
        if mc.mean_plddt_modeled_bb in df.columns:
            return df[df[mc.mean_plddt_modeled_bb] > self.cfg.min_plddt]
        elif mc.mean_plddt_all_atom in df.columns:
            return df[df[mc.mean_plddt_all_atom] > self.cfg.min_plddt]
        else:
            self._log.warning("No pLDDT column found, skipping pLDDT filter")
            return df

    @_log_filter("date filter")
    def _date_filter(self, df: MetadataDataFrame) -> MetadataDataFrame:
        if not mc.date in df.columns:
            self._log.warning("No date column found, skipping date filter")
            return df
        if self.cfg.min_date is None and self.cfg.max_date is None:
            return df

        if self.cfg.min_date is not None:
            min_date = pd.to_datetime(self.cfg.min_date, errors="coerce")
            if min_date is pd.NaT:
                self._log.warning(f"Invalid min_date: {self.cfg.min_date}")
                return df
            datetime_col = df[mc.date].apply(pd.to_datetime, errors="coerce")
            df = df[datetime_col >= min_date]

        if self.cfg.max_date is not None:
            max_date = pd.to_datetime(self.cfg.max_date, errors="coerce")
            if max_date is pd.NaT:
                self._log.warning(f"Invalid max_date: {self.cfg.max_date}")
                return df
            datetime_col = df[mc.date].apply(pd.to_datetime, errors="coerce")
            df = df[datetime_col <= max_date]

        return df

    @_log_filter("Redesign RMSD filter")
    def _redesign_rmsd_filter(self, df: MetadataDataFrame) -> MetadataDataFrame:
        if BestRedesignColumn.best_rmsd not in df.columns:
            return df
        return df[
            df[BestRedesignColumn.best_rmsd] <= self.cfg.redesigned_rmsd_threshold
        ]

    @property
    def all_filters(
        self,
    ) -> Dict[str, Callable[[MetadataDataFrame], MetadataDataFrame]]:
        return {
            "date": self._date_filter,
            "oligomeric": self._oligomeric_filter,
            "num_chains": self._num_chains_filter,
            "length": self._length_filter,
            "max_percent_unknown": self._max_percent_unknown_filter,
            "max_coil": self._max_coil_filter,
            "rog": self._rog_filter,
            "plddt": self._plddt_filter,
        }

    def filter_metadata(self, raw_csv: MetadataDataFrame) -> MetadataDataFrame:
        """
        Filter a metadata CSV according to DatasetConfig.
        """
        df = raw_csv.copy()

        for filter_name, filter_fn in self.all_filters.items():
            try:
                df = filter_fn(df)
            except KeyError as e:
                self._log.warning(
                    f"Skipping {filter_name} filter; missing respective column: {e}"
                )

        return df.sort_values(by=self.modeled_length_col, ascending=False)

    def check_row(self, csv_row: MetadataCSVRow):
        """
        Check if a single row of metadata passes the filter criteria. Raises DataError if fails.
        """
        prev_disabled = self._log.disabled
        self._log.disabled = True

        try:
            df = pd.DataFrame([csv_row])
            for filter_name, filter_fn in self.all_filters.items():
                if len(filter_fn(df)) == 0:
                    raise DataError(f"Failed {filter_name} filter")
        except DataError as e:
            self._log.disabled = prev_disabled
            raise e
        finally:
            self._log.disabled = prev_disabled
