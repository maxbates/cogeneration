import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from cogeneration.config.base import DatasetConfig
from cogeneration.type.dataset import MetadataColumn as mc
from cogeneration.type.dataset import MetadataDataFrame


class DatasetFilterer:
    def __init__(self, dataset_cfg: DatasetConfig):
        self.dataset_cfg = dataset_cfg
        self._log = logging.getLogger("DatasetFilterer")

    @property
    def modeled_length_col(self) -> mc:
        return self.dataset_cfg.modeled_trim_method.to_dataset_column()

    def _rog_filter(self, data_csv: MetadataDataFrame) -> MetadataDataFrame:
        """
        Filter by radius of gyration.
        """
        y_quant = pd.pivot_table(
            data_csv,
            values=mc.radius_gyration,
            index=self.modeled_length_col,
            aggfunc=lambda x: np.quantile(x, self.dataset_cfg.filter.rog_quantile),
        )
        x_quant = y_quant.index.to_numpy()
        y_quant = y_quant.radius_gyration.to_numpy()

        # Fit polynomial regressor
        poly = PolynomialFeatures(degree=4, include_bias=True)
        poly_features = poly.fit_transform(x_quant[:, None])
        poly_reg_model = LinearRegression()
        poly_reg_model.fit(poly_features, y_quant)

        # Calculate cutoff for all sequence lengths
        max_len = data_csv[self.modeled_length_col].max()
        pred_poly_features = poly.fit_transform(np.arange(max_len)[:, None])
        # Add a little more.
        pred_y = poly_reg_model.predict(pred_poly_features) + 0.1

        row_rog_cutoffs = data_csv[self.modeled_length_col].map(lambda x: pred_y[x - 1])
        return data_csv[data_csv[mc.radius_gyration] < row_rog_cutoffs]

    def _length_filter(self, data_csv: MetadataDataFrame) -> MetadataDataFrame:
        """Filter by sequence length."""
        return data_csv[
            (data_csv[self.modeled_length_col] >= self.dataset_cfg.filter.min_num_res)
            & (data_csv[self.modeled_length_col] <= self.dataset_cfg.filter.max_num_res)
        ]

    def _plddt_filter(self, data_csv: MetadataDataFrame) -> MetadataDataFrame:
        """Filter proteins which do not have the required minimum pLDDT."""
        # not used in the public multiflow codebase
        # TODO - pull out pLDDTs from structure, not available in current CSV
        # return data_csv[
        #     dc.num_confident_plddt
        #     > self.dataset_cfg.filter.min_num_confident_plddt
        # ]
        return data_csv

    def _max_coil_filter(self, data_csv: MetadataDataFrame) -> MetadataDataFrame:
        """Filter proteins which exceed max_coil_percent."""
        return data_csv[
            data_csv[mc.coil_percent] <= self.dataset_cfg.filter.max_coil_percent
        ]

    def _max_percent_unknown_filter(
        self, data_csv: MetadataDataFrame
    ) -> MetadataDataFrame:
        """Filter proteins which exceed `max_percent_residues_unknown`."""
        return data_csv[
            (data_csv[mc.seq_len] / data_csv[self.modeled_length_col])
            >= self.dataset_cfg.filter.max_percent_residues_unknown
        ]

    def filter_metadata(self, raw_csv: MetadataDataFrame) -> MetadataDataFrame:
        """
        Initial filtering of dataset.
        Does not filter redesigned / synthetic datasets.
        """
        filter_cfg = self.dataset_cfg.filter
        running_length = len(raw_csv)
        data_csv = raw_csv.copy()

        # monomer / oligomer
        data_csv = data_csv[data_csv[mc.oligomeric_detail].isin(filter_cfg.oligomeric)]
        if len(data_csv) < running_length:
            self._log.debug(
                f"{running_length} -> {len(data_csv)} examples after oligomeric filter"
            )
            running_length = len(data_csv)

        # number of chains
        data_csv = data_csv[data_csv[mc.num_chains].isin(filter_cfg.num_chains)]
        if len(data_csv) < running_length:
            self._log.debug(
                f"{running_length} ->{len(data_csv)} examples after num_chains filter"
            )
            running_length = len(data_csv)

        # length
        data_csv = self._length_filter(data_csv=data_csv)
        if len(data_csv) < running_length:
            self._log.debug(
                f"{running_length} ->{len(data_csv)} examples after length filter"
            )
            running_length = len(data_csv)

        # modelable residues
        data_csv = self._max_percent_unknown_filter(data_csv=data_csv)
        if len(data_csv) < running_length:
            self._log.debug(
                f"{running_length} ->{len(data_csv)} examples after % unknown filter"
            )
            running_length = len(data_csv)

        # max coil percent
        data_csv = self._max_coil_filter(data_csv=data_csv)
        if len(data_csv) < running_length:
            self._log.debug(
                f"{running_length} ->{len(data_csv)} examples after max coil filter"
            )
            running_length = len(data_csv)

        # radius of gyration
        data_csv = self._rog_filter(data_csv=data_csv)
        if len(data_csv) < running_length:
            self._log.debug(
                f"{running_length} -> {len(data_csv)} examples after rog filter"
            )
            running_length = len(data_csv)

        # pLDDT
        data_csv = self._plddt_filter(data_csv=data_csv)
        if len(data_csv) < running_length:
            self._log.debug(
                f"{running_length} -> {len(data_csv)} examples after pLDDT filter"
            )
            running_length = len(data_csv)

        # sort by modeled length
        data_csv = data_csv.sort_values(by=self.modeled_length_col, ascending=False)

        return data_csv
