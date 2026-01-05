import os
from typing import List

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from cogeneration.config.base import DataConfig
from cogeneration.dataset.datasets import BaseDataset
from cogeneration.dataset.protein_dataloader import LengthBatcher
from cogeneration.type.batch import BatchProp as bp
from cogeneration.type.task import DataTask
from varco.config import VarcoDatasetConfig
from varco.data import DataBatch, DataSample
from varco.tree_plan import BatchedTreePlan, TreePlan


class ProteinDataset(BaseDataset):
    """Wrapper to simplify BaseDataset and extract relevant features"""

    def __init__(
        self,
        cfg: VarcoDatasetConfig,
        eval: bool = False,
        use_test: bool = False,
    ):
        super().__init__(
            cfg=cfg,
            task=DataTask.inpainting,
            eval=eval,
            use_test=use_test,
        )

    def __getitem__(self, idx) -> DataSample:
        feats = super().__getitem__(idx)

        motif_mask = feats[bp.motif_mask].bool()
        res_mask = feats[bp.res_mask].int()
        chain_idx = feats[bp.chain_idx].int()

        # domains
        trans_1 = feats[bp.trans_1]
        rotmats_1 = feats[bp.rotmats_1]
        aatypes_1 = feats[bp.aatypes_1]

        # conditioning + confidence
        contact_conditioning = feats[bp.contact_conditioning]
        res_bfactor = feats[bp.res_bfactor]
        res_plddt = feats[bp.res_plddt]

        tree_plan = TreePlan.generate(motif_mask=motif_mask)
        tree_plan.validate()

        return DataSample(
            tree_plan=tree_plan,
            motif_mask=motif_mask,
            res_mask=res_mask,
            chain_idx=chain_idx,
            trans_1=trans_1,
            rotmats_1=rotmats_1,
            aatypes_1=aatypes_1,
            contact_conditioning=contact_conditioning,
            res_bfactor=res_bfactor,
            res_plddt=res_plddt,
        )


class ProteinDataLoader(DataLoader):
    """DataLoader for ProteinDataset"""

    def __init__(
        self,
        dataset: ProteinDataset,
        batch_size: int = 1,
        num_workers: int = max(1, os.cpu_count() - 2),
        **kwargs,
    ):
        # force custom collate_fn
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        super().__init__(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=ProteinDataLoader.collate_fn,
            prefetch_factor=None if num_workers == 0 else 2,
            persistent_workers=False,
            **kwargs,
        )

    @staticmethod
    def collate_fn(batch: List[DataSample]) -> DataBatch:
        # Special handling for tree collation
        plans = [item.tree_plan for item in batch]
        tree = BatchedTreePlan.collate(plans)

        motif_mask = torch.stack([item.motif_mask for item in batch])  # (B, N)
        res_mask = torch.stack([item.res_mask for item in batch])  # (B, N)
        chain_idx = torch.stack([item.chain_idx for item in batch])  # (B, N)
        trans_1 = torch.stack([item.trans_1 for item in batch])  # (B, N, 3)
        rotmats_1 = torch.stack([item.rotmats_1 for item in batch])  # (B, N, 3, 3)
        aatypes_1 = torch.stack([item.aatypes_1 for item in batch])  # (B, N)
        contact_conditioning = torch.stack(
            [item.contact_conditioning for item in batch]
        )  # (B, N, N)
        res_bfactor = torch.stack([item.res_bfactor for item in batch])  # (B, N)
        res_plddt = torch.stack([item.res_plddt for item in batch])  # (B, N)

        return DataBatch(
            tree=tree,
            motif_mask=motif_mask,
            res_mask=res_mask,
            chain_idx=chain_idx,
            trans_1=trans_1,
            rotmats_1=rotmats_1,
            aatypes_1=aatypes_1,
            contact_conditioning=contact_conditioning,
            res_bfactor=res_bfactor,
            res_plddt=res_plddt,
        )


class ProteinDataModule(pl.LightningDataModule):
    """DataModule for ProteinDataset"""

    def __init__(
        self,
        cfg: DataConfig,
        dataset: ProteinDataset,
    ):
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset

    def train_dataloader(self, rank=None, num_replicas=None) -> DataLoader:
        batch_sampler = LengthBatcher(
            sampler_cfg=self.cfg.sampler,
            metadata_csv=self.dataset.csv,
            modeled_length_col=self.dataset.cfg.modeled_trim_method.to_dataset_column(),
            rank=rank or 0,
            num_replicas=num_replicas or 1,
        )

        return ProteinDataLoader(
            dataset=self.dataset,
            batch_sampler=batch_sampler,
            num_workers=self.cfg.loader.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return ProteinDataLoader(
            dataset=self.dataset,
            batch_size=1,
            num_workers=0,
        )
