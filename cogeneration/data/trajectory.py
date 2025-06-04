from dataclasses import dataclass, field
from typing import List, Optional

import torch

from cogeneration.data import all_atom
from cogeneration.type.batch import BatchProp as bp
from cogeneration.type.batch import ModelPrediction
from cogeneration.type.batch import NoisyBatchProp as nbp
from cogeneration.type.batch import NoisyFeatures
from cogeneration.type.batch import PredBatchProp as pbp


def detach(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return x.detach()


def to_cpu(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return x.detach().cpu()


def to_cpu_clone(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return x.detach().clone().cpu()


@dataclass
class SamplingStep:
    """
    A single step in the sampling trajectory.
    For model states, this is the model's prediction at time `t`.
    For protein states, this is the interpolated structure + sequence at time `t`.
    """

    res_mask: torch.Tensor  # (B, N)
    trans: torch.Tensor  # (B, N, 3)
    rotmats: torch.Tensor  # (B, N, 3, 3)
    aatypes: torch.Tensor  # (B, N)
    torsions: Optional[torch.Tensor]  # (B, N, 7, 2)
    logits: Optional[torch.Tensor]  # model output only; (B, N, S) S={20,21}

    def to_cpu(self):
        """
        Move all tensors to CPU and detach.
        """
        self.res_mask = to_cpu(self.res_mask)
        self.trans = to_cpu(self.trans)
        self.rotmats = to_cpu(self.rotmats)
        self.aatypes = to_cpu(self.aatypes)
        self.torsions = to_cpu(self.torsions)
        self.logits = to_cpu(self.logits)

    @property
    def structure(self) -> torch.Tensor:
        """
        Returns the atom37 representation tensor shape (B, N, 37, 3).
        """
        return all_atom.atom37_from_trans_rot(
            trans=self.trans,
            rots=self.rotmats,
            torsions=self.torsions,
            aatype=self.aatypes,
            res_mask=self.res_mask,
            unknown_to_alanine=True,  # show UNK residues as alanine
        )

    def select_batch_idx(self, idx: torch.Tensor) -> "SamplingStep":
        """
        Select batch members by index `idx`.
        """
        return SamplingStep(
            res_mask=self.res_mask[idx],
            trans=self.trans[idx],
            rotmats=self.rotmats[idx],
            aatypes=self.aatypes[idx],
            torsions=None if self.torsions is None else self.torsions[idx],
            logits=None if self.logits is None else self.logits[idx],
        )

    @classmethod
    def from_values(
        cls,
        res_mask: torch.Tensor,
        trans: torch.Tensor,
        rotmats: torch.Tensor,
        aatypes: torch.Tensor,
        torsions: Optional[torch.Tensor],
        logits: Optional[torch.Tensor],
    ):
        """
        Safely create a SamplingStep
        """
        return cls(
            res_mask=detach(res_mask),
            trans=detach(trans),
            rotmats=detach(rotmats),
            aatypes=detach(aatypes),
            torsions=detach(torsions),
            logits=detach(logits),
        )

    @classmethod
    def from_batch(cls, batch: NoisyFeatures):
        """
        Create a SamplingStep from `_t` values in a batch
        """
        return cls.from_values(
            res_mask=batch[bp.res_mask],
            trans=batch[nbp.trans_t],
            rotmats=batch[nbp.rotmats_t],
            aatypes=batch[nbp.aatypes_t],
            torsions=batch.get(nbp.torsions_t, None),
            logits=None,
        )

    @classmethod
    def from_model_prediction(
        cls,
        pred: ModelPrediction,
        res_mask: torch.Tensor,
    ) -> "SamplingStep":
        """
        Create a SamplingStep from a model prediction.
        """
        return cls.from_values(
            res_mask=res_mask,
            trans=pred[pbp.pred_trans],
            rotmats=pred[pbp.pred_rotmats],
            aatypes=pred[pbp.pred_aatypes],
            torsions=pred.get(pbp.pred_torsions, None),
            logits=pred[pbp.pred_logits],
        )


@dataclass
class SamplingTrajectory:
    """
    A trajectory of inference sampling steps.

    Can be used for both model predictions over time, and saving intermediate interpolated protein states.
    """

    num_batch: int
    num_res: int
    num_tokens: int
    steps: List[SamplingStep] = field(default_factory=list)
    check_dimensions: bool = True

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, index: int) -> SamplingStep:
        return self.steps[index]

    def append(self, step: SamplingStep):
        step.to_cpu()
        self.steps.append(step)

    def select_batch_idx(self, idx: torch.Tensor):
        """
        Select batch members by index `idx`.
        Returns a new SamplingTrajectory with the selected steps.
        """
        new_num_batch = idx.shape[0]
        new_steps = [step.select_batch_idx(idx) for step in self.steps]
        return SamplingTrajectory(
            num_batch=new_num_batch,
            num_res=self.num_res,
            num_tokens=self.num_tokens,
            steps=new_steps,
            check_dimensions=self.check_dimensions,
        )

    @property
    def num_steps(self):
        return len(self.steps)

    @property
    def structure(self) -> torch.Tensor:
        """
        Returns structure / backbone tensor [num_batch, num_steps, sample_length, 37, 3]
        """
        t = torch.stack([step.structure for step in self.steps], dim=0).transpose(0, 1)
        if self.check_dimensions:
            expected_shape = (self.num_batch, self.num_steps, self.num_res, 37, 3)
            assert (
                t.shape == expected_shape
            ), f"Unexpected structure shape {t.shape}, expected {expected_shape}"
        return t

    @property
    def amino_acids(self) -> torch.Tensor:
        """
        Returns amino acid types tensor [num_batch, num_steps, sample_length]
        """
        t = (
            torch.stack([step.aatypes for step in self.steps], dim=0)
            .transpose(0, 1)
            .long()
        )
        if self.check_dimensions:
            expected_shape = (self.num_batch, self.num_steps, self.num_res)
            assert (
                t.shape == expected_shape
            ), f"Unexpected amino_acids shape {t.shape}, expected {expected_shape}"
        return t

    @property
    def logits(self) -> Optional[torch.Tensor]:
        """
        Returns logits tensor if available [num_batch, num_steps, sample_length, num_tokens]
        Currently only available for model steps, not protein steps, since we don't flow for logits.
        """
        if self.steps[0].logits is None:
            return None
        t = torch.stack([step.logits for step in self.steps], dim=0).transpose(0, 1)
        if self.check_dimensions:
            expected_shape = (
                self.num_batch,
                self.num_steps,
                self.num_res,
                self.num_tokens,
            )
            assert (
                t.shape == expected_shape
            ), f"Unexpected logits shape {t.shape}, expected {expected_shape}"
        return t
