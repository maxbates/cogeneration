# Adapted from Multiflow, Openfold, and AlphaFold2


# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Protein data type."""
import dataclasses
import os
import re
from typing import Any, Mapping, Optional, Union

import numpy as np
import numpy.typing as npt
from Bio.PDB.Chain import Chain

from cogeneration.data import residue_constants
from cogeneration.data.const import (
    ALPHANUMERIC,
    CHAIN_TO_INT,
    PDB_CHAIN_IDS,
    PDB_MAX_CHAINS,
)
from cogeneration.data.residue_constants import restypes_with_x

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class Protein:
    """Protein structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # 0-indexed number corresponding to the chain in the protein that this residue
    # belongs to.
    chain_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    def __post_init__(self):
        if len(np.unique(self.chain_index)) > PDB_MAX_CHAINS:
            raise ValueError(
                f"Cannot build an instance with more than {PDB_MAX_CHAINS} chains "
                "because these cannot be written to PDB format."
            )


def chain_str_to_int(chain_str: str) -> int:
    chain_int = 0
    if len(chain_str) == 1:
        return CHAIN_TO_INT[chain_str]
    for i, chain_char in enumerate(chain_str):
        chain_int += CHAIN_TO_INT[chain_char] + (i * len(ALPHANUMERIC))
    return chain_int


def process_chain(chain: Chain, chain_id: int) -> Protein:
    """Convert a PDB chain object into a AlphaFold Protein instance.

    Forked from alphafold.common.protein.py `_from_bio_structure()`

    WARNING: All non-standard residue types will be converted into UNK. All
        non-standard atoms will be ignored.

    Took out lines 94-97 which don't allow insertions in the PDB.
    Sabdab uses insertions for the chothia numbering so we need to allow them.

    Took out lines 110-112 since that would mess up CDR numbering.

    Args:
        chain: Instance of Biopython's chain class.

    Returns:
        Protein object with protein features.
    """
    assert isinstance(
        chain_id, int
    ), "chain_id must be an int, use `chain_str_to_int()`"

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    b_factors = []
    chain_ids = []
    for res in chain:
        # AF2 behavior is to not tolerate insertion codes.
        # We re-number the residues during processing, so these aren't deal breakers.
        # if res.id[2] != " ":
        #     raise ValueError(
        #         f"PDB/mmCIF contains an insertion code at chain {chain.id} and"
        #         f" residue index {res.id[1]}. These are not supported."
        #     )

        res_shortname = residue_constants.restype_3to1.get(res.resname, "X")
        restype_idx = residue_constants.restype_order.get(
            res_shortname, residue_constants.restype_num
        )
        pos = np.zeros((residue_constants.atom_type_num, 3))
        mask = np.zeros((residue_constants.atom_type_num,))
        res_b_factors = np.zeros((residue_constants.atom_type_num,))
        for atom in res:
            if atom.name not in residue_constants.atom_types:
                continue
            pos[residue_constants.atom_order[atom.name]] = atom.coord
            mask[residue_constants.atom_order[atom.name]] = 1.0
            res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor

        # TODO consider copying AF2 behavior for unknown atoms.
        #   Leads to e.g. metalic atoms being dropped.
        # # (note: missing in public FrameFlow, but it does [sometimes?] skip empty res...)
        # if np.sum(mask) < 0.5:
        #     # If no known atom positions are reported for the residue then skip it.
        #     continue

        aatype.append(restype_idx)
        atom_positions.append(pos)
        atom_mask.append(mask)
        residue_index.append(res.id[1])
        b_factors.append(res_b_factors)
        chain_ids.append(chain_id)

    # Unlike AF2, we expect to receive a valid int chain_id, and just pass it through.
    # We concat chains explicitly in caller after processing chains.

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=np.array(chain_ids),
        b_factors=np.array(b_factors),
    )


def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
    chain_end = "TER"
    return (
        f"{chain_end:<6}{atom_index:>5}      {end_resname:>3} "
        f"{chain_name:>1}{residue_index:>4}"
    )


def prot_to_pdb(prot: Protein, model=1, add_end=True) -> str:
    """Converts a `Protein` instance to a PDB string.

    Args:
      prot: The protein to convert to PDB.

    Returns:
      PDB string.
    """
    res_1to3 = lambda r: residue_constants.restype_1to3.get(
        restypes_with_x[int(r)], "UNK"
    )
    atom_types = residue_constants.atom_types

    pdb_lines = []

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(int)
    chain_index = prot.chain_index.astype(int)
    b_factors = prot.b_factors

    if np.any(aatype > residue_constants.restype_num):
        raise ValueError("Invalid aatypes.")

    # Construct a mapping from chain integer indices to chain ID strings.
    chain_ids = {}
    for i in np.unique(chain_index):  # np.unique gives sorted output.
        if i >= PDB_MAX_CHAINS:
            raise ValueError(
                f"The PDB format supports at most {PDB_MAX_CHAINS} chains."
            )
        chain_ids[i] = PDB_CHAIN_IDS[i]

    pdb_lines.append(f"MODEL     {model}")
    atom_index = 1
    last_chain_index = chain_index[0]
    # Add all atom sites.
    for i in range(aatype.shape[0]):
        # Close the previous chain if in a multichain PDB.
        if last_chain_index != chain_index[i]:
            pdb_lines.append(
                _chain_end(
                    atom_index,
                    res_1to3(aatype[i - 1]),
                    chain_ids[chain_index[i - 1]],
                    residue_index[i - 1],
                )
            )
            last_chain_index = chain_index[i]
            atom_index += 1  # Atom index increases at the TER symbol.

        res_name_3 = res_1to3(aatype[i])
        for atom_name, pos, mask, b_factor in zip(
            atom_types, atom_positions[i], atom_mask[i], b_factors[i]
        ):
            if mask < 0.5:
                continue

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            element = atom_name[0]  # Protein supports only C, N, O, S, this works.
            charge = ""
            # PDB is a columnar format, every space matters here!
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_ids[chain_index[i]]:>1}"
                f"{residue_index[i]:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1

    # Close the final chain.
    pdb_lines.append(
        _chain_end(
            atom_index,
            res_1to3(aatype[-1]),
            chain_ids[chain_index[-1]],
            residue_index[-1],
        )
    )
    pdb_lines.append("ENDMDL")
    if add_end:
        pdb_lines.append("END")

    # Pad all lines to 80 characters.
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return "\n".join(pdb_lines) + "\n"  # Add terminating newline.


def ideal_atom_mask(prot: Protein) -> np.ndarray:
    """Computes an ideal atom mask.

    `Protein.atom_mask` typically is defined according to the atoms that are
    reported in the PDB. This function computes a mask according to heavy atoms
    that should be present in the given sequence of amino acids.

    Args:
      prot: `Protein` whose fields are `numpy.ndarray` objects.

    Returns:
      An ideal atom mask.
    """
    return residue_constants.STANDARD_ATOM_MASK[prot.aatype]


def prot_from_atom37(
    atom37: npt.NDArray,
    atom37_mask: npt.NDArray,
    aatype: Optional[npt.NDArray] = None,
    chain_idx: Optional[npt.NDArray] = None,
    res_idx: Optional[npt.NDArray] = None,
    b_factors: Optional[npt.NDArray] = None,
):
    assert atom37.ndim == 3
    assert atom37.shape[-1] == 3
    assert atom37.shape[-2] == 37
    n = atom37.shape[0]

    if aatype is None:
        aatype = np.zeros(n, dtype=int)
    if res_idx is None:
        res_idx = np.arange(1, n + 1)
    if chain_idx is None:
        chain_idx = np.ones(n)
    if b_factors is None:
        b_factors = np.zeros([n, 37])

    return Protein(
        atom_positions=atom37,
        atom_mask=atom37_mask,
        aatype=aatype,
        residue_index=res_idx,
        chain_index=chain_idx,
        b_factors=b_factors,
    )


def write_prot_to_pdb(
    prot_pos: npt.NDArray,  # (N, 37, 3) or (T, N, 37, 3) if trajectory
    file_path: str,
    aatype: Optional[npt.NDArray] = None,
    chain_idx: Optional[npt.NDArray] = None,
    res_idx: Optional[npt.NDArray] = None,
    b_factors: Optional[npt.NDArray] = None,
    overwrite=False,
    no_indexing=False,
    backbone_only: bool = False,
) -> str:
    """
    Writes a protein (3D tensor) or a trajectory (N steps, 3D tensor) to a PDB file.
    The positions should be the full atom37 representation
    """
    if overwrite:
        max_existing_idx = 0
    else:
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path).strip(".pdb")

        try:
            existing_files = [x for x in os.listdir(file_dir) if file_name in x]
            max_existing_idx = max(
                [
                    int(re.findall(r"_(\d+).pdb", x)[0])
                    for x in existing_files
                    if re.findall(r"_(\d+).pdb", x)
                    if re.findall(r"_(\d+).pdb", x)
                ]
                + [0]
            )
        except FileNotFoundError:
            max_existing_idx = 0
    if not no_indexing:
        save_path = file_path.replace(".pdb", "") + f"_{max_existing_idx+1}.pdb"
    else:
        save_path = file_path

    if aatype is not None:
        assert aatype.ndim == prot_pos.ndim - 2

    with open(save_path, "w") as f:
        if prot_pos.ndim == 4:
            # Trajectory
            for t, pos37 in enumerate(prot_pos):
                atom37_mask = np.sum(np.abs(pos37), axis=-1) > 1e-7  # (N, 37)
                if backbone_only:
                    atom37_mask[:, 3:] = False

                prot = prot_from_atom37(
                    pos37,
                    atom37_mask,
                    aatype=aatype[t],
                    chain_idx=chain_idx,
                    res_idx=res_idx,
                    b_factors=b_factors,
                )
                pdb_prot = prot_to_pdb(prot, model=t + 1, add_end=False)
                f.write(pdb_prot)
        elif prot_pos.ndim == 3:
            # Single frame
            atom37_mask = np.sum(np.abs(prot_pos), axis=-1) > 1e-7  # (N, 37)
            if backbone_only:
                atom37_mask[:, 3:] = False

            prot = prot_from_atom37(
                prot_pos,
                atom37_mask,
                aatype=aatype,
                chain_idx=chain_idx,
                res_idx=res_idx,
                b_factors=b_factors,
            )
            pdb_prot = prot_to_pdb(prot, model=1, add_end=False)
            f.write(pdb_prot)
        else:
            raise ValueError(f"Invalid positions shape {prot_pos.shape}")
        f.write("END")

    return save_path
