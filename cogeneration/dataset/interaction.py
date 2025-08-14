from dataclasses import dataclass
from enum import Enum
from itertools import combinations, product
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Structure import Structure
from numpy import typing as npt
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from cogeneration.data.all_atom import atom37_from_trans_rot
from cogeneration.data.const import INT_TO_CHAIN
from cogeneration.data.protein import chain_str_to_int
from cogeneration.data.residue_constants import (
    atom_order,
    ligands_excluded,
    metal_types,
    nucleotide_resnames,
    unk_restype_index,
)
from cogeneration.type.batch import BatchFeatures, BatchProp
from cogeneration.type.dataset import ChainFeatures
from cogeneration.type.dataset import DatasetProteinColumn as dpc
from cogeneration.type.dataset import MetadataColumn as mc
from cogeneration.type.dataset import MetadataCSVRow

# clash / interaction constants
DIST_CLASH_ATOM_ANGSTROMS = 1.8
DIST_CLASH_BACKBONE_ANGSTROMS = 2.3
DIST_INTERACTION_ATOM = 4.5
DIST_INTERACTION_BACKBONE = 6.5
FRAC_RES_FOR_CLASH = 0.1
RESIDUE_CONTACT_THRESHOLD = 3  # interacts with >=3 distinct residues

# non-residue molecule constants
LARGE_MOLECULE_ATOM_THRESHOLD = 10  # DNA, fatty acids, small peptides etc.

# ComponentKey for atom groupings
ComponentKey = Tuple[int, Tuple[str, int, str]]  # (chain_id, (hetflag, resseq, icode))


def is_solution_resname(resname: str) -> bool:
    """Return True if residue name is considered solvent/buffer and should be excluded."""
    return resname.strip().upper() in ligands_excluded


def is_nucleotide_resname(resname: str) -> bool:
    """Return True if residue name represents a DNA/RNA nucleotide (3-letter code)."""
    return resname.strip().upper() in nucleotide_resnames


@dataclass(frozen=True)
class ChainAtom:
    """
    struct for atoms in a (non-residue or residue) chain

    note atoms don't know about their containing residue/molecule,
    so can't safely know if they are solvent, nucleic, etc.
    """

    chain_id: int
    res_index: int
    atom_index: int
    atom_name: str
    atom_element: str
    coord: npt.NDArray
    # Unique residue identifier from PDB (hetflag, resseq, icode)
    res_uid: Tuple[str, int, str]

    def __post_init__(self):
        # ensure clean names
        assert self.atom_name == self.atom_name.strip().upper()
        assert self.atom_element == self.atom_element.strip().upper()

    @property
    def is_metal(self) -> bool:
        return self.atom_element in metal_types

    def __hash__(self):
        return hash(
            (
                self.chain_id,
                self.res_index,
                self.atom_index,
                self.atom_name,
                self.atom_element,
                self.res_uid[0],
                self.res_uid[1],
                self.res_uid[2],
            )
        )

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, ChainAtom):
            return NotImplemented
        # Compare identity fields exactly and coordinates approximately
        return (
            self.chain_id == other.chain_id
            and self.res_index == other.res_index
            and self.atom_index == other.atom_index
            and self.atom_name == other.atom_name
            and self.atom_element == other.atom_element
            and self.res_uid == other.res_uid
            and np.allclose(self.coord, other.coord, atol=1e-2)
        )


@dataclass(frozen=True)
class Interaction:
    """struct for interactions between two chains"""

    chain_id: int
    chain_res_index: int
    chain_atom_idx: int
    other_id: int
    other_res_index: int
    other_atom_idx: int
    dist_ang: float

    @property
    def is_backbone(self) -> bool:
        return self.chain_atom_idx < 3 and self.other_atom_idx < 3

    @property
    def is_clash(self) -> bool:
        if self.is_backbone:
            return self.dist_ang < DIST_CLASH_BACKBONE_ANGSTROMS
        return self.dist_ang < DIST_CLASH_ATOM_ANGSTROMS


@dataclass
class ChainClash:
    """struct for aggregating clashes between two chains"""

    chain_id: int
    other_id: int
    chain_num_res: int
    residue_pairs: Set[Tuple[int, int]]  # (chain_res_index, other_res_index)

    @property
    def percent_residues_clash(self) -> float:
        return len(self.residue_pairs) / self.chain_num_res

    @property
    def is_clash(self) -> bool:
        # check if clash meets criteria, i.e. percent of residues exceeds threshold
        return self.percent_residues_clash >= FRAC_RES_FOR_CLASH


@dataclass
class MultimerInteractions:
    chain_feats: ChainFeatures
    interactions: List[Interaction]

    @staticmethod
    def find_chain_interactions(
        complex_feats: ChainFeatures,
        chain_i_id: int,
        chain_j_id: int,
        backbone_only: bool = False,
        only_standard_residues: bool = True,
        min_dist: float = 1e-3,  # >0 to ignore self
        max_dist: float = DIST_INTERACTION_ATOM,
    ) -> List[Interaction]:
        """
        KD-tree neighbor search of atoms between two chains.
        """
        # (n, 37, 3)
        chain_i_atoms = complex_feats[dpc.atom_positions][
            complex_feats[dpc.chain_index] == chain_i_id
        ]
        chain_j_atoms = complex_feats[dpc.atom_positions][
            complex_feats[dpc.chain_index] == chain_j_id
        ]
        if chain_i_atoms.shape[0] == 0 or chain_j_atoms.shape[0] == 0:
            return []

        if backbone_only:
            chain_i_atoms = chain_i_atoms[:, :3]
            chain_j_atoms = chain_j_atoms[:, :3]
        else:
            # atom14 to limit runtime
            chain_i_atoms = chain_i_atoms[:, :14]
            chain_j_atoms = chain_j_atoms[:, :14]

        # limit to interactions between standard residues
        valid_res_i = (
            complex_feats[dpc.aatype][complex_feats[dpc.chain_index] == chain_i_id]
            != unk_restype_index
        )
        valid_res_j = (
            complex_feats[dpc.aatype][complex_feats[dpc.chain_index] == chain_j_id]
            != unk_restype_index
        )

        # capture number of atoms for determining positions after flattening
        i_num_res, i_num_atoms, _ = chain_i_atoms.shape
        j_num_res, j_num_atoms, _ = chain_j_atoms.shape

        # flatten coords for KDtree
        chain_i_atoms = chain_i_atoms.reshape(-1, 3)
        chain_j_atoms = chain_j_atoms.reshape(-1, 3)

        # build KDtree on chain_j
        # query all points in chain_i within max_dist
        neighbors = cKDTree(chain_j_atoms).query_ball_point(x=chain_i_atoms, r=max_dist)

        interactions: List[Interaction] = []
        for idx_i, js in enumerate(neighbors):
            i_pos = chain_i_atoms[idx_i]
            for idx_j in js:
                j_pos = chain_j_atoms[idx_j]
                dist = np.linalg.norm(i_pos - j_pos)
                if dist < min_dist:
                    continue

                # recover residue+atom indices
                res_i, atm_i = divmod(idx_i, i_num_atoms)
                res_j, atm_j = divmod(idx_j, j_num_atoms)

                # skip interactions with non-standard residues
                if only_standard_residues:
                    if not valid_res_i[res_i] or not valid_res_j[res_j]:
                        continue

                interactions.append(
                    Interaction(
                        chain_id=chain_i_id,
                        chain_res_index=res_i,
                        chain_atom_idx=atm_i,
                        other_id=chain_j_id,
                        other_res_index=res_j,
                        other_atom_idx=atm_j,
                        dist_ang=float(dist),
                    )
                )

        return interactions

    def potential_chain_clashes(self, backbone_only: bool = True) -> List[ChainClash]:
        """Determine potential ChainClashes between chains. Check `ChainClash.is_clash`"""
        potential_clashes = {}
        for interaction in self.interactions:
            if backbone_only and not interaction.is_backbone:
                continue
            if interaction.is_clash:
                key = (interaction.chain_id, interaction.other_id)
                if key not in potential_clashes:
                    chain_mask = (
                        self.chain_feats[dpc.chain_index] == interaction.chain_id
                    )
                    chain_atoms = self.chain_feats[dpc.atom_positions][chain_mask]

                    potential_clashes[key] = ChainClash(
                        chain_id=interaction.chain_id,
                        other_id=interaction.other_id,
                        chain_num_res=chain_atoms.shape[0],
                        residue_pairs=set(),
                    )
                potential_clashes[key].residue_pairs.add(
                    (interaction.chain_res_index, interaction.other_res_index)
                )

        return list(potential_clashes.values())

    @property
    def unique_backbone_interactions(self) -> List[Interaction]:
        """Return unique backbone residue interactions (i.e. one per res-res pair)"""
        mapping = {}
        for interaction in self.interactions:
            if interaction.is_backbone:
                key = (
                    interaction.chain_id,
                    interaction.chain_res_index,
                    interaction.other_id,
                    interaction.other_res_index,
                )
                if key in mapping:
                    continue
                # keep the first one
                mapping[key] = interaction
        return list(mapping.values())

    @property
    def atom_interactions(self) -> List[Interaction]:
        return [
            interaction
            for interaction in self.interactions
            if not interaction.is_backbone
        ]

    @property
    def backbone_res_interacting(self) -> Set[Tuple[int, int]]:
        """
        Return a set of tuples (chain_id, res_index) for all residues interacting with
        the backbone of another chain.
        """
        return {
            (interaction.chain_id, interaction.chain_res_index)
            for interaction in self.unique_backbone_interactions
        }

    def grouped_backbone_interactions(self) -> Dict[Tuple[int, int], Set[Interaction]]:
        """
        Group backbone interactions by residue.
        Returns {(chain_id, res_index): set(interactions)}
        """
        grouped_interactions = {}
        for interaction in self.unique_backbone_interactions:
            if interaction.is_clash:
                continue
            if not interaction.is_backbone:
                continue
            if interaction.chain_id > interaction.other_id:
                continue

            key = (interaction.chain_id, interaction.chain_res_index)
            res_interactions = grouped_interactions.get(key, set())
            res_interactions.add(interaction)
            grouped_interactions[key] = res_interactions

        return grouped_interactions

    def residue_backbone_interaction_counts(self) -> Dict[Tuple[int, int], int]:
        """
        Count unique backbone residue-residue interactions for ALL interacting residues.

        Returns a mapping from (chain_id, res_index) -> number of unique interacting partners.

        Notes:
        - Excludes clashes
        - Counts unique residue partners (one per residue-residue pair)
        - Includes residues on both sides of an interacting pair (no early-chain bias)
        """
        counts: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}
        for interaction in self.unique_backbone_interactions:
            if interaction.is_clash:
                continue
            if not interaction.is_backbone:
                continue

            a_key = (interaction.chain_id, interaction.chain_res_index)
            b_key = (interaction.other_id, interaction.other_res_index)

            # Track unique partners for each residue
            counts.setdefault(a_key, set()).add(b_key)
            counts.setdefault(b_key, set()).add(a_key)

        # Convert partner sets to counts
        return {key: len(partners) for key, partners in counts.items()}

    def serialize_hot_spots(self) -> str:
        """
        Serialize hot spot residues (backbone interactions) to a string.
        Format: "<chain_id><res_index>:<num_interactions>,..."
        """
        return ",".join(
            f"{INT_TO_CHAIN[chain_id]}{res_index}:{len(interaction_set)}"
            for (
                chain_id,
                res_index,
            ), interaction_set in self.grouped_backbone_interactions().items()
        )

    def serialize_chain_interactions(self) -> str:
        """
        Serialize chain interactions to a string.
        Format: "<chain_a>:<chain_b>:<num_res_chain_a>:<num_res_chain_b>,...."
        """
        # count {(chain_a, chain_b): (set(res_a), set(res_b))}
        chain_chain_counts = {}
        for interaction in self.unique_backbone_interactions:
            chain_a = interaction.chain_id
            chain_b = interaction.other_id
            res_a = interaction.chain_res_index
            res_b = interaction.other_res_index

            # ensure chain_a < chain_b for consistent ordering
            if chain_a > chain_b:
                chain_a, chain_b = chain_b, chain_a
            chain_key = (chain_a, chain_b)

            if chain_key not in chain_chain_counts:
                chain_chain_counts[chain_key] = (set(), set())
            chain_chain_counts[chain_key][0].add(res_a)
            chain_chain_counts[chain_key][1].add(res_b)

        # format as "<chain_a>:<chain_b>:<num_bb_res_xing_a>:<num_bb_res_xing_b>,...."
        return ",".join(
            f"{INT_TO_CHAIN[chain_a]}:{INT_TO_CHAIN[chain_b]}:{len(set_a)}:{len(set_b)}"
            for (chain_a, chain_b), (set_a, set_b) in chain_chain_counts.items()
        )

    @classmethod
    def from_chain_feats(
        cls,
        complex_feats: ChainFeatures,
        metadata: MetadataCSVRow,
    ):
        chain_idxs = np.unique(complex_feats[dpc.chain_index])
        chain_pairs = [(i, j) for i, j in product(chain_idxs, chain_idxs) if i < j]

        all_interactions = []
        for i, j in chain_pairs:
            # Backbone interactions
            all_interactions.extend(
                MultimerInteractions.find_chain_interactions(
                    complex_feats=complex_feats,
                    chain_i_id=i,
                    chain_j_id=j,
                    backbone_only=True,
                    only_standard_residues=True,
                    min_dist=DIST_INTERACTION_ATOM,
                    max_dist=DIST_INTERACTION_BACKBONE,
                )
            )

            # All-atom interactions
            # Too slow, disabled
            # all_interactions.extend(
            #     MultimerInteractions.find_chain_interactions(
            #         complex_feats=complex_feats,
            #         chain_i_id=i,
            #         chain_j_id=j,
            #         backbone_only=False,
            #         only_standard_residues=True,
            #         max_dist=DIST_INTERACTION_ATOM,
            #     )
            # )

        return cls(
            chain_feats=complex_feats,
            interactions=all_interactions,
        )

    @classmethod
    def from_batch_feats(cls, feats: BatchFeatures):
        """
        Calculate MultimerInteractions from single item batch features, i.e. without batch dimension
        Probably should check if multimer before calling, but not
        """
        assert BatchProp.chain_idx in feats
        assert BatchProp.trans_1 in feats
        assert (
            feats[BatchProp.chain_idx].ndim == 1
        ), f"Expected (N,) tensor, got {feats[BatchProp.chain_idx].shape}"
        assert (
            feats[BatchProp.trans_1].ndim == 2
        ), f"Expected (N, 3) tensor, got {feats[BatchProp.trans_1].shape}"

        chain_feats: ChainFeatures = {
            dpc.atom_positions: atom37_from_trans_rot(
                trans=feats[BatchProp.trans_1].unsqueeze(0).detach().cpu(),
                rots=feats[BatchProp.rotmats_1].unsqueeze(0).detach().cpu(),
                torsions=feats[BatchProp.torsions_1].unsqueeze(0).detach().cpu(),
                aatype=feats[BatchProp.aatypes_1].unsqueeze(0).detach().cpu(),
                res_mask=feats[BatchProp.res_mask].unsqueeze(0).detach().cpu(),
            )
            .squeeze(0)
            .numpy(),
            dpc.chain_index: feats[BatchProp.chain_idx].detach().cpu().numpy(),
            dpc.aatype: feats[BatchProp.aatypes_1].detach().cpu().numpy(),
        }

        return cls.from_chain_feats(
            complex_feats=chain_feats,
            metadata=None,
        )

    def update_metadata(
        self,
        metadata: MetadataCSVRow,
    ):
        metadata[mc.chain_interactions] = self.serialize_chain_interactions()
        metadata[mc.num_backbone_interactions] = len(self.unique_backbone_interactions)
        metadata[mc.num_backbone_res_interacting] = len(self.backbone_res_interacting)
        # metadata[dc.num_atom_interactions] = len(self.atom_interactions)

        potential_clashes = self.potential_chain_clashes(backbone_only=True)
        # serialize chain clashes
        metadata[mc.chain_clashes] = ",".join(
            f"{INT_TO_CHAIN[chain_clash.chain_id]}:{INT_TO_CHAIN[chain_clash.other_id]}:{len(chain_clash.residue_pairs)}"
            for chain_clash in potential_clashes
        )
        metadata[mc.num_chains_clashing] = len(
            set(clash.chain_id for clash in potential_clashes if clash.is_clash)
        )


@dataclass(frozen=True)
class AtomInteraction(Interaction):
    """struct for interactions between a ChainAtom and an atom in another chain"""

    atom: ChainAtom

    @property
    def is_backbone(self) -> bool:
        return False


class NonResidueEntityType(Enum):
    METAL_ATOM = "metal_atom"
    SMALL_MOLECULE = "small_molecule"
    NUCLEIC_ACID_POLYMER = "nucleic_acid_polymer"
    OTHER_POLYMER = "other_polymer"


@dataclass
class NonProteinEntity:
    """Collection of atoms that belong to a non-residue component (e.g. metal ion, small molecule, DNA, RNA)"""

    entity_type: NonResidueEntityType
    components: Set[Tuple[int, Tuple[str, int, str]]]
    atoms: List[ChainAtom]
    chain_ids: Set[int]

    @property
    def num_atoms(self) -> int:
        return len(self.atoms)


@dataclass
class NonResidueInteractions:
    """
    Class for determining interactions with metals and non-residue chains (DNA, single atoms, polymers)
    """

    structure: Structure
    complex_feats: ChainFeatures

    def __post_init__(self):
        """
        Initialize non-protein entity detection and interaction counting.

        Steps overview:
        1) Collect all atoms that belong to non-protein residues (including ligands, nucleotides, and ions),
           excluding solution molecules (e.g., water, buffers). This captures ligands even if they share a
           PDB chain with protein residues.
        2) Index these atoms into "components" using a stable PDB residue identifier per chain:
           component_key := (chain_id, res_uid), where res_uid = (hetflag, resseq, icode). This enables
           counting per small-molecule residue and detection of nucleic-acid strands.
        3) Classify components into entities: metal ions (atoms), small molecules (components),
           nucleic acids (grouped per chain), and other polymers (grouped per chain).
        4) Build entity-residue contacts to count entity-level interactions (>= 3 unique residues).
        """

        # 1) Collect atoms from non-residue components (excluding solution molecules).
        self.all_non_res_chain_atoms = NonResidueInteractions.get_all_chain_atoms(
            structure=self.structure,
            chain_ids=None,
            filter_solution=True,
            only_non_residues=True,
        )

        # Chain IDs that contain at least one non-residue component
        self.chains_containing_nonres = set(
            atom.chain_id for atom in self.all_non_res_chain_atoms
        )

        # count metal atoms
        self.metal_atoms = self.get_all_metal_atoms(self.structure)

        # count solution atoms
        self.num_solution_molecules = self.count_solution_molecules(self.structure)

        # Find interactions between non-residue chain atoms and residue chains
        self.non_res_interactions = NonResidueInteractions.get_residue_interactions(
            atoms=self.all_non_res_chain_atoms,
            complex_feats=self.complex_feats,
        )

        # Build simplified entity view for external consumption (index entities and classifies components)
        self.entities = NonResidueInteractions.build_entities(
            structure=self.structure,
            atoms=self.all_non_res_chain_atoms,
        )

        # aggregate ChainAtom -> unique residue interactions
        # { ChainAtom:  { res_chain_id: set(res_index) } }
        self.atom_res_contacts: Dict[ChainAtom, Dict[int, Set[int]]] = {}
        for inter in self.non_res_interactions:
            self.atom_res_contacts.setdefault(inter.atom, {})
            self.atom_res_contacts[inter.atom].setdefault(inter.other_id, set())
            self.atom_res_contacts[inter.atom][inter.other_id].add(
                inter.other_res_index
            )

        # aggregate chain_id -> unique residue interactions
        # { non_res_chain_id: { res_chain_id: set(res_index) } }
        # TODO - ensure accounts for entities in chains that contain residues
        self.chain_res_contacts: Dict[int, Dict[int, Set[int]]] = {
            cid: {} for cid in self.chains_containing_nonres
        }
        for atom, res_contacts in self.atom_res_contacts.items():
            for other_id, res_idx_set in res_contacts.items():
                self.chain_res_contacts[atom.chain_id].setdefault(
                    other_id, set()
                ).update(res_idx_set)

    @property
    def num_non_residue_chains(self) -> int:
        """Number of non-atom entities, i.e. small molecules, nucleic acids, polymers"""
        return len([e for e in self.entities if len(e.atoms) > 1])

    @property
    def num_small_molecules(self) -> int:
        return len(self.entities_by_type(NonResidueEntityType.SMALL_MOLECULE))

    @property
    def num_nucleic_acid_polymers(self) -> int:
        return len(self.entities_by_type(NonResidueEntityType.NUCLEIC_ACID_POLYMER))

    @property
    def num_other_polymers(self) -> int:
        """Count OTHER polymers (not including nucleic acids)"""
        return len(self.entities_by_type(NonResidueEntityType.OTHER_POLYMER))

    @property
    def num_single_atom_chains(self) -> int:
        chain_counts: Dict[int, int] = {}
        for atom in self.all_non_res_chain_atoms:
            chain_counts[atom.chain_id] = chain_counts.get(atom.chain_id, 0) + 1
        return sum(1 for count in chain_counts.values() if count == 1)

    def count_solution_molecules(self, structure: Structure) -> int:
        """
        Count water / solution molecules in the structure.
        """
        count = 0

        for chain in structure.get_chains():
            for res in chain:
                # count water / solution atoms
                if is_solution_resname(res.get_resname()):
                    count += 1

        return count

    @staticmethod
    def get_all_chain_atoms(
        structure: Structure,
        chain_ids: Optional[Iterable[int]],
        filter_solution: bool = True,
        only_non_residues: bool = False,
    ) -> List[ChainAtom]:
        """
        Get all atoms in `chain_ids`, e.g. atoms in non-valid chains.
        Optionally filter to only non-residue atoms.
        """
        struct_chains = {
            chain_str_to_int(chain.id.upper()): chain
            for chain in structure.get_chains()
        }

        # optionally limit to `chain_ids`
        if chain_ids is not None:
            struct_chains = {
                chain_id: struct_chains[chain_id]
                for chain_id in struct_chains.keys()
                if chain_id in chain_ids
            }

        # collect all atoms as ChainAtoms
        # `process_chain` does not tolerate atoms not in atom_type, so it is avoided.
        all_non_res_chain_atoms: List[ChainAtom] = []
        for chain_id, chain in struct_chains.items():
            for res_idx, res in enumerate(chain):
                # drop water / solution atoms
                if filter_solution and is_solution_resname(res.get_resname()):
                    continue

                # If requested, only include non-amino-acid residues (hetero, nucleotides, metals, etc.)
                if only_non_residues and is_aa(res, standard=True):
                    continue

                # stable residue identifiers from PDB
                hetflag, resseq, icode = res.get_id()

                for atom_idx, atom in enumerate(res.get_atoms()):
                    # prefer PDB using PDB residue index for stability / correctness
                    pdb_res_index = int(resseq) if isinstance(resseq, int) else res_idx

                    all_non_res_chain_atoms.append(
                        ChainAtom(
                            chain_id=chain_id,
                            res_index=pdb_res_index,
                            atom_index=atom_idx,
                            atom_name=atom.name.strip().upper(),
                            atom_element=atom.element.strip().upper(),
                            coord=atom.coord,
                            res_uid=(hetflag, pdb_res_index, icode),
                        )
                    )

        return all_non_res_chain_atoms

    @staticmethod
    def get_all_metal_atoms(
        structure, chain_ids: Optional[List[int]] = None
    ) -> List[ChainAtom]:
        """
        Get all metal atoms in the structure, optionally filtered to `chain_ids`
        """
        return [
            atom
            for atom in NonResidueInteractions.get_all_chain_atoms(
                structure, chain_ids=chain_ids
            )
            if atom.is_metal
        ]

    @staticmethod
    def get_residue_interactions(
        atoms: List[ChainAtom],
        complex_feats: ChainFeatures,
    ) -> List[AtomInteraction]:
        """
        Find interactions between atoms and residue chains
        """
        non_res_chain_interactions: List[AtomInteraction] = []

        if len(atoms) > 0:
            atom_coords = np.vstack([h.coord for h in atoms])  # (H, 3)
            res_coords = complex_feats[dpc.atom_positions].reshape(-1, 3)  # (N * 37, 3)
            res_valid = complex_feats[dpc.aatype] != unk_restype_index  # (N, )

            # No separate handling for clashes
            neighbors = cKDTree(res_coords).query_ball_point(
                atom_coords, r=DIST_INTERACTION_ATOM
            )
            for atom_i, res_set in enumerate(neighbors):
                atom = atoms[atom_i]

                for res_j in res_set:
                    dist_ang = float(np.linalg.norm(atom.coord - res_coords[res_j]))

                    # skip self (i.e. atoms in ~ same position)
                    if dist_ang < 1e-2:
                        continue

                    res_idx, res_atom_idx = divmod(res_j, 37)
                    res_chain_id = int(complex_feats[dpc.chain_index][res_idx])

                    # skip non-residue interactions i.e. aatype == UNK
                    if not res_valid[res_idx]:
                        continue

                    non_res_chain_interactions.append(
                        AtomInteraction(
                            chain_id=atom.chain_id,
                            # important to use atom res/atom idx:
                            # dropped solution atoms => atom_i != atom.res_index
                            chain_res_index=atom.res_index,
                            chain_atom_idx=atom.atom_index,
                            other_id=int(res_chain_id),
                            other_res_index=res_idx,
                            other_atom_idx=res_atom_idx,
                            dist_ang=dist_ang,
                            atom=atoms[atom_i],
                        )
                    )

        return non_res_chain_interactions

    @staticmethod
    def build_entities(
        structure: Structure, atoms: List[ChainAtom]
    ) -> List[NonProteinEntity]:
        """
        Group non-protein content into clean entities:
        - Metals: one entity per metal-only component
        - Small molecules: one entity per non-metal, non-nucleotide component
        - Nucleic acids: one entity per chain, merging all nucleotide components
        - Other polymers: one entity per chain containing non-protein components not classified as nucleic acids
        """
        # Build components, i.e. atoms associated with a (chain_id, res_uid)
        component_atoms: Dict[ComponentKey, List[ChainAtom]] = {}
        for atom in atoms:
            component_atoms.setdefault((atom.chain_id, atom.res_uid), []).append(atom)

        # Map components to resname
        component_resname: Dict[ComponentKey, str] = {}
        for chain in structure.get_chains():
            cid = chain_str_to_int(chain.id.upper())
            for res in chain:
                hetflag, resseq, icode = res.get_id()
                resname = res.get_resname().strip().upper()
                key: ComponentKey = (
                    cid,
                    (
                        hetflag,
                        int(resseq) if isinstance(resseq, int) else resseq,
                        icode,
                    ),
                )
                if key in component_atoms and not is_solution_resname(resname):
                    component_resname[key] = resname

        entities: List[NonProteinEntity] = []

        processed_components: Set[ComponentKey] = set()
        nucleotide_keys: Set[ComponentKey] = set()

        # First pass: classify per-component entities (metals and small molecules) and collect nucleotide keys
        for key, comp_atoms in component_atoms.items():
            if len(comp_atoms) == 0:
                continue
            resname = component_resname.get(key, "")
            if is_nucleotide_resname(resname):
                nucleotide_keys.add(key)
                continue
            if all(a.is_metal for a in comp_atoms):
                entities.append(
                    NonProteinEntity(
                        entity_type=NonResidueEntityType.METAL_ATOM,
                        components={key},
                        atoms=list(comp_atoms),
                        chain_ids={key[0]},
                    )
                )
                processed_components.add(key)
                continue
            # default to small molecule component (non-nucleotide, non-metal)
            entities.append(
                NonProteinEntity(
                    entity_type=NonResidueEntityType.SMALL_MOLECULE,
                    components={key},
                    atoms=list(comp_atoms),
                    chain_ids={key[0]},
                )
            )
            processed_components.add(key)

        # Second pass: nucleic acids grouped per chain (polymer entity)
        chain_to_na_components: Dict[int, List[ComponentKey]] = {}
        for key in nucleotide_keys:
            chain_to_na_components.setdefault(key[0], []).append(key)
        for chain_id, comp_keys in chain_to_na_components.items():
            chain_atoms: List[ChainAtom] = []
            for ck in comp_keys:
                chain_atoms.extend(component_atoms.get(ck, []))
            if len(chain_atoms) == 0:
                continue
            entities.append(
                NonProteinEntity(
                    entity_type=NonResidueEntityType.NUCLEIC_ACID_POLYMER,
                    components=set(comp_keys),
                    atoms=chain_atoms,
                    chain_ids={chain_id},
                )
            )
            processed_components.update(comp_keys)

        # Third pass: other polymers per chain (unclassified non-protein components that aren't metals, small molecules, or nucleic acids)
        all_component_keys = set(component_atoms.keys())
        leftover_by_chain: Dict[int, List[ComponentKey]] = {}
        for key in all_component_keys:
            if key in processed_components or key in nucleotide_keys:
                continue
            comp_atoms = component_atoms.get(key, [])
            if len(comp_atoms) == 0:
                continue
            leftover_by_chain.setdefault(key[0], []).append(key)

        for chain_id, comp_keys in leftover_by_chain.items():
            chain_atoms: List[ChainAtom] = []
            for ck in comp_keys:
                chain_atoms.extend(component_atoms.get(ck, []))
            if len(chain_atoms) == 0:
                continue
            entities.append(
                NonProteinEntity(
                    entity_type=NonResidueEntityType.OTHER_POLYMER,
                    components=set(comp_keys),
                    atoms=chain_atoms,
                    chain_ids={chain_id},
                )
            )

        return entities

    def entities_by_type(
        self, entity_type: NonResidueEntityType
    ) -> List[NonProteinEntity]:
        """Get entities by type"""
        return [e for e in self.entities if e.entity_type == entity_type]

    def _entity_residue_contacts(
        self, entity: NonProteinEntity
    ) -> Set[Tuple[int, int]]:
        """Get contacts for all atoms in entity"""
        contacts: Set[Tuple[int, int]] = set()
        for atom in entity.atoms:
            if atom in self.atom_res_contacts:
                for other_id, res_idx_set in self.atom_res_contacts[atom].items():
                    for res_idx in res_idx_set:
                        contacts.add((other_id, res_idx))
        return contacts

    @property
    def metals_interacting(self) -> List[ChainAtom]:
        """Return metal atoms with sufficient unique residue contacts (>= threshold)."""
        result: List[ChainAtom] = []
        for entity in self.entities_by_type(NonResidueEntityType.METAL_ATOM):
            contacts = self._entity_residue_contacts(entity)
            if len(contacts) >= RESIDUE_CONTACT_THRESHOLD:
                result.extend(entity.atoms)
        return result

    @property
    def num_macromolecule_interactions(self) -> int:
        """Count interacting macromolecule entities (entity-level count)."""
        count = 0
        for entity in self.entities:
            if len(entity.atoms) < LARGE_MOLECULE_ATOM_THRESHOLD:
                continue
            if len(self._entity_residue_contacts(entity)) >= RESIDUE_CONTACT_THRESHOLD:
                count += 1
        return count

    @property
    def num_small_molecule_interactions(self) -> int:
        """Count interacting small-molecule components using entities."""
        count = 0
        for entity in self.entities_by_type(NonResidueEntityType.SMALL_MOLECULE):
            if len(self._entity_residue_contacts(entity)) >= RESIDUE_CONTACT_THRESHOLD:
                count += 1
        return count

    @property
    def num_nucleic_acid_interactions(self) -> int:
        """Count interacting nucleic-acid polymer entities using entities."""
        count = 0
        for entity in self.entities_by_type(NonResidueEntityType.NUCLEIC_ACID_POLYMER):
            if len(self._entity_residue_contacts(entity)) >= RESIDUE_CONTACT_THRESHOLD:
                count += 1
        return count

    @property
    def mediated_chain_interactions(self) -> List[Tuple[int, int, int]]:
        """
        Find protein-protein interactions likely mediated by non-residue atoms.
        Returns tuples (non_res_chain_id, prot_chain_a, prot_chain_b).
        """
        mediated: List[Tuple[int, int, int]] = []

        # for each non-res chain, find protein chains with sufficient contacts
        for chain_id, prot_contacts in self.chain_res_contacts.items():
            if not chain_id in self.chains_containing_nonres:
                continue
            valid_prots = [
                prot_id
                for prot_id, res_set in prot_contacts.items()
                if len(res_set) >= RESIDUE_CONTACT_THRESHOLD
            ]
            # produce each unordered pair once, if 2+ chains interact with non-res chain
            for a, b in combinations(sorted(valid_prots), 2):
                mediated.append((chain_id, a, b))

        return mediated

    @classmethod
    def from_chain_feats(
        cls,
        complex_feats: ChainFeatures,  # match chain_ids + positions in `structure`
        structure: Structure,
    ):
        return cls(
            structure=structure,
            complex_feats=complex_feats,
        )

    def update_metadata(self, metadata: MetadataCSVRow):
        # tally non-residue counts
        metadata[mc.num_non_residue_chains] = self.num_non_residue_chains
        metadata[mc.num_single_atom_chains] = self.num_single_atom_chains
        metadata[mc.num_solution_molecules] = self.num_solution_molecules
        metadata[mc.num_metal_atoms] = len(self.metal_atoms)
        metadata[mc.num_small_molecules] = self.num_small_molecules
        metadata[mc.num_nucleic_acid_polymers] = self.num_nucleic_acid_polymers
        metadata[mc.num_other_polymers] = self.num_other_polymers

        # interacting components
        metadata[mc.num_metal_interactions] = len(self.metals_interacting)
        metadata[mc.num_macromolecule_interactions] = (
            self.num_macromolecule_interactions
        )
        metadata[mc.num_small_molecule_interactions] = (
            self.num_small_molecule_interactions
        )
        metadata[mc.num_nucleic_acid_interactions] = self.num_nucleic_acid_interactions

        # check for chain-chain interactions mediated by non-residues
        metadata[mc.num_mediated_interactions] = len(self.mediated_chain_interactions)
