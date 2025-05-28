from dataclasses import dataclass
from itertools import combinations, product
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Structure import Structure
from numpy import typing as npt
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from cogeneration.data.protein import chain_str_to_int
from cogeneration.data.residue_constants import (
    atom_order,
    metal_types,
    solutions,
    unk_restype_index,
)
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


@dataclass
class ChainAtom:
    """struct for atoms in a (non-residue or residue) chain"""

    chain_id: int
    res_index: int
    atom_index: int
    atom_name: str
    atom_element: str
    coord: npt.NDArray

    def __post_init__(self):
        # cleanup
        self.atom_name = self.atom_name.strip().upper()
        self.atom_element = self.atom_element.strip().upper()

    @property
    def is_metal(self) -> bool:
        return self.atom_element in metal_types

    @property
    def atom_type(self) -> Optional[int]:
        if self.is_metal:
            return None
        if self.atom_name not in atom_order:
            return None
        return atom_order[self.atom_name]

    def __hash__(self):
        return hash((self.chain_id, self.res_index, self.atom_index, self.atom_name))


@dataclass
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
        KD‑tree neighbor search of atoms between two chains.
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

    @property
    def chain_clashes(self) -> List[ChainClash]:
        """Determine ChainClashes between chains"""
        # aggregate clash interactions per chain interaction
        potential_clashes = {}
        for interaction in self.interactions:
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
                potential_clashes[key].num_clashes += 1
                potential_clashes[key].residue_pairs.add(
                    (interaction.chain_res_index, interaction.other_res_index)
                )

        # return a ChainClash if exceeds thresholds
        return [
            potential
            for potential in list(potential_clashes.values())
            if potential.percent_residues_clash >= FRAC_RES_FOR_CLASH
        ]

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
            f"{chain_a}:{chain_b}:{len(set_a)}:{len(set_b)}"
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
            # Too slow, disabled  TODO re-enable
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

    def update_metadata(
        self,
        metadata: MetadataCSVRow,
    ):
        metadata[mc.chain_interactions] = self.serialize_chain_interactions()
        metadata[mc.num_backbone_interactions] = len(self.unique_backbone_interactions)
        metadata[mc.num_backbone_res_interacting] = len(self.backbone_res_interacting)
        # metadata[dc.num_atom_interactions] = len(self.atom_interactions)

        metadata[mc.num_chains_clashing] = len(
            set(clash.chain_id for clash in self.chain_clashes)
        )


@dataclass
class AtomInteraction(Interaction):
    """struct for interactions between a ChainAtom and an atom in another chain"""

    atom: ChainAtom

    @property
    def is_backbone(self) -> bool:
        return False


@dataclass
class NonResidueInteractions:
    """
    Class for determining interactions with metals and non-residue chains (DNA, single atoms, polymers)
    """

    structure: Structure
    complex_feats: ChainFeatures

    def __post_init__(self):
        # Determine non-residue chains
        # valid chains are already present in `complex_feats`.
        all_chain_ids = [
            chain_str_to_int(chain.id.upper()) for chain in self.structure.get_chains()
        ]
        valid_chain_ids = set(self.complex_feats[dpc.chain_index])
        self._non_res_chain_ids = {
            cid for cid in all_chain_ids if cid not in valid_chain_ids
        }

        # count all atoms per non-residue chain
        self.all_non_res_chain_atoms = NonResidueInteractions.get_all_chain_atoms(
            structure=self.structure, chain_ids=self.non_res_chain_ids
        )

        # count solution atoms
        self.num_solution_molecules = self.count_solution_molecules(self.structure)

        # Find interactions between non-residue chain atoms and residue chains
        self.non_res_interactions = NonResidueInteractions.get_residue_interactions(
            atoms=self.all_non_res_chain_atoms,
            complex_feats=self.complex_feats,
        )

        # count atoms per chain
        self.non_res_chain_atom_counts = {}
        for atom in self.all_non_res_chain_atoms:
            self.non_res_chain_atom_counts.setdefault(atom.chain_id, 0)
            self.non_res_chain_atom_counts[atom.chain_id] += 1

        # count ChainAtom -> unique residue interactions
        # { ChainAtom:  { other_id: set(res_index) } }
        self.atom_res_contacts: Dict[ChainAtom, Dict[int, Set[int]]] = {}
        for inter in self.non_res_interactions:
            self.atom_res_contacts.setdefault(inter.atom, {})
            self.atom_res_contacts[inter.atom].setdefault(inter.other_id, set())
            self.atom_res_contacts[inter.atom][inter.other_id].add(
                inter.other_res_index
            )

        # count chain_id -> unique residue interactions
        # { non_res_chain_id: { other_id: set(res_index) } }
        self.chain_res_contacts: Dict[int, Dict[int, Set[int]]] = {
            cid: {} for cid in self.non_res_chain_ids
        }
        for atom, res_contacts in self.atom_res_contacts.items():
            for other_id, res_idx_set in res_contacts.items():
                self.chain_res_contacts[atom.chain_id].setdefault(
                    other_id, set()
                ).update(res_idx_set)

        # count metal interactions, includes metal in residue chains
        # keep separate, because includes metals in valid chains
        metal_atoms = self.get_all_metal_atoms(self.structure)
        metal_interactions = NonResidueInteractions.get_residue_interactions(
            atoms=metal_atoms,
            complex_feats=self.complex_feats,
        )
        self.metal_res_contacts: Dict[ChainAtom, Dict[int, Set[int]]] = {}
        for inter in metal_interactions:
            self.metal_res_contacts.setdefault(inter.atom, {})
            self.metal_res_contacts[inter.atom].setdefault(inter.other_id, set())
            self.metal_res_contacts[inter.atom][inter.other_id].add(
                inter.other_res_index
            )

    @property
    def non_res_chain_ids(self) -> Set[int]:
        return self._non_res_chain_ids

    @property
    def num_non_residue_chains(self) -> int:
        return len(self.non_res_chain_ids)

    @property
    def num_single_atom_chains(self) -> int:
        return sum(1 for count in self.non_res_chain_atom_counts.values() if count == 1)

    def count_solution_molecules(self, structure: Structure) -> int:
        """
        Count water / solution molecules in the structure.
        """
        count = 0

        for chain in structure.get_chains():
            for res in chain:
                # count water / solution atoms
                if res.get_resname().strip() in solutions:
                    count += 1

        return count

    @staticmethod
    def get_all_chain_atoms(
        structure: Structure,
        chain_ids: Optional[Iterable[int]],
        filter_solution: bool = True,
    ) -> List[ChainAtom]:
        """
        Get all atoms in `chain_ids`, e.g. atoms in non-valid chains
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
                if filter_solution and res.get_resname().strip() in solutions:
                    continue

                # allow residues to fall through, in case getting atoms for entire structure,
                # even though not expected for non-res chains.
                # Also continue with nucleotides, which also have res.id[0] == " ".
                # if res.id[0] == " ":
                #     if not is_aa(res, standard=True):
                #         pass

                for atom_idx, atom in enumerate(res.get_atoms()):
                    all_non_res_chain_atoms.append(
                        ChainAtom(
                            chain_id=chain_id,
                            res_index=res_idx,
                            atom_index=atom_idx,
                            atom_name=atom.name,
                            atom_element=atom.element,
                            coord=atom.coord,
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

    @property
    def metals_interacting(self) -> List[ChainAtom]:
        """return ChainAtoms for metals across all chains with sufficient residue contacts"""
        metal_atoms = self.get_all_metal_atoms(self.structure)
        result: List[ChainAtom] = []

        for metal_atom in metal_atoms:
            contacts = self.metal_res_contacts.get(metal_atom, {})
            res_interactions = sum(len(res_set) for res_set in contacts.values())
            if res_interactions >= RESIDUE_CONTACT_THRESHOLD:
                result.append(metal_atom)

        return result

    @property
    def macromolecule_chains_interacting(self) -> List[int]:
        """
        Return chain_ids for non-res chains with:
           - sufficient atoms to be a macromolecule
           - sufficient residue contacts
        """
        result: List[int] = []

        for cid in self.non_res_chain_ids:
            # filter to large molecules
            chain_atom_count = self.non_res_chain_atom_counts.get(cid, 0)
            if chain_atom_count < LARGE_MOLECULE_ATOM_THRESHOLD:
                continue

            # count unique residue interactions for atoms in the chain
            chain_uniq_res_interactions: Set[Tuple[int, int]] = set()
            for atom in self.all_non_res_chain_atoms:
                if atom.chain_id != cid:
                    continue
                if atom not in self.atom_res_contacts:
                    continue

                for other_id, other_res_idx_set in self.atom_res_contacts[atom].items():
                    for other_res_idx in other_res_idx_set:
                        chain_uniq_res_interactions.add((other_id, other_res_idx))

            if len(chain_uniq_res_interactions) >= RESIDUE_CONTACT_THRESHOLD:
                result.append(cid)

        return result

    @property
    def mediated_chain_interactions(self) -> List[Tuple[int, int, int]]:
        """
        Find protein-protein interactions likely mediated by non-residue atoms.
        Returns tuples (non_res_chain_id, prot_chain_a, prot_chain_b).
        """
        mediated: List[Tuple[int, int, int]] = []

        # for each non-res chain, find protein chains with sufficient contacts
        for chain_id, prot_contacts in self.chain_res_contacts.items():
            if not chain_id in self.non_res_chain_ids:
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
        # tally non-residue interactions
        metadata[mc.num_non_residue_chains] = self.num_non_residue_chains
        metadata[mc.num_single_atom_chains] = self.num_single_atom_chains
        metadata[mc.num_solution_molecules] = self.num_solution_molecules
        metadata[mc.num_metal_interactions] = len(self.metals_interacting)
        metadata[mc.num_macromolecule_interactions] = len(
            self.macromolecule_chains_interacting
        )

        # check for chain-chain interactions mediated by non-residues
        metadata[mc.num_mediated_interactions] = len(self.mediated_chain_interactions)
