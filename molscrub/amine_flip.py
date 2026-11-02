from rdkit import Chem
import numpy as np
import math
from .utils import rotation_matrix

from dataclasses import dataclass, field
from typing import Literal
from collections import defaultdict


@dataclass
class AmineGroup:
    """
    Dataclass to store substituents of non-ring
    amine groups. 
    """
    mol: Chem.Mol
    n_idx: int # index of nitrogen atom
    num_subs: Literal[3,4]  # number of atoms linked to M. can only be 3 or 4. 
    sub_idxs: list[int | None] # ranked idxs by size. One can be None, depending if amine has 3 or 4 subs. 

    # filled in automatically
    equivalent_subs: list[list[int]] = field(init=False, default_factory=list)
    _ranks: list[int] = field(init=False, repr=False, default_factory=list)

    def __post_init__(self):
        # symmetry classes: equivalent atoms share the same rank
        self._ranks = list(Chem.CanonicalRankAtoms(self.mol, breakTies=False))

        real_idxs = [i for i in self.sub_idxs if i is not None]

        # sort by size, then by symmetry rank so ties are deterministic
        idxs = sorted(real_idxs, key=lambda i: (self._get_sub_size(i), self._ranks[i]))

        if self.num_subs == 4:
            self.sub_idxs = idxs
        elif self.num_subs == 3:
            self.sub_idxs = [idxs[0], None, idxs[1], idxs[2]]  # None = sub to be flipped
        else:
            raise ValueError("Invalid number of substituents in amine group")

        # group substituents whose attachment atoms are symmetry-equivalent
        classes = defaultdict(list)
        for i in real_idxs:
            classes[self._ranks[i]].append(i)
        self.equivalent_subs = [group for group in classes.values() if len(group) > 1]

    @property
    def has_symmetry(self) -> bool:
        """True if at least two substituents are identical."""
        return bool(self.equivalent_subs)

    def are_equivalent(self, idx1: int, idx2: int) -> bool:
        """True if the substituents rooted at idx1 and idx2 are identical."""
        return self._ranks[idx1] == self._ranks[idx2]

    def _get_sub_size(self, idx):
        """
        gets number of attached atoms to given index in mol
        """
        group = get_substituent_tree(self.mol, idx, self.n_idx)
        return len(group)

# non-ring amine flips
def find_n_nonRing_substituents(mol: Chem.Mol) -> list[AmineGroup]:
    """
    Find a positively charged N atom in a non-ring
    with two substituents (non-ring atoms).
    Return (n_idx, sub1_idx, sub2_idx) if found, else None.
    """
    smarts = '[NX4H1,NX3]'
    patt = Chem.MolFromSmarts(smarts)

    matches = mol.GetSubstructMatches(patt)

    amines = []

    for match in matches:
        n_idx = match[0]
        n_atom = mol.GetAtomWithIdx(n_idx)

        neighbors = n_atom.GetNeighbors()
        sub_neighbors = [a for a in neighbors if not a.IsInRing()]

        if len(sub_neighbors) == 3: # 3 amine substituents
            sub1_idx = sub_neighbors[0].GetIdx()
            sub2_idx = sub_neighbors[1].GetIdx()
            sub3_idx = sub_neighbors[2].GetIdx()
            amines.append(AmineGroup(mol, n_idx, len(sub_neighbors), [sub1_idx, sub2_idx, sub3_idx]))
        elif len(sub_neighbors) == 4: # case with 4 substituent
            sub1_idx = sub_neighbors[0].GetIdx()
            sub2_idx = sub_neighbors[1].GetIdx()
            sub3_idx = sub_neighbors[2].GetIdx()
            sub4_idx = sub_neighbors[3].GetIdx()
            amines.append(AmineGroup(mol, n_idx, len(sub_neighbors), [sub1_idx, sub2_idx, sub3_idx, sub4_idx]))
        else:
            raise ValueError("Something went wrong here")

    return amines

def find_n_ring_substituents(mol):
    """
    Find a positively charged N atom in a ring
    with two substituents (non-ring atoms).
    Return (n_idx, sub1_idx, sub2_idx) if found, else None.
    """
    smarts = '[NRX4H1,NRX3]'
    patt = Chem.MolFromSmarts(smarts)

    matches = mol.GetSubstructMatches(patt)

    amines = []

    for match in matches:
        n_idx = match[0]
        n_atom = mol.GetAtomWithIdx(n_idx)

        neighbors = n_atom.GetNeighbors()
        ring_neighbors = [a for a in neighbors if a.IsInRing()]
        sub_neighbors = [a for a in neighbors if not a.IsInRing()]

        if len(ring_neighbors) == 2 and len(sub_neighbors) == 2:
            sub1_idx = sub_neighbors[0].GetIdx()
            sub2_idx = sub_neighbors[1].GetIdx()
            amines.append((n_idx, sub1_idx, sub2_idx))
        elif len(ring_neighbors) == 2 and len(sub_neighbors) == 1: # case with 1 substituent
            sub1_idx = sub_neighbors[0].GetIdx()
            amines.append((n_idx, sub1_idx))

    return amines

def get_substituent_tree(mol, root_idx, exclude_idx):
    """
    Return all atom indices in the substituent connected to root_idx,
    excluding the path back to exclude_idx (typically the nitrogen).
    """
    visited = set()
    stack = [root_idx]
    substituent = []

    while stack:
        atom_idx = stack.pop()
        if atom_idx in visited:
            continue
        visited.add(atom_idx)
        substituent.append(atom_idx)
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx != exclude_idx and nbr_idx not in visited:
                stack.append(nbr_idx)

    return substituent

def swap_substituents(mol, coords, n_idx, sub1_idx, sub2_idx):

    """
    Rotate the two substituents around the axis bisecting the angle between the two ring atoms bonded to nitrogen.

    coords: numpy array of shape (N_atoms, 3)
    n_idx: index of nitrogen atom
    sub1_idx: index of first substituent atom
    sub2_idx: index of second substituent atom; may be None

    Returns a copy of the modified coords.
    """
    coords_new = coords.copy()

    # Find the two ring atoms connected to nitrogen
    ring_neighbors = []
    nitrogen = mol.GetAtomWithIdx(n_idx)
    neighbors = nitrogen.GetNeighbors()
    for neighbor in neighbors:
        idx = neighbor.GetIdx()
        if idx not in [sub1_idx, sub2_idx]:
            ring_neighbors.append(neighbor.GetIdx())

    if len(ring_neighbors) != 2:
        raise ValueError("Could not identify two ring atoms bonded to nitrogen")

    r1 = coords[ring_neighbors[0]]
    r2 = coords[ring_neighbors[1]]
    n = coords[n_idx]

    # Define vectors
    v1 = r1 - n
    v2 = r2 - n

    # Axis is bisector of v1 and v2
    bisector = v1 / np.linalg.norm(v1) + v2 / np.linalg.norm(v2)
    bisector /= np.linalg.norm(bisector)

    # Get full substituent trees
    group1 = get_substituent_tree(mol, sub1_idx, n_idx)
    if sub2_idx is not None:
        group2 = get_substituent_tree(mol, sub2_idx, n_idx)
    else:
        group2 = []

    for sub_group in [group1, group2]:
        for atom_idx in sub_group:
            vec = coords[atom_idx] - n
            rotated_vec = np.dot(rotation_matrix(bisector, math.pi), vec)
            coords_new[atom_idx] = n + rotated_vec

    return coords_new


def get_ring_atoms(mol, n_idx):
    """
    Given the nitrogen index, find the ring it's part of and return the ring atom indices.
    """
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if n_idx in ring and (len(ring) == 5 or len(ring) == 6):
            return ring
    return None

# for debbuging .
def molToXYZ(mol, coords):
    """
    Convert an RDKit mol and numpy array of coordinates to XYZ format string.
    """
    lines = [str(mol.GetNumAtoms()), "XYZ generated from RDKit"]
    for atom, pos in zip(mol.GetAtoms(), coords):
        sym = atom.GetSymbol()
        x, y, z = pos
        lines.append(f"{sym} {x:.6f} {y:.6f} {z:.6f}")
    return "\n".join(lines)

# testing
# from rdkit.Chem import SDMolSupplier
# sdf_path = "amine.sdf"  
# suppl = SDMolSupplier(sdf_path, removeHs=False)

# for mol in suppl:
#     if mol is None:
#         continue

#     match = find_n_ring_substituents(mol)
#     if match:
#         n_idx, sub1_idx, sub2_idx = match
#         ring_atoms = get_ring_atoms(mol, n_idx)
#         conf = mol.GetConformer()
#         coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])

#         new_coords = swap_substituents(coords, mol, n_idx, sub1_idx, sub2_idx, ring_atoms)

#         print(f"Original coords for substituents: {coords[sub1_idx]}, {coords[sub2_idx]}")
#         print(f"Swapped coords for substituents: {new_coords[sub1_idx]}, {new_coords[sub2_idx]}")
#         break
#     else:
#         print("No matching nitrogen found.")
