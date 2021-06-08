import math
import numpy as np
import tensorflow as tf
from rdkit import Chem

class GraphDataset(tf.keras.utils.Sequence):

    def __init__(self,
                 X,
                 y=None,
                 shuffle=True,
                 batch_size=64,
                 max_atoms=9,
                 atoms=['C','N','O','H','Br','Cl','F','I','P','S','Si'],
                 bonds=['single','double','triple','aromatic']):

        self.X = X
        self.y = y
        self.num_examples = len(self.X)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.max_atoms = max_atoms
        self.atoms = atoms
        self.bonds = [b.upper() for b in bonds]
        self.atom_mapping = create_atom_mapping(atoms)
        self.bond_mapping = create_bond_mapping(bonds)
        self.atom_dim = len(atoms) + 1
        self.bond_dim = len(bonds) + 1
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(self.num_examples / self.batch_size)

    def on_epoch_end(self):
        self.index = 0
        self.num_passed_examples = 0
        if self.shuffle:
            np.random.shuffle(self.X)

    def __getitem__(self, _):
        adj_batch, feat_batch = [], []
        while len(adj_batch) < self.batch_size:
            graph = self.smiles_to_graph(self.X[self.index])

            if graph is not None:
                adj_batch.append(graph[0])
                feat_batch.append(graph[1])
                self.num_passed_examples += 1

            self.index += 1
            if self.num_examples <= self.index:
                self.index = 0
                if self.num_passed_examples < self.batch_size:
                    raise RuntimeError('Failed to obtain a single batch')

        return np.array(adj_batch), np.array(feat_batch)

    def smiles_to_graph(self, smiles):
        """Converts SMILES string to graph ((A, X) tuple))"""
        m = smiles_to_mol(smiles, catch_errors=True)

        # check that mol is not None and also pass different criteria
        if (
            m is None or
            m.GetNumAtoms() > self.max_atoms or
            not all([a.GetSymbol() in self.atoms for a in m.GetAtoms()]) or
            not all([b.GetBondType().name in self.bonds for b in m.GetBonds()])
        ):
            return None

        # initialize features (X) and adjacency (A) matrices
        X = np.zeros((self.max_atoms, self.atom_dim))
        A = np.zeros((self.bond_dim, self.max_atoms, self.max_atoms))

        # loop over atoms in mol object and add to X and A
        for atom in m.GetAtoms():
            i = atom.GetIdx()
            atom_type = self.atom_mapping[atom.GetSymbol()]
            X[i] = np.eye(self.atom_dim)[atom_type]
            for neighbor in atom.GetNeighbors():
                j = neighbor.GetIdx()
                bond = m.GetBondBetweenAtoms(i, j)
                bond_type = self.bond_mapping[bond.GetBondType().name]
                A[bond_type, [i, j], [j, i]] = 1

        # where no bond, add 1 to last channel (indicating "no bond")
        A[-1, np.sum(A, axis=0) == 0] = 1
        # where no atom(_type), add 1 to last column (indicating "non-atom")
        X[np.where(X.sum(axis=1) == 0)[0], -1] = 1
        return A, X

    def graph_to_mol(self, graph, sanitize=False):
        """Converts graph to RDKit Mol object"""
        adjacency = np.copy(graph[0])
        features = np.copy(graph[1])

        mol = Chem.RWMol()

        # if argmax(features, axis=1) == self.atom_dim-1, atom is predicted as "None"
        # remove this non-atom from the features and adjacency tensors
        atom_idx = np.where(np.argmax(features, axis=1) != self.atom_dim-1)[0]
        features = features[atom_idx]
        adjacency = adjacency[:, atom_idx][:, :, atom_idx]

        # loop over atoms and add to mol object
        for atom_i in np.argmax(features, axis=1):
            atom = Chem.Atom(self.atom_mapping[atom_i])
            _ = mol.AddAtom(atom)

        # obtain bond type for atom_i and atom_j
        (bonds_ij, atoms_i, atoms_j) = np.where(np.triu(adjacency) == 1)
        # loop over bonds (between atom_i and atom_j) and add to mol objcet.
        for (bond_ij, atom_i, atom_j) in zip(bonds_ij, atoms_i, atoms_j):
            if atom_i == atom_j or bond_ij >= self.bond_dim-1:
                continue
            bond_type = self.bond_mapping[bond_ij]
            mol.AddBond(int(atom_i), int(atom_j), bond_type)

        if sanitize:
            Chem.SanitizeMol(mol)

        return mol


def create_atom_mapping(symbols):
    mapping = {}
    index = 0
    for symbol in symbols:
        if not mapping.get(symbol):
            mapping[symbol] = index
            mapping[index] = symbol
            index += 1
    return mapping

def create_bond_mapping(types):
    mapping = {}
    index = 0
    for t in types:
        t = t.upper()
        if not mapping.get(t):
            mapping[t] = index
            mapping[index] = Chem.BondType.names[t]
            index += 1
    return mapping

def smiles_to_mol(smiles, catch_errors=True):
    """Generates RDKit Mol object from a SMILES string"""
    if catch_errors:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        flag = Chem.SanitizeMol(mol, catchErrors=True)
        if flag != Chem.SanitizeFlags.SANITIZE_NONE:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^flag)
            return mol
    return Chem.MolFromSmiles(smiles, sanitize=True)
