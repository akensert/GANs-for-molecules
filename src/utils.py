from IPython.display import SVG
from rdkit.Chem import Draw
from rdkit import Chem
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def draw_mol(mol, size=(400, 300)):
    try:
        mol = Draw.rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=True)
    except Exception as e:
        print(e)
        mol = Draw.rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=False)

    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(*size)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return SVG(drawer.GetDrawingText())

def save_mol(mol, path='images/mol.png', size=(400, 300)):
    try:
        mol = Draw.rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=True)
    except Exception as e:
        print(e)
        mol = Draw.rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=False)

    drawer = Draw.MolDraw2DCairo(*size)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    drawer.WriteDrawingText(path)

def draw_mol_with_labels(mol, atom_idx=True, bond_idx=False, size=(400, 300)):
    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(*size)
    drawer.drawOptions().addAtomIndices=True if atom_idx else False
    drawer.drawOptions().addBondIndices=True if bond_idx else False
    drawer.SetLineWidth(3)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return SVG(drawer.GetDrawingText())

def check_novelty(mol_gen, data):
    smiles_gen = Chem.MolToSmiles(mol_gen)
    for s in data.smiles:
        if s == smiles_gen:
            return False
    return True
