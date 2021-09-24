import sys
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np


def label(mol):
    for i, at in enumerate(mol.GetAtoms()):
        lbl = str(i)
        at.SetProp('atomLabel', lbl)
    return mol


mol = Chem.MolFromSmiles('CNCNC')
mol

label(mol)

Adj = Chem.GetAdjacencyMatrix(mol)
Adj

C = [1, 0]
N = [0, 1]
X = np.array([C, N, C, N, C])
X