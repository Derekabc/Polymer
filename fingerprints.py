
# from rdkit import Chem
# from rdkit.Chem import Draw
# from rdkit.Chem.Draw import rdMolDraw2D
# from rdkit.Chem import rdDepictor
# rdDepictor.SetPreferCoordGen(True)
# from rdkit.Chem.Draw import IPythonConsole
# from IPython.display import SVG
# import rdkit
# print(rdkit.__version__)

# diclofenac = Chem.MolFromSmiles('O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl')
# d2d = rdMolDraw2D.MolDraw2DSVG(350,300)
# d2d.DrawMolecule(diclofenac)
# d2d.FinishDrawing()
# SVG(d2d.GetDrawingText())


import sys
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
# IPythonConsole.ipython_useSVG=True  #< set this to False if you want PNGs instead of SVGs

# mol = Chem.MolFromSmiles('CCCC')
# mol


# smiles_lists = ['C(C(=O)O)N', 'N[C@@H](CC1=CC=CC=C1)C(O)=O',
#             'O=C([C@H](CC1=CNC=N1)N)O', 'C(C[C@@H](C(=O)O)N)S']
# mol_list = []
# for smiles in smiles_lists:
#     mol = Chem.MolFromSmiles(smiles)
#     mol_list.append(mol)

# img = Draw.MolsToGridImage(mol_list, molsPerRow=4)
# img


# # pattern = Chem.MolFromSmiles('S')
# # pattern = Chem.MolFromSmiles('C(=O)O')
# pattern = Chem.MolFromSmiles('CC(N)C')
# Chem.MolFromSmiles('CC(N)C')

# for mol in mol_list:
#     print(mol.HasSubstructMatch(pattern))

glycine = mol_list[0]
glycine

bi= {}
fp = AllChem.GetMorganFingerprintAsBitVect(glycine, 2, nBits=1024, bitInfo=bi)
fp_arr = np.zeros((1,))
DataStructs.ConvertToNumpyArray(fp, fp_arr)
fp_arr
np.nonzero(fp_arr)
list(fp.GetOnBits())


prints = [(glycine, x, bi) for x in fp.GetOnBits()]
Draw.DrawMorganBits(prints, molsPerRow=4, legends=[str(x) for x in fp.GetOnBits()])


cysteine = mol_list[3]
img = Draw.MolsToGridImage([glycine, cysteine], molsPerRow=2)
img

fp2 = AllChem.GetMorganFingerprintAsBitVect(cysteine, 2, nBits=1024, bitInfo=bi)
print("glycine:", list(fp.GetOnBits()))
print("cysteine", list(fp2.GetOnBits()))

common = set(fp.GetOnBits()) & set(fp2.GetOnBits())
print("common", common)