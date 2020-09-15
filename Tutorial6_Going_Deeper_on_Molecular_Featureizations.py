#%%
# we explore the different featurization methods available 
# for molecules. These featurization methods include:
# ConvMolFeaturizer,
# WeaveFeaturizer,
# CircularFingerprints
# RDKitDescriptors
# BPSymmetryFunction
# CoulombMatrix
# CoulombMatrixEig
# AdjacencyFingerprints
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import deepchem as dc
import numpy as np
from rdkit import Chem
import deepchem.feat as feat
from deepchem.utils import conformers

""" 
We use H2PO4 as a running example throughout this 
tutorial. Many of the featurization methods use conformers or 
the molecules. A conformer can be generated using the 
ConformerGenerator class in deepchem.utils.conformers. 
"""

example_smile = "CCC"
example_mol = Chem.MolFromSmiles(example_smile)

#%%

# RDKitDescriptors 
# featurizes a molecule by computing descriptors values for 
# specified descriptors. Intrinsic to the featurizer is a set 
# of allowed descriptors, which can be accessed using 
RDKitDescriptors.allowedDescriptors.
for descriptor in feat.RDKitDescriptors.allowedDescriptors:
    print(descriptor)

rdkit_desc = feat.RDKitDescriptors()
features = rdkit_desc._featurize(example_mol)

print('The number of descriptors present are: ', len(features))
#%%
# BPSymmetryFunction
# Behler-Parinello Symmetry function or BPSymmetryFunction 
# featurizes a molecule by computing the atomic number and 
# coordinates for each atom in the molecule. The features 
# can be used as input for symmetry functions, 
# like `RadialSymmetry`, `DistanceMatrix` and `DistanceCutoff`.
# These functions can be found in `deepchem.feat.coulomb_matrices`
from deepchem.feat.molecule_featurizers.bp_symmetry_function_input import BPSymmetryFunctionInput

engine = conformers.ConformerGenerator(max_conformers=1)
example_mol_new = engine.generate_conformers(example_mol)

"""
The featurizer takes in `max_atoms` as an argument. 
As input, it takes in a conformer of the molecule and computes:

1. coordinates of every atom in the molecule (in Bohr units)
2. the atomic numbers for all atoms.

These features are concantenated and padded with zeros to 
account for different number of atoms, across molecules.
"""
bp_sym = BPSymmetryFunctionInput(max_atoms=20)
features = bp_sym._featurize(mol=example_mol)
print(features)

atomic_numbers = features[:, 0]
from collections import Counter

unique_numbers = Counter(atomic_numbers)
print(unique_numbers)

# %%
# CoulombMatrix
# It featurizes a molecule by computing the coulomb matrices 
# for different conformers of the molecule, and returning it 
# as a list.
# CoulombMatrix is invariant to molecular rotation and translation, 
# since the interatomic distances or atomic numbers do not change. 

# A Coulomb matrix tries to encode the energy structure of a 
# molecule. The matrix is symmetric, with the off-diagonal 
# elements capturing the Coulombic repulsion between pairs of 
# atoms and the diagonal elements capturing atomic energies 
# using the atomic numbers. 
# The featurizer takes in max_atoms as an argument and also 
# has options for removing hydrogens from the molecule 
# (remove_hydrogens), generating additional random coulomb 
# matrices(randomize), and getting only the upper triangular 
# matrix (upper_tri).

example_smile = "CCC"
example_mol = Chem.MolFromSmiles(example_smile)

engine = conformers.ConformerGenerator(max_conformers=1)
example_mol = engine.generate_conformers(example_mol)

print("Number of available conformers for propane: ", len(example_mol.GetConformers()))

coulomb_mat = feat.CoulombMatrix(max_atoms=20, randomize=False, remove_hydrogens=False, upper_tri=False)
features = coulomb_mat._featurize(mol=example_mol)

print(len(example_mol.GetConformers()) == len(features))
print(len(example_mol.GetConformers()), len(features))
# %%
# CoulombMatrixEig
# However the CoulombMatrix is not invariant to random permutations 
# of the atom's indices. To deal with this, the CoulumbMatrixEig 
# featurizer was introduced, which uses the eigenvalue spectrum of 
# the columb matrix, and is invariant to random permutations of 
# the atom's indices.
# CoulombMatrixEig inherits from CoulombMatrix and featurizes 
# a molecule by first computing the coulomb matrices for different
# conformers of the molecule and then computing the eigenvalues 
# for each coulomb matrix. These eigenvalues are then padded to 
# account for variation in number of atoms across molecules. 
# The featurizer takes in max_atoms as an argument and also 
# has options for removing hydrogens from the molecule 
# (remove_hydrogens), generating additional random coulomb 
# matrices(randomize).

example_smile = "CCC"
example_mol = Chem.MolFromSmiles(example_smile)

engine = conformers.ConformerGenerator(max_conformers=1)
example_mol = engine.generate_conformers(example_mol)

print("Number of available conformers for propane: ", len(example_mol.GetConformers()))

coulomb_mat_eig = feat.CoulombMatrixEig(max_atoms=20, randomize=False, remove_hydrogens=False)
features = coulomb_mat_eig._featurize(mol=example_mol)
# %%
