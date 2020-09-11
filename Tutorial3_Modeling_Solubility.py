#%%
import deepchem as dc
import tensorflow as tf
import numpy as numpy
from tensorflow.keras.layers import Reshape, Conv2D, Flatten, Dense
from deepchem.utils.save import load_from_disk

data_file = 'data/delaney-processed.csv'
dataset = load_from_disk(data_file)
print("Columns of dataset: %s" % str(dataset.columns.values))
print("Number of examples in dataset: %s" % str(dataset.shape[0]))

# %%
import tempfile
from rdkit import Chem
from rdkit.Chem import Draw
from itertools import islice
from IPython.display import Image, display

def display_images(filenames):
    for file in filenames:
        display(Image(file))

def mols_to_pngs(mols, basename='test'):
    filenames = []
    for i, mol in enumerate(mols):
        filename = "%s%d.png" % (basename, i)
        Draw.MolToFile(mol, filename)
        filenames.append(filename)

    return filenames
    
# %%
# num_to_display = 14
# molecules = []
# for _, data in islice(dataset.iterrows(), num_to_display):
#     molecules.append(Chem.MolFromSmiles(data["smiles"]))
# # display_images(mols_to_pngs(molecules))
# print(molecules)
m = Chem.rdmolfiles.MolFromMolFile('h2po4.mol')
display_images(mols_to_pngs([m]))

print(m.GetPos())

#%%
featurizer = dc.feat.CircularFingerprint(size=1024)
loader = dc.data.CSVLoader(
      tasks=["measured log solubility in mols per litre"], 
      feature_field="smiles",
      featurizer=featurizer)
dataset = loader.create_dataset(data_file)
splitter = dc.splits.ScaffoldSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset)