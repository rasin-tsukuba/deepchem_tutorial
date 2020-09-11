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
# m = Chem.rdmolfiles.MolFromMolFile('h2po4.mol')
# display_images(mols_to_pngs([m]))

# print(m.GetPos())

#%%
featurizer = dc.feat.CircularFingerprint(size=1024)
loader = dc.data.CSVLoader(
      tasks=["measured log solubility in mols per litre"], 
      feature_field="smiles",
      featurizer=featurizer)
dataset = loader.create_dataset(data_file)
splitter = dc.splits.ScaffoldSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset)

train_mols = [Chem.MolFromSmiles(compound)
              for compound in train_dataset.ids]

valid_mols = [Chem.MolFromSmiles(compound)
                for compound in valid_dataset.ids]

#%%
# One common transformation applied to data is to normalize it 
# to have zero-mean and unit-standard-deviation. We will apply 
# this transformation to the log-solubility 

transformers = [
    dc.trans.NormalizationTransformer(transform_y=True, dataset=train_dataset)
]

for dataset in [train_dataset, valid_dataset, test_dataset]:
    for transformer in transformers:
        dataset = transformer.transform(dataset)

#%%
# fitting simple learning models to our data. 
# deepchem provides a number of machine-learning model classes.
# SklearnModel that wraps any machine-learning model available in scikit-learn 
# we will start by building a simple random-forest regressor that attempts 
# to predict the log-solubility from our computed ECFP4 features.
from sklearn.ensemble import RandomForestRegressor

# we instantiate the SklearnModel object
sklearn_model = RandomForestRegressor(n_estimators=100)
# then call the fit() method on the train_dataset we constructed above
model = dc.models.SklearnModel(sklearn_model)
model.fit(train_dataset)

#%%
# We next evaluate the model on the validation set 
# to see its predictive power.
# deepchem provides the Evaluator class to facilitate this process. 
from deepchem.utils.evaluate import Evaluator

metric = dc.metrics.Metric(dc.metrics.r2_score)
# reate a new Evaluator instance and call the 
# compute_model_performance() method.
evaluator = Evaluator(model, valid_dataset, transformers)
r2score = evaluator.compute_model_performance([metric])
print(r2score)

#%%
# We now build a series of SklearnModels with different choices 
# for n_estimators and max_features and evaluate performance on 
# the validation set.
def rf_model_builder(n_estimators, max_features, model_dir):
    sklearn_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
    )
    return dc.models.SklearnModel(sklearn_model, model_dir)

params_dict = {
    "n_estimators": [10, 100],
    "max_features": ["auto", "sqrt", "log2", None],
}

optimizer = dc.hyper.GridHyperparamOpt(rf_model_builder)
best_rf, best_rf_hyperparams, all_rf_results = optimizer.hyperparam_search(
    params_dict, train_dataset, valid_dataset, transformers,
    metric=metric)

#%%

rf_test_evaluator = Evaluator(best_rf, test_dataset, transformers)
rf_test_r2score = rf_test_evaluator.compute_model_performance([metric])
print("RF Test set R^2 %f" % (rf_test_r2score["r2_score"]))

#%% 
# construct a simple network
import numpy.random
import numpy as np

params_dict = {"learning_rate": np.power(10., np.random.uniform(-5, -3, size=1)),
               "decay": np.power(10, np.random.uniform(-6, -4, size=1)),
               "nb_epoch": [20] }
n_features = train_dataset.get_data_shape()[0]

def model_builder(learning_rate, decay, nb_epoch, model_dir):
    model = dc.models.MultitaskRegressor(
        1, n_features, layer_sizes=[1000], dropouts=[0.25],
        batch_size=50, learning_rate=learning_rate, decay=decay,
        nb_epoch=nb_epoch, model_dir=model_dir)
    return model

optimizer = dc.hyper.GridHyperparamOpt(model_builder)
best_dnn, best_dnn_hyperparams, all_dnn_results = optimizer.hyperparam_search(
    params_dict, train_dataset, valid_dataset, transformers,
    metric=metric)

#%%
dnn_test_evaluator = Evaluator(best_dnn, test_dataset, transformers)
dnn_test_r2score = dnn_test_evaluator.compute_model_performance([metric])
print("DNN Test set R^2 %f" % (dnn_test_r2score["r2_score"]))

#%%
import matplotlib.pyplot as plt
task = "measured log solubility in mols per litre"
predicted_test = best_rf.predict(test_dataset)
true_test = test_dataset.y
plt.scatter(predicted_test, true_test)
plt.xlabel('Predicted log-solubility in mols/liter')
plt.ylabel('True log-solubility in mols/liter')
plt.title(r'RF- predicted vs. true log-solubilities')
plt.show()

#%%

task = "measured log solubility in mols per litre"
predicted_test = best_dnn.predict(test_dataset)
true_test = test_dataset.y
plt.scatter(predicted_test, true_test)
plt.xlabel('Predicted log-solubility in mols/liter')
plt.ylabel('True log-solubility in mols/liter')
plt.title(r'DNN predicted vs. true log-solubilities')
plt.show()