#%%
import deepchem as dc
from deepchem.models.graph_models import GraphConvModel

# Load Tox21 dataset
# 8k compounds on 12 different targets
tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21(
    featurizer='GraphConv',
    reload=False
)

train_dataset, valid_dataset, test_dataset = tox21_datasets
n_tasks = len(tox21_tasks)

# print(valid_dataset.y)
#%%
batch_size = 50
for ind, (X_b, y_b, w_b, ids_b) in enumerate(valid_dataset.iterbatches(batch_size, pad_batches=True, deterministic=True)):
    print(X_b[0])
    break


#%%
model = GraphConvModel(n_tasks, batch_size=50, mode='classification')
print(model)
num_epochs = 10
losses = []
for i in range(num_epochs):
    loss = model.fit(train_dataset, nb_epoch=1)
    print("Epoch %d loss: %f" % (i, loss))
    losses.append(loss)

# %%
import numpy as np
metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
print("Training ROC-AUC Score: %f" % train_scores["mean-roc_auc_score"])
valid_scores = model.evaluate(valid_dataset, [metric], transformers)
print("Validation ROC-AUC Score: %f" % valid_scores["mean-roc_auc_score"])
test_scores = model.evaluate(test_dataset, [metric], transformers)
print("Validation ROC-AUC Score: %f" % test_scores["mean-roc_auc_score"])

# %%
# Graph Convolutions require the structure of the molecule
# in question and a vector of features for every atom that
# describes the local cheimcal environment

# `atom_features` holds a feature vector of length 75 for each atom
# `degree_slice` is an indexing convenience that makes it easy to 
# locate atoms from all molecules with a given degree
# `membership` determines the membership of atoms in molecules
# `deg_adjs` is a list that contains adjacency lists grouped by atom degree

from deepchem.metrics import to_one_hot
from deepchem.feat.mol_graphs import ConvMol

def data_generator(dataset, predict=False, pad_batches=True):
    # iterbatches: Get an object that iterates over minibatches from the dataset.
    for ind, (X_b, y_b, w_b, ids_b) in enumerate(dataset.iterbatches(
                                                batch_size, 
                                                pad_batches=pad_batches, 
                                                deterministic=True)):
        # Concatenates list of ConvMol’s into one mol object that 
        # can be used to feed into tensorflow placeholders.
        # agglomerate_mols -> mols: ConvMol objects to be combined into one molecule.
        multiConvMol = ConvMol.agglomerate_mols(X_b)
        # get_atom_features: Returns canonicalized version of atom features
        inputs = [multiConvMol.get_atom_features(), 
                    multiConvMol.deg_slice,
                    np.array(multiConvMol.membership)]

        # Returns adjacency lists grouped by atom degree.            
        for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
            inputs.append(multiConvMol.get_deg_adjacency_lists()[i])
        labels = [to_one_hot(y_b.flatten(), 2).reshape(-1, n_tasks, 2)]
        weights = [w_b]
        yield (inputs, labels, weights)

#%%

# `GraphConv` layer: This layer implements the graph convolution.
# The graph convolution combines per-node feature vectures in a 
# nonlinear fashion with the feature vectors for neighboring nodes.
# This "blends" information in local neighborhoods of a graph.

# `GraphPool` layer: This layer does a max-pooling over the feature 
# vectors of atoms in a neighborhood.

# `GraphGather` layer: Many graph convolutional networks manipulate 
# feature vectors per graph-node. For a molecule for example, each 
# node might represent an atom, and the network would manipulate 
# atomic feature vectors that summarize the local chemistry of the 
# atom. However, at the end of the application, we will likely want 
# to work with a molecule level feature representation. This layer 
# creates a graph level feature vector by combining all the 
# node-level feature vectors.

from deepchem.models.layers import GraphConv, GraphPool, GraphGather
import tensorflow as tf
import tensorflow.keras.layers as layers

batch_size = 50

class KerasModel(tf.keras.Model):
    def __init__(self):
        super(KerasModel, self).__init__()

        self.gc1 = GraphConv(128, activation_fn=tf.nn.tanh)
        self.batch_norm1 = layers.BatchNormalization()
        self.gp1 = GraphPool()

        self.gc2 = GraphConv(128, activation_fn=tf.nn.tanh)
        self.batch_norm2 = layers.BatchNormalization()
        self.gp2 = GraphPool()

        self.dense1 = layers.Dense(256, activation=tf.nn.tanh)
        self.batch_norm3 = layers.BatchNormalization()
        self.readout = GraphGather(batch_size=batch_size, activation_fn=tf.nn.tanh)

        self.dense2 = layers.Dense(n_tasks * 2)
        self.logits = layers.Reshape((n_tasks, 2))
        self.softmax = layers.Softmax()

    def call(self, inputs):
        gc1_output = self.gc1(inputs)
        batch_norm1_output = self.batch_norm1(gc1_output)
        gp1_output = self.gp1([batch_norm1_output] + inputs[1:])

        gc2_output = self.gc2([gp1_output] + inputs[1:])
        batch_norm2_output = self.batch_norm2(gc2_output)
        gp2_output = self.gp2([batch_norm2_output] + inputs[1:])

        dense1_output = self.dense1(gp2_output)
        batch_norm3_output = self.batch_norm3(dense1_output)
        readout_output = self.readout([batch_norm1_output] + inputs[1:])

        logits_output = self.logits(self.dense2(readout_output))
        return self.softmax(logits_output)

loss = dc.models.losses.CategoricalCrossEntropy()
model = dc.models.KerasModel(KerasModel(), loss=loss)

#%%
import numpy as np
num_epochs = 100
losses = []
for i in range(num_epochs):
    loss = model.fit_generator(data_generator(train_dataset))
    print("Epoch %d loss: %f" % (i, loss))
    losses.append(loss)


# %%
import matplotlib.pyplot as plot

plot.title("Keras Version")
plot.ylabel("Loss")
plot.xlabel("Epoch")
x = range(num_epochs)
y = losses
plot.scatter(x, y)
plot.show()


metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)

def reshape_y_pred(y_true, y_pred):
    """
    GraphConv always pads batches, so we need to remove the predictions
    for the padding samples.  Also, it outputs two values for each task
    (probabilities of positive and negative), but we only want the positive
    probability.
    """
    n_samples = len(y_true)
    return y_pred[:n_samples, :, 1]
    

print("Evaluating model")
train_predictions = model.predict_on_generator(data_generator(train_dataset, predict=True))
train_predictions = reshape_y_pred(train_dataset.y, train_predictions)
train_scores = metric.compute_metric(train_dataset.y, train_predictions, train_dataset.w)
print("Training ROC-AUC Score: %f" % train_scores)

valid_predictions = model.predict_on_generator(data_generator(valid_dataset, predict=True))
valid_predictions = reshape_y_pred(valid_dataset.y, valid_predictions)
valid_scores = metric.compute_metric(valid_dataset.y, valid_predictions, valid_dataset.w)
print("Valid ROC-AUC Score: %f" % valid_scores)
# %%
