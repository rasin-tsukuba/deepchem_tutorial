#%%
import deepchem as dc
import numpy as np
import tensorflow as tf
print(dc.__version__)

#%%
data = np.random.random((4, 4))
labels = np.random.random((4, ))

from deepchem.data.datasets import NumpyDataset

dataset = NumpyDataset(data, labels)
#print(dataset.X, dataset.y)

# for x, y, _, _ in dataset.itersamples():
#     print(x, y)

# print(dataset.ids, dataset.w)

w = np.random.random((4, ))
dataset_with_weight = NumpyDataset(data, labels, w)
# print(dataset_with_weight.w)

#%%
## MNIST Example
import tensorflow_datasets as tfds

data_dir = '/tmp/tfds'

mnist_data, info = tfds.load(name='mnist', batch_size=-1, data_dir=data_dir, with_info=True)
mnist_data = tfds.as_numpy(mnist_data)
train_data, test_data = mnist_data['train'], mnist_data['test']
num_labels = info.features['label'].num_classes
h, w, c = info.features['image'].shape
num_pixels = h * w * c

train_images, train_labels = train_data['image'], train_data['label']
print(train_images.shape)
train_images = np.reshape(train_images, (len(train_images), num_pixels))
print(train_images.shape)
train_labels = tf.one_hot(train_labels, num_labels)

test_images, test_labels = test_data['image'], test_data['label']
test_images = np.reshape(test_images, (len(test_images), num_pixels))
test_labels = tf.one_hot(test_labels, num_labels)


train = NumpyDataset(train_images, train_labels)
#valid = NumpyDataset(mnist.validation.images, mnist.validation.labels)
#%%
import matplotlib.pyplot as plt

sample = np.reshape(train.X[5], (28, 28))
plt.imshow(sample)
plt.show()

#%%
# Converting a Numpy Array to tf.data.dataset()
import tensorflow as tf

data_small = np.random.random((4, 5))
label_small = np.random.random((4, ))
dataset = tf.data.Dataset.from_tensor_slices((data_small, label_small))

#%%
# Extracting the numpy dataset from tf.data
numpy_data = np.zeros((4, 5))
numpy_label = np.zeros((4, ))

for data, label in dataset:
    numpy_data[counter, :] = data
    numpy_label[counter] = label

dataset_ = NumpyDataset(numpy_data, num_labels)

#%%
# Converting NumpyDataset to tf.data
tf_dataset = tf.data.Dataset.from_tensor_slices((dataset_.X, dataset_.y))

#%%
# Using Splitters to split DeepChem Datasets

import os
import deepchem as dc
current_dir = os.path.dirname(os.path.realpath('__file__'))
input_data = os.path.join(current_dir, 'example.csv')

tasks = ['log-solubility']
featurizer = dc.feat.CircularFingerprint(size=1024)
loader = dc.data.CSVLoader(tasks=tasks, feature_field="smiles", featurizer = featurizer)
dataset = loader.create_dataset(input_data)
print(dataset)

#%%
## Index Splitter

from deepchem.splits.splitters import IndexSplitter

splitter = IndexSplitter()
train_data, valid_data, test_data = splitter.split(dataset)
print(train_data[0])

# %%
## Specified Splitter
import deepchem as dc
from deepchem.splits.splitters import SpecifiedSplitter

tasks = ['log-solubility']
featurizer = dc.feat.CircularFingerprint(size=1024)
loader = dc.data.CSVLoader(tasks=tasks, feature_field="smiles", featurizer=featurizer)
dataset = loader.featurize('user_specified_example.csv')

split_field = 'split'

splitter = SpecifiedSplitter('user_specified_example.csv', split_field)
train_data,valid_data,test_data=splitter.split(dataset)
print(train_data,valid_data,test_data)
# %%
## Indice Splitter
import deepchem as dc
from deepchem.splits.splitters import IndiceSplitter

tasks = ['log-solubility']
featurizer = dc.feat.CircularFingerprint(size=1024)
loader = dc.data.CSVLoader(tasks=tasks, feature_field="smiles", featurizer=featurizer)
dataset = loader.featurize('user_specified_example.csv')

splitter=IndiceSplitter(valid_indices=[7],test_indices=[9])
splitter.split(dataset)
train_data,valid_data,test_data=splitter.split(dataset)
print(train_data,valid_data,test_data)

# %%
## Random Group Splitter
import deepchem as dc
from deepchem.splits.splitters import RandomGroupSplitter

def load_solubility_data():
    featurizer = dc.feat.CircularFingerprint(size=1024)
    tasks = ['log-solubility']
    task_type = "regression"
    loader = dc.data.CSVLoader(tasks=tasks, feature_field="smiles", featurizer=featurizer)

    return loader.featurize("example.csv")

groups = [0, 4, 1, 2 ,3, 7, 0, 3, 1, 0]
solubility_dataset = load_solubility_data()
splitter = RandomGroupSplitter(groups=groups)

train_idxs, valid_idxs, test_idxs = splitter.split(solubility_dataset)
