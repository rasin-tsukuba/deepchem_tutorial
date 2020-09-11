#%%
import deepchem as dc
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Reshape, Conv2D, Flatten, Dense

mnist = tf.keras.datasets.mnist.load_data("mnist.npz")


#%%
train_images = mnist[0][0].reshape((-1, 28, 28, 1)) / 255
valid_images = mnist[1][0].reshape((-1, 28, 28, 1)) / 255
train = dc.data.NumpyDataset(train_images, mnist[0][1])
valid = dc.data.NumpyDataset(valid_images, mnist[1][1])

keras_model = tf.keras.Sequential([
    Conv2D(filters=32, kernel_size=5, activation=tf.nn.relu),
    Conv2D(filters=64, kernel_size=5, activation=tf.nn.relu),
    Flatten(),
    Dense(1024, activation=tf.nn.relu),
    Dense(10)
])

model = dc.models.KerasModel(keras_model, dc.models.losses.SparseSoftmaxCrossEntropy())

model.fit(train, nb_epoch=2)
# %%
## Pytorch Model
# train_images = mnist[0][0].reshape((-1, 1, 28, 28)) / 255
# valid_images = mnist[1][0].reshape((-1, 1, 28, 28)) / 255
# train = dc.data.NumpyDataset(train_images, mnist[0][1])
# valid = dc.data.NumpyDataset(valid_images, mnist[1][1])

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()

#         self.conv1 = nn.Conv2d(1, 32, 5)
#         self.conv2 = nn.Conv2d(32, 64, 5)

#         self.fc1 = nn.Linear(64 * 20 * 20, 1024)
#         self.fc2 = nn.Linear(1024, 10)
    
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))

#         x = x.view(-1, self.num_flat_features(x))
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)

#         return x
    
#     def num_flat_features(self, x):
#         size = x.size()[1:]
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features

# torch_model = LeNet()
# print(torch_model)

# model = nn.Sequential(
#     nn.Conv2d(1, 32, 5),
#     nn.ReLU(),
#     nn.Conv2d(32, 64, 5),
#     nn.ReLU(),
#     nn.Flatten(),
#     nn.Linear(64 * 20 * 20, 1024),
#     nn.ReLU(),
#     nn.Linear(1024, 10)
# )
# print(torch.cuda.is_available())

# print(model)
# model = model.cuda()
# model = dc.models.TorchModel(model, dc.models.losses.SparseSoftmaxCrossEntropy(), device=0)
# model.fit(train, nb_epoch=2)
# %%
