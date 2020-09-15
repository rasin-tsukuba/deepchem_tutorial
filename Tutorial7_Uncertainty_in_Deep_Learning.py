#%%
# We'll use the SAMPL dataset from the MoleculeNet suite to 
# run our experiments in this tutorial. Let's load up our 
# dataset for our experiments, and then make some uncertainty 
# predictions.

import deepchem as dc
import numpy as np
import matplotlib.pyplot as plt

tasks, datasets, transformers = dc.molnet.load_sampl(reload=False)
train_dataset, valid_dataset, test_dataset = datasets

# uncertainty (bool) – if True, include extra outputs and 
# loss terms to enable the uncertainty in outputs to be 
# predicted
# This instructs it to add features to the model that are 
# needed for estimating uncertainty.
model = dc.models.MultitaskRegressor(len(tasks), 
                                    1024, 
                                    uncertainty=True)

model.fit(train_dataset, nb_epoch=200)
# we call predict_uncertainty() instead of predict() to 
# produce the output. 
# `y_pred` is the predicted outputs. 
# `y_std` is another array of the same shape, where each 
# element is an estimate of the uncertainty (standard deviation) 
# of the corresponding element in y_pred.
y_pred, y_std = model.predict_uncertainty(test_dataset)
# %%
# Intuitively, it is a measure of how much we can trust the 
# predictions. More formally, we expect that the true value 
# of whatever we are trying to predict should usually be within 
# a few standard deviations of the predicted value.

x = np.linspace(0, 5, 10)
y = 0.15*x + np.random.random(10)
plt.scatter(x, y)
fit = np.polyfit(x, y, 1)
line_x = np.linspace(-1, 6, 2)
plt.plot(line_x, np.poly1d(fit)(line_x))
plt.show()

# %%
