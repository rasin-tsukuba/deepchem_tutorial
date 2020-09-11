#%%
# The goal is to demonstrate how multitask methods can provide 
# improved performance in situations with little or 
# very unbalanced data.

"""
The MUV dataset is a challenging benchmark in molecular design 
that consists of 17 different "targets" where there are only 
a few "active" compounds per target. There are 93,087 compounds
in total, yet no task has more than 30 active compounds, and 
many have even less. Training a model with such a small number 
of positive examples is very challenging. Multitask models 
address this by training a single model that predicts all the 
different targets at once. If a feature is useful for predicting 
one task, it often is useful for predicting several other tasks 
as well. Each added task makes it easier to learn important 
features, which improves performance on other tasks
"""

import deepchem as dc
import numpy as np

tasks, datasets, transformers = dc.molnet.load_muv(split='random')
train_dataset, valid_dataset, test_dataset = datasets

#%%
# create a MultitaskClassifier
n_tasks = len(tasks)
n_features = train_dataset.get_data_shape()[0]
model = dc.models.MultitaskClassifier(n_tasks, n_features)
model.fit(train_dataset)

#%%
# We loop over the 17 tasks and compute the ROC AUC for each one. 
# To ensure we have enough data to compute a meaningful result, 
# we only compute the score for tasks with at least three 
# positive samples.

y_true = test_dataset.y
y_pred = model.predict(test_dataset)
metric = dc.metrics.roc_auc_score

for i in range(n_tasks):
    if np.sum(y_true[:, i])>2:
        score = metric(dc.metrics.to_one_hot(y_true[:, i]), y_pred[:, 1])
        print(tasks[i], score)
    else:
        print(tasks[i], 'Not enough positives in test set')