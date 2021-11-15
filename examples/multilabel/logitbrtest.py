'''
Created on 31 jan 2021

@author: Yonatan-Carlos Carranza-Alarcon

About how to use the Imprecise Binary relevance with logistic base classifier
'''
from classifip.dataset import arff
from classifip.models.mlc.logitbr import Logit_BR
from classifip.models.mlc.mlcncc import MLCNCC
import timeit
import numpy as np

# We start by creating the binary relevance classifier
model = Logit_BR(DEBUG=True)

# Loading data set
nb_labels = 2
dataArff = arff.ArffFile()
dataArff.load("labels2.arff")

idx_inst = np.random.choice(range(10), 1)
print("\nGround-truth observed before missing process")
print(dataArff.data[idx_inst[0]][-nb_labels:])

# Learning
missing_pct = 0.6
# missing % percentage of values of label
MLCNCC.missing_labels_learn_data_set(learn_data_set=dataArff, nb_labels=nb_labels, missing_pct=missing_pct)

# Learning model
model.learn(learn_data_set=dataArff, nb_labels=nb_labels)

# Evaluation
test = model.evaluate([dataArff.data[idx_inst[0]][:-nb_labels]])

# Print interval probabilities
print("\nInterval marginal probabilities")
print(test[0][0])

# Print precise probabilities
print("\nPrecise marginal probabilities")
for prob_label in test[0][1]:
    print(prob_label)

# The output is a list of probability intervals, we can print each instance :
print("\nPrediction using maximality decision vs ground-truth observed")
print(test[0][0].multilab_dom(), dataArff.data[idx_inst[0]][-nb_labels:])
