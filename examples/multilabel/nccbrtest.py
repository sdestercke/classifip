"""
Created on 25 august 2016

@author: Sebastien Destercke

About how to use the Naive Credal Classifier Binary Relevance.

"""

print("Example of multilabel prediciton with NCC BR - data set yeast \n")

print("Data loading \n")
from classifip.dataset import arff

data = arff.ArffFile()
data.load("yeast.arff")
data.discretize(discmet='eqfreq', numint=5)
dataset = 'yeast'
nblab = 14

# We start by creating an instance of the base classifier we want to use
print("Model creation and learning \n")
from classifip.models.mlc import nccbr
from classifip.models.mlc.mlcncc import MLCNCC

model = nccbr.NCCBR()

# Learning
missing_pct = 0.0
# missing % percentage of values of label
MLCNCC.missing_labels_learn_data_set(learn_data_set=data, nb_labels=nblab, missing_pct=missing_pct)
model.learn(data, nblab)

import numpy as np

for s in np.arange(0.1, 1.5, 0.1):
    # Evaluation : we can set the parameters of the classifier
    test = model.evaluate([row[0:len(row) - nblab] for row in data.data[0:1]], ncc_epsilon=0.001, ncc_s_param=s)

    # The output is a list of probability intervals, we can print each instance :
    print("Probability intervals obtained for each label on the first test instance \n", s)
    print(test[0])
    print(test[0].multilab_dom())

test = model.evaluate([row[0:len(row) - nblab] for row in data.data[0:1]], ncc_epsilon=0.001, ncc_s_param=0.0)
print("Precise probability obtained for each label on the first test instance \n")
print(test[0])
print(test[0].multilab_dom())
