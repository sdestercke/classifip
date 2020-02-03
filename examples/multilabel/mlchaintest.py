'''
Created on 20 may 2019

@author: Yonatan-Carlos Carranza-Alarcon

About how to use the Imprecise multilabel chaining
'''
from classifip.dataset import arff
from classifip.models.mlc.chainncc import MLChaining
import timeit

dataArff = arff.ArffFile()
dataArff.load("emotions.arff")
dataArff.discretize(discmet='eqfreq', numint=5)
nb_labels = 5

# Test instances
new_instances = [row[0:len(row) - nb_labels] for row in dataArff.data[10:11]]

# We start by creating a model
model = MLChaining()
model.learn(dataArff, nb_labels)  # , seed_random_label=134)

probabilities, chain = model.evaluate(new_instances, ncc_epsilon=0.001, ncc_s_param=1)

print("Probability intervals obtained for each label on the first test instance \n")
print(probabilities[0])

print("Label true", dataArff.data[0:1][0][-nb_labels:], '\n')
print("Label predicts: ", chain, '\n')

