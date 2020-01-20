'''
Created on 20 may 2019

@author: Yonatan-Carlos Carranza-Alarcon

About how to use the Imprecise multilabel chaining
'''
from classifip.dataset import arff

dataArff = arff.ArffFile()
dataArff.load("yeast.arff")
dataArff.discretize(discmet='eqfreq', numint=5)
dataset='yeast'

# We start by creating a model
from classifip.models.mlcncc import MLCNCC
model = MLCNCC()
nblab = 14
model.learn(dataArff, nblab) #, seed_random_label=134)
probs, chain = model.evaluate([row[0:len(row) - nblab] for row in dataArff.data[0:1]], ncc_epsilon=0.001, ncc_s_param=1)

print("Probability intervals obtained for each label on the first test instance \n")
print(probs[0])

print("Label predicts")
print(chain, '\n')

print("Label true")
print(dataArff.data[0:1][0][-nblab:], '\n')
