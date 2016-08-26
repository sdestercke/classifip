'''
Created on 25 august 2016

@author: Sebastien Destercke

About how to use the Naive Credal Classifier Binary Relevance.

'''

print("Example of multilabel prediciton with NCC BR - data set yeast \n")

print("Data loading \n")
from classifip.dataset import arff

data=arff.ArffFile()
data.load("yeast.arff")
data.discretize(discmet='eqfreq',numint=5)
dataset='yeast'
nblab=14

# We start by creating an instance of the base classifier we want to use
print("Model creation and learning \n")
from classifip.models import nccbr
model=nccbr.NCCBR()

# Learning
model.learn(data,nblab)

# Evaluation : we can set the parametersof the classifier
test = model.evaluate([row[0:len(row)-nblab] for row in data.data[0:10]],ncc_epsilon=0.001,ncc_s_param=1)

# The output is a list of probability intervals, we can print each instance :
print("Probability intervals obtained for each label on the first test instance \n")
print test[0]


