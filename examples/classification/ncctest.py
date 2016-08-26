'''
Created on 25 august 2016

@author: Sebastien Destercke

About how to use the Naive Credal Classifier.

'''

import classifip

# We start by creating an instance of the base classifier we want to use
print("Example of Ordinal regression learning - data set LEV \n")
model = classifip.models.ncc.NCC()
dataArff= classifip.dataset.arff.ArffFile()

# Training / Evaluating data set
dataArff.load('LEV_eqfreq_dis.arff')

# Learning
model.learn(dataArff)

# Evaluation : we can set the parametersof the classifier
test = model.evaluate([dataArff.data[2]],ncc_epsilon=0.001,ncc_s_param=3)

# The output is a list of probability intervals, we can print each instance :
print("Probability intervals obtained for the tested instance \n")
print test[0]
print("\nPrediction using interval dominance criterion with 0/1 costs \n")
print test[0].getintervaldomdecision()
print("\nIndex of Prediction using maximin criterion with 0/1 costs \n")
print test[0].getmaximindecision()


