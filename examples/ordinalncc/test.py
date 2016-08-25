'''
Created on 25 august 2016

@author: Sebastien Destercke

About how to use the Ordinal regression classifier.

'''

import classifip

# We start by creating an instance of the base classifier we want to use
print("Example of Ordinal regression learning - data set LEV \n")
model = classifip.models.nccof.NCCOF()
dataArff= classifip.dataset.arff.ArffFile()

# Training / Evaluating data set
dataArff.load('LEV_eqfreq_dis.arff')

# Learning
model.learn(dataArff)

# Evaluation : we can set the parameters native to the base classifier 
test = model.evaluate([dataArff.data[2]],ncc_epsilon=0.001,ncc_s_param=[6])

# The output is a list of BinaryTree, we can print each instance :
print("P-box obtained for the tested instance \n")
print test[0][0]
print("\nPrediction using maximality criterion with 0/1 costs \n")
print test[0][0].getmaximaldecision()
print("\nIndex of Prediction using maximin criterion with 0/1 costs \n")
print test[0][0].getmaximindecision()


