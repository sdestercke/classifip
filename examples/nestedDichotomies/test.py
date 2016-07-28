'''
Created on 22 juin 2016

@author: Gen

About how to use the Nested Dichotomies classifier.

'''

import classifip

# We start by creating an instance of the base classifier we want to use  
base = classifip.models.ncc.NCC()
dataArff= classifip.dataset.arff.ArffFile()

# Training / Evaluating data set
dataArff.load('LEV_eqfreq_dis.arff')

# Initialization of the Nested Dichotomies classifier
classifier = classifip.models.nestedDichotomies.NestedDichotomies(classifier=base,label=dataArff.attribute_data['class'])

# Build the dichotomy structure. For more options, see the examples of the class BinaryTree
classifier.build()

# Learning
classifier.learn(dataArff)

# Evaluation : we can set the parameters native to the base classifier 
test = classifier.evaluate([dataArff.data[2]],ncc_epsilon=0.001,ncc_s_param=[2],maxi=False)

# The output is a list of BinaryTree, we can print each instance :

test[0].printProba()

# that we can convert to an interval-valued probabilities
print test[0].toIntervalsProbability()

# The decision is taken by computing lower expectation, see the examples of CredalSet
print test[0].getmaximaldecision()


