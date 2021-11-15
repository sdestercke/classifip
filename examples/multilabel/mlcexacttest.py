'''
Created on 20 may 2019

@author: Yonatan-Carlos Carranza-Alarcon

About how to use the Imprecise multilabel chaining
'''
from classifip.dataset import arff
from classifip.models.mlc.exactncc import MLCNCCExact
import timeit

dataArff = arff.ArffFile()
dataArff.load("emotions.arff")
dataArff.discretize(discmet='eqfreq', numint=5)
nb_labels = 6

# Test instances
new_instances = [row[0:len(row) - nb_labels] for row in dataArff.data[1:2]]

# Inference exact with complexity minimal
model = MLCNCCExact()
model.learn(dataArff, nb_labels=nb_labels)

start = timeit.default_timer()
solution_exact_1 = model.evaluate(new_instances, ncc_epsilon=0.001, ncc_s_param=2)
print('Solution with minimum complexity:', solution_exact_1, timeit.default_timer() - start)
start = timeit.default_timer()
solution_exact_2 = model.evaluate_exact(new_instances, ncc_epsilon=0.001, ncc_s_param=2)
print('Solution with maximum complexity:', solution_exact_2, timeit.default_timer() - start)

# exec('def foo(a, b):  return a + b')
# print(foo(1, 2))

