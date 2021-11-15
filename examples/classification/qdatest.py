'''
Created on 18 may 2018

@author: Yonatan-Carlos Carranza-Alarcon

Imprecise Gaussian Discriminant

multi-classes classification
'''

from classifip.models.qda import LinearDiscriminant
from classifip.dataset.uci_data_set import export_data_set

# We start by creating an instance of the base classifier we want to use
print("Example of Imprecise Linear Discriminant Analyse for multi-classes - Data set IRIS \n")
model = LinearDiscriminant(solver_matlab=False, DEBUG=False)
data = export_data_set('iris.data')

# Learning
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].tolist()
model.learn(X=X, y=y, ell=5)

# Evaluation : we can set the method for minimize convex problem with quadratic
test = model.evaluate(query=X[2], method="quadratic")

# The output is a list of probability intervals, we can print each instance :
print("\nPrediction using interval dominance criterion with 0/1 costs + quadratic method\n")
print(test)
print(model.get_bound_cond_probability())

# Evaluation : we can set the method for minimize convex problem with non-linear
test = model.evaluate(query=X[2], method="nonlinear")

# The output is a list of probability intervals, we can print each instance :
print("\nPrediction using interval dominance criterion with 0/1 costs + nonlinear method\n")
print(test)
print(model.get_bound_cond_probability())


