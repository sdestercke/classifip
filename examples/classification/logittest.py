'''
Created on 26 jan 2021

@author: Yonatan-Carlos Carranza-Alarcon

Imprecise Binary Logistic regression (using glmnet_python)
'''

from classifip.models.logit import BinaryILogisticLasso
from classifip.dataset.uci_data_set import export_data_set
import numpy as np

# We start by creating an instance of the base classifier we want to use
print("Example of Imprecise Binary Logistic classifier - Data set IRIS \n")
model = BinaryILogisticLasso(DEBUG=True)
data = export_data_set('iris.data')

# selecting just two class for binary classification
data = data.loc[data['4'].isin(['Iris-setosa', 'Iris-virginica'])]

# Learning
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].tolist()
model.learn(X=X, y=y)

# Evaluation : we can set the method for minimize convex problem with quadratic
test = model.evaluate(test_dataset=X[np.array([2, 80]), :])

# # The output is a list of probability intervals, we can print each instance :
print("\nInterval of probabilities of first instance\n")
print(test[0])
# print(model.get_bound_cond_probability())

# The output is a list of probability intervals, we can print each instance :
print("\nPrediction using maximality decision vs ground-truth observed\n")
print(model.get_maximality_from_credal(test[0]), y[2])
