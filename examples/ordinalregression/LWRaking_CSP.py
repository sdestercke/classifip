import classifip
import numpy as np

# We start by creating an instance of the base classifier we want to use
print("Example of Ordinal regression learning - data set LEV \n")
# model = classifip.models.nccof.NCCOF()
model = classifip.models.ncclr.NCCLR()
dataArff= classifip.dataset.arff.ArffFile()
dataArff.load("/Users/salmuz/Downloads/datasets/iris_dense.xarff")
dataArff.discretize(discmet="eqfreq", numint=10)

# Learning
model.learn(dataArff)

# Evaluation : we can set the parameters native to the base classifier
test = model.evaluate([dataArff.data[2]], ncc_s_param=8)

print("Tested instance")
print(dataArff.data[2])

print("\nP-box obtained for the tested instance of classifier 2")
print(test[0][0])

print("\nIndex of Prediction using maximal criterion with 0/1 costs")
print(test[0][0].getmaximaldecision())

print("\nIndex of Prediction using maximal criterion with L1 costs")
# Matrix utilities for ranking
ranking_utility = np.array([[2, 1, 0], [1, 2, 1], [0, 1, 2]])
for rk_test in test[0]:
    print(rk_test.getmaximaldecision(ranking_utility))

print("\nConstraint Satisfy Problem for making inference(reduction/propagation)")
classifip.models.ncclr.inference_ranking_csp(test)