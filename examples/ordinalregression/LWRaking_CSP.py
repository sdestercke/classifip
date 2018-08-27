import classifip
import numpy as np

# We start by creating an instance of the base classifier we want to use
print("Example of Ordinal regression learning - data set LEV \n")
# model = classifip.models.nccof.NCCOF()
model = classifip.models.ncclr.NCCLR()
dataArff= classifip.dataset.arff.ArffFile()
dataArff.load("/Users/salmuz/Downloads/datasets/iris_dense.xarff")
dataArff.discretize(discmet="eqfreq", numint=5)

# Learning
model.learn(dataArff)

# Evaluation : we can set the parameters native to the base classifier
test = model.evaluate([dataArff.data[100]], ncc_s_param=8)

print("Tested instance")
print(dataArff.data[100])

print("\nP-box obtained for the tested instance of classifier 2")
for clazz, classifier in test[0].items():
    print("\nClassifier label-wise ranking decomposition %s" % clazz)
    print(classifier)

print("\nIndex of Prediction using maximal criterion with 0/1 costs")
print(test[0]['L1'].getmaximaldecision())

print("\nIndex of Prediction using maximal criterion with L1 costs")
# Matrix utilities for ranking
ranking_utility = np.array([[2, 1, 0], [1, 2, 1], [0, 1, 2]])
for clazz, classifier in test[0].items():
    print(classifier.getmaximaldecision(ranking_utility))

print("\nConstraint Satisfy Problem for making inference(reduction/propagation)")
inference_csp = classifip.models.ncclr.inference_ranking_csp(test)
if inference_csp is None : print("\nWithout response CSP ranking.")
else: print("\nCSP ranking response:", inference_csp)