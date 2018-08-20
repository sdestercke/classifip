

import classifip
import numpy as np

# We start by creating an instance of the base classifier we want to use
print("Example of Ordinal regression learning - data set LEV \n")
# model = classifip.models.nccof.NCCOF()
model = classifip.models.ncclr.NCCLR()
dataArff= classifip.dataset.arff.ArffFile()
dataArff.load("/Users/salmuz/Downloads/datasets/iris_dense.xarff")
dataArff.discretize(discmet="eqfreq", numint=5)
model.learn(dataArff)


# Training / Evaluating data set
# dataArff.load('LEV_eqfreq_dis.arff')

# # Learning
# model.learn(dataArff)
#
# # Evaluation : we can set the parameters native to the base classifier
# test = model.evaluate([dataArff.data[2]], ncc_s_param=12)
#
# # The output is a list of p-boxes, we can print each instance :
# print("P-box obtained for the tested instance \n")
# print(test[0])
# plower = test[0].lproba[1]
# pupper = test[0].lproba[0]
#
# idxLower = np.where(plower <= 0.5)
# idxUpper = np.where(pupper >= 0.5)
#
# print(idxLower, idxUpper)

