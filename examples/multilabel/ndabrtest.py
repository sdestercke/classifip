from classifip.dataset import arff
import numpy as np
from classifip.models.mlc import ndabr
from classifip.models.mlc import knnnccbr
from classifip.models.mlc.mlcncc import MLCNCC

print("Example of multilabel prediciton with NCC BR - data set yeast \n")

print("Data loading \n")

dataArff = arff.ArffFile()
dataArff.load("labels5.arff")
dataset = 'labels5'
nblab = 5

# We start by creating an instance of the base classifier we want to use
print("Model creation and learning \n")


def normalize(dataArff, n_labels, method='minimax'):
    from classifip.utils import normalize_minmax
    np_data = np.array(dataArff.data, dtype=np.float64)
    np_data = np_data[..., :-n_labels]
    if method == "minimax":
        np_data = normalize_minmax(np_data)
    else:
        from sklearn import preprocessing
        np_data = preprocessing.scale(np_data)
        print("testing:", np.mean(np_data[:, 0]), np.var(np_data[:, 0]))
    dataArff.data = [np_data[i].tolist() + dataArff.data[i][-n_labels:]
                     for i in range(len(np_data))]


print("old", dataArff.data[0])
normalize(dataArff, nblab, "none")
print("new", dataArff.data[0])

model = ndabr.NDABR()
model.learn(dataArff, nblab)
print(model.evaluate([row[0:len(row) - nblab] for row in dataArff.data[0:1]]))
print(dataArff.data[0][-nblab:])

dataArff_disc = dataArff.make_clone()
dataArff_disc.discretize(discmet='eqfreq', numint=5)
model = knnnccbr.KNN_NCC_BR()
model.learn(dataArff, nblab, learn_disc_set=dataArff_disc)
prd = model.evaluate([(dataArff.data[0], dataArff_disc.data[0])], k=5, ncc_s_param=0.5)
skeptic, precise, precise_proba = prd[0]
print(skeptic.multilab_dom())
print(precise.multilab_dom())
print(dataArff.data[0][-nblab:])
