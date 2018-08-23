import classifip
import random
import sys
from classifip.models.ncclr import inference_ranking_csp

model = classifip.models.ncclr.NCCLR()
dataArff= classifip.dataset.arff.ArffFile()
dataArff.load("/Users/salmuz/Downloads/datasets/iris_dense.xarff")
seed = random.randrange(sys.maxsize)
max_ncc_s_param = 10
print("Seed generated for system is %5d." % seed)
for nb_int in range(5,11):
    print("Number interval for discreteness %5d." % nb_int)
    dataArff.discretize(discmet="eqfreq", numint=nb_int)
    training, testing = classifip.evaluation.train_test_split(dataArff, test_pct=0.2, random_seed=seed)
    for ncc_imprecise in range(2, max_ncc_s_param + 1):
        print("Level imprecision %5d." % ncc_imprecise)
        cv_kFold = classifip.evaluation.k_fold_cross_validation(dataArff, 10)
        for set_train, set_test in cv_kFold:
            model.learn(set_train)
            evaluation = model.evaluate(set_test.data, ncc_s_param=ncc_imprecise)
            inference_ranking_csp(evaluation)

#  model.learn(dataArff)
# test = model.evaluate([dataArff.data[2]], ncc_s_param=8)
# print(dataArff.data[2])
# model.ranking(test)
#