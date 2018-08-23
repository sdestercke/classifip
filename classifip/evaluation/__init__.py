from . import costMatrix

import random

def k_fold_cross_validation(data, K, randomise=False, random_seed=None, structured=False):
    """
    Generates K (training, validation) pairs from the items in X.
    Each pair is a partition of X, where validation is an iterable
    of length len(X)/K. So each training iterable is of length (K-1)*len(X)/K.

    :param data: the observed class value
    :type data: :class:`~classifip.dataset.arff.ArffFile`
    :param K: number of folds
    :type K: integer
    :param randomise: randomise or not data set before splitting
    :type randomise: boolean
    :param random_seed: set the seed for the randomisation to reproduce
        identical splits if needed 
    :type random_seed: integer
    :param structured: evenly split between output class (not implemented)
    :type structured: boolean 
    :returns: iterable over training/evluation pairs
    :rtype: list of :class:`~classifip.dataset.arff.ArffFile`
    
    ..todo::
    
        * implement the structured k-fold validation
    """
    if randomise:
        import random
        if random_seed != None:
            random.seed(random_seed)
        random.shuffle(data.data)
    datatr = data.make_clone()
    datatst = data.make_clone()
    for k in range(K):
        datatr.data = [x for i, x in enumerate(data.data) if i % K != k]
        datatst.data = [x for i, x in enumerate(data.data) if i % K == k]
        yield datatr, datatst


def train_test_split(data, test_pct = 0.5, random_seed=None):
    """
       Generates partition (training, testing) pairs from the items in X
    """
    training = data.make_clone()
    if random_seed is not None:
        random.seed(random_seed)
        random.shuffle(training.data)
    testing = training.make_clone()
    idx_end_train = int(len(data.data) * (1 - test_pct))
    training.data = training.data[:idx_end_train]
    testing.data = testing.data[idx_end_train:]
    return training, testing
