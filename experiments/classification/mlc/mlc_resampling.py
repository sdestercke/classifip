from mlc_common import init_dataset, get_nb_labels_class
from classifip.evaluation import train_test_split
import random, numpy as np


def generate_seeds(nb_seeds):
    return [random.randrange(pow(2, 20)) for _ in range(nb_seeds)]


def generate_name(name, resampling, pct_test):
    test_full_name = "%s_test_%s_%s.arff" % (name, resampling, pct_test)
    train_full_name = "%s_train_%s_%s.arff" % (name, resampling, pct_test)
    return test_full_name, train_full_name


def saving_data_sets(data_learning, out_path, pct_test, file_train_name, file_test_name, seed, i_sampling):
    is_valid = True
    is_valid_count = 0
    while is_valid:
        data_training, data_test = \
            train_test_split(data_learning, test_pct=pct_test, random_seed=seed)
        # testing data set saving
        data_test.comment = "Testing re-sampling generation at " + str(pct_test) + "%."
        data_test.comment += "With seed " + str(seed) + "."
        f_write = open(out_path + file_test_name, "w")
        f_write.write(data_test.write())
        f_write.close()
        # training data set saving
        data_training.comment = "Training re-sampling generation at " + str(1 - pct_test) + "%."
        data_training.comment += "With seed " + str(seed) + "."
        f_write = open(out_path + file_train_name, "w")
        f_write.write(data_training.write())
        f_write.close()
        # valid data with two class minimal
        nb_labels = get_nb_labels_class(data_training)
        _data = np.array(data_training.data)
        is_valid = False
        for i in range(nb_labels):
            values, counts = np.unique(_data[:, -(i+1)], return_counts=True)
            if len(np.unique(values)) != 2 or not np.all(counts > 1):
                is_valid = True
                break
        seed = generate_seeds(1)[0]
        is_valid_count += 1
    if is_valid_count > 1:
        print("[new-SEEDs-TEST] ", pct_test, i_sampling, seed, flush=True)


def create_dataset_by_percentage(in_path, out_path, pct_test, nb_samplings, dataset="emotions",
                                 scaling=True, nb_labels=None):
    seeds = generate_seeds(nb_samplings)
    data_learning, _ = init_dataset(in_path, remove_features=[], scaling=scaling, nb_labels=nb_labels)
    print("[SEEDs-GENERATION-PCT-TEST] ", pct_test, seeds, flush=True)
    for i in range(nb_samplings):
        str_pct_training = str(int(round((1 - pct_test) * 100)))
        file_test_name, file_train_name = generate_name(dataset, str(i + 1), str_pct_training)
        saving_data_sets(data_learning, out_path, pct_test, file_train_name, file_test_name, seeds[i], i)


NB_SAMPLINGS = 50
dataset = "medical"
in_path = "/Users/salmuz/Downloads/datasets_mlc/" + dataset + ".arff"
out_path = "/Users/salmuz/Downloads/"
for pct in np.arange(0.1, 1, 0.1):
    create_dataset_by_percentage(in_path, out_path, pct, NB_SAMPLINGS, dataset=dataset,
                                 scaling=False, nb_labels=45)
